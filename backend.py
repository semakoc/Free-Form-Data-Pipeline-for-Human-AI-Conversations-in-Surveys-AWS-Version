# Backend for AWS deployment of the free-form human-AI conversation survey pipeline.
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import datetime
import boto3
import io
import csv
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- CONFIG ---
MODEL_NAME = "gpt-4o-2024-08-06"
SESSION_TIMEOUT_SECONDS = 15 * 60  # 15 minutes

# Set up OpenAI
client = OpenAI(api_key="...")

# Set up S3 for AWS deployment
s3 = boto3.client("s3")
bucket_name = "futureus-demo-chatlogs"
log_filename = "chatlog.csv"

# In-memory session storage: {(participant_id, response_id): {"messages": [...], "last_active": timestamp}}
all_sessions = {}

def trim_history(messages, max_exchanges=10):
    if not messages:
        return messages
    system_msg = messages[0]
    convo = messages[1:]
    if len(convo) > 2 * max_exchanges:
        convo = convo[-2 * max_exchanges:]
    return [system_msg] + convo

@app.route("/")
def serve_ui():
    return send_file("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}

    user_input = data.get("message", "").strip()
    response_id = data.get("response_id", "none")
    participant_id = data.get("participant_id", "anonymous")
    stimuli = data.get("stimuli", "unknown")
    print(f" {response_id}: {user_input} (Stimuli: {stimuli})")

    session_key = (participant_id, response_id)
    now = time.time()

    # If new or timed out, reset session
    if (
        session_key not in all_sessions or
        now - all_sessions[session_key]["last_active"] > SESSION_TIMEOUT_SECONDS
    ):
        system_prompt = (
            f"You are a nonjudgmental assistant helping the user reflect on this moral stimuli: '{stimuli}'. " 
            f"Keep your responses short—just a few sentences—and easy to understand, like you're texting a thoughtful friend. "
            f"Use plain language at an 8th-grade reading level. Avoid using bullet points or lists unless they truly make things clearer. "
            f"Be supportive, reflective, and help the user think things through calmly."
        )
        messages = [{"role": "system", "content": system_prompt}]
        all_sessions[session_key] = {
            "messages": messages,
            "last_active": now
        }

    messages = all_sessions[session_key]["messages"]

    # Inject starting message (AWS deployment)
    if user_input.upper() == "START_CONVERSATION":
        user_input = f"Help me decide what I should do. {stimuli}"

    # Append and trim
    messages.append({"role": "user", "content": user_input})
    messages = trim_history(messages)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        bot_reply = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": bot_reply})
    except Exception as e:
        bot_reply = f" Error generating response: {str(e)}"

    # Update session state
    all_sessions[session_key]["messages"] = messages
    all_sessions[session_key]["last_active"] = now

    # Prepare log entry (stored in S3 for AWS deployment)
    timestamp = datetime.datetime.now().isoformat()
    new_row = [timestamp, MODEL_NAME, participant_id, response_id, stimuli, user_input, bot_reply]

    try:
        # Retrieve existing CSV from S3
        try:
            obj = s3.get_object(Bucket=bucket_name, Key=log_filename)
            existing_data = obj['Body'].read().decode('utf-8')
        except s3.exceptions.NoSuchKey:
            existing_data = ""

        # Prepare CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        if not existing_data:
            writer.writerow(["timestamp", "model", "participant_id", "response_id", "stimuli", "user_input", "bot_reply"])

        writer.writerow(new_row)
        final_data = existing_data + output.getvalue() if existing_data else output.getvalue()

        # Upload to S3 (AWS deployment)
        s3.put_object(Bucket=bucket_name, Key=log_filename, Body=final_data.encode('utf-8'))
        print(" Chat row appended to chatlog.csv")

    except Exception as e:
        print(f" CSV logging failed: {e}")

    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
