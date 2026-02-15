# === MORAL STIMULI CHATBOT (REPLIT VERSION) ===
# This version has been modified from the AWS deployment
# (which used EC2, S3, and Cloudflare Tunnels)
# to run entirely inside Replit using local CSV logging.

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import datetime, os, time, csv, json

# --- GLOBAL PROMPT & INPUT DEFINITIONS ---

# --- Customize your own system prompt and initial user input template --- 

SYSTEM_PROMPT_TEMPLATE = (
    "You are a nonjudgmental assistant helping the user reflect. "
    "Keep replies short (3–5 sentences)."
)

INITIAL_USER_INPUT_TEMPLATE = "Help me decide what I should do. {stimuli}" 

# --- Flask setup (unchanged structure) ---
app = Flask(__name__)
CORS(app)

# --- MODEL & SESSION CONFIG ---
# Replit-friendly environment variable call (was hardcoded or pulled from AWS env vars)
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SESSION_TIMEOUT_SECONDS = 15 * 60
all_sessions = {}

# --- OpenAI client ---
# CHANGED: reads key from Replit Secrets instead of EC2 environment or local file
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def trim_history(messages, max_exchanges=10):
    """Keep only recent conversation turns."""
    if not messages:
        return []
    system_msg = messages[0]
    convo = messages[1:]
    if len(convo) > 2 * max_exchanges:
        convo = convo[-2 * max_exchanges:]
    return [system_msg] + convo


@app.route("/")
def serve_ui():
    # CHANGED: simple static file serve (no AWS S3 hosting)
    return send_file("frontend.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_input = (data.get("message") or "").strip()
    response_id = data.get("response_id", "none")
    participant_id = data.get("participant_id", "anonymous")
    stimuli = data.get("stimuli", "default_stimuli")

    print(f"{response_id}: {user_input} (Stimuli: {stimuli})")

    session_key = (participant_id, response_id)
    now = time.time()

    # --- Session initialization ---
    # Replit keeps everything in memory during runtime
    if (session_key not in all_sessions
            or now - all_sessions[session_key]["last_active"]
            > SESSION_TIMEOUT_SECONDS):
        system_prompt = (
            "You are a nonjudgmental assistant helping the user reflect on this moral stimuli. "
            "Keep replies short (3–5 sentences), and end with a gentle reflective question."
        )
        messages = [{"role": "system", "content": system_prompt}]
        all_sessions[session_key] = {"messages": messages, "last_active": now, "system_prompt": system_prompt}

    messages = all_sessions[session_key]["messages"]
    system_prompt = all_sessions[session_key].get("system_prompt", messages[0]["content"] if messages and messages[0].get("role") == "system" else "")

    # CHANGED: simple reset command for starting conversation
    if user_input.upper() == "START_CONVERSATION":
        user_input = f"Help me decide what I should do. {stimuli}"

    # GPT response logic (unchanged)
    messages.append({"role": "user", "content": user_input})
    messages = trim_history(messages)
    bot_reply = "[Error: no response]"

    try:
        resp = client.chat.completions.create(model=MODEL_NAME,
                                              messages=messages)
        bot_reply = resp.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": bot_reply})
    except Exception as e:
        bot_reply = f"Error generating response: {e}"

    all_sessions[session_key]["messages"] = messages
    all_sessions[session_key]["last_active"] = now

    # --- LOGGING ---
    # CHANGED: replaced AWS S3 upload with local CSV logging for Replit
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": MODEL_NAME,
        "participant_id": participant_id,
        "response_id": response_id,
        "stimuli": stimuli,
        "system_prompt": system_prompt,
        "user_input": user_input,
        "bot_reply": bot_reply,
    }

    csv_filename = "chatlog.csv"
    fieldnames = [
        "timestamp",
        "model",
        "participant_id",
        "response_id",
        "stimuli",
        "system_prompt",
        "user_input",
        "bot_reply",
    ]
    header_line = ",".join(fieldnames)
    file_exists = os.path.isfile(csv_filename)
    file_empty = not file_exists or (os.path.getsize(csv_filename) == 0)
    needs_header = file_empty
    if file_exists and not file_empty:
        with open(csv_filename, "r", newline="", encoding="utf-8") as f:
            first_line = (f.readline() or "").strip()
        needs_header = not first_line.startswith("timestamp,")

    if needs_header and file_exists and not file_empty:
        with open(csv_filename, "r", newline="", encoding="utf-8") as f:
            existing = f.read()
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            f.write(header_line + "\n")
            f.write(existing)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(record)
    else:
        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(record)

    # CHANGED: removed S3 write confirmation printout
    return jsonify({"response": bot_reply})


if __name__ == "__main__":
    # CHANGED: runs on port 5000 (required for Replit)
    app.run(host="0.0.0.0", port=5000)
