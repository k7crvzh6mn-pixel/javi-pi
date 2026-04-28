import os
import json
import time
import random
import tempfile
import subprocess
import urllib.request
import urllib.parse
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openwakeword.model import Model
from faster_whisper import WhisperModel
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
WAKE_WORD_MODEL   = "/home/trips0007/javi-pi/venv/lib/python3.13/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
PIPER_BIN         = "/home/trips0007/javi-pi/tts/piper/piper"
PIPER_MODEL       = "/home/trips0007/javi-pi/tts/en_US-amy-medium.onnx"
RESPEAKER_NAME    = "reSpeaker"
SAMPLE_RATE       = 16000
WAKE_THRESHOLD    = 0.5
WHISPER_MODEL     = "tiny"
BT_WARMUP_MS      = 600
BT_TAIL_MS        = 400  # silence after speech so end doesn't get clipped
POST_SPEAK_DELAY  = 0.6
FOLLOW_UP_SECS    = 10
SILENCE_SECS      = 1.5
ENERGY_THRESHOLD  = 1200
MIN_SPEECH_SECS   = 0.6
MAX_RECORD_SECS   = 12
HOME_CITY         = "Corpus Christi, TX"

SLEEP_PHRASES = {"nevermind", "never mind", "go to sleep", "goodbye", "that's all", "stop listening"}

GREETINGS = [
    "What's up?",
    "Yeah?",
    "Go ahead.",
    "What do you need?",
    "I'm listening.",
    "What's up, Charles?",  # occasional only
]

THINKING_PHRASES = [
    "Let me check that.",
    "Give me a sec.",
    "Let me look that up.",
    "One moment.",
    "On it.",
]

# ── Init ──────────────────────────────────────────────────────────────────────
client       = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
whisper      = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
oww          = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
conversation = []
_device_index = 0

# ── Web tools ─────────────────────────────────────────────────────────────────
def get_weather(city: str = HOME_CITY) -> str:
    try:
        encoded = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded}?format=3"
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read().decode().strip()
    except Exception as e:
        return f"Couldn't fetch weather: {e}"

def web_search(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        answer = data.get("AbstractText") or data.get("Answer") or ""
        if not answer:
            # Fall back to top related topic
            topics = data.get("RelatedTopics", [])
            answer = topics[0].get("Text", "") if topics else ""
        return answer or "I couldn't find a clear answer for that."
    except Exception as e:
        return f"Search failed: {e}"

def get_sports_score(team: str) -> str:
    try:
        encoded = urllib.parse.quote(team)
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        games = data.get("events", [])
        for game in games:
            name = game.get("name", "").lower()
            if team.lower() in name:
                status = game["status"]["type"]["description"]
                competitors = game["competitions"][0]["competitors"]
                scores = {c["team"]["displayName"]: c["score"] for c in competitors}
                return f"{game['name']}: {scores} — {status}"
        return f"No current game found for {team}."
    except Exception as e:
        return f"Couldn't fetch scores: {e}"

# Tool definitions for Claude API
TOOLS = [
    {
        "name": "get_weather",
        "description": f"Get current weather for a city. Defaults to {HOME_CITY}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Corpus Christi, TX'"}
            },
            "required": []
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current facts, news, scores, or anything Claude doesn't know.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_sports_score",
        "description": "Get live NBA game scores for a team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Team name, e.g. 'Lakers'"}
            },
            "required": ["team"]
        }
    }
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "web_search": web_search,
    "get_sports_score": get_sports_score,
}

# ── Audio ─────────────────────────────────────────────────────────────────────
def find_respeaker():
    for i, d in enumerate(sd.query_devices()):
        if RESPEAKER_NAME.lower() in d["name"].lower():
            return i
    raise RuntimeError("ReSpeaker not found — is it plugged in?")

def record_with_vad(device_index, max_seconds=MAX_RECORD_SECS):
    chunk_dur     = 0.3
    chunk_samples = int(SAMPLE_RATE * chunk_dur)
    silent_needed = int(SILENCE_SECS / chunk_dur)
    max_chunks    = int(max_seconds / chunk_dur)
    recorded      = []
    speech_started = False
    silent_chunks  = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        device=device_index) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk    = chunk.flatten()
            energy   = np.abs(chunk).mean()
            if energy > ENERGY_THRESHOLD:
                speech_started = True
                silent_chunks  = 0
                recorded.extend(chunk)
            elif speech_started:
                silent_chunks += 1
                recorded.extend(chunk)
                if silent_chunks >= silent_needed:
                    break

    if len(recorded) < int(SAMPLE_RATE * MIN_SPEECH_SECS):
        return np.array([], dtype=np.int16)
    return np.array(recorded, dtype=np.int16)

def transcribe(audio_array):
    if len(audio_array) < SAMPLE_RATE * 0.3:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio_array)
        segments, _ = whisper.transcribe(f.name, language="en")
        text = " ".join(s.text for s in segments).strip()
        os.unlink(f.name)
    return text

def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tmp],
        input=text.encode(),
        stderr=subprocess.DEVNULL,
        check=True,
    )
    rate, data = wav.read(tmp)
    warmup = np.zeros(int(rate * BT_WARMUP_MS / 1000), dtype=data.dtype)
    tail   = np.zeros(int(rate * BT_TAIL_MS / 1000), dtype=data.dtype)
    wav.write(tmp, rate, np.concatenate([warmup, data, tail]))
    subprocess.run(["pw-play", tmp], stderr=subprocess.DEVNULL, check=True)
    os.unlink(tmp)
    time.sleep(POST_SPEAK_DELAY)

def listen_for_wake_word(device_index):
    chunk_size = 1280
    print("Listening for 'Hey Jarvis'...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk_size, device=device_index) as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            prediction = oww.predict(chunk.flatten().astype(np.int16))
            if list(prediction.values())[0] >= WAKE_THRESHOLD:
                oww.reset()
                return

# ── Claude with tool use ──────────────────────────────────────────────────────
def ask_claude(text):
    conversation.append({"role": "user", "content": text})
    messages = list(conversation)

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=(
                "You are Jarvis, a helpful home assistant on a Raspberry Pi owned by Charles, "
                f"who lives in {HOME_CITY}. Keep answers concise — responses are spoken aloud. "
                "No markdown, bullet points, or special characters. "
                "Do not address the user by name in every response — only occasionally if natural. "
                "Use the available tools for weather, sports scores, or any current information."
            ),
            tools=TOOLS,
            messages=messages,
        )

        # If Claude wants to call a tool, run it and loop back
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn     = TOOL_FUNCTIONS.get(block.name)
                    result = fn(**block.input) if fn else "Tool not found."
                    print(f"  Tool: {block.name}({block.input}) → {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            reply = response.content[0].text
            conversation.append({"role": "assistant", "content": reply})
            return reply

# ── Query handler ─────────────────────────────────────────────────────────────
def is_simple_question(text):
    if len(text.split()) <= 6:
        return True
    return any(p in text.lower() for p in
               ["what is", "what's", "who is", "who's", "how much", "how many",
                "what time", "plus", "minus", "times", "divided", "spell"])

def handle_query(text):
    if not text:
        return False
    print(f"  You: {text}")
    if any(p in text.lower() for p in SLEEP_PHRASES):
        speak("Going to sleep. Say Hey Jarvis when you need me.")
        return "sleep"
    if not is_simple_question(text):
        speak(random.choice(THINKING_PHRASES))
    reply = ask_claude(text)
    print(f"  Jarvis: {reply}")
    speak(reply)
    return True

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global _device_index
    _device_index = find_respeaker()
    print(f"ReSpeaker at index {_device_index}")
    speak("Jarvis is ready.")

    while True:
        try:
            listen_for_wake_word(_device_index)
            time.sleep(0.6)  # let "hey jarvis" finish

            audio = record_with_vad(_device_index)
            text  = transcribe(audio)

            if not text:
                speak(random.choice(GREETINGS))
                audio = record_with_vad(_device_index, max_seconds=FOLLOW_UP_SECS)
                text  = transcribe(audio)

            result = handle_query(text)
            if result == "sleep" or not result:
                continue

            # Follow-up window — no wake word needed
            while True:
                audio = record_with_vad(_device_index, max_seconds=FOLLOW_UP_SECS)
                if len(audio) == 0:
                    print("  Follow-up window closed.")
                    break
                text   = transcribe(audio)
                result = handle_query(text)
                if result == "sleep" or not result:
                    break

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
