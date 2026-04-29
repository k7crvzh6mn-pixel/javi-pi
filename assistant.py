import os
import json
import time
import re
import random
import tempfile
import subprocess
import threading
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openwakeword.model import Model
from faster_whisper import WhisperModel
import anthropic

XVF_HOST = "/home/trips0007/xvf_host/xvf_host"

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
WAKE_WORD_MODEL   = "/home/trips0007/javi-pi/venv/lib/python3.13/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
PIPER_BIN         = "/home/trips0007/javi-pi/tts/piper/piper"
PIPER_MODEL       = "/home/trips0007/javi-pi/tts/en_US-amy-medium.onnx"
HINTS_FILE        = "/home/trips0007/javi-pi/whisper_hints.txt"
RESPEAKER_NAME    = "reSpeaker"
SAMPLE_RATE       = 16000
WAKE_THRESHOLD    = 0.5
WHISPER_MODEL     = "tiny"
BT_WARMUP_MS      = 600
BT_TAIL_MS        = 400
POST_SPEAK_DELAY  = 0.6
FOLLOW_UP_SECS    = 10
SILENCE_SECS      = 1.5
ENERGY_THRESHOLD  = 1200
MIN_SPEECH_SECS   = 0.6
MAX_RECORD_SECS   = 12
HOME_CITY         = "Corpus Christi, TX"
BT_SPEAKER_MAC    = "E8:09:59:10:A8:BD"

SLEEP_PHRASES   = {"nevermind", "never mind", "go to sleep", "goodbye", "that's all", "stop listening"}
RESET_PHRASES   = {"start fresh", "start over", "new conversation", "clear history", "reset"}
STATUS_PHRASES  = {"what can you do", "what are your capabilities", "help", "list your features"}

GREETINGS = [
    "What's up?",
    "Yeah?",
    "Go ahead.",
    "What do you need?",
    "I'm listening.",
    "What's up, Charles?",
]

THINKING_PHRASES = [
    "Let me check that.",
    "Give me a sec.",
    "Let me look that up.",
    "One moment.",
    "On it.",
]

# ── Init ──────────────────────────────────────────────────────────────────────
client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
whisper       = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
oww           = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
conversation  = []
_device_index = 0
_timers       = []  # active timer threads
_keepalive    = None
_led_available = os.path.exists(XVF_HOST)

# ── Whisper hint learning ─────────────────────────────────────────────────────
_BASE_HINTS = (
    "Lakers, Celtics, Warriors, Bulls, Heat, Knicks, Spurs, Nets, Clippers, Nuggets, "
    "Suns, Mavericks, Bucks, Sixers, Raptors, Rockets, Pistons, Pacers, "
    "S&P 500, NASDAQ, Dow Jones, Apple, Tesla, Google, Amazon, Microsoft, NVIDIA, "
    "weather, Corpus Christi, Jarvis, nevermind, go to sleep, timer, volume"
)

def load_hints() -> str:
    base = _BASE_HINTS
    if os.path.exists(HINTS_FILE):
        extras = open(HINTS_FILE).read().strip()
        if extras:
            base += ", " + extras.replace("\n", ", ")
    return base

def save_hint(word: str):
    word = word.strip().strip(".,!?")
    if not word:
        return
    existing = open(HINTS_FILE).read() if os.path.exists(HINTS_FILE) else ""
    if word.lower() not in existing.lower():
        with open(HINTS_FILE, "a") as f:
            f.write(word + "\n")
        print(f"  Hint saved: {word}")

# ── ReSpeaker LED ─────────────────────────────────────────────────────────────
# XVF3800 LED controlled via xvf_host CLI (raw USB ctrl_transfer not supported by firmware)
# LED_EFFECT: 0=off, 3=single color, 4=DOA direction tracking
def _init_led():
    if _led_available:
        _xvf("LED_EFFECT", 0)
        print("  LED: initialized")
    else:
        print(f"  LED: xvf_host not found at {XVF_HOST}")

def _xvf(*args):
    try:
        subprocess.run(
            [XVF_HOST, "-u", "usb"] + [str(a) for a in args],
            capture_output=True, timeout=3,
            cwd="/home/trips0007/xvf_host"
        )
    except Exception as e:
        print(f"  LED error: {e}")

def led(state: str):
    if not _led_available:
        return
    if state == "listen":
        _xvf("LED_EFFECT", 4)            # DOA spinning — follows your voice
    elif state == "think":
        _xvf("LED_EFFECT", 3)            # solid color while processing
        _xvf("LED_COLOR", 0x000060)      # dim blue
    elif state == "off":
        _xvf("LED_EFFECT", 0)            # all off

# ── Tools ─────────────────────────────────────────────────────────────────────
def get_weather(city: str = HOME_CITY) -> str:
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=3"
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read().decode().strip()
    except Exception as e:
        return f"Weather unavailable: {e}"

def get_stock(symbol: str) -> str:
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}?interval=1d&range=2d"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        meta   = data["chart"]["result"][0]["meta"]
        price  = meta.get("regularMarketPrice", 0)
        prev   = meta.get("chartPreviousClose", price)
        change = price - prev
        pct    = (change / prev * 100) if prev else 0
        direction = "up" if change >= 0 else "down"
        return f"{symbol.upper()} is ${price:.2f}, {direction} {abs(pct):.1f}% today."
    except Exception as e:
        return f"Stock data unavailable: {e}"

def get_news(topic: str = "top") -> str:
    try:
        feeds = {
            "top":       "http://feeds.bbci.co.uk/news/rss.xml",
            "us":        "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
            "tech":      "http://feeds.bbci.co.uk/news/technology/rss.xml",
            "sports":    "http://feeds.bbci.co.uk/sport/rss.xml",
            "business":  "http://feeds.bbci.co.uk/news/business/rss.xml",
        }
        url = feeds.get(topic.lower(), feeds["top"])
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            root = ET.fromstring(r.read())
        items = root.findall(".//item")[:4]
        headlines = [item.find("title").text for item in items if item.find("title") is not None]
        return "Here are the top headlines: " + ". ".join(headlines)
    except Exception as e:
        return f"News unavailable: {e}"

def web_search(query: str) -> str:
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        answer = data.get("AbstractText") or data.get("Answer") or ""
        if not answer:
            topics = data.get("RelatedTopics", [])
            answer = topics[0].get("Text", "") if topics else ""
        return answer or "I couldn't find a clear answer for that."
    except Exception as e:
        return f"Search failed: {e}"

def get_sports_score(team: str) -> str:
    def games_on_date(day_str, team_lower):
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day_str}"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        found = []
        for game in data.get("events", []):
            if team_lower in game.get("name", "").lower():
                status      = game["status"]["type"]["description"]
                competitors = game["competitions"][0]["competitors"]
                scores      = {c["team"]["displayName"]: c["score"] for c in competitors}
                score_str   = ", ".join(f"{k} {v}" for k, v in scores.items())
                found.append(f"{game['shortName']} ({score_str}) — {status}")
        return found

    try:
        team_lower = team.lower()
        results, upcoming = [], []

        for delta in range(0, -8, -1):
            day = (date.today() + timedelta(days=delta)).strftime("%Y%m%d")
            results.extend(games_on_date(day, team_lower))

        for delta in range(1, 4):
            day = (date.today() + timedelta(days=delta)).strftime("%Y%m%d")
            upcoming.extend(games_on_date(day, team_lower))

        if not results and not upcoming:
            return f"No recent or upcoming games found for {team} in the last week."

        summary = ""
        if results:
            summary += "Recent games: " + " | ".join(results[:3])
        if upcoming:
            summary += (" Next up: " if summary else "Upcoming: ") + upcoming[0]
        return summary
    except Exception as e:
        return f"Couldn't fetch scores: {e}"

def get_datetime() -> str:
    now = datetime.now()
    return now.strftime("It's %I:%M %p on %A, %B %d, %Y.")

TOOLS = [
    {"name": "get_weather",
     "description": f"Current weather for a city. Default: {HOME_CITY}.",
     "input_schema": {"type": "object", "properties": {
         "city": {"type": "string"}}, "required": []}},
    {"name": "get_stock",
     "description": "Current stock price and daily change for a ticker symbol.",
     "input_schema": {"type": "object", "properties": {
         "symbol": {"type": "string", "description": "Ticker, e.g. AAPL, TSLA, SPY"}}, "required": ["symbol"]}},
    {"name": "get_news",
     "description": "Latest news headlines. Topics: top, us, tech, sports, business.",
     "input_schema": {"type": "object", "properties": {
         "topic": {"type": "string"}}, "required": []}},
    {"name": "web_search",
     "description": "Search the web for anything not covered by other tools.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_sports_score",
     "description": "NBA game scores and schedule for a team, last 7 days + upcoming.",
     "input_schema": {"type": "object", "properties": {
         "team": {"type": "string"}}, "required": ["team"]}},
    {"name": "get_datetime",
     "description": "Current local time and date.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
]

TOOL_FUNCTIONS = {
    "get_weather":      get_weather,
    "get_stock":        get_stock,
    "get_news":         get_news,
    "web_search":       web_search,
    "get_sports_score": get_sports_score,
    "get_datetime":     get_datetime,
}

# ── Audio ─────────────────────────────────────────────────────────────────────
# ── Speaker keepalive ─────────────────────────────────────────────────────────
def start_keepalive():
    def _loop():
        silence = np.zeros(int(44100 * 0.1), dtype=np.int16)  # 100ms of silence
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, 44100, silence)
            silent_file = f.name
        while True:
            time.sleep(90)  # every 90s to prevent auto-shutoff
            subprocess.run(["pw-play", silent_file], capture_output=True)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

def find_respeaker():
    for i, d in enumerate(sd.query_devices()):
        if RESPEAKER_NAME.lower() in d["name"].lower():
            return i
    raise RuntimeError("ReSpeaker not found — is it plugged in?")

def ensure_bluetooth():
    for attempt in range(10):
        result = subprocess.run(
            ["bluetoothctl", "info", BT_SPEAKER_MAC],
            capture_output=True, text=True
        )
        if "Connected: yes" in result.stdout:
            print("  Bluetooth connected.")
            return
        print(f"  Bluetooth not connected, attempt {attempt + 1}/10...")
        subprocess.run(["bluetoothctl", "connect", BT_SPEAKER_MAC],
                       capture_output=True, timeout=10)
        time.sleep(3)
    print("  Warning: Bluetooth speaker could not be connected.")

def record_with_vad(device_index, max_seconds=MAX_RECORD_SECS,
                    energy_threshold=ENERGY_THRESHOLD, min_speech_secs=MIN_SPEECH_SECS):
    chunk_dur      = 0.3
    chunk_samples  = int(SAMPLE_RATE * chunk_dur)
    silent_needed  = int(SILENCE_SECS / chunk_dur)
    max_chunks     = int(max_seconds / chunk_dur)
    recorded       = []
    speech_started = False
    silent_chunks  = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        device=device_index) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk    = chunk.flatten()
            energy   = np.abs(chunk).mean()
            if energy > energy_threshold:
                speech_started = True
                silent_chunks  = 0
                recorded.extend(chunk)
            elif speech_started:
                silent_chunks += 1
                recorded.extend(chunk)
                if silent_chunks >= silent_needed:
                    break

    if len(recorded) < int(SAMPLE_RATE * min_speech_secs):
        return np.array([], dtype=np.int16)
    return np.array(recorded, dtype=np.int16)

def transcribe(audio_array):
    if len(audio_array) < SAMPLE_RATE * 0.3:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio_array)
        segments, _ = whisper.transcribe(f.name, language="en", initial_prompt=load_hints())
        text = " ".join(s.text for s in segments).strip()
        os.unlink(f.name)
    return text

def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tmp],
        input=text.encode(), stderr=subprocess.DEVNULL, check=True,
    )
    rate, data = wav.read(tmp)
    warmup = np.zeros(int(rate * BT_WARMUP_MS / 1000), dtype=data.dtype)
    tail   = np.zeros(int(rate * BT_TAIL_MS / 1000), dtype=data.dtype)
    wav.write(tmp, rate, np.concatenate([warmup, data, tail]))
    for attempt in range(2):
        result = subprocess.run(["pw-play", tmp], capture_output=True)
        if result.returncode == 0:
            break
        print(f"  pw-play failed (attempt {attempt+1}), reconnecting Bluetooth...")
        subprocess.run(["bluetoothctl", "connect", BT_SPEAKER_MAC],
                       capture_output=True, timeout=10)
        time.sleep(3)
    os.unlink(tmp)
    time.sleep(POST_SPEAK_DELAY)

def set_volume(direction: str):
    sink_id = subprocess.run(
        ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
        capture_output=True, text=True
    ).stdout
    m = re.search(r"[\d.]+", sink_id)
    vol = float(m.group()) if m else 0.5
    if "up" in direction or "louder" in direction or "higher" in direction:
        new_vol = min(1.0, vol + 0.15)
        speak("Turning it up.")
    else:
        new_vol = max(0.0, vol - 0.15)
        speak("Turning it down.")
    subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{new_vol:.2f}"])

def set_timer(seconds: int, label: str = "Timer"):
    def _timer():
        time.sleep(seconds)
        speak(f"{label} is done!")
    t = threading.Thread(target=_timer, daemon=True)
    t.start()
    _timers.append(t)

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

# ── Claude ────────────────────────────────────────────────────────────────────
def ask_claude(text):
    conversation.append({"role": "user", "content": text})
    messages = list(conversation)

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=(
                f"You are Jarvis, a helpful home assistant owned by Charles in {HOME_CITY}. "
                "Responses are spoken aloud — keep them concise and conversational. "
                "No markdown, bullet points, or special characters. "
                "Don't address the user by name in every response. "
                "Use tools for any current or real-world information."
            ),
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    fn     = TOOL_FUNCTIONS.get(block.name)
                    result = fn(**block.input) if fn else "Tool not found."
                    print(f"  Tool: {block.name}({block.input}) → {result[:80]}")
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

def parse_timer(text):
    m = re.search(r"(\d+)\s*(second|minute|hour|sec|min|hr)s?", text.lower())
    if not m:
        return None, None
    n = int(m.group(1))
    unit = m.group(2)
    if unit.startswith("s"):
        return n, "seconds"
    elif unit.startswith("m"):
        return n * 60, "minutes"
    elif unit.startswith("h"):
        return n * 3600, "hours"
    return None, None

def handle_query(text):
    global conversation
    if not text:
        return False

    print(f"  You: {text}")
    low = text.lower()

    # Sleep
    if any(p in low for p in SLEEP_PHRASES):
        speak("Going to sleep. Say Hey Jarvis when you need me.")
        return "sleep"

    # Reset conversation
    if any(p in low for p in RESET_PHRASES):
        conversation = []
        speak("Done, starting fresh.")
        return True

    # Status / capabilities
    if any(p in low for p in STATUS_PHRASES):
        speak("I can check weather, stocks, news, sports scores, set timers, "
              "control volume, answer questions, and search the web.")
        return True

    # Volume
    if "volume" in low or "louder" in low or "quieter" in low or "turn it up" in low or "turn it down" in low:
        set_volume(low)
        return True

    # Timer
    if "timer" in low or "remind me in" in low:
        secs, unit = parse_timer(text)
        if secs:
            n_str = text[text.lower().find(re.search(r"\d+", text.lower()).group()):].split()[0]
            set_timer(secs, "Your timer")
            speak(f"Timer set.")
            return True

    # Learn corrections
    correction = None
    for pattern in [r"no[,]? i said (.+)", r"i meant (.+)", r"remember (.+)", r"the word is (.+)"]:
        m = re.search(pattern, low)
        if m:
            correction = m.group(1).strip()
            break
    if correction:
        save_hint(correction)
        speak(f"Got it, I'll remember that.")
        return True

    # Call Claude in parallel with thinking phrase
    reply_box = [None]
    def fetch():
        reply_box[0] = ask_claude(text)

    claude_thread = threading.Thread(target=fetch)
    claude_thread.start()

    if not is_simple_question(text):
        speak(random.choice(THINKING_PHRASES))

    claude_thread.join()
    print(f"  Jarvis: {reply_box[0]}")
    speak(reply_box[0])
    return True

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global _device_index
    _device_index = find_respeaker()
    print(f"ReSpeaker at index {_device_index}")

    ensure_bluetooth()
    start_keepalive()
    _init_led()
    speak("Jarvis is ready.")
    while True:
        try:
            # Idle — LED off while waiting for wake word
            listen_for_wake_word(_device_index)

            # Wake word heard — light up in DOA mode while recording
            led("listen")
            time.sleep(0.6)

            audio = record_with_vad(_device_index)
            text  = transcribe(audio)

            if not text:
                # Still need input — stay in listen mode, greet, record again
                speak(random.choice(GREETINGS))
                audio = record_with_vad(_device_index, max_seconds=FOLLOW_UP_SECS)
                text  = transcribe(audio)

            # Got text — solid while thinking + speaking
            led("think")
            result = handle_query(text)
            led("off")

            if result == "sleep" or not result:
                continue

            # Follow-up window — listen mode between replies
            while True:
                led("listen")
                audio = record_with_vad(_device_index, max_seconds=FOLLOW_UP_SECS,
                                        energy_threshold=800, min_speech_secs=0.4)
                if len(audio) == 0:
                    led("off")
                    print("  Follow-up window closed.")
                    break
                text = transcribe(audio)
                led("think")
                result = handle_query(text)
                led("off")
                if result == "sleep" or not result:
                    break

        except KeyboardInterrupt:
            led("off")
            print("\nShutting down.")
            break
        except Exception as e:
            led("off")
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
