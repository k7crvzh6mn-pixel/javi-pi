import os
import time
import random
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openwakeword.model import Model
from faster_whisper import WhisperModel
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
WAKE_WORD_MODEL    = "/home/trips0007/javi-pi/venv/lib/python3.13/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
PIPER_BIN          = "/home/trips0007/javi-pi/tts/piper/piper"
PIPER_MODEL        = "/home/trips0007/javi-pi/tts/en_US-amy-medium.onnx"
RESPEAKER_NAME     = "reSpeaker"
SAMPLE_RATE        = 16000
WAKE_THRESHOLD     = 0.5
WHISPER_MODEL      = "tiny"
BT_WARMUP_MS       = 400
FOLLOW_UP_SECS     = 10   # seconds to keep listening after a response
SILENCE_SECS       = 1.5  # seconds of quiet that ends a recording
ENERGY_THRESHOLD   = 500  # RMS threshold for speech detection
MAX_RECORD_SECS    = 12

SLEEP_PHRASES = {"nevermind", "never mind", "go to sleep", "goodbye", "that's all", "stop listening"}

GREETINGS = [
    "What do you need, Charles?",
    "What's up, Charles?",
    "Yeah, Charles?",
    "What can I do for you, Charles?",
    "What's up?",
    "Go ahead.",
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

# ── Audio helpers ─────────────────────────────────────────────────────────────
def find_respeaker():
    for i, d in enumerate(sd.query_devices()):
        if RESPEAKER_NAME.lower() in d["name"].lower():
            return i
    raise RuntimeError("ReSpeaker not found — is it plugged in?")

def record_with_vad(device_index, max_seconds=MAX_RECORD_SECS):
    """Record until silence is detected after speech, or max_seconds is hit."""
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
    padded = np.concatenate([warmup, data])
    wav.write(tmp, rate, padded)
    subprocess.run(["pw-play", tmp], stderr=subprocess.DEVNULL, check=True)
    os.unlink(tmp)

def is_simple_question(text):
    words = text.split()
    if len(words) <= 6:
        return True
    simple = ["what is", "what's", "who is", "who's", "how much", "how many",
              "what time", "plus", "minus", "times", "divided", "spell"]
    return any(p in text.lower() for p in simple)

def listen_for_wake_word(device_index):
    chunk_size = 1280
    print("Listening for 'Hey Jarvis'...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk_size, device=device_index) as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            prediction = oww.predict(chunk.flatten().astype(np.int16))
            score = list(prediction.values())[0]
            if score >= WAKE_THRESHOLD:
                print(f"  Wake word detected (score={score:.2f})")
                oww.reset()
                return

# ── Conversation handler ──────────────────────────────────────────────────────
def ask_claude(text):
    conversation.append({"role": "user", "content": text})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=(
            "You are Jarvis, a helpful home assistant on a Raspberry Pi owned by Charles. "
            "Keep answers concise — you're being spoken aloud. "
            "Do not use markdown, bullet points, or special characters."
        ),
        messages=conversation,
    )
    reply = response.content[0].text
    conversation.append({"role": "assistant", "content": reply})
    return reply

def handle_query(text):
    """Transcribe, check for sleep, optionally say filler, then respond."""
    if not text:
        return False  # nothing heard

    print(f"  You said: {text}")

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
    device_index = find_respeaker()
    print(f"ReSpeaker found at device index {device_index}")
    speak("Jarvis is ready.")

    while True:
        try:
            listen_for_wake_word(device_index)

            # Small pause so "hey jarvis" finishes before we start recording
            time.sleep(0.4)

            # Record what comes right after the wake word
            audio = record_with_vad(device_index)
            text  = transcribe(audio)

            if not text:
                # No question — greet and wait for follow-up
                speak(random.choice(GREETINGS))
                audio = record_with_vad(device_index, max_seconds=FOLLOW_UP_SECS)
                text  = transcribe(audio)

            result = handle_query(text)
            if result == "sleep" or not result:
                continue

            # Follow-up window — keep listening without wake word
            while True:
                audio = record_with_vad(device_index, max_seconds=FOLLOW_UP_SECS)
                if len(audio) < SAMPLE_RATE * 0.3:
                    print("  Follow-up window expired, back to listening.")
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
