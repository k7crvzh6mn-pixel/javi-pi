import os
import io
import time
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from openwakeword.model import Model
from faster_whisper import WhisperModel
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
WAKE_WORD_MODEL   = "/home/trips0007/javi-pi/venv/lib/python3.13/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"
RESPEAKER_NAME    = "reSpeaker"       # partial match against sounddevice device names
SAMPLE_RATE       = 16000
WAKE_THRESHOLD    = 0.5               # 0.0–1.0, higher = less sensitive
RECORD_SECONDS    = 6                 # how long to record after wake word
WHISPER_MODEL     = "tiny"            # tiny/base/small — tiny is fastest on Pi

# ── Init ──────────────────────────────────────────────────────────────────────
client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
whisper       = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
oww           = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
conversation  = []  # keeps context across turns

def find_respeaker():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if RESPEAKER_NAME.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    raise RuntimeError("ReSpeaker not found — is it plugged in?")

def record_audio(seconds, device_index):
    print(f"  Recording {seconds}s...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device_index,
    )
    sd.wait()
    return audio.flatten()

def transcribe(audio_array):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio_array)
        segments, _ = whisper.transcribe(f.name, language="en")
        text = " ".join(s.text for s in segments).strip()
        os.unlink(f.name)
    return text

def ask_claude(text):
    conversation.append({"role": "user", "content": text})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system="You are Javi, a helpful home assistant on a Raspberry Pi. Keep answers concise — you're being spoken aloud.",
        messages=conversation,
    )
    reply = response.content[0].text
    conversation.append({"role": "assistant", "content": reply})
    return reply

def speak(text):
    # Use piper TTS if available, otherwise fall back to espeak
    if os.system("which piper > /dev/null 2>&1") == 0:
        os.system(f'echo "{text}" | piper --model en_US-lessac-medium --output_raw | pw-play --rate=22050 -')
    else:
        os.system(f'espeak -s 150 "{text}"')

def listen_for_wake_word(device_index):
    chunk_size = 1280  # 80ms at 16kHz — openWakeWord expects this
    print("Listening for 'Hey Jarvis'...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk_size, device=device_index) as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            audio_chunk = chunk.flatten().astype(np.int16)
            prediction  = oww.predict(audio_chunk)
            score       = list(prediction.values())[0]
            if score >= WAKE_THRESHOLD:
                print(f"  Wake word detected! (score={score:.2f})")
                oww.reset()
                return

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    device_index = find_respeaker()
    print(f"ReSpeaker found at device index {device_index}")
    speak("Javi is ready.")

    while True:
        try:
            listen_for_wake_word(device_index)
            speak("Yeah?")
            audio   = record_audio(RECORD_SECONDS, device_index)
            text    = transcribe(audio)
            if not text:
                speak("I didn't catch that.")
                continue
            print(f"  You said: {text}")
            reply   = ask_claude(text)
            print(f"  Javi: {reply}")
            speak(reply)
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
