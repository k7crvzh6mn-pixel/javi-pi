import os
import time
import tempfile
import subprocess
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
RECORD_SECONDS    = 6
WHISPER_MODEL     = "tiny"
TTS_RATE          = 22050
# Silence padding (samples) prepended to every clip to wake up Bluetooth
BT_WARMUP_MS      = 400

# ── Init ──────────────────────────────────────────────────────────────────────
client       = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
whisper      = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
oww          = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
conversation = []

def find_respeaker():
    for i, d in enumerate(sd.query_devices()):
        if RESPEAKER_NAME.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    raise RuntimeError("ReSpeaker not found — is it plugged in?")

def record_audio(seconds, device_index):
    print(f"  Recording {seconds}s...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16", device=device_index)
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
        system=(
            "You are Jarvis, a helpful home assistant on a Raspberry Pi. "
            "Keep answers concise — you're being spoken aloud. "
            "Do not use markdown, bullet points, or special characters."
        ),
        messages=conversation,
    )
    reply = response.content[0].text
    conversation.append({"role": "assistant", "content": reply})
    return reply

def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name

    # Generate speech with Piper
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tmp],
        input=text.encode(),
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Read wav, prepend silence to wake Bluetooth from low-power state
    rate, data = wav.read(tmp)
    os.unlink(tmp)
    warmup = np.zeros(int(rate * BT_WARMUP_MS / 1000), dtype=data.dtype)
    padded = np.concatenate([warmup, data])

    # Play through default PipeWire sink (Soundcore 2)
    sd.play(padded, samplerate=rate)
    sd.wait()

def listen_for_wake_word(device_index):
    chunk_size = 1280  # 80ms at 16kHz — required by openWakeWord
    print("Listening for 'Hey Jarvis'...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk_size, device=device_index) as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            prediction = oww.predict(chunk.flatten().astype(np.int16))
            score = list(prediction.values())[0]
            if score >= WAKE_THRESHOLD:
                print(f"  Wake word detected! (score={score:.2f})")
                oww.reset()
                return

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    device_index = find_respeaker()
    print(f"ReSpeaker found at device index {device_index}")
    speak("Jarvis is ready.")

    while True:
        try:
            listen_for_wake_word(device_index)
            speak("Yes?")
            audio = record_audio(RECORD_SECONDS, device_index)
            text  = transcribe(audio)
            if not text:
                speak("I didn't catch that.")
                continue
            print(f"  You said: {text}")
            reply = ask_claude(text)
            print(f"  Jarvis: {reply}")
            speak(reply)
        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
