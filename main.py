# main.py
import sys
from asr import transcribe
from nlp import load_qa, match_question
from tts import speak

def main(audio_path):
    print("🎧 Transcribing…")
    text = transcribe(audio_path)
    print("🗣️ You said:", text)

    qa = load_qa()
    answer = match_question(text, qa)
    print("🤖 Bot answers:", answer)

    print("🔊 Generating speech…")
    out = speak(answer)
    print("✅ Spoken reply saved to", out)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/audio.wav")
        sys.exit(1)
    main(sys.argv[1])
