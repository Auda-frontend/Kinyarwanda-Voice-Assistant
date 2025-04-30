# tts_vits_pipeline.py
import os
from transformers import pipeline

# ① Instantiate your MMS-VITS TTS pipeline
tts_pipe = pipeline(
    task="text-to-speech",
    model="facebook/mms-tts-kin",   # Kinyarwanda model
    framework="pt",                  # PyTorch
    device=-1                        # CPU; set to 0 for GPU
)

def speak(text: str, out_dir="spoken_outputs") -> str:
    """
    Synthesize `text` with facebook/mms-tts-kin via the HF pipeline.
    Saves a WAV file and returns its path.
    """
    os.makedirs(out_dir, exist_ok=True)
    # ② Get audio array + sample rate
    result = tts_pipe(text)
    audio_ts = result["audio"]                  # a np.ndarray shape [1, time]
    sr       = result["sampling_rate"]          # e.g. 16000 :contentReference[oaicite:0]{index=0}

    # ③ Write out as a WAV
    import soundfile as sf
    filename = f"vits-{int(os.times()[4])}.wav"
    path     = os.path.join(out_dir, filename)
    sf.write(path, audio_ts.squeeze(), samplerate=sr)
    return path

if __name__ == "__main__":
    # quick sanity check
    path = speak("Ni meza, murakoze!", lang="rw")
    print("Saved TTS to", path)
