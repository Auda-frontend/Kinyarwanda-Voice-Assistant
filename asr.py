# asr.py
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

# ─── Model & processor setup ────────────────────────────────────────────────
MODEL_PATH = "benax-rw/KinyaWhisper"
LANG_TOKEN_ID = 50259  # <|rw|> for Kinyarwanda
TARGET_SR = 16000

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# Remove any built-in forced_decoder_ids so we can pass our own
gen_config = model.generation_config
gen_config.forced_decoder_ids = None

def transcribe(audio_path: str) -> str:
    """
    Load an audio file, resample/mixdown to 16 kHz mono,
    then run KinyaWhisper to return the Kinyarwanda transcription.
    """
    # 1. Load & resample
    waveform, sr = torchaudio.load(audio_path)  # -> Tensor[ch, time]
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # to mono
    if sr != TARGET_SR:
        waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
    
    # 2. Prepare features
    audio_input = waveform.squeeze(0)  # shape [time]
    inputs = processor(audio_input, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])
    
    # 3. Force Kinyarwanda decode
    decoder_input_ids = torch.tensor([[LANG_TOKEN_ID]], dtype=torch.long)
    
    # 4. Generate
    predicted_ids = model.generate(
        inputs["input_features"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        generation_config=gen_config,
        max_new_tokens=5,
        no_repeat_ngram_size=1,
        suppress_tokens=[],
    )
    
    # 5. Decode & return
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
