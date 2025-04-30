---
license: mit
datasets:
- benax-rw/my_kinyarwanda_dataset
language:
- rw
metrics:
- wer
base_model: openai/whisper-small
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags:
- kinyarwanda
- asr
- whisper
- low-resource
- fine-tuning
- benax-technologies
- transformers
- torchaudio
- speech-recognition
model-index:
- name: KinyaWhisper
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: KinyaWhisper Custom Dataset
      type: custom
      config: kinyarwanda
    metrics:
    - name: WER
      type: wer
      value: 51.85
---

## 🗣️ KinyaWhisper
KinyaWhisper is a fine-tuned version of OpenAI’s Whisper model for automatic speech recognition (ASR) in Kinyarwanda. It was trained on 102 manually labeled .wav files and serves as a reproducible baseline for speech recognition in low-resource, indigenous languages.

## 🔧 Usage

To run inference on your own audio files using the fine-tuned KinyaWhisper model:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned KinyaWhisper model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("your_audio.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)
```

## 🏋️ Taining Details
•	Model: openai/whisper-small
•	Epochs: 80
•	Batch size: 4
•	Learning rate: 1e-5
•	Optimizer: Adam
•	Final loss: 0.00024
•	WER: 51.85%

## ⚠️Limitations
The model was trained on a small dataset (102 samples). It performs best on short, clear Kinyarwanda utterances and may struggle with longer or noisy audio. This is an early-stage educational model, not yet suitable for production use.

## 📚 Citation

If you use this model, please cite:

```bibtex
@misc{baziramwabo2025kinyawhisper,
  author       = {Gabriel Baziramwabo},
  title        = {KinyaWhisper: Fine-Tuning Whisper for Kinyarwanda ASR},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/benax-rw/KinyaWhisper}},
  note         = {Version 1.0}
}
```
## 📬 Contact
Maintained by Gabriel Baziramwabo. 
✉️ gabriel@benax.rw
🔗 https://benax.rw
