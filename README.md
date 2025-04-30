Kinyarwanda Voice Assistant
🗣️ ASR + 🤖 NLP + 🔊 TTS


Features
🎙️ Speech-to-text for Kinyarwanda

❓ Question matching from knowledge base

🔊 Text-to-speech responses

Quick Start

python main.py audio.wav

Files

File	Purpose
asr.py	Speech recognition
nlp.py	Question answering
tts.py	Speech synthesis

Installation

Clone repo

Install requirements:

pip install -r requirements.txt

Usage
Record audio (16kHz WAV recommended)

Run:
python main.py your_audio.wav

Get text + voice response

Output
🎧 Transcribing...  
🗣️ You said: "Muraho"  
🤖 Bot answers: "Muraho nawe"  
🔊 Generating speech...  
✅ Spoken reply saved to output.mp3  

Requirements
Python 3.8+
FFmpeg