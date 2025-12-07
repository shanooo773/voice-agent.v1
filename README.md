# Voice Agent - Open Source AI Doctor

A voice-based medical consultation agent using **completely open-source** models via Hugging Face Transformers.

## 🌟 Features

- **Voice-Only Interaction**: No image processing, pure voice conversation
- **100% Open Source**: No paid APIs required
- **Local Processing**: All models run locally
- **Components**:
  - 🎤 **STT**: OpenAI Whisper (via Transformers)
  - 🧠 **LLM**: Qwen/Llama/GPT-OSS (via Transformers)
  - 🔊 **TTS**: Dia2 (via Transformers)
  - 🎨 **UI**: Gradio

## 📋 Models Used

1. **Speech-to-Text**: `openai/whisper-large-v3`
2. **Language Model**: `Qwen/Qwen2.5-7B-Instruct` (or alternatives)
3. **Text-to-Speech**: `diacritical/dia2-base` (English) or `diacritical/dia2-multilingual` (Urdu support)

## 🚀 Project Setup Guide

This guide provides step-by-step instructions to set up your project environment across macOS, Linux, and Windows.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installing System Dependencies](#installing-system-dependencies)
   - [macOS](#macos)
   - [Linux](#linux)
   - [Windows](#windows)
3. [Setting Up a Python Virtual Environment](#setting-up-a-python-virtual-environment)
   - [Using Pipenv](#using-pipenv)
   - [Using pip and venv](#using-pip-and-venv)
   - [Using Conda](#using-conda)
4. [Running the Application](#running-the-application)

## System Requirements

- Python 3.10 or higher (3.12 recommended)
- At least 8GB RAM (16GB+ recommended for larger models)
- 10GB+ free disk space for models

## Installing System Dependencies

### macOS

1. **Install Homebrew** (if not already installed):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install system dependencies:**

   ```bash
   brew install ffmpeg portaudio
   ```


### Linux
For Debian-based distributions (e.g., Ubuntu):

1. **Update the package list:**

   ```bash
   sudo apt update
   ```

2. **Install system dependencies:**

   ```bash
   sudo apt install ffmpeg portaudio19-dev python3-dev
   ```

### Windows

#### Install FFmpeg:
1. Visit the official FFmpeg download page: [FFmpeg Downloads](https://ffmpeg.org/download.html)
2. Navigate to the Windows builds section and download the latest static build.

#### Extract and Set Up FFmpeg:
1. Extract the downloaded ZIP file to a folder (e.g., `C:\ffmpeg`).
2. Add the `bin` directory to your system's PATH:
   - Search for "Environment Variables" in the Start menu.
   - Click on "Edit the system environment variables."
   - In the System Properties window, click on "Environment Variables."
   - Under "System variables," select the "Path" variable and click "Edit."
   - Click "New" and add the path to the `bin` directory (e.g., `C:\ffmpeg\bin`).
   - Click "OK" to apply the changes.

#### Install PortAudio:
1. Download the PortAudio binaries from the official website: [PortAudio Downloads](http://www.portaudio.com/download.html)
2. Follow the installation instructions provided on the website.

---

## Setting Up a Python Virtual Environment

### Using Pipenv
1. **Install Pipenv (if not already installed):**  
```
pip install pipenv
```

2. **Install Dependencies with Pipenv:** 

```
pipenv install
```

3. **Activate the Virtual Environment:** 

```
pipenv shell
```

---

### Using `pip` and `venv`
#### Create a Virtual Environment:
```
python -m venv venv
```

#### Activate the Virtual Environment:
**macOS/Linux:**
```
source venv/bin/activate
```

**Windows:**
```
venv\Scripts\activate
```

#### Install Dependencies:
```
pip install -r requirements.txt
```

---

### Using Conda
#### Create a Conda Environment:
```
conda create --name myenv python=3.11
```

#### Activate the Conda Environment:
```
conda activate myenv
```

#### Install Dependencies:
```
pip install -r requirements.txt
```

---

## Running the Application

### Option 1: Run Gradio UI (Recommended)

```bash
python gradio_app.py
```

Then open your browser to `http://127.0.0.1:7860`

### Option 2: Use Individual Modules

#### Test LLM (Brain of the Doctor):
```bash
python brain_of_the_doctor.py
```

#### Test STT (Voice of the Patient):
```bash
python voice_of_the_patient.py
```

#### Test TTS (Voice of the Doctor):
```bash
python voice_of_the_doctor.py
```

## 📂 Project Structure

```
voice-agent.v1/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── stt.py               # Whisper STT implementation
│   ├── llm.py               # Open-source LLM implementation
│   ├── tts.py               # Dia2 TTS implementation
│   ├── audio_utils.py       # Audio recording/playback
│   └── agent.py             # Voice Agent pipeline
├── brain_of_the_doctor.py   # LLM wrapper (updated)
├── voice_of_the_patient.py  # STT wrapper (updated)
├── voice_of_the_doctor.py   # TTS wrapper (updated)
├── gradio_app.py            # Gradio UI (voice-only)
├── requirements.txt         # Python dependencies
├── Pipfile                  # Pipenv configuration
└── README.md                # This file
```

## 🔧 Troubleshooting

### Model Download Issues
Models are downloaded automatically from Hugging Face on first run. This may take time depending on your internet connection.

### Memory Issues
If you encounter out-of-memory errors, try:
- Using a smaller model like `sahil2801/gpt-oss`
- Reducing batch size or sequence length
- Using CPU instead of GPU (set `device=-1` in the code)

### Audio Issues
- Ensure your microphone permissions are enabled
- Check that PortAudio is installed correctly
- Try adjusting the sample rate in `audio_utils.py`

## 📝 Notes

- **No API Keys Required**: All models run locally
- **No Image Processing**: This version is voice-only
- **Privacy**: All data stays on your machine
- **Hardware**: Works on CPU, but GPU recommended for faster inference

## 🆚 Changes from Previous Version

### Removed:
- ❌ GROQ API (STT + Vision)
- ❌ ElevenLabs TTS
- ❌ gTTS
- ❌ OpenAI API
- ❌ Image processing (PIL, base64 encoding)
- ❌ Vision models
- ❌ speech_recognition library
- ❌ pydub library
- ❌ pyaudio library

### Added:
- ✅ Transformers pipelines
- ✅ PyTorch
- ✅ sounddevice for audio I/O
- ✅ soundfile for audio file handling
- ✅ Local Whisper STT
- ✅ Open-source LLM (Qwen/Llama)
- ✅ Dia2 TTS

## License

This project is for educational purposes.
