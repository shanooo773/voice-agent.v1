# Open-Source Voice Agent

A fully open-source voice agent using Hugging Face Transformers for Speech-to-Text (Whisper), Language Model (Qwen/Llama), and Text-to-Speech (MMS-TTS).

## Architecture

```
Microphone Input →
Whisper (STT) →
GPT-OSS / Llama-3 / Qwen (LLM) →
MMS-TTS (TTS) →
Speakers
```

## Features

- **100% Open-Source**: No paid APIs required (no OpenAI, no ElevenLabs, no Grok)
- **No API Keys**: All models run locally using Hugging Face Transformers
- **Modular Design**: Clean separation of STT, LLM, and TTS components
- **Async Support**: Both synchronous and asynchronous voice loops
- **Customizable**: Easy to swap models for different languages or performance needs

## Models Used

### Speech-to-Text (STT)
- **Model**: `openai/whisper-large-v3`
- **Library**: Hugging Face Transformers
- Accurate multilingual speech recognition

### Large Language Model (LLM)
Choose one of:
- `Qwen/Qwen2.5-7B-Instruct` (default)
- `meta-llama/Llama-3-8b-instruct`
- `sahil2801/gpt-oss`

### Text-to-Speech (TTS)
- **English**: `facebook/mms-tts-eng`
- **Urdu**: `facebook/mms-tts-urd`
- **Library**: Hugging Face Transformers

## Installation

### Prerequisites

#### macOS

1. **Install Homebrew** (if not already installed):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install FFmpeg and PortAudio:**

   ```bash
   brew install ffmpeg portaudio
   ```

#### Linux

For Debian-based distributions (e.g., Ubuntu):

1. **Update the package list**

   ```bash
   sudo apt update
   ```

2. **Install FFmpeg and PortAudio:**

   ```bash
   sudo apt install ffmpeg portaudio19-dev
   ```

#### Windows

1. **Download FFmpeg**:
   - Visit [FFmpeg Downloads](https://ffmpeg.org/download.html)
   - Download and extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to your system PATH

2. **Install PortAudio**:
   - Download from [PortAudio Downloads](http://www.portaudio.com/download.html)
   - Follow installation instructions

### Python Environment Setup

#### Using pip and venv

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**

   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   **Optional Dependencies:**
   - For Gradio UI: `pip install -r requirements-ui.txt`
   - For FastAPI support: `pip install -r requirements-api.txt`

#### Using Pipenv

1. **Install Pipenv** (if not already installed):

   ```bash
   pip install pipenv
   ```

2. **Install Dependencies:**

   ```bash
   pipenv install
   ```

3. **Activate the Virtual Environment:**

   ```bash
   pipenv shell
   ```

#### Using Conda

1. **Create a Conda Environment:**

   ```bash
   conda create --name voice-agent python=3.12
   ```

2. **Activate the Conda Environment:**

   ```bash
   conda activate voice-agent
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the voice agent with default settings:

```bash
python main.py
```

The agent will:
1. Prompt you to press Enter to start recording
2. Record your voice for 5 seconds
3. Transcribe your speech using Whisper
4. Generate a response using the LLM
5. Convert the response to speech using MMS-TTS
6. Play the audio response

### Command-Line Options

```bash
# Use a different LLM model
python main.py --llm-model meta-llama/Llama-3-8b-instruct

# Use Urdu TTS
python main.py --tts-model facebook/mms-tts-urd

# Record for 10 seconds
python main.py --record-duration 10

# Use async mode
python main.py --async

# Add a system prompt
python main.py --system-prompt "You are a helpful medical assistant."
```

### Advanced Usage

#### Custom System Prompt

```bash
python main.py --system-prompt "You are a professional assistant. Keep your responses concise and helpful."
```

#### Different Model Combinations

```bash
# Use GPT-OSS for LLM
python main.py --llm-model sahil2801/gpt-oss

# Use Llama-3 for LLM
python main.py --llm-model meta-llama/Llama-3-8b-instruct
```

## Project Structure

```
voice-agent.v1/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── stt.py              # Whisper Speech-to-Text
│   ├── llm.py              # Open-source LLM
│   ├── tts.py              # MMS Text-to-Speech
│   ├── audio_utils.py      # Audio recording and playback
│   └── agent.py            # Voice agent orchestration
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── Pipfile                 # Pipenv dependencies
└── README.md              # This file
```

## Module Documentation

### src/stt.py
Handles speech-to-text using Whisper:
- `WhisperSTT`: Class for transcribing audio files

### src/llm.py
Manages the language model:
- `OpenSourceLLM`: Class for generating text responses

### src/tts.py
Handles text-to-speech using MMS-TTS:
- `MMSTTS`: Class for synthesizing speech from text

### src/audio_utils.py
Audio utilities:
- `AudioRecorder`: Record audio from microphone
- `AudioPlayer`: Play audio files

### src/agent.py
Main voice agent orchestration:
- `VoiceAgent`: Combines STT → LLM → TTS pipeline
- Supports both sync and async modes

## Troubleshooting

### Audio Recording Issues

If you encounter issues with audio recording:

1. **Check PortAudio installation**:
   ```bash
   # macOS
   brew list portaudio
   
   # Linux
   dpkg -l | grep portaudio
   ```

2. **Test microphone**:
   ```bash
   # macOS
   rec -r 16000 -c 1 test.wav
   
   # Linux
   arecord -d 5 -r 16000 -f S16_LE test.wav
   ```

### Model Loading Issues

If models fail to load:

1. **Check internet connection** (required for first-time model download)
2. **Increase timeout** or retry
3. **Check disk space** (models can be several GB)

### Performance Issues

For better performance:

1. **Use GPU** if available (modify `device=-1` to `device=0` in source files)
2. **Use smaller models**:
   - STT: `openai/whisper-base` or `openai/whisper-small`
   - LLM: `sahil2801/gpt-oss` (smaller than Qwen or Llama)

## License

This project uses open-source models and libraries. Please refer to individual model licenses on Hugging Face.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Credits

- **Whisper**: OpenAI
- **Qwen**: Alibaba Cloud
- **Llama**: Meta
- **MMS-TTS**: Facebook/Meta
- **Hugging Face**: For the Transformers library

