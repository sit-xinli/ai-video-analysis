# AI Video Analysis for Accessibility

This project is an AI-powered system designed to analyze video and audio content to provide real-time situational awareness and suggest safe actions, primarily for wheelchair users. It integrates multiple AI models to understand visual and auditory information, generating comprehensive scene descriptions and actionable advice.

## Key Features

- **Audio Analysis**:
  - **Speech-to-Text**: Transcribes spoken language from audio using OpenAI's Whisper model.
  - **Sound Event Detection**: Identifies ambient sounds (e.g., traffic, sirens, speech) using a Hugging Face Audio Spectrogram Transformer (AST) model.

- **Visual Analysis**:
  - **Keyframe Extraction**: Extracts significant frames from video files at set intervals.
  - **Image Captioning**: Generates detailed descriptions of visual scenes using OpenAI's GPT-4 Vision model.

- **Integrated Scene Description**:
  - Combines audio transcripts, sound events, and visual captions to create a holistic understanding of the environment.
  - Leverages GPT-4 to generate a final, coherent description of the scene in Japanese.

- **Actionable Advice**:
  - Based on the analysis, the system suggests the next safe action (e.g., "Go forward," "Stop," "Look around") to help users navigate their surroundings safely.

## Technology Stack

This project utilizes the following key libraries and models:

- **AI Models**:
  - OpenAI GPT-4 (Vision, Transcription)
  - Hugging Face Transformers (Audio Spectrogram Transformer for sound classification)
- **Core Libraries**:
  - `openai`: For interacting with OpenAI APIs.
  - `transformers`: For using Hugging Face models.
  - `torch`, `torchaudio`, `torchvision`: For deep learning tasks.
  - `moviepy`, `ffmpeg`: For video and audio processing.
  - `opencv-python`: For image and video manipulation.
  - `pydub`, `webrtcvad`: For audio segmentation and voice activity detection.
- **Environment Management**:
  - `uv`: For Python package management.

## Setup

Follow these steps to set up and run the project.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd ai-video-analysis
```

### 2. Install Dependencies

This project uses `uv` for package management. Make sure you have it installed.

```bash
# Install dependencies from pyproject.toml
uv pip install -r requirements.txt
```
*(Note: If `requirements.txt` is not present, generate it from `pyproject.toml` or install dependencies directly.)*

### 3. Configure Environment Variables

You need to provide API keys for the services used.

1.  Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
2.  Open the `.env` file and add your API keys:
    ```
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    OPENAI_BASE_URL="your_openai_base_url_here" # Optional
    HF_TOKEN="your_huggingface_token_here"   # Optional
    ```

## Usage

To analyze a video or audio file, run one of the Python scripts with the target file.

```bash
# Example using v2-1-segements-openai.py
python v2-1-segements-openai.py
```

Make sure to modify the script to point to your desired input video file. The script will print the analysis results, including scene descriptions and suggested actions, to the console.

## Script Evolution

The project has evolved through several scripts, each with a different focus:

-   **`v1.py`**: Initial version combining local models (BLIP for image captioning) and OpenAI for transcription.
-   **`v1.1-openai4caption.py`**: Replaced local image captioning with OpenAI's GPT-4 Vision for more accurate and context-aware visual descriptions. Introduced voice activity detection (`webrtcvad`) to filter out silent audio chunks.
-   **`v2-segements-openai.py`**: Refined the analysis pipeline to process the video in consecutive segments. It now aggregates information from all segments before generating a final summary and action, providing better contextual understanding.
-   **`v1.2-noaudio-openai.py`**: Added functionality to handle videos without audio tracks.
-   **`v1.3-noimage-openai.py`**: Added support for analyzing audio-only files (e.g., `.mp3`).

The most advanced and recommended script for general use is **`v2-1-segements-openai.py`** as it provides the most comprehensive, context-aware analysis.