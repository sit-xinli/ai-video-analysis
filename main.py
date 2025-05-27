import os
import cv2
import torch
from openai import OpenAI
import torchaudio
import numpy as np
from pydub import AudioSegment
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from PIL import Image
import tempfile
from moviepy import VideoFileClip
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForAudioClassification
)
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if OPENAI_API_KEY is None or OPENAI_BASE_URL is None:
    raise ValueError("Please set the OPENAI_API_KEY and OPENAI_URL environment variables.")
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Load Hugging Face audio classifier
audio_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
audio_model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
audio_model.eval()

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Helper: extract audio from video
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    video.audio.write_audiofile(audio_path, logger=None)
    return audio_path

# Segment audio
def segment_audio(audio_path, chunk_length_ms=10000):
    audio = AudioSegment.from_file(audio_path)
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# Transcribe speech
def transcribe_chunk(audio_chunk):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        audio_chunk.export(tmp.name, format="mp3")
        with open(tmp.name, "rb") as audio_file:
            transcript =client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", 
                file=audio_file,
                response_format="json",
                #language="ja"
            )
        print(transcript)
        return transcript.text

# --- Audio Event Detection via Hugging Face AST ---
def detect_events(audio_chunk, top_k=3):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        audio_chunk.export(tmp.name, format="wav")
        waveform, sr = torchaudio.load(tmp.name)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    inputs = audio_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = audio_model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top_indices = torch.topk(probs, top_k).indices
    labels = [audio_model.config.id2label[i.item()] for i in top_indices]
    return labels

# Extract frames from video
def extract_key_frames(video_path, interval_s=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frames = []
    success, frame = cap.read()
    while success:
        if int(frame_count % (fps * interval_s)) == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    return frames

# Caption images
def caption_image(img):
    inputs = blip_processor(images=img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

# Combine all into GPT prompt
def generate_scene_description(transcript, events, image_captions):
    prompt = f"""You are an AI assistant describing scenes from video.
Speech: "{transcript}"
Sounds detected: {', '.join(events)}
Visuals: {', '.join(image_captions)}
With all the context including all language, describe what is happening in the scene in Japanese."""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Full video analysis pipeline
def analyze_video(video_path):
    interval_s = 3  # Segment length in seconds
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    print(f"Analyzing video: {video_path}")
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        raise ValueError("Unsupported video format. Please provide a valid video file.")

    audio_path = extract_audio(video_path)
    audio_chunks = segment_audio(audio_path, chunk_length_ms=interval_s * 1000)
    frames = extract_key_frames(video_path, interval_s)
    results = []
    for i, chunk in enumerate(audio_chunks):
        print(f"Processing segment {i+1}/{len(audio_chunks)}...")
        transcript = transcribe_chunk(chunk)
        events = detect_events(chunk)
        if i < len(frames):
            caption = caption_image(frames[i])
        else:
            caption = "No frame available"
        scene = generate_scene_description(transcript, events, [caption])
        results.append((i, transcript, events, caption, scene))
    return results

results = analyze_video("04-22-141130.mp4")

for i, transcript, events, caption, scene in results:
    print(f"--- Segment {i} ---")
    print(f"Transcript: {transcript}")
    print(f"Audio Events: {events}")
    print(f"Visual Caption: {caption}")
    print(f"Scene Description: {scene}")