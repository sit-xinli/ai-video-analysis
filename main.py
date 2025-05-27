import os
import cv2
import torch
import openai
import torchaudio
import numpy as np
from pydub import AudioSegment
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from PIL import Image
import tempfile
#import moviepy.editor as mp
from moviepy import VideoFileClip
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load PANNs
@torch.no_grad()
def load_panns_model():
    model = torch.hub.load('qiuqiangkong/panns_demo', 'Cnn14', pretrained=True)
    model.eval()
    labels = torch.hub.load('qiuqiangkong/panns_demo', 'get_labels')
    return model, labels

panns_model, panns_labels = load_panns_model()

# Helper: extract audio from video
def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

# Segment audio
def segment_audio(audio_path, chunk_length_ms=10000):
    audio = AudioSegment.from_file(audio_path)
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# Transcribe speech
def transcribe_chunk(audio_chunk):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        audio_chunk.export(tmp.name, format="mp3")
        with open(tmp.name, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
        return transcript['text']

# Detect audio events
def detect_events(audio_chunk, top_k=3):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        audio_chunk.export(tmp.name, format="wav")
        waveform, sr = torchaudio.load(tmp.name)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(sr, 32000)
        waveform = resampler(waveform)
    input_tensor = waveform.unsqueeze(0)
    output = panns_model(input_tensor)
    scores = torch.sigmoid(output['clipwise_output'])[0].cpu().numpy()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [panns_labels[i] for i in top_indices if scores[i] > 0.2]

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
Describe what is happening in the scene."""
    response = openai.ChatCompletion.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Full video analysis pipeline
def analyze_video(video_path):
    audio_path = extract_audio(video_path)
    audio_chunks = segment_audio(audio_path)
    frames = extract_key_frames(video_path, interval_s=10)
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
    print()