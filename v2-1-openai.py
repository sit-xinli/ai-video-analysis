import os
import cv2
import torch
from openai import OpenAI
import torchaudio
import numpy as np
from pydub import AudioSegment
from torchvision import transforms
from PIL import Image
import tempfile
from moviepy import VideoFileClip
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForAudioClassification
)
import webrtcvad
import wave
import os
import dotenv
import base64

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

def has_speech(mp3_path, aggressiveness=2, min_speech_frames=5):
    vad = webrtcvad.Vad(aggressiveness)

    # Load and convert MP3 to 16kHz mono PCM WAV
    audio = AudioSegment.from_file(mp3_path).set_frame_rate(16000).set_channels(1)
    pcm_audio = audio.raw_data
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_size = int(sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample (16-bit audio)

    speech_frames = 0
    num_frames = len(pcm_audio) // frame_size

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = pcm_audio[start:end]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            speech_frames += 1
        if speech_frames >= min_speech_frames:
            return True
    return False

# Transcribe speech
def transcribe_chunk(audio_chunk):    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        audio_chunk.export(tmp.name, format="mp3")
        if has_speech(tmp.name):
            with open(tmp.name, "rb") as audio_file:
                    transcript =client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe", 
                        file=audio_file,
                        response_format="json",
                        #language="ja"
                    )
                    return transcript.text
        else:
            print("No speech detected in this audio chunk.")
            return ""            

# --- Audio Event Detection via Hugging Face AST ---
def detect_events(audio_chunk, top_k=5):
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
            width, height = img.shape[1], img.shape[0]
            # if width or height is larger than 800, then resize
            if width > height and width > 800:
                scale = 800 / width
                img = cv2.resize(img, (800, int(height * scale)))
            elif height > width and height > 800:
                scale = 800 / height
                img = cv2.resize(img, (int(width * scale), 800))
            # Convert the frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            frames.append(base64_frame)
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    return frames

# Caption images: TBD -- use CHATGPT for image captioning
def caption_image(img, speech, evts):
    content = [ {"type": "text", "text": f"Speech: {speech}.\n\nSounds detected: {', '.join(evts)} " } ]
    content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
            })
    
    response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You is analysing the images. please give captions to the image paying attention to both images and text."},
                {"role": "user", "content": content}
            ],
            max_tokens=300
        )
    return response.choices[0].message.content

# generate scene description using GPT
def generate_scene_description(prompt):
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# Full video analysis pipeline
def analyze_video(video_path):
    interval_s = 10  # Segment length in seconds
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
        transcript = transcribe_chunk(chunk) # Transcribe audio chunk use CHATGPT
        events = detect_events(chunk) # Detect audio events use Hugging Face AST
        if i < len(frames):
            caption = caption_image(frames[i], transcript, events) # Caption image using BLIP--will be change to CHATGPT
        else:
            caption = "No frame available"
        #scene = generate_scene_description(transcript, events, [caption])
        results.append((i, transcript, events, caption))
    return results

#results = analyze_video("2025-05-05 220122.mp4")
results = analyze_video("2025-05-27-180908.mp4")

prompt = """
You are an AI assistant describing scenes from video to a disabled person with wheelchair.
With all the context including in these consecutive SEGEMENTs, firstly describe what is happening in the scene.
Then concisely advice the next action to avoid risk for the disabled person in Japanese.
"""
# Combine all into GPT prompt
for i, transcript, events, caption in results:
    prompt = f"""{prompt} 
    --- Segment {i} ---
    Speech detected: "{transcript}"
    Sounds detected: {', '.join(events)}
    Visual description: {caption} """
print(prompt)
response = generate_scene_description(prompt)
print(f"\n\nScene Description: {response}")