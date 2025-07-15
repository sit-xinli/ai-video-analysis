import os
import re
import cv2
import torch
from openai import OpenAI
import torchaudio
import json
from pydub import AudioSegment
from PIL import Image
import numpy as np
import tempfile
from moviepy import VideoFileClip
from transformers import (
    AutoProcessor, AutoModelForAudioClassification
)
import webrtcvad
import os
import dotenv
import base64
import gradio as gr
from google import genai
from google.genai import types

# Load environment variables from .env file
dotenv.load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
if OPENAI_API_KEY is None or OPENAI_BASE_URL is None or HF_TOKEN is None:
    raise ValueError("Please set the OPENAI_API_KEY, OPENAI_URL, and HF_TOKEN environment variables.")
   
client_openai = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# Load Hugging Face audio classifier
audio_processor = AutoProcessor.from_pretrained("suhacan/ast-finetuned-audioset-10-10-0.4593-finetuned-gtzan")
audio_model = AutoModelForAudioClassification.from_pretrained("suhacan/ast-finetuned-audioset-10-10-0.4593-finetuned-gtzan")
audio_model.eval()

# Gemini APIキーを環境変数から取得
client_gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Helper: profile video to get frame count and duration (seconds)
def profile_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    print(f"Frame count: {frame_count}, FPS: {fps}, Duration: {duration:.2f} seconds")

    return (frame_count, duration)

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
def speech_branch(audio_chunk, language="en"):
    result = "no speech detected"
    prompt = """You are an AI assistant transcribing audio for a disabled person with wheelchair. 
    Please automatically determine the language and transcribe the audio accurately and clearly, 
    paying attention to any important details or context that may be relevant to the disabled person.
    If the audio is not clear or contains no speech, return a message "no speech" indicating that no speech was detected."""

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        audio_chunk.export(tmp.name, format="mp3")
        if has_speech(tmp.name):
            with open(tmp.name, "rb") as audio_file:
                transcript =client_openai.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", 
                    file=audio_file,
                    prompt = prompt, # Optional
                    )
            if transcript.text is None or transcript.text.strip() == "no speech":
                print("No speech detected in this audio chunk.")
            else:
                result = transcript.text
                print(f">>>translate>>>> {result}")
        else:
            print("No speech detected in this audio chunk.")

        return result            

# --- Audio Event Detection via Hugging Face AST ---
def acoustic_event_branch(audio_chunk, top_k=5):
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
def extract_key_frames(video_path, num_of_frames=5, start_time=0, end_time=None):
    frames = []
    frame_dims = []

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Could not retrieve FPS from the video.")

    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    end_frame = int(end_time * fps) if end_time is not None else float('inf')
    
    if end_time is not None and end_frame < start_frame:
        raise ValueError("End time must be greater than start time.")
    
    success, frame = cap.read()
    interval = round((end_frame - start_frame) / num_of_frames) if end_frame > start_frame else 1
    frame_count = start_frame
    while success and frame_count < end_frame:
        if (frame_count - start_frame) % interval == 0:
            img = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            width, height = img.shape[1], img.shape[0]
            if width > height and width > 800:
                scale = 800 / width
                img = cv2.resize(img, (800, int(height * scale)))
            elif height > width and height > 800:
                scale = 800 / height
                img = cv2.resize(img, (int(width * scale), 800))
            
            frame_dims.append((img.shape[1], img.shape[0]))
            _, buffer = cv2.imencode('.jpg', img)
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            frames.append(base64_frame)

        success, frame = cap.read()
        frame_count += 1
    cap.release()
    return frames, frame_dims

# Caption images
def visual_branch(frames, speech, evts):
    content = [ {"type": "text", "text": f"Speech: {speech}.\n\nSounds detected: {', '.join(evts)} " } ]
    for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })
    
    response = client_openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", 
                 "content": """You are an AI assistant describing scenes from video to a disabled person with wheelchair. Your task is to analyze the provided images and identify hazard. The hazard can be anything that may cause harm or risk to the disabled person with wheelchair. The typical hazard include curbs, steps, uneven road surface, obstacles, dangerous objects, or unsafe conditions. First, describe the scene according to hazards to a disabled person with wheelchair. Then, provide the exact bounding box coordinates for the identified hazard object in the most hazard image. The coordinates should be normalized from 0.0 to 1.0 for x, y, width, and height, relative to the image dimensions. The output format MUST be a JSON object with the following structure:
                  {
                    "Description": "A detailed description of the scene, including actions, objects, and people, and the primary hazard.",
                    "HazardRegion": {
                        "index": <index of the most representative image>,
                        "score": <confidence score of the hazard region>,
                        "caption": "<caption for the detected hazard object>",
                        "box": [x, y, width, height]
                    }
                  }
                  """
                },
                {"role": "user", "content": content}
            ],
            max_tokens=1000
        )
    return response.choices[0].message.content

# generate scene description using GPT
def generate_scene_description(prompt):
    
    response = client_openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
        
    )
    return response.choices[0].message.content.strip()

def gemini_detect_objects(image, hazard_type):
    """
    Detect objects in the image using Gemini API.
    Returns a list of detected objects with their bounding boxes and confidence scores.
    
    the format of the response is expected to be a JSON string with bounding boxes
    [
        {
            "box_2d": [0.1604, 0.5990, 0.6050, 0.9896],
            "label": "overturned vehicle"
        },
        {
            "box_2d": [0.0000, 0.0000, 1.0000, 1.0000],
            "label": "person"
        }
    ]
    """
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', image_rgb)
    image = types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")

    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            f"Analyze the image and detect objects --- '{hazard_type}'. Return bounding boxes in normalized coordinates (0.0 to 1.0) relative to the image dimensions.", 
            image
        ],
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    print(f"Detected bounding boxes: {response.text}")
    bounding_boxes = json.loads(response.text)
    
    return bounding_boxes

# Full video analysis pipeline
def analyze_video(video_path,  segments_of_video, frames_per_segment=5, language="en"):
    
    _, duration = profile_video(video_path)

    # if video duration is less than 3 seconds, use 1 second interval
    interval_s = round(duration/segments_of_video) if duration > 3 else 1

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    print(f"Analyzing video: {video_path}")
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        raise ValueError("Unsupported video format. Please provide a valid video file.")

    audio_path = extract_audio(video_path)
    audio_chunks = segment_audio(audio_path, chunk_length_ms=interval_s * 1000)
    
    results = []
    all_frames = []
    all_frame_dims = []
    
    for i, chunk in enumerate(audio_chunks):
        print(f"Processing segment {i+1}/{len(audio_chunks)}...")

        events = acoustic_event_branch(chunk)
        transcript = speech_branch(chunk, language)

        frames, frame_dims = extract_key_frames(video_path, 
                                    num_of_frames=frames_per_segment, 
                                    start_time=i * interval_s, 
                                    end_time=(i + 1) * interval_s
                                )
        all_frames.append(frames)
        all_frame_dims.append(frame_dims)
        if len(frames) > 0:
            visual_message = visual_branch(frames, transcript, events)
            try:
               visual_json = json.loads(visual_message)
               print(f"JSON response for segment {i}:\n {visual_message} \n")
            except json.JSONDecodeError:
               print(f"Invalid JSON response for segment \n {visual_message} \n")
               visual_json = {
                   "Description": visual_message
               }

        results.append((i, transcript, events, visual_json))
    
    return results, all_frames, all_frame_dims

def main_process(video_path, init_prompt, language, segments_of_video, frames_per_segment):
    results, all_frames, all_frame_dims = analyze_video(video_path, int(segments_of_video), int(frames_per_segment), language)
    
    content_prompt = init_prompt
    segmented_results_text = ""
    most_important_hazard_region = None
    hazard_segment_index = -1

    for i, (segment_index, transcript, events, visual_message) in enumerate(results):
        segmented_results_text += f"--- SEGMENT {segment_index} ---\n"
        segmented_results_text += f"Detected Speech: {transcript}\n"
        segmented_results_text += f"Detected Sounds: {', '.join(events)}\n"
        visual_description = visual_message.get("Description", "N/A")
        segmented_results_text += f"Visual Description: {visual_description}\n\n"

        hazard_region = visual_message.get("HazardRegion", None)            
        if hazard_region and (most_important_hazard_region is None or hazard_region.get("score", 0) > most_important_hazard_region.get("score", 0)):
            most_important_hazard_region = hazard_region
            hazard_segment_index = i
                    
    content_prompt = init_prompt + "\n\n" + segmented_results_text
    response = generate_scene_description(content_prompt)

    hazard_info = "No hazard detected."
    hazard_image_with_box = all_frames[0][0]
    if most_important_hazard_region and hazard_segment_index != -1:
        hazard_info = f"{most_important_hazard_region}"
        frame_index = most_important_hazard_region.get("index", 0)
        if frame_index < len(all_frames[hazard_segment_index]):
            base64_frame = all_frames[hazard_segment_index][frame_index]
            img_data = base64.b64decode(base64_frame)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) #keep as BGR format 
            
            img_width, img_height = all_frame_dims[hazard_segment_index][frame_index]
            most_important_hazard_type = most_important_hazard_region.get("caption", "Hazard")
            bounding_boxes_with_labels = gemini_detect_objects(img, most_important_hazard_type)
        
            hazard_image_with_box = img.copy() #keep BGR ordewr for cv2 drawing
            for bounding_box_with_label in bounding_boxes_with_labels:
                bounding_box = bounding_box_with_label.get("box_2d", [0, 0, 0, 0])
                label = bounding_box_with_label.get("label", "Unknown")

                # Extract bounding box coordinates, note the order is [y1, x1, y2, x2]
                y1, x1, y2, x2 = bounding_box
                # convert normalize coordinates to the image dimensions
                x1, y1 = int(x1 * img_width), int(y1 * img_height)
                x2, y2 = int(x2 * img_width), int(y2 * img_height)                           
                # Draw rectangle and label only for the most important hazard type
                cv2.rectangle(hazard_image_with_box, (x1, y1), (x2, y2), (0,255,0), 3)
                cv2.putText(hazard_image_with_box, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    return segmented_results_text, response, cv2.cvtColor(hazard_image_with_box, cv2.COLOR_BGR2RGB), hazard_info

# Gradio Interface
prompt_templates = {
    "Default": """You are an AI assistant describing scenes from video to a disabled person with wheelchair. With all the context including in these consecutive SEGEMENTS, Detected Speech and Visual description is more important than Detected Sounds.
Firstly, please describe what is happening in the scene, and then to avoid risk for the disabled person with wheelchair, please advice the NextAction.
The FORMAT of the output MUST be:
{
  "Description": "A detailed description of the scene, including actions, objects, and people.",
  "NextAction": "Go forward | Turn left | Turn right | Stop | Go backward | Wait | Look around | Run away | Navigate to location | Avoid obstacle | Adjust speed | Follow person | Return to charger | Emergency stop | Open door | Call elevator | Adjust seat | Send alert | Share location | Request help | Voice command mode | Daily schedule | Entertainment mode"
}""",
    "Default(日本語)": """あなたはAIアシスタントで、車椅子に乗った障害者にビデオのシーンを説明します。
これらの連続したSEGEMENTSに含まれるすべてのコンテキストでは、検出された音声よりも、検出された音声と視覚的な説明の方が重要です。
まず、そのシーンで何が起こっているかを日本語で説明し、次に障害者の危険を回避するために、次のアクションをアドバイスしてください。

出力の書式は次のようにしなければならない（MUST）:
{
  "Description": "行動、物、人を含むシーンの詳細な描写",
  "Next_Action": "進む" | "左に曲がる" | "右に曲がる" | "止まる" | "後進する" | "待つ" | "周囲を見渡す" | "逃げる"
}""",
    "Safety First": """Analyze the video for any potential safety hazards. 
If any are found, describe them in detail and suggest a safe course of action.
If no hazards are found, state that the scene appears safe.""",
    "Safety First(日本語)": """ビデオを分析して、潜在的な安全上の危険を特定してください。   
もし危険が見つかった場合は、それらを詳細に説明し、安全な行動を提案してください。
危険が見つからない場合は、シーンが安全であると述べてください。"""
}

def update_prompt(template_name):
    return prompt_templates[template_name]

with gr.Blocks() as demo:
    gr.Markdown("# マルチモデルAIビデオ解析")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Video Input")
            gr.Markdown("Upload a video and provide a prompt to analyze the scene. The AI will provide a description and suggest the next action.")
            video_input = gr.Video(label="Input Video", sources=["upload", "webcam"])

            with gr.Accordion("Parameters", open=False):
                language_input = gr.Dropdown(
                    label="Language for Transcription",
                    choices=["en", "ja", "zh", "ko", "fr", "de", "es"],
                    value="ja",
                )
                segments_of_video_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=1,
                    label="Number of Segments to Analyze",
                )
                frames_per_segment_input = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5,
                    label="Number of Frames per Segment",
                )
            
            prompt_template_dropdown = gr.Dropdown(
                label="Prompt Templates",
                choices=list(prompt_templates.keys()),
                value="Default"
            )
            
            init_prompt_input = gr.Textbox(
                label="Initial Prompt",
                value=prompt_templates["Default"],
                lines=10
            )
            
            prompt_template_dropdown.change(
                fn=update_prompt,
                inputs=prompt_template_dropdown,
                outputs=init_prompt_input
            )
            
            analyze_button = gr.Button("Analyze Video")

        with gr.Column(scale=1):
            gr.Markdown("## Analysis Results")
            gr.Markdown("The most impportant hazard scene and its hazard region shown. The description hazard scene and nex action for a wheelchair user is recommended.")
            image_output = gr.Image(label="Hazard Region")
            hazard_info_output = gr.Textbox(label="Hazard Details")
            final_response_output = gr.Textbox(label="Final Response")
            segmented_output = gr.Textbox(label="Segmented Analysis", lines=15)

    analyze_button.click(
        fn=main_process,
        inputs=[
            video_input, 
            init_prompt_input, 
            language_input, 
            segments_of_video_input, 
            frames_per_segment_input
        ],
        outputs=[segmented_output, final_response_output, image_output, hazard_info_output]
    )

if __name__ == "__main__":
    demo.launch()