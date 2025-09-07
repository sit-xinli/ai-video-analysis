import os
import re
import cv2
import torch
from openai import OpenAI
import torchaudio
import json
from pydub import AudioSegment
from PIL import Image,ExifTags
import numpy as np
import tempfile
from moviepy import VideoFileClip
from transformers import (
    AutoProcessor, AutoModelForAudioClassification,AutoModelForZeroShotObjectDetection
)
import webrtcvad
import os
import dotenv
import base64
import gradio as gr
from google import genai
from google.genai import types
from pymediainfo import MediaInfo

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

# GroundingDINO
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "IDEA-Research/grounding-dino-base"
# Load the pretrained DETR model and processor
zeroshot_processor = AutoProcessor.from_pretrained(model_id)
zeroshot_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


#get rotation for video
def get_rotation(video_path):
    media_info = MediaInfo.parse(video_path)
    for track in media_info.tracks:
        if track.track_type == "Video":
            return int(float(track.rotation))  # May return None if not present
    return 0


# Helper: profile video to get frame count and duration (seconds)
def profile_video(video_path):

    rotation = get_rotation(video_path)
    print(f"^^^^^^^^^^^^^^^^^^^^^^^Rotation: {rotation} degrees ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    cap = cv2.VideoCapture(video_path)    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    print(f"Frame count: {frame_count}, FPS: {fps}, Duration: {duration:.2f} seconds")

    return (frame_count, duration, rotation)

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
    prompt = """ 
You are an AI assistant that **transcribes audio for a person who uses a wheelchair**.

Your tasks are:

1. **Automatically detect the spoken language** in the provided audio.
2. **Transcribe the speech accurately and clearly** into text, preserving the meaning and important details relevant to the user.
3. If the audio is **unclear, contains only noise, or no speech is detected**, return the exact string:

   ```
   no speech
   ```

### **Output Format (strict JSON):**

```json
{
  "Language": "<detected language code or name, e.g., 'English' or 'Japanese'>",
  "Transcription": "<transcribed speech text OR 'no speech'>"
}
```
"""

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
def extract_key_frames(video_path, num_of_frames=5, start_time=0, end_time=None, rotation=0):
    frames = []
    #frame_dims = []

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
            img = frame
            if rotation == 90 : 
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rotation ==270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            width, height = img.shape[1], img.shape[0]
            if width > height and width > 800:
                scale = 800 / width
                img = cv2.resize(img, (800, int(height * scale)))
            elif height > width and height > 800:
                scale = 800 / height
                img = cv2.resize(img, (int(width * scale), 800))
            
            #frame_dims.append((img.shape[1], img.shape[0]))
            _, buffer = cv2.imencode('.jpg', img)
            """
            Even though buffer is a NumPy array (from cv2.imencode), 
            base64.b64encode() automatically treats it as bytes 
            because NumPy arrays support the buffer protocol. 
            So you don’t need to explicitly call .tobytes() — base64.b64encode() handles it internally.
            """
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            frames.append(base64_frame)

        success, frame = cap.read()
        frame_count += 1
    cap.release()
    
    return frames

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
                 "content": """
                  You are an AI assistant that analyzes video frames for a person using a wheelchair.
Your task is to:

1. **Identify hazards** in the provided images. A hazard is anything that could cause risk or difficulty for a wheelchair user (e.g., curbs, steps, stairs, uneven road surfaces, holes, obstacles, clutter, dangerous objects, moving vehicles, slippery areas, or blocked paths).
2. **Describe the scene** with a clear and detailed explanation of the environment, including relevant people, objects, actions, and the main hazard.
3. **Provide bounding box coordinates** for the most hazardous object(s) in the most relevant image. Use normalized values between `0.0` and `1.0` for `[x, y, width, height]`, where `(x, y)` is the top-left corner of the bounding box relative to the image dimensions.
4. If multiple hazards are present, include them as separate entries in the `HazardRegion` list, ordered by severity or immediacy of risk.

### **Output Format (strict JSON):**

```json
{
  "Description": "A detailed description of the scene, focusing on hazards relevant to a wheelchair user. Mention objects, people, actions, and the primary hazard clearly.",
  "HazardRegion": [
    {
      "index": <integer index of the image where the hazard is most visible>,
      "score": <float between 0.0 and 1.0 indicating confidence>,
      "caption": "Short caption describing the hazard object or condition",
      "box": [x, y, width, height]
    }
  ]
}
```
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
"""
GroudingDINO require image in RGB color order"
size = (original_width, original_height)

if (original_width > original_height and width < height ) or (original_width < original_height and width > height)
  oratate image 90 or 270

"""
def grounding_dino(image, caption):
    """
    Detect objects in the image using GroundingDINO API.
    Returns a list of detected objects with their bounding boxes and confidence scores.
    
    the format of the response is expected to be a JSON string with bounding boxes
    [
        scores:[], 
        labels:[], 
        boxes:[]
    ]
    """
    
    # Preprocess the image
    inputs = zeroshot_processor(images=image, text=caption, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = zeroshot_model(**inputs)

    scores = outputs.logits.softmax(-1)[..., :-1].max(-1).values
    threshold = scores.quantile(0.2)  # Keep top 80% confident predictions

    # Post-process the results (keep only high-confidence predictions) 
    if hasattr(image, "shape"):
        height, width = image.shape[:2]
        target_sizes = torch.tensor([[height, width]])
    elif hasattr(image, "size"):  #if have shape, the size if the whole size of image (channel*w*h)
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        
    
    results = zeroshot_processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, box_threshold=threshold, text_threshold=0.3)[0]
    
    return results

# Full video analysis pipeline
def analyze_video(video_path,  segments_of_video, frames_per_segment=5, language="en"):
    
    _, duration, rotation = profile_video(video_path)

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
    #all_frame_dims = []
    
    for i, chunk in enumerate(audio_chunks):
        print(f"Processing segment {i+1}/{len(audio_chunks)}...")

        events = acoustic_event_branch(chunk)
        transcript = speech_branch(chunk, language)

        frames = extract_key_frames(video_path, 
                                    num_of_frames=frames_per_segment, 
                                    start_time=i * interval_s, 
                                    end_time=(i + 1) * interval_s, 
                                    rotation=rotation)
        all_frames.append(frames)
        #all_frame_dims.append(frame_dims)
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
    
    return results, all_frames, rotation

def main_process(video_path, init_prompt, language, segments_of_video, frames_per_segment):
    results, all_frames, rotation = analyze_video(video_path, int(segments_of_video), int(frames_per_segment), language)
    
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
            #old_width, old_height = all_frame_dims[hazard_segment_index][frame_index]
            
            img_data = base64.b64decode(base64_frame)
            np_arr = np.frombuffer(img_data, np.uint8) #Creates a 1D NumPy array from the raw bytes buffer.
            #convert from a NumPy array back into an actual image matrix that OpenCV can work with,kept as BGR format 
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            most_important_hazard_type = most_important_hazard_region.get("caption", "Hazard")
            hazard_image_with_box = img.copy() #keep BGR order for cv2 drawing
            

            bounding_boxes = grounding_dino(cv2.cvtColor(hazard_image_with_box, cv2.COLOR_BGR2RGB), most_important_hazard_type)
            for score, label, box in zip(bounding_boxes["scores"], bounding_boxes["labels"], bounding_boxes["boxes"]):
                xmin, ymin, xmax, ymax = map(int, box.cpu().numpy())
                if len(label) > 0 :                                        
                    # Draw rectangle and label only for the most important hazard type
                    cv2.rectangle(hazard_image_with_box, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
                    cv2.putText(hazard_image_with_box, f"{label}:{score:.2f}", (xmin+2, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    return segmented_results_text, response, cv2.cvtColor(hazard_image_with_box, cv2.COLOR_BGR2RGB), hazard_info

# Gradio Interface
prompt_templates = {
    "Default": """
You are an AI assistant that describes video scenes to a person who uses a wheelchair.
Use all provided context from the consecutive **SEGMENTS**.
When generating the output:

* Prioritize **Detected Speech** and **Visual Descriptions** over other sounds.
* First, give a **detailed description** of what is happening in the scene (including people’s actions, objects, environment, and spatial relations).
* Then, to ensure the safety and convenience of the wheelchair user, recommend the most appropriate **NextAction**.

The output **must strictly follow** this JSON format (no extra text outside the JSON):

```json
{
  "Description": "A detailed description of the hazard, including actions, objects, and relations.",
  "NextAction": "Go forward | Turn left | Turn right | Stop | Go backward | Wait | Look around | Run away | Navigate to location | Avoid obstacle | Adjust speed | Follow person | Return to charger | Emergency stop | Open door | Call elevator | Request help | Daily schedule"
}
```
""",
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