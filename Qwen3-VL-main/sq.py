import os
import base64
import warnings
import time
import numpy as np
import decord
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import cv2

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ===================== Core Configuration =====================
os.environ['DASHSCOPE_API_KEY'] = 'xxx'  # Replace with your actual API Key
os.environ['OPENAI_BASE_HTTP_API_URL'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DASH_MODEL_ID = 'qwen-vl-plus'
LOCAL_VIDEO_PATH = r'D:\XW2\yolov13\main\video_youtobe\youtobe1.mp4'
OUTPUT_VIDEO_PATH = r'D:\XW2\yolov13\main\video_youtobe\annotated_video.mp4'  # Output annotated video path
FONT_PATH = "simhei.ttf"  # Local font file path (for drawing Chinese, comment out if not available)

# [Key Prompt] Require output annotation box coordinates + ID + judgment result
PROMPT_TEMPLATE = """Please strictly output in the following format, do not add any extra text:
1. For {batch_num} frames of video, perform vehicle object tracking, assign a unique ID to each vehicle;
2. Judgment rule: same ID vehicle annotation box gradually变小 → away from screen (meets condition);
3. Output format (one vehicle per line, fields separated by |):
ID|start frame|end frame|start box(W1,H1,X1,Y1)|end box(W2,H2,X2,Y2)|meets condition or not(yes/no)
4. If no vehicles, output "none", nothing else.
Note: Annotation box format is (width, height, top-left X, top-left Y), based on 640x360 resolution."""

# ===================== Utility Functions =====================
def extract_video_frames(video_path, max_frames=147):
    """Extract video frames + get original resolution (for video synthesis)"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    sample_step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, sample_step))[:max_frames]

    frames_np = vr.get_batch(frame_indices).asnumpy()
    frames_pil = []
    frames_original = []  # Store original resolution frames for video synthesis
    for frame in frames_np:
        # Scaled frames for API analysis
        img_pil = Image.fromarray(frame).resize((640, 360), Image.Resampling.LANCZOS)
        frames_pil.append(img_pil)
        # Original frames for annotation synthesis
        frames_original.append(frame)

    # Get original video resolution
    original_h, original_w = frames_np[0].shape[:2]
    print(f"Video basic info:")
    print(f"Total frames: {total_frames} | Sampled frames: {len(frames_pil)} | Sample step: {sample_step}")
    print(f"FPS: {fps} | Original resolution: {original_w}x{original_h}")
    print(f"Sampled original frame indices: {frame_indices}")
    return frames_pil, frames_original, frame_indices, fps, (original_w, original_h)

def image_to_base64(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def parse_model_output(output_str):
    """Parse vehicle annotation info from model output"""
    car_info = []
    if output_str.strip() == "none":
        return car_info
    for line in output_str.strip().split("\n"):
        parts = line.split("|")
        if len(parts) != 6:
            continue
        car_id, start_idx, end_idx, box1, box2, is_match = parts
        # Parse annotation box
        try:
            w1, h1, x1, y1 = map(int, box1.strip("()").split(","))
            w2, h2, x2, y2 = map(int, box2.strip("()").split(","))
            car_info.append({
                "id": car_id,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "box1": (x1, y1, w1, h1),
                "box2": (x2, y2, w2, h2),
                "is_match": is_match == "yes"
            })
        except:
            continue
    return car_info

def draw_annotation_on_frame(frame_np, car_info_list, frame_idx, original_size, scaled_size=(640, 360)):
    """Draw annotation boxes and IDs on original frame"""
    original_w, original_h = original_size
    scaled_w, scaled_h = scaled_size
    # Calculate scale ratio
    scale_x = original_w / scaled_w
    scale_y = original_h / scaled_h

    # Convert to PIL for drawing
    frame_pil = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(frame_pil)
    # Load font (use default font if font file not available)
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except:
        font = ImageFont.load_default(size=20)

    # Draw annotation box for each car
    for car in car_info_list:
        if car["start_idx"] <= frame_idx <= car["end_idx"]:
            # Select annotation box for current frame
            if frame_idx == car["start_idx"]:
                x, y, w, h = car["box1"]
            elif frame_idx == car["end_idx"]:
                x, y, w, h = car["box2"]
            else:
                continue  # Only annotate start and end frames

            # Scale back to original resolution
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Color differentiation: meets condition (away from screen) is red, otherwise green
            color = (255, 0, 0) if car["is_match"] else (0, 255, 0)
            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            # Draw ID and judgment result
            text = f"ID:{car['id']} {'[yes]' if car['is_match'] else '[no]'}"
            draw.text((x, y-25), text, fill=color, font=font)

    return np.array(frame_pil)

def generate_annotated_video(frames_original, frame_indices, car_info_all, fps, original_size, output_path):
    """Generate video with annotations"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, original_size)

    # Build mapping from frame index to annotation info
    frame_to_cars = {}
    for car in car_info_all:
        for idx in range(car["start_idx"], car["end_idx"] + 1):
            if idx not in frame_to_cars:
                frame_to_cars[idx] = []
            frame_to_cars[idx].append(car)

    # Draw frame by frame and write to video
    for frame_idx, frame_np in zip(frame_indices, frames_original):
        annotated_frame = draw_annotation_on_frame(frame_np, frame_to_cars.get(frame_idx, []), frame_idx, original_size)
        # Convert to BGR format (cv2 requirement)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        out.write(annotated_frame_bgr)

    out.release()
    print(f"Annotated video saved to: {output_path}")

def call_api_with_batch_frames(batch_frames, batch_indices, prompt):
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
        timeout=300
    )

    content = [{"type": "text", "text": prompt}]
    for idx, (frame, frame_num) in enumerate(zip(batch_frames, batch_indices)):
        frame_b64 = image_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": frame_b64}
        })
        print(f"Added frame {idx+1} in batch (original index: {frame_num})")

    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=DASH_MODEL_ID,
            messages=[{"role": "user", "content": content}],
            max_tokens=8192,
            temperature=0.0,
            stream=False
        )
        cost_time = round(time.time() - start_time, 2)
        print(f"This batch inference completed, time: {cost_time}s")
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"API call failed: {type(e).__name__} - {str(e)}")

def batch_process_all_frames(frames_pil, frames_original, frame_indices, fps, original_size, batch_size=49):
    total_away_cars = 0
    all_car_info = []  # Store all vehicle annotation info

    total_batches = (len(frames_pil) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(frames_pil))
        batch_frames = frames_pil[start:end]
        batch_indices = frame_indices[start:end]

        print(f"\n===== Processing batch {batch_idx+1}/{total_batches} (total {len(batch_frames)} frames) =====")
        batch_prompt = PROMPT_TEMPLATE.format(batch_num=len(batch_frames))
        batch_result = call_api_with_batch_frames(batch_frames, batch_indices, batch_prompt)

        # Parse this batch's vehicle info
        batch_car_info = parse_model_output(batch_result)
        all_car_info.extend(batch_car_info)
        # Count this batch's vehicles meeting condition
        batch_away_num = sum([1 for car in batch_car_info if car["is_match"]])
        total_away_cars += batch_away_num
        print(f"This batch vehicles away from screen: {batch_away_num}")

    # Generate annotated video
    generate_annotated_video(frames_original, frame_indices, all_car_info, fps, original_size, OUTPUT_VIDEO_PATH)
    return total_away_cars

# ===================== Main Execution Logic =====================
if __name__ == "__main__":
    try:
        start_total = time.time()
        # Step 1: Extract frames (with original frames + resolution)
        print("===== Start extracting video key frames =====")
        frames_pil, frames_original, frame_indices, fps, original_size = extract_video_frames(LOCAL_VIDEO_PATH, max_frames=147)
        extract_time = round(time.time() - start_total, 2)
        print(f"Video frame extraction completed, time: {extract_time}s")

        # Step 2: Batch analysis + generate annotated video
        print("\n===== Start batch analysis and generate annotated video =====")
        total_away_cars = batch_process_all_frames(frames_pil, frames_original, frame_indices, fps, original_size, batch_size=49)

        # Step 3: Output results
        total_cost = round(time.time() - start_total, 2)
        print("\n" + "="*80)
        print(f"Total vehicles away from screen in video: {total_away_cars}")
        print(f"Annotated video path: {OUTPUT_VIDEO_PATH}")
        print("="*80)
        print(f"\nAll processing completed, total time: {total_cost}s")

    except Exception as e:
        print(f"\nExecution failed: {str(e)}")