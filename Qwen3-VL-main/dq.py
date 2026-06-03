import os
import base64
import warnings
import time
import numpy as np
import decord
from decord import VideoReader, cpu
from PIL import Image
from openai import OpenAI

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ===================== Core Configuration =====================
os.environ['DASHSCOPE_API_KEY'] = 'xxxx'  # Replace with your Key
os.environ['OPENAI_BASE_HTTP_API_URL'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DASH_MODEL_ID = 'qwen-vl-plus'
LOCAL_VIDEO_PATH = r'D:\XW2\yolov13\main\video_youtobe\youtobe1.mp4'

# Restructured Prompt: force binding original index + minimal format + must analyze all frames
PROMPT_TEMPLATE = """Please strictly execute the following instructions, do not add any extra text:
1. Analyze the following {batch_num} frames of video, count vehicles for each frame (van type not considered);
2. Output format (one frame per line, no other content): Original frame index X: Y vehicles;
3. Frames with less than 5 vehicles must also be output, do not skip;
4. Only output the above format content, no titles, descriptions, or list symbols."""

# ===================== Utility Functions =====================
def extract_video_frames(video_path, max_frames=147):
    """Extract video frames (keeping original logic)"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    sample_step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, sample_step))[:max_frames]

    frames_np = vr.get_batch(frame_indices).asnumpy()
    frames_pil = []
    for frame in frames_np:
        img = Image.fromarray(frame)
        img = img.resize((640, 360), Image.Resampling.LANCZOS)
        frames_pil.append(img)

    print(f"Video total frames: {total_frames} | Sampled frames: {len(frames_pil)} | Sample step: {sample_step}")
    print(f"Sampled original frame indices: {frame_indices}")
    return frames_pil, frame_indices

def image_to_base64(image):
    """Convert image to Base64"""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

def call_api_with_batch_frames(batch_frames, batch_indices, prompt):
    """Call API for single batch of frames"""
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
        timeout=180
    )

    # Construct request content: text prompt first, then images
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
            max_tokens=4096,
            temperature=0.0,
            stream=False
        )
        cost_time = round(time.time() - start_time, 2)
        print(f"This batch inference completed, time: {cost_time}s")
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"API call failed: {type(e).__name__} - {str(e)}")

def batch_process_all_frames(frames_pil, frame_indices, batch_size=49):
    """Batch process all frames and aggregate results"""
    all_results = []
    total_batches = (len(frames_pil) + batch_size - 1) // batch_size  # Calculate total batches

    for batch_idx in range(total_batches):
        # Split batches
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(frames_pil))
        batch_frames = frames_pil[start:end]
        batch_indices = frame_indices[start:end]

        print(f"\n===== Processing batch {batch_idx+1}/{total_batches} (total {len(batch_frames)} frames) =====")
        # Generate batch-specific prompt (clarify batch frame count)
        batch_prompt = PROMPT_TEMPLATE.format(batch_num=len(batch_frames))
        # Call API
        batch_result = call_api_with_batch_frames(batch_frames, batch_indices, batch_prompt)
        all_results.append(batch_result)

    # Aggregate all batch results
    final_result = "\n".join(all_results)
    # Filter frames with >= 5 vehicles (optional: can also let model filter directly, here as secondary backup)
    filtered_result = []
    for line in final_result.split("\n"):
        line = line.strip()
        if "：" in line and "辆车" in line:
            try:
                frame_num = line.split("：")[0].replace("原始帧序号", "")
                car_num = int(line.split("：")[1].replace("辆车", ""))
                if car_num >= 5:
                    filtered_result.append(line)
            except:
                continue  # Skip lines with format errors

    return final_result, filtered_result

# ===================== Main Execution Logic =====================
if __name__ == "__main__":
    try:
        start_total = time.time()
        # Step 1: Extract all frames
        print("===== Start extracting video key frames =====")
        frames, frame_indices = extract_video_frames(LOCAL_VIDEO_PATH, max_frames=2000)
        extract_time = round(time.time() - start_total, 2)
        print(f"Video frame extraction completed, time: {extract_time}s")

        # Step 2: Batch call API and aggregate
        print("\n===== Start batch calling API to analyze frame content =====")
        all_raw_result, filtered_result = batch_process_all_frames(frames, frame_indices, batch_size=2000)

        # Step 3: Output results
        total_cost = round(time.time() - start_total, 2)
        print("\n" + "="*80)
        print("Raw statistics for all frames (including frames with < 5 vehicles):")
        print("="*80)
        print(all_raw_result)

        print("\n" + "="*80)
        print("Filtered frames (vehicle count >= 5):")
        print("="*80)
        for line in filtered_result:
            print(line)
        print(f"\nAll processing completed, total time: {total_cost}s")

    except Exception as e:
        print(f"\nExecution failed: {str(e)}")