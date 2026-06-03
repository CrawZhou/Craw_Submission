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
os.environ['DASHSCOPE_API_KEY'] = 'xxx'  # Replace with your actual API Key
os.environ['OPENAI_BASE_HTTP_API_URL'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DASH_MODEL_ID = 'qwen-vl-plus'
A_VIDEO_PATH = r'D:\XW2\yolov13\main\video_youtobe\youtobe1.mp4'  # Original video A
B_VIDEO_PATH = r'D:\XW2\yolov13\main\highway_msu\msu_42.mp4'       # Clip video B
FRAME_SAMPLE_RATE = 1  # Sample rate (1 frame per 30 frames)
SIMILARITY_THRESHOLD = 0.8  # Similarity threshold
TOP_K_CANDIDATES = 5  # Keep Top3 candidate results (as reference when no match)
API_RETRY_TIMES = 2  # API call retry times on failure

# ===================== Utility Functions =====================
def extract_video_frames(video_path, sample_rate=5, max_frames=200):
    """Extract key frames from video (with error handling + empty frame filtering)"""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        raise Exception(f"Failed to open video {video_path}: {str(e)}")

    total_frames = len(vr)
    if total_frames == 0:
        raise Exception(f"Video {video_path} has no valid frames")

    fps = vr.get_avg_fps()
    frame_indices = []
    frames_pil = []

    # Extract frames by sample rate (with boundary check)
    for idx in range(0, total_frames, sample_rate):
        if len(frames_pil) >= max_frames or idx >= total_frames:
            break
        try:
            frame_np = vr[idx].asnumpy()
            if frame_np.size == 0:
                continue
            frame_pil = Image.fromarray(frame_np).resize((640, 360), Image.Resampling.LANCZOS)
            frames_pil.append(frame_pil)
            frame_indices.append(idx)
        except:
            continue

    if len(frames_pil) == 0:
        raise Exception(f"Cannot extract valid frames from video {video_path}")

    print(f"Video {os.path.basename(video_path)} processed")
    print(f"Total frames: {total_frames} | Sampled frames: {len(frames_pil)} | FPS: {fps:.2f}")
    print(f"Sampled frame indices: {frame_indices[:5]}... (total {len(frame_indices)} frames)")
    return frames_pil, frame_indices, fps

def image_to_base64(image):
    """Convert image to Base64 (with error handling)"""
    try:
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        raise Exception(f"Failed to convert image to Base64: {str(e)}")

def get_frame_similarity_by_api(client, frame_a_pil, frame_b_pil, retry_times=2):
    """Call API to calculate similarity (with retry mechanism + result validation)"""
    a_b64 = image_to_base64(frame_a_pil)
    b_b64 = image_to_base64(frame_b_pil)

    prompt = """Please strictly compare the visual content similarity of these two video frames, output only a number between 0-1 (1=identical, 0=completely different),
do not output any text, units, or punctuation, e.g.: 0.98"""

    # Retry mechanism
    for retry in range(retry_times + 1):
        try:
            completion = client.chat.completions.create(
                model=DASH_MODEL_ID,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": a_b64}},
                        {"type": "image_url", "image_url": {"url": b_b64}}
                    ]
                }],
                max_tokens=10,
                temperature=0.0,
                timeout=600
            )

            # Result validation
            score_str = completion.choices[0].message.content.strip()
            # Remove non-numeric characters (in case model outputs extra content)
            score_str = ''.join([c for c in score_str if c in '0123456789.'])
            score = float(score_str)
            # Constrain range to 0-1
            score = max(0.0, min(1.0, score))
            return score

        except ValueError:
            if retry < retry_times:
                print(f"Similarity parsing failed (retry {retry+1}/{retry_times}): {score_str}")
                time.sleep(1)
                continue
            return 0.0
        except Exception as e:
            if retry < retry_times:
                print(f"API call failed (retry {retry+1}/{retry_times}): {str(e)}")
                time.sleep(1)
                continue
            return 0.0

def locate_clip_in_video_by_api(a_frames, a_indices, a_fps, b_frames, b_indices):
    """Locate clip in video based on API (optimized: TopK candidates + progress bar + timing stats)"""
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
        timeout=120
    )

    print("\n===== Start calling API for video clip relocation =====")
    b_length = len(b_frames)
    if b_length == 0:
        raise Exception("Clip B has no valid frames")
    if len(a_frames) < b_length:
        raise Exception("Sampled frames in original video A is less than in clip B, cannot match")

    # Store all candidate results (start index + average similarity)
    candidates = []
    total_checks = len(a_frames) - b_length + 1
    start_time = time.time()

    # Iterate through all possible start positions in A
    for a_start_idx in range(total_checks):
        current_similarities = []
        # Compare each frame of B clip
        for b_idx in range(b_length):
            a_frame = a_frames[a_start_idx + b_idx]
            b_frame = b_frames[b_idx]
            sim = get_frame_similarity_by_api(client, a_frame, b_frame, API_RETRY_TIMES)
            current_similarities.append(sim)

        # Calculate average similarity and save candidate
        avg_sim = np.mean(current_similarities)
        candidates.append((a_start_idx, avg_sim))

        # Progress hint (output every 10 times to reduce screen clutter)
        if a_start_idx % max(1, total_checks//10) == 0 or a_start_idx == total_checks-1:
            elapsed = time.time() - start_time
            progress = (a_start_idx + 1) / total_checks * 100
            print(f"Progress: {progress:.1f}% | Checked {a_start_idx+1}/{total_checks} | Elapsed: {elapsed:.2f}s")

    # Sort by similarity and take TopK
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:TOP_K_CANDIDATES]

    # Determine best match
    best_match_a_start_idx = -1
    best_avg_similarity = 0.0
    if top_candidates and top_candidates[0][1] >= SIMILARITY_THRESHOLD:
        best_match_a_start_idx = top_candidates[0][0]
        best_avg_similarity = top_candidates[0][1]

    # Calculate final start and end frames
    if best_match_a_start_idx == -1:
        return -1, -1, 0.0, a_fps, top_candidates

    a_start_frame = a_indices[best_match_a_start_idx]
    a_end_frame = a_indices[best_match_a_start_idx + b_length - 1]
    return a_start_frame, a_end_frame, best_avg_similarity, a_fps, top_candidates

# ===================== Main Execution Logic =====================
if __name__ == "__main__":
    try:
        total_start = time.time()
        # Step 1: Extract key frames from A and B videos
        print("===== Extract key frames from original video A =====")
        a_frames, a_indices, a_fps = extract_video_frames(A_VIDEO_PATH, FRAME_SAMPLE_RATE)

        print("\n===== Extract key frames from clip video B =====")
        b_frames, b_indices, b_fps = extract_video_frames(B_VIDEO_PATH, FRAME_SAMPLE_RATE)

        # Step 2: Call API to locate clip position
        a_start_frame, a_end_frame, similarity, fps, top_candidates = locate_clip_in_video_by_api(
            a_frames, a_indices, a_fps, b_frames, b_indices
        )

        # Step 3: Output results
        total_cost = round(time.time() - total_start, 2)
        print("\n" + "="*80)
        print("Video clip relocation result based on API (reviewer baseline)")
        print("="*80)

        if a_start_frame == -1:
            print(f"No matching clip found with threshold ({SIMILARITY_THRESHOLD})")
            # Output TopK candidates (for reference)
            if top_candidates:
                print(f"\nTop{TOP_K_CANDIDATES} candidate matches (lower similarity):")
                for i, (start_idx, sim) in enumerate(top_candidates):
                    if sim < 0.1:
                        continue
                    frame = a_indices[start_idx]
                    time_stamp = round(frame / fps, 2)
                    print(f"  Candidate {i+1}: Video A frame {frame} ({time_stamp}s) | Similarity: {sim:.2%}")
        else:
            start_time = round(a_start_frame / fps, 2)
            end_time = round(a_end_frame / fps, 2)
            print(f"Matching clip found!")
            print(f"Start frame in A: {a_start_frame} | Start time: {start_time}s")
            print(f"End frame in A: {a_end_frame} | End time: {end_time}s")
            print(f"Average matching similarity: {similarity:.2%}")
            print(f"Original video FPS: {fps:.2f} FPS")

        print(f"\nTotal processing completed, total time: {total_cost}s")

    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
        # Print detailed traceback (for debugging)
        import traceback
        traceback.print_exc()