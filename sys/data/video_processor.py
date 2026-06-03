import os
import cv2
from typing import Optional, Tuple, List
from utils.logger import get_logger

logger = get_logger("VideoProcessor")


class VideoProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (1280, 640),
        fps: int = 30,
        enable_denoise: bool = False
    ):
        self.target_size = target_size
        self.fps = fps
        self.enable_denoise = enable_denoise

    def process(self, input_path: str, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.target_size)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, self.target_size)

            if self.enable_denoise:
                resized = cv2.fastNlMeansDenoisingColored(
                    resized, None, h=10, hColor=10,
                    templateWindowSize=7, searchWindowSize=21
                )

            out.write(resized)
            frame_count += 1

        cap.release()
        out.release()

        logger.info(f"Processed video: {input_path} -> {output_path} ({frame_count} frames)")
        return output_path

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')
    ) -> int:
        os.makedirs(output_dir, exist_ok=True)

        count = 0
        for filename in os.listdir(input_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue

            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                self.process(input_path, output_path)
                count += 1
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        logger.info(f"Batch processing complete: {count} videos")
        return count


def convert_video_format(
    input_path: str,
    output_path: str,
    codec: str = 'libx264'
) -> str:
    import subprocess

    cmd = f'ffmpeg -i "{input_path}" -vcodec {codec} -acodec aac -y "{output_path}"'
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    return output_path


class VideoClipper:
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def clip_by_frame_range(
        self,
        video_path: str,
        output_path: str,
        start_frame: int,
        end_frame: int
    ) -> str:
        import subprocess

        start_time = start_frame / self.fps
        duration = (end_frame - start_frame + 1) / self.fps

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        cmd = (
            f'ffmpeg -i "{video_path}" '
            f'-ss {start_time:.3f} -t {duration:.3f} '
            f'-c:v libx264 -c:a aac -y "{output_path}"'
        )

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode('utf-8', errors='ignore')}")
            raise RuntimeError(f"Failed to clip video: {video_path} [{start_frame}, {end_frame}]")

        logger.info(f"Clipped: {video_path} [{start_frame}, {end_frame}] -> {output_path}")
        return output_path

    def clip_multiple(
        self,
        video_path: str,
        output_dir: str,
        frame_ranges: List[Tuple[int, int]],
        prefix: str = "clip"
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        output_paths = []
        for i, (start, end) in enumerate(frame_ranges):
            output_path = os.path.join(output_dir, f"{prefix}_{i:04d}_f{start}-{end}.mp4")
            self.clip_by_frame_range(video_path, output_path, start, end)
            output_paths.append(output_path)

        logger.info(f"Clipped {len(frame_ranges)} clips to {output_dir}")
        return output_paths

    def clip_vsus(
        self,
        video_path: str,
        msu_ranges: List[Tuple[int, int]],
        output_dir: str
    ) -> List[str]:
        return self.clip_multiple(video_path, output_dir, msu_ranges, prefix="vsu")