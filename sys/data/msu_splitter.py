from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json

from .frame_parser import FrameParser
from config.settings import MSUConfig
from utils.logger import get_logger

logger = get_logger("MSUSplitter")


class MSUSplitter:
    def __init__(self, config: MSUConfig = None):
        self.config = config or MSUConfig()
        self.msu_ranges = []
        self.msu_durations = []
        self.filtered_ids = set()
        self._obj_frames = {}  # object_id -> list of frames (from object_frame_statistics format)

    def split(
        self,
        frame_info: Dict[str, List[Dict]]
    ) -> Tuple[List[Tuple[int, int]], List[Dict[int, Tuple[int, int]]]]:
        parser = FrameParser(frame_info)

        # Convert to object_frame_statistics format to match main's logic
        self._obj_frames = parser.to_object_frame_statistics()

        self.filtered_ids = self._filter_low_frequency_objects()
        logger.info(f"Filtered objects: {len(self._obj_frames)} -> {len(self.filtered_ids)}")

        first_break = self._find_first_break_point()
        self.msu_ranges.append((0, first_break))
        self.msu_durations.append(self._build_duration_map(0, first_break))

        self._dynamic_split(first_break)

        for i, (s, e) in enumerate(self.msu_ranges):
            logger.info(f"MSU-{i+1}: [{s}, {e}] {e - s + 1} frames")

        return self.msu_ranges, self.msu_durations

    def _filter_low_frequency_objects(self) -> Set[int]:
        """Filter objects that appear in fewer than filter_min_frames."""
        filtered = set()
        for oid, frames in self._obj_frames.items():
            if len(frames) >= self.config.filter_min_frames:
                filtered.add(oid)
        return filtered

    def _id_set_at_frame(self, frame: int) -> Set[int]:
        """Get set of object IDs that appear at given frame."""
        result = set()
        for oid, frames in self._obj_frames.items():
            if oid in self.filtered_ids and frame in frames:
                result.add(oid)
        return result

    def _find_first_break_point(self) -> int:
        """Find first break point - matches main/video_split.py init() logic."""
        # Get frames sorted
        all_frames = sorted(int(k) for k in self._obj_frames.keys())
        if not all_frames:
            return 0

        # Get init_ids at frame 0 (or first frame in data structure)
        init_ids = self._id_set_at_frame(0)
        total_init = len(init_ids) if init_ids else 1

        disjoint_idx = None
        tmp_idx = None

        frames = sorted(set(f for frames in self._obj_frames.values() for f in frames))
        max_data_frame = frames[-1] if frames else 0

        for frame in frames:
            curr_ids = self._id_set_at_frame(frame)

            if frame > 0 and total_init > 0:
                overlap_ratio = len(init_ids & curr_ids) / total_init

                if disjoint_idx is None and init_ids.isdisjoint(curr_ids):
                    disjoint_idx = frame
                if tmp_idx is None and overlap_ratio <= self.config.init_overlap_threshold:
                    tmp_idx = frame

                if disjoint_idx is not None and tmp_idx is not None:
                    break

        # Build initial duration
        init_duration = {}
        for oid, frames in self._obj_frames.items():
            if oid in self.filtered_ids:
                init_duration[oid] = (min(frames), max(frames))

        candidate = tmp_idx or disjoint_idx or max_data_frame

        if candidate - 0 + 1 < self.config.min_window_length:
            candidate = min(0 + self.config.min_window_length - 1, max_data_frame)

        if disjoint_idx is not None and tmp_idx is not None:
            if (disjoint_idx - tmp_idx) / tmp_idx > self.config.init_disjoint_ratio:
                candidate = tmp_idx if tmp_idx - 0 + 1 >= self.config.min_window_length else \
                           min(tmp_idx + (self.config.min_window_length - (tmp_idx - 0 + 1)), max_data_frame)
            else:
                candidate = disjoint_idx if disjoint_idx - 0 + 1 >= self.config.min_window_length else \
                           min(disjoint_idx + (self.config.min_window_length - (disjoint_idx - 0 + 1)), max_data_frame)

        return candidate

    def _build_duration_map(
        self,
        start: int,
        end: int
    ) -> Dict[int, Tuple[int, int]]:
        """Build duration map for objects in given frame range."""
        duration_map = {}

        for oid, frames in self._obj_frames.items():
            if oid not in self.filtered_ids:
                continue

            # Find frames within [start, end]
            in_range = [f for f in frames if start <= f <= end]
            if in_range:
                duration_map[oid] = (min(in_range), max(in_range))

        return duration_map

    def _check_stats(self, durations: Dict[int, Tuple[int, int]]) -> Dict[Tuple[float, float], int]:
        """Check statistics of durations - matches main/check_stats logic."""
        if not durations:
            return {}

        duration_list = [last - first for first, last in durations.values()]
        if not duration_list:
            return {}

        min_dur = min(duration_list)
        max_dur = max(duration_list)

        if min_dur == max_dur:
            return {(float(min_dur), float(max_dur)): len(duration_list)}

        n_bins = 6
        bin_width = (max_dur - min_dur) / n_bins
        bins = defaultdict(int)

        for dur in duration_list:
            bin_idx = min(int((dur - min_dur) / bin_width), n_bins - 1)
            bin_start = min_dur + bin_idx * bin_width
            bin_end = bin_start + bin_width
            bins[(bin_start, bin_end)] += 1

        return {k: v for k, v in bins.items() if v > 0}

    def _dynamic_split(self, first_break: int) -> None:
        """Dynamic split - matches main/dynamic_split logic."""
        all_frames = sorted(set(f for frames in self._obj_frames.values() for f in frames))
        max_frame = all_frames[-1] if all_frames else 0

        init_duration = self.msu_durations[0]
        init_hist = self._check_stats(init_duration)

        if init_hist:
            mode_bin = max(init_hist.items(), key=lambda x: x[1])[0]
            guidance_length = max(
                int((mode_bin[0] + mode_bin[1]) / 2),
                self.config.min_window_length
            )
        else:
            guidance_length = max(first_break + 1, self.config.min_window_length)

        current = first_break + 1

        while current <= max_frame:
            remaining_frames = max_frame - current + 1

            expansion_factor = 1
            window_len = guidance_length * expansion_factor
            window_len = min(window_len, remaining_frames, self.config.max_window_length)
            window_len = max(window_len, self.config.min_window_length)
            end = current + window_len - 1

            # Try expansion
            expanded = False
            for _ in range(self.config.max_expansion_factor):
                duration = self._build_duration_map(current, end)
                if not duration:
                    break

                hist = self._check_stats(duration)
                if not hist:
                    break

                bin_items = sorted(hist.items(), key=lambda x: x[0])
                counts = [count for (_, _), count in bin_items]
                max_count = max(counts) if counts else 0
                total_objects = sum(counts)

                if total_objects == 0:
                    break

                current_ratio = max_count / total_objects
                max_bin = max(hist.items(), key=lambda x: x[1])[0]
                max_bin_length = max_bin[1] - max_bin[0]

                if current_ratio > self.config.expansion_threshold and expansion_factor < self.config.max_expansion_factor:
                    new_expansion_factor = expansion_factor + 1
                    new_window_len = guidance_length * new_expansion_factor
                    new_window_len = max(new_window_len, max_bin_length)
                    new_window_len = min(new_window_len, remaining_frames, self.config.max_window_length)
                    new_end = current + new_window_len - 1

                    if new_end > end and new_end <= max_frame:
                        expanded = True
                        expansion_factor = new_expansion_factor
                        end = new_end
                    else:
                        break
                else:
                    break

            # Window adjustment if not expanded
            if not expanded:
                current_duration = self._build_duration_map(current, end)
                if current_duration:
                    current_hist = self._check_stats(current_duration)
                    if current_hist:
                        max_bin = max(current_hist.items(), key=lambda x: x[1])[0]
                        max_bin_length = max_bin[1] - max_bin[0]

                        adjusted_length = max_bin_length if max_bin_length >= self.config.min_window_length else self.config.min_window_length
                        adjusted_length = min(adjusted_length, self.config.max_window_length, remaining_frames)

                        if adjusted_length != window_len:
                            end = current + adjusted_length - 1
                            window_len = adjusted_length

            # Final constraint check
            final_window_length = end - current + 1
            if final_window_length < self.config.min_window_length:
                new_end = current + self.config.min_window_length - 1
                end = new_end if new_end <= max_frame else max_frame
            elif final_window_length > self.config.max_window_length:
                end = current + self.config.max_window_length - 1

            # Record MSU
            self.msu_ranges.append((current, end))
            self.msu_durations.append(self._build_duration_map(current, end))

            # Update guidance length
            final_hist = self._check_stats(self.msu_durations[-1])
            if final_hist:
                mode_bin = max(final_hist.items(), key=lambda x: x[1])[0]
                new_guidance = int((mode_bin[0] + mode_bin[1]) / 2)
                guidance_length = max(
                    int(guidance_length * 0.5 + new_guidance * 0.5),
                    self.config.min_window_length
                )

            current = end + 1

    def get_msu_for_frame(self, frame_id: int) -> int:
        for i, (start, end) in enumerate(self.msu_ranges):
            if start <= frame_id <= end:
                return i
        return -1


def load_msu_splits(json_path: str, frame_info_path: str = None) -> Tuple[List, List]:
    if frame_info_path:
        parser = FrameParser()
        parser.load_from_json(frame_info_path)
        splitter = MSUSplitter()
        return splitter.split(parser.frame_info)
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('msu_ranges', []), data.get('msu_durations', [])