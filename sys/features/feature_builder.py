from typing import Dict, List, Tuple, Optional
import numpy as np

from .F_builder import FMatrixBuilder
from .I_builder import IMatrixBuilder
from .T_builder import TMatrixBuilder
from data.frame_parser import FrameParser
from data.msu_splitter import MSUSplitter
from config.settings import Config
from utils.logger import get_logger

logger = get_logger("FeatureBuilder")


class FeatureBuilder:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.f_builder = FMatrixBuilder(extract_keypoints=True)
        self.i_builder = IMatrixBuilder(self.config.interaction)
        self.t_builder = TMatrixBuilder()
        self.msu_splitter = MSUSplitter(self.config.msu)

        self._msu_ranges = None
        self._msu_durations = None
        self._F = None
        self._I = None
        self._T = None
        self._raw_trajectories = None

    def build(
        self,
        frame_info: Dict[str, List[Dict]],
        skip_splits: bool = False
    ) -> Tuple[List, List, List]:
        if not skip_splits:
            self._msu_ranges, self._msu_durations = self.msu_splitter.split(frame_info)
            logger.info(f"MSU split done: {len(self._msu_ranges)} MSUs")
        else:
            if self._msu_ranges is None or self._msu_durations is None:
                raise ValueError("MSU splits not provided")

        self._F, self._raw_trajectories = self.f_builder.build(
            frame_info, self._msu_ranges, self._msu_durations
        )
        logger.info(f"F matrix done: {len(self._F)} MSUs")

        self._I, _ = self.i_builder.build(self._raw_trajectories)
        logger.info(f"I matrix done: {len(self._I)} MSUs")

        self._T = self.t_builder.build(self._msu_ranges, self._msu_durations)
        logger.info(f"T matrix done: {len(self._T)} MSUs")

        return self._F, self._I, self._T

    def set_msu_splits(
        self,
        msu_ranges: List[Tuple[int, int]],
        msu_durations: List[Dict[int, Tuple[int, int]]]
    ) -> None:
        self._msu_ranges = msu_ranges
        self._msu_durations = msu_durations

    @property
    def F(self) -> List:
        return self._F

    @property
    def I(self) -> List:
        return self._I

    @property
    def T(self) -> List:
        return self._T

    @property
    def msu_ranges(self) -> List:
        return self._msu_ranges

    @property
    def msu_durations(self) -> List:
        return self._msu_durations

    def get_msu_count(self) -> int:
        return len(self._F) if self._F else 0

    def get_msu_features(self, msu_idx: int) -> Optional[Dict]:
        if self._F is None or msu_idx >= len(self._F):
            return None

        return {
            'msu_idx': msu_idx,
            'F': self._F[msu_idx],
            'I': self._I[msu_idx],
            'T': self._T[msu_idx],
            'msu_range': self._msu_ranges[msu_idx] if self._msu_ranges else None,
            'msu_duration': self._msu_durations[msu_idx] if self._msu_durations else None
        }

    def to_numpy(self) -> Dict:
        return {
            'F': [np.array(f) for f in self._F] if self._F else [],
            'I': [np.array(i) for i in self._I] if self._I else [],
            'T': [np.array(t) for t in self._T] if self._T else []
        }

    def summary(self) -> str:
        if self._F is None:
            return "Feature matrices not built"

        lines = [
            "=" * 50,
            "MSU Feature Matrix Summary",
            "=" * 50,
            f"MSU count: {len(self._F)}",
            ""
        ]

        for i in range(len(self._F)):
            f_shape = (len(self._F[i]), len(self._F[i][0]) if self._F[i] else 0) if self._F[i] else (0, 0)
            i_shape = self._I[i].shape if self._I[i] is not None else (0, 0)
            t_shape = (len(self._T[i]), 2) if self._T[i] else (0, 0)

            lines.append(f"MSU-{i}: F={f_shape}, I={i_shape}, T={t_shape}")

        return "\n".join(lines)