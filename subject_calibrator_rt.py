"""
Calibração per-subject em tempo real — TEV9 / C37 (sem pose).

Adaptado de DeteccaoFadigaAgentic/SALTE_INFERENCE/subject_calibrator_rt.py:
- Removidos campos de pose (pitch/yaw/roll) em SubjectBaseline e CalibratedFrame
- Z-Norm aplicada apenas a EAR e MAR

Fluxo:
- Warm-up: coleta frames por `search_sec` segundos (padrão 120s)
- Ao final, seleciona segmento de `baseline_sec` com maior EAR médio (C6-V2)
- Calcula baseline (mean, std) para EAR e MAR
- Após calibração, Z-Normaliza cada frame em tempo real

Ref:
- C5:  Z-Norm per subject (nunca global)
- C6-V2: Calibração pelo segmento de maior EAR nos primeiros 120s
- C13: PERCLOS sobre EAR raw (nunca Z-normalizado — preservado downstream)
- C37: 15 features, sem pose
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from .feature_extractor_rt import RTFrameFeatures
except ImportError:
    from feature_extractor_rt import RTFrameFeatures


@dataclass
class CalibrationConfig:
    """Configuração da calibração per-subject (C37)."""

    fps: int = 30
    search_sec: float = 120.0       # C6-V2: primeiros 120s
    baseline_sec: float = 30.0      # Segmento de baseline
    stride_sec: float = 1.0         # Stride para busca do melhor segmento
    min_face_ratio: float = 0.90    # Mínimo de face no segmento
    fallback_to_first: bool = True  # Se nenhum segmento válido, usa os primeiros

    # Per-feature min_std fisiológicos — evita Z-norm explodir quando
    # a pessoa fica muito parada durante o warm-up.
    ear_min_std: float = 0.015      # EAR varia ±0.015 só com blinks naturais
    mar_min_std: float = 0.01       # MAR variação mínima


@dataclass
class SubjectBaseline:
    """Resultado da calibração de um sujeito (C37 — sem pose)."""

    ear_mean: float
    ear_std: float
    mar_mean: float
    mar_std: float
    ear_p90_raw: float = 0.0   # P90 do EAR raw para debug/PERCLOS fallback
    is_valid: bool = True
    segment_start: int = 0
    segment_end: int = 0


@dataclass
class CalibratedFrame:
    """Frame com sinais raw E Z-normalizados (C37 — sem pose).

    - `*_raw`:   valores originais (para PERCLOS e blink detection — C13)
    - `*_znorm`: Z-normalizados per-subject (para agregação de janela)
    """

    timestamp_ms: float
    frame_idx: int
    face_detected: bool

    # Raw (C13)
    ear_avg_raw: float
    ear_l_raw: float
    ear_r_raw: float
    mar_raw: float

    # Z-Normalized (C5)
    ear_avg_znorm: float
    mar_znorm: float


class RTSubjectCalibrator:
    """Calibrador per-subject em tempo real (C37)."""

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.cfg = config or CalibrationConfig()
        self._warmup_buffer: List[RTFrameFeatures] = []
        self._baseline: Optional[SubjectBaseline] = None
        self._search_frames = int(self.cfg.search_sec * self.cfg.fps)

    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> Optional[SubjectBaseline]:
        return self._baseline

    @property
    def warmup_progress(self) -> float:
        if self._baseline is not None:
            return 1.0
        return min(len(self._warmup_buffer) / max(self._search_frames, 1), 1.0)

    def push(self, frame_feats: RTFrameFeatures) -> Optional[CalibratedFrame]:
        """Alimenta um frame durante o warm-up.

        Retorna:
          - None enquanto estiver em warm-up
          - CalibratedFrame após calibração (e para todos os frames seguintes)
        """
        if self._baseline is not None:
            return self._apply_znorm(frame_feats)

        self._warmup_buffer.append(frame_feats)

        if len(self._warmup_buffer) >= self._search_frames:
            self._compute_baseline()
            return self._apply_znorm(frame_feats)

        return None

    def force_calibrate(self) -> bool:
        """Força calibração com os frames coletados até agora.

        Retorna True se conseguiu calibrar, False se dados insuficientes.
        """
        min_frames = int(self.cfg.baseline_sec * self.cfg.fps * 0.5)
        if len(self._warmup_buffer) < min_frames:
            return False
        self._compute_baseline()
        return self._baseline is not None

    def calibrate(self, frame_feats: RTFrameFeatures) -> CalibratedFrame:
        """Aplica Z-Norm a um frame. Requer calibração prévia."""
        if self._baseline is None:
            raise RuntimeError(
                "Calibrador não está calibrado. "
                "Use push() durante warm-up ou force_calibrate()."
            )
        return self._apply_znorm(frame_feats)

    def _compute_baseline(self) -> None:
        """C6-V2: busca o segmento de baseline_sec com maior EAR médio."""
        buf = self._warmup_buffer
        fps = self.cfg.fps
        baseline_frames = int(self.cfg.baseline_sec * fps)
        stride_frames = max(int(self.cfg.stride_sec * fps), 1)

        if len(buf) < baseline_frames:
            baseline_frames = len(buf)

        best_ear = -1.0
        best_start = 0

        for start in range(0, len(buf) - baseline_frames + 1, stride_frames):
            segment = buf[start : start + baseline_frames]

            face_ratio = sum(1 for f in segment if f.face_detected) / len(segment)
            if face_ratio < self.cfg.min_face_ratio:
                continue

            ear_vals = [f.ear_avg for f in segment if f.face_detected]
            if not ear_vals:
                continue
            ear_mean = float(np.mean(ear_vals))

            if ear_mean > best_ear:
                best_ear = ear_mean
                best_start = start

        if best_ear < 0 and self.cfg.fallback_to_first:
            best_start = 0

        segment = buf[best_start : best_start + baseline_frames]
        valid_frames = [f for f in segment if f.face_detected]

        if not valid_frames:
            # Último recurso
            valid_frames = [f for f in buf if f.face_detected]
            if not valid_frames:
                self._baseline = SubjectBaseline(
                    ear_mean=0.3, ear_std=0.05,
                    mar_mean=0.1, mar_std=0.05,
                    is_valid=False,
                    segment_start=0,
                    segment_end=0,
                )
                return

        ears = np.array([f.ear_avg for f in valid_frames])
        mars = np.array([f.mar for f in valid_frames])
        ears_clean = ears[(ears > 0.05) & (ears < 0.80)]
        if len(ears_clean) > 10:
            ears = ears_clean

        # P90 excluindo blinks (para debug/fallback PERCLOS)
        all_valid = [f for f in buf if f.face_detected]
        all_ears = np.array([f.ear_avg for f in all_valid])
        if len(all_ears) > 0:
            median_ear = float(np.median(all_ears))
            blink_thresh = median_ear * 0.7
            open_ears = all_ears[all_ears >= blink_thresh]
            if len(open_ears) > 10:
                ear_p90 = float(np.percentile(open_ears, 90))
            else:
                ear_p90 = float(np.percentile(all_ears, 90))
        else:
            ear_p90 = float(ears.mean())

        self._baseline = SubjectBaseline(
            ear_mean=float(ears.mean()),
            ear_std=max(float(ears.std()), self.cfg.ear_min_std),
            mar_mean=float(mars.mean()),
            mar_std=max(float(mars.std()), self.cfg.mar_min_std),
            ear_p90_raw=ear_p90,
            is_valid=True,
            segment_start=best_start,
            segment_end=best_start + baseline_frames,
        )

    def _apply_znorm(self, f: RTFrameFeatures) -> CalibratedFrame:
        """Aplica Z-Norm usando o baseline calculado."""
        b = self._baseline
        assert b is not None

        if not f.face_detected:
            return CalibratedFrame(
                timestamp_ms=f.timestamp_ms,
                frame_idx=f.frame_idx,
                face_detected=False,
                ear_avg_raw=0.0, ear_l_raw=0.0, ear_r_raw=0.0,
                mar_raw=0.0,
                ear_avg_znorm=0.0, mar_znorm=0.0,
            )

        return CalibratedFrame(
            timestamp_ms=f.timestamp_ms,
            frame_idx=f.frame_idx,
            face_detected=True,
            ear_avg_raw=f.ear_avg,
            ear_l_raw=f.ear_l,
            ear_r_raw=f.ear_r,
            mar_raw=f.mar,
            ear_avg_znorm=(f.ear_avg - b.ear_mean) / b.ear_std,
            mar_znorm=(f.mar - b.mar_mean) / b.mar_std,
        )
