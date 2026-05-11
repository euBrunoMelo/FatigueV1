"""
Window factory online — TEV9 / C37 (15 features, sem pose).

Adaptado de DeteccaoFadigaAgentic/SALTE_INFERENCE/window_factory_rt.py:
- Removido: pitch_mean, pitch_std, yaw_std, roll_std (4 features de pose)
- Removido: import de FEATURE_NAMES_19 do model_loader
- Mantido: microsleep_count / microsleep_total_ms para overlay/logging
  (NÃO entram no vetor do modelo, mas alimentam debug e regras)

Correções herdadas do original (mantidas):
- FIX 1: EAR velocity sobre EAR raw suavizado (median k=5)
- FIX 2: PERCLOS baseline = mean EAR do best-segment (alinha com FFV5)
- FIX 3: Blink velocity onset = local max PRE-blink no sinal suavizado
- FIX 4: Microsleep filters (median, blink overlap, purity)
- FIX 5: C22 clamp [0.01, 5] EAR/s
- FIX 6: perclos_p80_max = pico de rolling 5s (não max binário)
- FIX-RT-2 (C32): Z-Score Clamp [-3, +3] em features Z-normed

Respeita:
- C5:  Z-Norm per-subject para EAR/MAR stats
- C13: PERCLOS sobre EAR raw (nunca Z-normalizado)
- C21: Velocidades em EAR/s
- C22: Bounded derived features (blink_vel in [0.01, 5])
- C32: Z-Score clamp RT em [-3, +3]
- C37: 15 features, sem pose
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .subject_calibrator_rt import CalibratedFrame
except ImportError:
    from subject_calibrator_rt import CalibratedFrame


# C37: 15 features na ordem do inference_config.json do TEV9
FEATURE_NAMES_15 = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_vel_mean",
    "ear_vel_std",
    "mar_mean",
    "blink_count",
    "blink_rate_per_min",
    "blink_mean_dur_ms",
    "perclos_p80_mean",
    "perclos_p80_max",
    "blink_closing_vel_mean",
    "blink_opening_vel_mean",
    "long_blink_pct",
    "blink_regularity",
]

BLINK_CLOSE_FACTOR = 0.65
BLINK_OPEN_FACTOR = 0.80


@dataclass
class RTWindowConfig:
    fps: int = 30
    window_sec: float = 15.0
    stride_infer: int = 60
    min_valid_ratio: float = 0.80
    # PERCLOS: offline FFV5 usa 0.80 (FHWA P80); no RT com picamera2 a
    # calibração captura pico de alerta e a operação é mais relaxada,
    # gap de ~20% no EAR. Fator 0.65 compensa esse domain shift.
    perclos_factor: float = 0.65
    # C32: clamp Z-scores em [-zscore_clamp, +zscore_clamp].
    # ±3σ cobre 99.7% da distribuição; ear_z=-3 ainda é sinal forte de Danger.
    zscore_clamp: float = 3.0


class OnlineWindowFactory:
    """Mantém buffer de CalibratedFrames e emite janelas agregadas de 15 features (TEV9).

    Usa:
      - Campos *_znorm para: EAR stats, MAR
      - Campos *_raw   para: EAR velocity, PERCLOS, blinks, microsleeps
    """

    def __init__(self, config: Optional[RTWindowConfig] = None) -> None:
        self.cfg = config or RTWindowConfig()
        self.window_frames = int(self.cfg.window_sec * self.cfg.fps)
        self._buffer: List[CalibratedFrame] = []
        self._since_last_emit = 0

        # FIX 2: baseline PERCLOS externo (ear_mean do best-segment)
        self._perclos_baseline_ear: Optional[float] = None

    def set_perclos_baseline(self, ear_baseline: float) -> None:
        """Define baseline EAR para PERCLOS (FIX 2)."""
        self._perclos_baseline_ear = ear_baseline

    @staticmethod
    def _median_filter(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
        """Median filter puro numpy. Remove spikes de 1-2 frames."""
        if len(arr) < kernel:
            return arr.copy()
        half_k = kernel // 2
        padded = np.pad(arr, half_k, mode="edge")
        result = np.empty_like(arr)
        for i in range(len(arr)):
            result[i] = np.median(padded[i : i + kernel])
        return result

    def push(self, frame: CalibratedFrame) -> Optional[Dict[str, float]]:
        """Adiciona frame ao buffer. Retorna dict de features quando janela pronta."""
        self._buffer.append(frame)
        if len(self._buffer) > self.window_frames:
            self._buffer = self._buffer[-self.window_frames:]

        self._since_last_emit += 1
        if (
            len(self._buffer) < self.window_frames
            or self._since_last_emit < self.cfg.stride_infer
        ):
            return None

        self._since_last_emit = 0
        return self._aggregate_current_window()

    def _aggregate_current_window(self) -> Optional[Dict[str, float]]:
        if len(self._buffer) < self.window_frames:
            return None

        window = self._buffer[-self.window_frames:]
        face_mask = np.array([f.face_detected for f in window], dtype=bool)
        valid_ratio = float(face_mask.mean())
        if valid_ratio < self.cfg.min_valid_ratio:
            return None

        # Z-Normed (EAR stats, MAR)
        ear_z = np.array([f.ear_avg_znorm for f in window], dtype=np.float32)
        mar_z = np.array([f.mar_znorm for f in window], dtype=np.float32)

        # FIX-RT-2 (C32): Z-Score Clamp
        ZSCORE_CLAMP = self.cfg.zscore_clamp
        ear_z = np.clip(ear_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)
        mar_z = np.clip(mar_z, -ZSCORE_CLAMP, ZSCORE_CLAMP)

        # Raw (EAR velocity, PERCLOS, blinks)
        ear_raw = np.array([f.ear_avg_raw for f in window], dtype=np.float32)

        # FIX 1+5: median filter antes de velocidades
        ear_smooth = self._median_filter(ear_raw)

        feats: Dict[str, float] = {}

        # === EAR stats (Z-Normed, clampado C32) ===
        feats["ear_mean"] = float(ear_z.mean())
        feats["ear_std"] = float(ear_z.std())
        feats["ear_min"] = float(ear_z.min())

        # === EAR velocity (raw suavizado — FIX 1) ===
        ear_vel_raw = np.diff(ear_smooth, prepend=ear_smooth[0]) * self.cfg.fps
        feats["ear_vel_mean"] = float(ear_vel_raw.mean())
        feats["ear_vel_std"] = float(ear_vel_raw.std())

        # === MAR (Z-Normed, clampado C32) ===
        feats["mar_mean"] = float(mar_z.mean())

        # === Blink stats (raw + smooth — C13, FIX 3+5) ===
        blink_stats = self._compute_blink_stats(ear_raw, ear_smooth)
        feats.update(blink_stats)

        # === PERCLOS + microsleeps (raw — C13, FIX 2+4+6) ===
        perclos_stats, micro_stats = self._compute_perclos_and_microsleeps(
            ear_raw, ear_smooth, face_mask
        )
        feats.update(perclos_stats)
        feats.update(micro_stats)  # overlay/logging, NÃO vai no vetor do modelo

        # Validação: todas as 15 features do modelo devem estar presentes
        missing = [f for f in FEATURE_NAMES_15 if f not in feats]
        if missing:
            raise RuntimeError(
                f"WindowFactoryRT não preencheu todas as features: "
                f"faltando {missing}"
            )

        return feats

    def _compute_blink_stats(
        self, ear_raw: np.ndarray, ear_smooth: np.ndarray
    ) -> Dict[str, float]:
        """Detecção de blinks usando EAR raw (C13).

        FIX 3: onset via local max PRE-blink no sinal suavizado.
        FIX 5: velocidades clampadas em [0.01, 5] EAR/s (C22).
        """
        fps = self.cfg.fps
        window_sec = len(ear_raw) / max(fps, 1)

        zeros = {
            "blink_count": 0.0,
            "blink_rate_per_min": 0.0,
            "blink_mean_dur_ms": 0.0,
            "blink_closing_vel_mean": 0.0,
            "blink_opening_vel_mean": 0.0,
            "long_blink_pct": 0.0,
            "blink_regularity": 0.0,
        }

        valid_ear = ear_raw[~np.isnan(ear_raw)]
        if len(valid_ear) == 0:
            return zeros

        win_mean = float(valid_ear.mean())
        close_thresh = max(BLINK_CLOSE_FACTOR * win_mean, 0.20)
        open_thresh = max(BLINK_OPEN_FACTOR * win_mean, 0.28)

        blink_state = "open"
        blink_start: Optional[int] = None
        segments: List[Tuple[int, int]] = []
        for i, ear in enumerate(ear_raw):
            if np.isnan(ear):
                continue
            if blink_state == "open" and ear < close_thresh:
                blink_state = "closed"
                blink_start = i
            elif blink_state == "closed" and ear > open_thresh:
                blink_state = "open"
                if blink_start is not None:
                    segments.append((blink_start, i))
                blink_start = None
        if blink_state == "closed" and blink_start is not None:
            segments.append((blink_start, len(ear_raw)))

        min_blink_frames = 2
        blinks = [(s, e) for (s, e) in segments if (e - s) >= min_blink_frames]

        n_blinks = len(blinks)
        blink_count = float(n_blinks)
        blink_rate = (blink_count / window_sec) * 60.0 if window_sec > 0 else 0.0

        blink_durs_ms: List[float] = []
        closing_vels: List[float] = []
        opening_vels: List[float] = []
        blink_starts: List[int] = []
        long_blinks = 0

        pre_blink_lookback = max(int(fps * 0.2), 3)

        for s, e in blinks:
            seg = ear_raw[s:e]
            if len(seg) == 0:
                continue
            peak_offset = int(np.nanargmin(seg))
            peak_frame = s + peak_offset
            ear_peak = float(ear_smooth[peak_frame])

            # FIX 3+5: onset/offset via ear_smooth
            lookback_start = max(0, s - pre_blink_lookback)
            pre_blink_region = ear_smooth[lookback_start : s + 1]
            if len(pre_blink_region) > 0:
                onset_local_idx = int(np.nanargmax(pre_blink_region))
                onset_frame = lookback_start + onset_local_idx
                ear_onset = float(ear_smooth[onset_frame])
            else:
                onset_frame = s
                ear_onset = float(ear_smooth[s])

            post_blink_end = min(len(ear_smooth), e + pre_blink_lookback)
            post_blink_region = ear_smooth[e - 1 : post_blink_end]
            if len(post_blink_region) > 0:
                offset_local_idx = int(np.nanargmax(post_blink_region))
                offset_frame = (e - 1) + offset_local_idx
                ear_offset = float(ear_smooth[offset_frame])
            else:
                offset_frame = e - 1
                ear_offset = float(ear_smooth[e - 1])

            duration_frames = e - s
            duration_ms = duration_frames / max(fps, 1) * 1000.0
            blink_durs_ms.append(duration_ms)

            closing_frames = max(peak_frame - onset_frame, 1)
            closing_vel = abs(ear_onset - ear_peak) / (closing_frames / max(fps, 1))

            opening_frames = max(offset_frame - peak_frame, 1)
            opening_vel = abs(ear_offset - ear_peak) / (opening_frames / max(fps, 1))

            # FIX 5: clamp fisiológico [0.01, 5] EAR/s
            closing_vel = float(np.clip(closing_vel, 0.01, 5.0))
            opening_vel = float(np.clip(opening_vel, 0.01, 5.0))

            closing_vels.append(closing_vel)
            opening_vels.append(opening_vel)

            if duration_ms > 300.0:
                long_blinks += 1

            blink_starts.append(s)

        blink_mean_dur_ms = float(np.mean(blink_durs_ms)) if blink_durs_ms else 0.0
        closing_mean = float(np.mean(closing_vels)) if closing_vels else 0.0
        opening_mean = float(np.mean(opening_vels)) if opening_vels else 0.0
        long_blink_pct = float(long_blinks) / float(n_blinks) if n_blinks > 0 else 0.0

        if len(blink_starts) >= 3:
            ibis = np.diff(sorted(blink_starts))
            blink_reg = float(np.std(ibis) / (np.mean(ibis) + 1e-6))
        else:
            blink_reg = 0.0

        return {
            "blink_count": blink_count,
            "blink_rate_per_min": blink_rate,
            "blink_mean_dur_ms": blink_mean_dur_ms,
            "blink_closing_vel_mean": closing_mean,
            "blink_opening_vel_mean": opening_mean,
            "long_blink_pct": long_blink_pct,
            "blink_regularity": blink_reg,
        }

    def _compute_perclos_and_microsleeps(
        self,
        ear_raw: np.ndarray,
        ear_smooth: np.ndarray,
        face_mask: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """PERCLOS P80 e microsleeps usando EAR raw (C13 + C25).

        FIX 2: baseline = mean EAR do best-segment (fallback: P90 intra-janela).
        FIX 4: microsleeps filtrados (median, blink overlap, purity).
        FIX 6: perclos_p80_max = pico de rolling 5s.
        """
        valid = face_mask & ~np.isnan(ear_raw)
        valid_ear = ear_raw[valid]
        if len(valid_ear) == 0:
            perclos_stats = {"perclos_p80_mean": 0.0, "perclos_p80_max": 0.0}
            micro_stats = {"microsleep_count": 0.0, "microsleep_total_ms": 0.0}
            return perclos_stats, micro_stats

        if self._perclos_baseline_ear is not None:
            baseline_ear = self._perclos_baseline_ear
        else:
            baseline_ear = float(np.percentile(valid_ear, 90))

        threshold = baseline_ear * self.cfg.perclos_factor

        closed = (ear_raw < threshold) & valid
        closed_f = closed.astype(np.float64)

        fps = self.cfg.fps

        perclos_mean = float(np.nanmean(closed_f))

        # FIX 6: perclos_p80_max = pico da rolling 5s
        sub_sec = 5.0
        sub_frames = max(int(sub_sec * fps), 1)
        if len(closed_f) >= sub_frames:
            cs = np.cumsum(closed_f)
            cs = np.insert(cs, 0, 0.0)
            rolling = (cs[sub_frames:] - cs[:-sub_frames]) / sub_frames
            perclos_max = float(np.max(rolling)) if len(rolling) > 0 else perclos_mean
        else:
            perclos_max = perclos_mean

        min_ms = 500.0
        min_frames = int(min_ms / 1000.0 * fps)

        microsleeps = 0
        total_ms = 0.0

        closed_smooth = (ear_smooth < threshold) & valid

        if closed_smooth.any():
            padded = np.concatenate([[False], closed_smooth.astype(bool), [False]])
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            blink_intervals = []
            closed_raw = (ear_raw < threshold) & valid
            if closed_raw.any():
                bp = np.concatenate([[False], closed_raw.astype(bool), [False]])
                bd = np.diff(bp.astype(np.int8))
                b_starts = np.where(bd == 1)[0]
                b_ends = np.where(bd == -1)[0]
                for bs, be in zip(b_starts.tolist(), b_ends.tolist()):
                    dur = be - bs
                    if 2 <= dur < min_frames:
                        blink_intervals.append((bs, be))

            for s, e in zip(starts.tolist(), ends.tolist()):
                length = e - s
                if length < min_frames:
                    continue

                blink_overlap = 0
                for bs, be in blink_intervals:
                    overlap_start = max(s, bs)
                    overlap_end = min(e, be)
                    if overlap_end > overlap_start:
                        blink_overlap += overlap_end - overlap_start

                if length > 0 and blink_overlap / length > 0.5:
                    continue

                seg_closed = closed_smooth[s:e]
                purity = float(seg_closed.mean()) if len(seg_closed) > 0 else 0.0
                if purity < 0.80:
                    continue

                microsleeps += 1
                total_ms += length / max(fps, 1) * 1000.0

        perclos_stats = {
            "perclos_p80_mean": perclos_mean,
            "perclos_p80_max": perclos_max,
        }
        micro_stats = {
            "microsleep_count": float(microsleeps),
            "microsleep_total_ms": float(total_ms),
        }
        return perclos_stats, micro_stats
