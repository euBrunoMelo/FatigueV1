# ==============================================================
# SALTE Edge Runtime — TEV9 (C37: 15 features)
# ==============================================================
# Arquivo unificado para deploy no Raspberry Pi.
# Contém: FRAME_COL, CircularFrameBuffer, RealtimeWindowAggregator,
#          SubjectCalibratorEdge, EdgeOptimizedInference,
#          InferenceGuardrails, BlinkRuleEngine, SALTEEdgePipeline
#
# Dependências: numpy, onnxruntime, opencv-contrib-python
# Artefatos:    best_model.onnx + best_model.onnx.data + inference_config.json
# ==============================================================

import os
import json
import time
import logging
import numpy as np
from typing import Optional, Tuple, List, NamedTuple
from collections import deque

logger = logging.getLogger('SALTE')

# ==============================================================
# Feature Definitions (C37: 15 features, no pose)
# ==============================================================

FEATURE_NAMES = [
    'ear_mean', 'ear_std', 'ear_min',
    'ear_vel_mean', 'ear_vel_std',
    'mar_mean',
    'blink_count', 'blink_rate_per_min', 'blink_mean_dur_ms',
    'perclos_p80_mean', 'perclos_p80_max',
    'blink_closing_vel_mean', 'blink_opening_vel_mean',
    'long_blink_pct', 'blink_regularity',
]
N_FEATURES = 15

# ==============================================================
# Frame-Level Column Indices (C37: 14 columns, no pose)
# ==============================================================

FRAME_COL = {
    'ear_avg':           0,
    'ear_velocity':      1,
    'mar':               2,
    'face_detected':     3,
    'invalid_segment':   4,
    'blink_id':          5,
    'blink_duration_ms': 6,
    'blink_closing_vel': 7,
    'blink_opening_vel': 8,
    'is_long_blink':     9,
    'perclos_p80':       10,
    'is_microsleep':     11,
    'microsleep_id':     12,
    'frame_idx':         13,
}
N_FRAME_FEATURES = 14

# ==============================================================
# CircularFrameBuffer — O(1) push, pre-allocated
# ==============================================================

class CircularFrameBuffer:
    def __init__(self, window_frames: int = 450, stride: int = 60):
        self.window_frames = window_frames
        self.stride = stride
        self.buffer = np.full((window_frames, N_FRAME_FEATURES),
                              np.nan, dtype=np.float32)
        self.write_pos = 0
        self.frames_since_last_inference = 0
        self.total_frames = 0

    def push(self, frame_features: np.ndarray):
        idx = self.write_pos % self.window_frames
        self.buffer[idx] = frame_features
        self.write_pos += 1
        self.total_frames += 1
        self.frames_since_last_inference += 1

    def ready(self) -> bool:
        return (self.total_frames >= self.window_frames and
                self.frames_since_last_inference >= self.stride)

    def get_window(self) -> np.ndarray:
        self.frames_since_last_inference = 0
        start = self.write_pos % self.window_frames
        return np.roll(self.buffer, -start, axis=0)

# ==============================================================
# RealtimeWindowAggregator — 14 frame cols → 15 features
# ==============================================================

class RealtimeWindowAggregator:
    def __init__(self, fps: int = 30, min_valid_ratio: float = 0.80):
        self.fps = fps
        self.min_valid_ratio = min_valid_ratio

    def aggregate(self, window: np.ndarray) -> Optional[np.ndarray]:
        n_frames = len(window)
        C = FRAME_COL

        face_ok = window[:, C['face_detected']] > 0.5
        seg_ok = window[:, C['invalid_segment']] < 0.5
        not_nan = ~np.isnan(window[:, C['ear_avg']])
        valid = face_ok & seg_ok & not_nan

        n_valid = valid.sum()
        if n_valid / n_frames < self.min_valid_ratio:
            return None

        v = window[valid]
        feats = np.zeros(N_FEATURES, dtype=np.float32)

        # G1: EAR stats (indices 0-2)
        ear = v[:, C['ear_avg']]
        feats[0] = np.mean(ear)
        feats[1] = np.std(ear, ddof=1) if len(ear) > 1 else 0.0
        feats[2] = np.min(ear)

        # G2: EAR velocity (indices 3-4)
        ev = v[:, C['ear_velocity']]
        ev_clean = ev[~np.isnan(ev)]
        feats[3] = np.mean(ev_clean) if len(ev_clean) > 0 else 0.0
        feats[4] = np.std(ev_clean, ddof=1) if len(ev_clean) > 1 else 0.0

        # G3: MAR (index 5)
        feats[5] = np.mean(v[:, C['mar']])

        # G5: Blink básico (indices 6-8)
        blink_ids_all = v[:, C['blink_id']]
        blink_mask = blink_ids_all > 0.5
        blink_frames = v[blink_mask]
        window_sec = n_frames / self.fps

        if len(blink_frames) > 0:
            unique_blinks = np.unique(blink_frames[:, C['blink_id']])
            unique_blinks = unique_blinks[unique_blinks > 0.5]
            n_blinks = len(unique_blinks)

            feats[6] = n_blinks
            feats[7] = (n_blinks / window_sec) * 60 if window_sec > 0 else 0.0

            durations, closing_vels, opening_vels = [], [], []
            long_count = 0
            blink_first_frames = []

            for bid in unique_blinks:
                bid_mask = blink_frames[:, C['blink_id']] == bid
                bid_data = blink_frames[bid_mask]

                for frame in bid_data:
                    if not np.isnan(frame[C['blink_duration_ms']]):
                        if frame[C['blink_duration_ms']] > 0:
                            durations.append(frame[C['blink_duration_ms']])
                        break

                for frame in bid_data:
                    if not np.isnan(frame[C['blink_closing_vel']]):
                        if frame[C['blink_closing_vel']] > 0:
                            closing_vels.append(frame[C['blink_closing_vel']])
                        break

                for frame in bid_data:
                    if not np.isnan(frame[C['blink_opening_vel']]):
                        if frame[C['blink_opening_vel']] > 0:
                            opening_vels.append(frame[C['blink_opening_vel']])
                        break

                for frame in bid_data:
                    if not np.isnan(frame[C['is_long_blink']]):
                        if frame[C['is_long_blink']] > 0.5:
                            long_count += 1
                        break

                fidx = bid_data[0][C['frame_idx']]
                if not np.isnan(fidx):
                    blink_first_frames.append(fidx)

            feats[8] = np.mean(durations) if durations else 0.0

            # G7: Blink morphology (indices 11-14)
            feats[11] = np.mean(closing_vels) if closing_vels else 0.0
            feats[12] = np.mean(opening_vels) if opening_vels else 0.0
            feats[13] = long_count / n_blinks if n_blinks > 0 else 0.0

            if len(blink_first_frames) >= 3:
                starts = np.sort(blink_first_frames)
                ibis = np.diff(starts)
                feats[14] = ibis.std() / (ibis.mean() + 1e-8)
            else:
                feats[14] = 0.0  # FIX: was feats[18]
        else:
            feats[6] = 0.0
            feats[7] = 0.0
            feats[8] = 0.0
            feats[11] = 0.0
            feats[12] = 0.0
            feats[13] = 0.0
            feats[14] = 0.0  # FIX: was feats[18]

        # G6: PERCLOS (indices 9-10)
        perclos = v[:, C['perclos_p80']]
        perclos_clean = perclos[~np.isnan(perclos)]
        feats[9] = np.mean(perclos_clean) if len(perclos_clean) > 0 else 0.0
        feats[10] = np.max(perclos_clean) if len(perclos_clean) > 0 else 0.0

        return feats

# ==============================================================
# SubjectCalibratorEdge — Boot-time Z-Norm (FIX: 3 cols, no pose)
# ==============================================================

class SubjectCalibratorEdge:
    # FIX: was [0,1,2,3,4,5] with head_pitch/yaw/roll
    ZNORM_INDICES = [0, 1, 2]  # ear_avg, ear_velocity, mar
    ZNORM_NAMES = ['ear_avg', 'ear_velocity', 'mar']

    def __init__(self, fps: int = 30, cal_window_sec: float = 30.0,
                 search_sec: float = 120.0, min_face_ratio: float = 0.90):
        self.fps = fps
        self.cal_frames = int(cal_window_sec * fps)
        self.search_frames = int(search_sec * fps)
        self.stride = self.cal_frames // 2
        self.min_face_ratio = min_face_ratio

        self._warmup_buffer = np.full(
            (self.search_frames, N_FRAME_FEATURES), np.nan, dtype=np.float32)
        self._warmup_count = 0
        self._cal_mean = np.zeros(len(self.ZNORM_INDICES), dtype=np.float32)
        self._cal_std = np.ones(len(self.ZNORM_INDICES), dtype=np.float32)
        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def warmup_progress(self) -> float:
        return 1.0 if self._calibrated else min(self._warmup_count / self.search_frames, 1.0)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._calibrated:
            if self._warmup_count < self.search_frames:
                self._warmup_buffer[self._warmup_count] = frame
                self._warmup_count += 1
            if self._warmup_count >= self.search_frames:
                self._calibrate()
            return frame

        out = frame.copy()
        for i, col_idx in enumerate(self.ZNORM_INDICES):
            if not np.isnan(out[col_idx]):
                out[col_idx] = (out[col_idx] - self._cal_mean[i]) / self._cal_std[i]
        return out

    def _calibrate(self):
        buf = self._warmup_buffer[:self._warmup_count]
        C = FRAME_COL
        best_start, best_ear = 0, -np.inf

        for start in range(0, len(buf) - self.cal_frames + 1, self.stride):
            end = start + self.cal_frames
            segment = buf[start:end]
            face_ratio = np.mean(segment[:, C['face_detected']] > 0.5)
            if face_ratio < self.min_face_ratio:
                continue
            valid = (segment[:, C['face_detected']] > 0.5) & \
                    (segment[:, C['invalid_segment']] < 0.5)
            if valid.sum() < self.cal_frames * 0.5:
                continue
            mean_ear = np.nanmean(segment[valid, C['ear_avg']])
            if mean_ear > best_ear:
                best_ear = mean_ear
                best_start = start

        cal_segment = buf[best_start:best_start + self.cal_frames]
        valid = (cal_segment[:, C['face_detected']] > 0.5) & \
                (cal_segment[:, C['invalid_segment']] < 0.5)
        valid_frames = cal_segment[valid]

        if len(valid_frames) > 10:
            for i, col_idx in enumerate(self.ZNORM_INDICES):
                vals = valid_frames[:, col_idx]
                vals_clean = vals[~np.isnan(vals)]
                if len(vals_clean) > 1:
                    self._cal_mean[i] = np.mean(vals_clean)
                    self._cal_std[i] = max(np.std(vals_clean), 1e-8)
                else:
                    self._cal_mean[i] = 0.0
                    self._cal_std[i] = 1.0

        self._calibrated = True
        logger.info(f"Calibration complete (frame {best_start})")

# ==============================================================
# EdgeOptimizedInference — ONNX on Pi (FIX: features_15)
# ==============================================================

class EdgeOptimizedInference:
    def __init__(self, artifacts_dir: str, infer_threads: int = 2):
        with open(os.path.join(artifacts_dir, 'inference_config.json')) as f:
            config = json.load(f)

        self.model_type = config['model_type']
        self.threshold = config['threshold']
        self.feature_names = config['feature_names']
        self.training_stats = config.get('training_stats', {})

        self.scale_mean = np.array(config.get('scaler_mean', []), dtype=np.float32)
        self.scale_std = np.array(config.get('scaler_scale', []), dtype=np.float32)
        self.scale_indices = np.array(config.get('scaler_indices', []))

        if self.model_type == 'mlp':
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = infer_threads
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            model_path = os.path.join(artifacts_dir, 'best_model.onnx')
            self.session = ort.InferenceSession(
                model_path, sess_options, providers=['CPUExecutionProvider'])
            self._input_name = self.session.get_inputs()[0].name
            self._input_buffer = np.zeros((1, len(self.feature_names)), dtype=np.float32)

    def _scale(self, features: np.ndarray) -> np.ndarray:
        x = features.copy().astype(np.float32)
        if len(self.scale_indices) > 0:
            x[self.scale_indices] = (x[self.scale_indices] - self.scale_mean) / self.scale_std
        return x

    def predict(self, features_15: np.ndarray) -> Tuple[int, float]:
        x_scaled = self._scale(features_15)  # FIX: was features_19
        self._input_buffer[0] = x_scaled
        logits = self.session.run(None, {self._input_name: self._input_buffer})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        prob_danger = float(probs[0, 1])
        label = 1 if prob_danger >= self.threshold else 0
        return label, prob_danger

# ==============================================================
# BlinkRuleEngine — Camada 1: regras robustas a ângulo
# ==============================================================

class BlinkRuleEngine:
    """Regras baseadas em features temporais de blink (robustas a ângulo).
    Opera independentemente do MLP. Nunca reduz alerta — só aumenta."""

    def __init__(self,
                 min_blink_rate: float = 10.0,
                 max_long_blink_pct: float = 0.40,
                 max_blink_dur_ms: float = 400.0,
                 max_regularity: float = 1.5):
        self.min_blink_rate = min_blink_rate
        self.max_long_blink_pct = max_long_blink_pct
        self.max_blink_dur_ms = max_blink_dur_ms
        self.max_regularity = max_regularity

    def evaluate(self, features_15: np.ndarray) -> dict:
        alerts = []
        warnings = []

        blink_count = features_15[6]
        blink_rate = features_15[7]
        blink_dur = features_15[8]
        long_pct = features_15[13]
        regularity = features_15[14]

        # R1: Taxa baixa
        if blink_rate < self.min_blink_rate and blink_count > 0:
            alerts.append(f'LOW_RATE:{blink_rate:.1f}/min')

        # R2: Piscadas longas
        if long_pct > self.max_long_blink_pct and blink_count >= 2:
            alerts.append(f'LONG_BLINKS:{long_pct:.0%}')

        # R3: Duração alta
        if blink_dur > self.max_blink_dur_ms and blink_count > 0:
            alerts.append(f'SLOW_BLINKS:{blink_dur:.0f}ms')

        # R4: Sem piscada (15s) — microsleep provável
        if blink_count == 0:
            alerts.append('NO_BLINKS_15S')

        # R5: Irregularidade
        if regularity > self.max_regularity and blink_count >= 3:
            warnings.append(f'IRREGULAR:{regularity:.2f}')

        # Decisão
        if blink_count == 0:  # R4 sozinha = DANGER
            level = 'DANGER'
        elif len(alerts) >= 2:
            level = 'DANGER'
        elif len(alerts) == 1:
            level = 'WATCH'
        else:
            level = 'SAFE'

        return {
            'level': level,
            'alerts': alerts,
            'warnings': warnings,
            'label': 1 if level == 'DANGER' else 0,
        }

# ==============================================================
# InferenceGuardrails — 4 camadas de segurança
# ==============================================================

class InputValidation(NamedTuple):
    valid: bool
    reason: str = ""

class InferenceGuardrails:
    FEATURE_BOUNDS = {
        'ear_mean': (-5.0, 5.0),
        'perclos_p80_mean': (0.0, 1.0),
        'blink_rate_per_min': (0.0, 120.0),
    }

    def __init__(self, confidence_low=0.40, confidence_high=0.60,
                 temporal_vote_window=5, drift_z_threshold=3.0):
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.temporal_vote_window = temporal_vote_window
        self.drift_z_threshold = drift_z_threshold
        self.prediction_history: List[int] = []

    def validate_input(self, features: np.ndarray) -> InputValidation:
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return InputValidation(False, "NaN/Inf")
        for i, name in enumerate(FEATURE_NAMES):
            if name in self.FEATURE_BOUNDS:
                lo, hi = self.FEATURE_BOUNDS[name]
                if features[i] < lo or features[i] > hi:
                    return InputValidation(False, f"{name}={features[i]:.3f} OOR")
        return InputValidation(True)

    def is_confident(self, confidence: float) -> bool:
        return not (self.confidence_low < confidence < self.confidence_high)

    def temporal_vote(self, label: int) -> int:
        self.prediction_history.append(label)
        ws = self.temporal_vote_window
        if len(self.prediction_history) < ws:
            return label
        recent = self.prediction_history[-ws:]
        return 1 if sum(recent) > ws // 2 else 0

    def detect_drift(self, recent_features: np.ndarray,
                     training_stats: dict) -> bool:
        drifted = False
        for i, name in enumerate(FEATURE_NAMES):
            if name in training_stats:
                t_mean = training_stats[name]['mean']
                t_std = training_stats[name]['std']
                if t_std > 0:
                    z = abs(np.mean(recent_features[:, i]) - t_mean) / t_std
                    if z > self.drift_z_threshold:
                        drifted = True
        return drifted

# ==============================================================
# SALTEEdgePipeline — Pipeline completa (2 camadas)
# ==============================================================

class SALTEEdgePipeline:
    """Pipeline edge: Camada 1 (regras de blink) + Camada 2 (MLP ONNX).
    Resultado final = MAX(regras, mlp) — nunca reduz alerta."""

    DRIFT_CHECK_INTERVAL = 50

    def __init__(self, artifacts_dir: str, fps: int = 30,
                 window_sec: float = 15.0, stride_infer: int = 60):
        self.inference = EdgeOptimizedInference(artifacts_dir)
        self.guardrails = InferenceGuardrails()
        self.blink_rules = BlinkRuleEngine()
        self.calibrator = SubjectCalibratorEdge(fps=fps)
        self.buffer = CircularFrameBuffer(
            window_frames=int(window_sec * fps), stride=stride_infer)
        self.aggregator = RealtimeWindowAggregator(fps=fps)

        self._last_label = 0
        self._last_confidence = 0.5
        self._features_history = deque(maxlen=self.DRIFT_CHECK_INTERVAL)
        self._prediction_count = 0

        logger.info(f"SALTE Edge Pipeline initialized")
        logger.info(f"  Model: {self.inference.model_type}, threshold={self.inference.threshold:.4f}")
        logger.info(f"  Features: {N_FEATURES}, Buffer: {self.buffer.window_frames}f, stride={self.buffer.stride}")

    def process_frame(self, frame_features: np.ndarray) -> Optional[dict]:
        """Push 1 frame (14 floats). Returns prediction dict when window ready."""
        calibrated = self.calibrator.process_frame(frame_features)
        self.buffer.push(calibrated)

        if not self.buffer.ready():
            return None

        window = self.buffer.get_window()
        features = self.aggregator.aggregate(window)

        if features is None:
            return {'label': self._last_label, 'confidence': self._last_confidence,
                    'source': 'WINDOW_INVALID', 'calibrated': self.calibrator.is_calibrated}

        return self._predict(features)

    def _predict(self, features: np.ndarray) -> dict:
        t0 = time.perf_counter()

        # Input validation
        input_val = self.guardrails.validate_input(features)
        if not input_val.valid:
            return {'label': self._last_label, 'confidence': self._last_confidence,
                    'source': f'INPUT_REJECTED:{input_val.reason}',
                    'calibrated': self.calibrator.is_calibrated}

        # Camada 1: Regras de blink (sempre ativa)
        rules_result = self.blink_rules.evaluate(features)

        # Camada 2: MLP ONNX
        mlp_label, mlp_confidence = self.inference.predict(features)

        # Fusão: MAX(regras, mlp) — nunca reduz alerta
        final_label = max(rules_result['label'], mlp_label)

        # Confidence guard
        if not self.guardrails.is_confident(mlp_confidence):
            mlp_label = self._last_label

        # Temporal voting
        voted_label = self.guardrails.temporal_vote(final_label)

        latency_ms = (time.perf_counter() - t0) * 1000

        self._last_label = final_label
        self._last_confidence = mlp_confidence
        self._prediction_count += 1

        # Drift check periódico
        self._features_history.append(features)
        drift = None
        if (self._prediction_count % self.DRIFT_CHECK_INTERVAL == 0 and
                len(self._features_history) >= self.DRIFT_CHECK_INTERVAL):
            drift = self.guardrails.detect_drift(
                np.array(self._features_history), self.inference.training_stats)

        return {
            'label': final_label,
            'voted_label': voted_label,
            'confidence': mlp_confidence,
            'mlp_label': mlp_label,
            'rules_level': rules_result['level'],
            'rules_alerts': rules_result['alerts'],
            'latency_ms': latency_ms,
            'drift': drift,
            'calibrated': self.calibrator.is_calibrated,
        }
