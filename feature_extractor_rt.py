"""
Extrator de features em tempo real — TEV9 / C37 (15 features, sem pose).

Adaptado de DeteccaoFadigaAgentic/SALTE_INFERENCE/feature_extractor_rt.py:
- Removido: HeadPoseSanitizer, solvePnP, POSE_LANDMARKS, FACE_3D_MODEL
- Removido: head_pitch / head_yaw / head_roll em RTFrameFeatures
- Mantido: EAR (L/R/avg + velocity), MAR, face_detected

C37 — modelo TEV9 não usa pose. Rodar pose aqui é puro custo de CPU
que não entra no vetor final (janela agrega só EAR stats + MAR + blinks).

Dependências: OpenCV, numpy, onnxruntime
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False


class LandmarkBackend(Protocol):
    """Interface mínima para um backend de landmarks.

    Deve retornar, para cada frame:
      - `landmarks`: array [N, 2] em coordenadas normalizadas (0–1)
      - `has_face`: bool
    """

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        ...


@dataclass
class RTExtractorConfig:
    fps: int = 30
    max_interp_gap_ms: float = 300.0


@dataclass
class RTFrameFeatures:
    """Features per-frame (C37: sem pose)."""

    timestamp_ms: float
    frame_idx: int
    ear_l: float
    ear_r: float
    ear_avg: float
    ear_velocity: float
    mar: float
    face_detected: bool


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear_from_landmarks(
    landmarks: np.ndarray, eye_indices: Iterable[int]
) -> float:
    """EAR clássico (Soukupová & Čech). `landmarks` em coords normalizadas."""
    pts = np.asarray([landmarks[i] for i in eye_indices], dtype=np.float64)
    v1 = _dist(pts[1], pts[5])
    v2 = _dist(pts[2], pts[4])
    hz = _dist(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hz) if hz > 1e-6 else 0.0


# Índices compatíveis com FFV5 (468-landmark mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_MAX = 0.8  # Limite fisiológico para conter glitches do FaceMesh.

MOUTH_UPPER = [82, 13, 312]
MOUTH_LOWER = [87, 14, 317]
MOUTH_CORNERS = [78, 308]


def compute_mar_from_landmarks(landmarks: np.ndarray) -> float:
    """MAR aproximado em coordenadas normalizadas, espelha fórmula da FFV5."""

    def _pt(idx: int) -> np.ndarray:
        return np.asarray(landmarks[idx], dtype=np.float64)

    v1 = _dist(_pt(MOUTH_UPPER[0]), _pt(MOUTH_LOWER[0]))
    v2 = _dist(_pt(MOUTH_UPPER[1]), _pt(MOUTH_LOWER[1]))
    v3 = _dist(_pt(MOUTH_UPPER[2]), _pt(MOUTH_LOWER[2]))
    hz = _dist(_pt(MOUTH_CORNERS[0]), _pt(MOUTH_CORNERS[1]))
    return (v1 + v2 + v3) / (3.0 * hz) if hz > 1e-6 else 0.0


class RealTimeFeatureExtractor:
    """Extrator frame-a-frame — TEV9 / C37 (sem pose)."""

    def __init__(
        self,
        backend: LandmarkBackend,
        config: Optional[RTExtractorConfig] = None,
    ) -> None:
        self.backend = backend
        self.cfg = config or RTExtractorConfig()
        self._frame_idx = 0
        self._prev_ear_avg: float = 0.0

    def process_frame(self, frame_bgr: np.ndarray) -> RTFrameFeatures:
        landmarks, has_face = self.backend.process(frame_bgr)

        if landmarks is None or not has_face:
            ear_velocity = (0.0 - self._prev_ear_avg) * self.cfg.fps
            feats = RTFrameFeatures(
                timestamp_ms=self._frame_idx * (1000.0 / self.cfg.fps),
                frame_idx=self._frame_idx,
                ear_l=0.0,
                ear_r=0.0,
                ear_avg=0.0,
                ear_velocity=ear_velocity,
                mar=0.0,
                face_detected=False,
            )
            self._prev_ear_avg = 0.0
        else:
            ear_l = float(
                np.clip(
                    compute_ear_from_landmarks(landmarks, LEFT_EYE),
                    0.0,
                    EAR_MAX,
                )
            )
            ear_r = float(
                np.clip(
                    compute_ear_from_landmarks(landmarks, RIGHT_EYE),
                    0.0,
                    EAR_MAX,
                )
            )
            ear_avg = (ear_l + ear_r) / 2.0
            ear_velocity = (ear_avg - self._prev_ear_avg) * self.cfg.fps
            mar = compute_mar_from_landmarks(landmarks)

            feats = RTFrameFeatures(
                timestamp_ms=self._frame_idx * (1000.0 / self.cfg.fps),
                frame_idx=self._frame_idx,
                ear_l=ear_l,
                ear_r=ear_r,
                ear_avg=ear_avg,
                ear_velocity=ear_velocity,
                mar=mar,
                face_detected=True,
            )
            self._prev_ear_avg = ear_avg

        self._frame_idx += 1
        return feats


class DummyBackend:
    """Backend de landmarks de exemplo. Não detecta nada — testes de integração."""

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        _ = frame_bgr
        return None, False


# ── ONNXFaceMeshBackend (BlazeFace + FaceMesh ONNX) ──────────────────────────


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class ONNXFaceMeshBackend:
    """Backend de landmarks via BlazeFace + FaceMesh ONNX (sem mediapipe).

    Pipeline:
      1. BlazeFace ONNX detecta face bbox (128x128 input)
      2. Crop + expand face region com 25% margin
      3. FaceMesh ONNX extrai 468 landmarks 3D (256x256 input)
      4. Landmarks convertidos para [0,1] relativos ao frame original
    """

    BLAZEFACE_INPUT_SIZE = 128
    FACEMESH_INPUT_SIZE = 256
    FACEMESH_NUM_LANDMARKS = 468
    CROP_MARGIN = 0.25

    def __init__(
        self,
        detector_path: str = "blazeface_detector.onnx",
        mesh_path: str = "face_mesh_landmark.onnx",
        *,
        min_face_score: float = 0.5,
    ) -> None:
        if not HAS_ONNX:
            raise RuntimeError(
                "onnxruntime não está instalado. "
                "Instale com `pip install onnxruntime` para ONNXFaceMeshBackend."
            )

        detector_path = str(Path(detector_path).resolve())
        mesh_path = str(Path(mesh_path).resolve())

        self._detector = ort.InferenceSession(detector_path)
        self._mesh = ort.InferenceSession(mesh_path)
        self._detector_input_name = self._detector.get_inputs()[0].name
        self._mesh_input_name = self._mesh.get_inputs()[0].name
        self._min_face_score = min_face_score

    def process(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], bool]:
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return None, False

        bbox = self._detect_face(frame_bgr)
        if bbox is None:
            return None, False

        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        pad_w = bw * self.CROP_MARGIN
        pad_h = bh * self.CROP_MARGIN
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        crop = frame_bgr[int(y1) : int(y2), int(x1) : int(x2)]
        if crop.size == 0:
            return None, False

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(
            crop_rgb,
            (self.FACEMESH_INPUT_SIZE, self.FACEMESH_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        mesh_in = (crop_resized.astype(np.float32) / 255.0).reshape(
            1, self.FACEMESH_INPUT_SIZE, self.FACEMESH_INPUT_SIZE, 3
        )

        mesh_out = self._mesh.run(None, {self._mesh_input_name: mesh_in})[0]
        flat = mesh_out[0, 0, 0, : self.FACEMESH_NUM_LANDMARKS * 3]
        lms_crop = flat.reshape(self.FACEMESH_NUM_LANDMARKS, 3)

        x_crop = lms_crop[:, 0] / self.FACEMESH_INPUT_SIZE
        y_crop = lms_crop[:, 1] / self.FACEMESH_INPUT_SIZE

        crop_w_orig = float(x2 - x1)
        crop_h_orig = float(y2 - y1)
        x_frame = (x1 + x_crop * crop_w_orig) / w
        y_frame = (y1 + y_crop * crop_h_orig) / h

        landmarks = np.stack([x_frame, y_frame], axis=1).astype(np.float32)
        return landmarks, True

    def _detect_face(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """Roda BlazeFace, retorna bbox (x1,y1,x2,y2) em coords de frame, ou None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(
            rgb,
            (self.BLAZEFACE_INPUT_SIZE, self.BLAZEFACE_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        inp = (inp.astype(np.float32) / 255.0).reshape(
            1, self.BLAZEFACE_INPUT_SIZE, self.BLAZEFACE_INPUT_SIZE, 3
        )

        reg, cls = self._detector.run(None, {self._detector_input_name: inp})
        scores = _sigmoid(cls[0, :, 0])
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < self._min_face_score:
            return None

        r = reg[0, best_idx, :]
        xc = float(r[0])
        yc = float(r[1])
        bw = float(r[2])
        bh = float(r[3])
        bw = max(bw, 8.0)
        bh = max(bh, 8.0)
        x1_128 = max(0, xc - bw / 2)
        y1_128 = max(0, yc - bh / 2)
        x2_128 = min(self.BLAZEFACE_INPUT_SIZE, xc + bw / 2)
        y2_128 = min(self.BLAZEFACE_INPUT_SIZE, yc + bh / 2)

        scale_x = w / self.BLAZEFACE_INPUT_SIZE
        scale_y = h / self.BLAZEFACE_INPUT_SIZE
        x1 = x1_128 * scale_x
        y1 = y1_128 * scale_y
        x2 = x2_128 * scale_x
        y2 = y2_128 * scale_y

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        return (x1, y1, x2, y2)
