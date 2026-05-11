"""
Host script — SALTE TEV9 / C37.

Pipeline:
  picamera2 (IMX500 como câmera comum)
    → ONNXFaceMeshBackend (BlazeFace + FaceMesh, CPU)
    → RealTimeFeatureExtractor        (per-frame: EAR, MAR, face_detected)
    → RTSubjectCalibrator             (warm-up 120s, z-norm per-subject)
    → OnlineWindowFactory             (15s window → 15 features TEV9)
    → EdgeOptimizedInference          (ONNX MLP, selective scaling, threshold 0.52)
    → BlinkRuleEngine + InferenceGuardrails
    → print / log

Uso:
  python run_host.py                          # default, headless
  python run_host.py --display                # janela com overlay
  python run_host.py --fps 30 --width 640 --height 480
  python run_host.py --warmup-sec 60          # reduz warm-up para testes
  python run_host.py --video v1.mp4 --display --sync-fps-from-video --warmup-sec 60
                                                # teste local (MP4 via OpenCV)

Teclas (modo --display):
  q  sair
  c  força calibração com frames coletados até agora

Assume que estão no mesmo diretório:
  best_model.onnx (+ .data), inference_config.json   ← TEV9 MLP
  blazeface_detector.onnx, face_mesh_landmark.onnx   ← face mesh
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# --- Imports do pipeline ---------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from feature_extractor_rt import (  # noqa: E402
    ONNXFaceMeshBackend,
    RealTimeFeatureExtractor,
    RTExtractorConfig,
    RTFrameFeatures,
)
from subject_calibrator_rt import (  # noqa: E402
    CalibrationConfig,
    RTSubjectCalibrator,
)
from window_factory_rt import (  # noqa: E402
    FEATURE_NAMES_15,
    OnlineWindowFactory,
    RTWindowConfig,
)
from salte_edge_runtime import (  # noqa: E402
    BlinkRuleEngine,
    EdgeOptimizedInference,
    InferenceGuardrails,
)

logger = logging.getLogger("SALTE.host")


# --- Câmera ---------------------------------------------------------------


class CameraBackend:
    """Wrapper picamera2 (IMX500), arquivo de vídeo ou cv2.VideoCapture (webcam dev)."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        use_picamera: bool,
        video_path: Optional[Path] = None,
        loop_video: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._picam: Optional[object] = None
        self._cv2_cap: Optional[cv2.VideoCapture] = None
        self._is_video_file = False
        self._loop_video = loop_video
        self._video_exhausted = False

        if video_path is not None:
            path = video_path.resolve()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(
                    "Falha ao abrir o vídeo com OpenCV. Verifique o caminho, o codec "
                    "(prefira H.264 em MP4) e se o arquivo não está corrompido."
                )
            self._cv2_cap = cap
            self._is_video_file = True
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = vw if vw > 0 else width
            self.height = vh if vh > 0 else height
            logger.info(
                f"Vídeo: {path.name} ({self.width}x{self.height}), "
                f"loop={'sim' if loop_video else 'não'}"
            )
            return

        if use_picamera:
            try:
                from picamera2 import Picamera2  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "picamera2 não disponível. Instale picamera2 no Pi ou rode "
                    "sem --picamera para usar a webcam local."
                ) from e

            picam = Picamera2()
            # IMX500 como câmera comum (NN on-sensor não é usada).
            # RGB888 entra como BGR no buffer numpy — tratamos em read().
            config = picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": float(fps)},
            )
            picam.configure(config)
            picam.start()
            # Warm-up do sensor
            time.sleep(1.0)
            self._picam = picam
            logger.info(f"Picamera2 iniciada: {width}x{height}@{fps}")
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if not cap.isOpened():
                raise RuntimeError("Falha ao abrir cv2.VideoCapture(0)")
            self._cv2_cap = cap
            logger.info(f"cv2.VideoCapture iniciada: {width}x{height}@{fps}")

    def is_exhausted(self) -> bool:
        """True após EOF em modo arquivo com --no-loop-video."""
        return self._video_exhausted

    def read(self) -> Optional[np.ndarray]:
        """Retorna frame BGR numpy [H,W,3] ou None em falha."""
        if self._picam is not None:
            # picamera2 com format=RGB888 retorna array H,W,3 — os bytes
            # vêm em ordem RGB, precisamos inverter para BGR (OpenCV)
            arr = self._picam.capture_array()  # type: ignore[attr-defined]
            if arr is None:
                return None
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if self._cv2_cap is not None:
            ok, frame = self._cv2_cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame

            if self._is_video_file:
                if self._loop_video:
                    self._cv2_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok2, frame2 = self._cv2_cap.read()
                    if ok2 and frame2 is not None and frame2.size > 0:
                        return frame2
                    logger.warning("Não foi possível reiniciar o vídeo após EOF.")
                    return None
                self._video_exhausted = True
                return None

            return None

        return None

    def close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._cv2_cap is not None:
            self._cv2_cap.release()


# --- Pipeline -------------------------------------------------------------


def build_feature_vector(feats: dict) -> np.ndarray:
    """Monta o vetor de 15 floats na ordem do FEATURE_NAMES_15."""
    vec = np.zeros(len(FEATURE_NAMES_15), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES_15):
        vec[i] = float(feats.get(name, 0.0))
    return vec


def ear_closed_threshold_for_frame(
    calibrator: RTSubjectCalibrator,
    provisional_ear_threshold: float,
) -> float:
    """Limiar EAR abaixo do qual o olho é tratado como fechado (blink)."""
    b = calibrator.baseline
    if calibrator.is_calibrated and b is not None and b.is_valid:
        return max(0.7 * b.ear_mean, 1e-6)
    return provisional_ear_threshold


def classify_eye_perceptual_state(
    feats: RTFrameFeatures,
    calibrator: RTSubjectCalibrator,
    provisional_ear_threshold: float,
) -> Tuple[str, Optional[float]]:
    """sem_rosto | olho_fechado | olho_aberto e limiar EAR usado (None se sem rosto)."""
    if not feats.face_detected:
        return "sem_rosto", None
    t = ear_closed_threshold_for_frame(calibrator, provisional_ear_threshold)
    if feats.ear_avg < t:
        return "olho_fechado", t
    return "olho_aberto", t


def update_eye_state_logging(
    feats_frame: RTFrameFeatures,
    calibrator: RTSubjectCalibrator,
    provisional_ear_threshold: float,
    log_every_frame: bool,
    last_state_holder: Dict[str, Optional[str]],
) -> None:
    """INFO em transição de estado; DEBUG opcional a cada frame."""
    state, thresh = classify_eye_perceptual_state(
        feats_frame, calibrator, provisional_ear_threshold
    )
    if log_every_frame:
        if feats_frame.face_detected and thresh is not None:
            logger.debug(
                "estado_olhos frame=%d %s EAR=%.4f limiar=%.4f",
                feats_frame.frame_idx,
                state,
                feats_frame.ear_avg,
                thresh,
            )
        else:
            logger.debug(
                "estado_olhos frame=%d %s",
                feats_frame.frame_idx,
                state,
            )
    prev = last_state_holder.get("v")
    if state == prev:
        return
    last_state_holder["v"] = state
    if feats_frame.face_detected and thresh is not None:
        logger.info(
            "estado_olhos frame=%d %s EAR=%.4f limiar=%.4f",
            feats_frame.frame_idx,
            state,
            feats_frame.ear_avg,
            thresh,
        )
    else:
        logger.info("estado_olhos frame=%d %s", feats_frame.frame_idx, state)


def main() -> int:
    parser = argparse.ArgumentParser(description="SALTE TEV9 host")
    parser.add_argument("--model-dir", default=str(HERE),
                        help="Diretório com best_model.onnx + inference_config.json")
    parser.add_argument("--detector", default=str(HERE / "blazeface_detector.onnx"))
    parser.add_argument("--mesh", default=str(HERE / "face_mesh_landmark.onnx"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-sec", type=float, default=120.0,
                        help="Duração do warm-up para calibração (padrão 120s)")
    parser.add_argument("--baseline-sec", type=float, default=30.0,
                        help="Tamanho do segmento de baseline (C6-V2)")
    parser.add_argument("--window-sec", type=float, default=15.0)
    parser.add_argument("--stride-frames", type=int, default=60)
    parser.add_argument("--display", action="store_true",
                        help="Abre janela OpenCV com overlay (não-headless)")
    parser.add_argument("--picamera", action="store_true",
                        help="Usa picamera2 (Raspberry Pi + IMX500) em vez da webcam local")
    parser.add_argument(
        "--video",
        default=None,
        metavar="PATH",
        help="Caminho para vídeo (MP4 etc.). OpenCV; use --fps alinhado ao arquivo ou "
             "--sync-fps-from-video.",
    )
    parser.add_argument(
        "--no-loop-video",
        action="store_true",
        help="Com --video: encerra ao fim do arquivo em vez de repetir.",
    )
    parser.add_argument(
        "--sync-fps-from-video",
        action="store_true",
        help="Com --video: usa o FPS reportado pelo arquivo no pipeline (se > 1).",
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override do threshold MLP (padrão: inference_config.json)")
    parser.add_argument(
        "--eye-provisional-threshold",
        type=float,
        default=0.22,
        help="EAR limiar provisório olho aberto/fechado durante warm-up (sem baseline). "
             "Após calibração: max(0.7×ear_mean do baseline, 1e-6), alinhado à lógica de blink.",
    )
    parser.add_argument(
        "--log-eye-every-frame",
        action="store_true",
        help="Log DEBUG de sem_rosto/olho_aberto/olho_fechado a cada frame (ative --log-level DEBUG).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    video_path: Optional[Path] = None
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.is_file():
            raise RuntimeError(f"Arquivo de vídeo não encontrado: {video_path}")

    effective_fps = args.fps
    if args.sync_fps_from_video:
        if video_path is None:
            logger.warning("--sync-fps-from-video sem --video: opção ignorada.")
        else:
            cap_probe = cv2.VideoCapture(str(video_path))
            if not cap_probe.isOpened():
                raise RuntimeError(
                    "Não foi possível abrir o vídeo para ler FPS. Verifique o codec "
                    "(prefira H.264 em MP4)."
                )
            vfps = float(cap_probe.get(cv2.CAP_PROP_FPS))
            cap_probe.release()
            if vfps > 1.0:
                effective_fps = int(round(vfps))
                logger.info(
                    f"FPS do vídeo: {vfps:.3f} → usando {effective_fps} no pipeline"
                )
            else:
                logger.warning(
                    f"FPS do vídeo inválido ({vfps}); mantendo --fps={args.fps}."
                )

    # --- Init engine ---
    inference = EdgeOptimizedInference(artifacts_dir=args.model_dir)
    if args.threshold is not None:
        inference.threshold = args.threshold
    logger.info(
        f"MLP carregado: {inference.model_type}, threshold={inference.threshold:.4f}, "
        f"{len(inference.feature_names)} features"
    )

    if inference.feature_names != FEATURE_NAMES_15:
        logger.warning(
            "Ordem de features do inference_config.json difere de FEATURE_NAMES_15. "
            f"Config: {inference.feature_names}"
        )

    rules = BlinkRuleEngine()
    guardrails = InferenceGuardrails()

    # --- Init extractor + calibrator + window ---
    backend = ONNXFaceMeshBackend(
        detector_path=args.detector,
        mesh_path=args.mesh,
    )
    extractor = RealTimeFeatureExtractor(
        backend=backend,
        config=RTExtractorConfig(fps=effective_fps),
    )
    calibrator = RTSubjectCalibrator(
        config=CalibrationConfig(
            fps=effective_fps,
            search_sec=args.warmup_sec,
            baseline_sec=args.baseline_sec,
        )
    )
    window = OnlineWindowFactory(
        config=RTWindowConfig(
            fps=effective_fps,
            window_sec=args.window_sec,
            stride_infer=args.stride_frames,
        )
    )

    # --- Câmera ---
    use_picamera = video_path is None and args.picamera
    cam = CameraBackend(
        width=args.width,
        height=args.height,
        fps=effective_fps,
        use_picamera=use_picamera,
        video_path=video_path,
        loop_video=not args.no_loop_video,
    )

    # --- Sinais ---
    stop = {"flag": False}

    def _sigint(*_):
        logger.info("SIGINT recebido — encerrando")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    # --- Loop principal ---
    logger.info("Iniciando loop. Warm-up ~%.0fs antes da primeira inferência.",
                args.warmup_sec + args.window_sec)

    last_log = time.monotonic()
    frames_seen = 0
    frames_in_window = 0
    predictions = 0
    prev_warmup_pct = -1
    perclos_baseline_set = False
    last_eye_state: Dict[str, Optional[str]] = {"v": None}
    face_lost_frames = 0
    FACE_LOST_ALERT_FRAMES = max(1, int(1.5 * effective_fps))

    try:
        while not stop["flag"]:
            frame = cam.read()
            if frame is None:
                if cam.is_exhausted():
                    logger.info("Vídeo terminado.")
                    break
                time.sleep(0.01)
                continue

            frames_seen += 1
            frames_in_window += 1

            feats_frame = extractor.process_frame(frame)
            if not feats_frame.face_detected:
                face_lost_frames += 1
                if (
                    calibrator.is_calibrated
                    and face_lost_frames == FACE_LOST_ALERT_FRAMES
                ):
                    logger.warning(
                        "ALERTA: rosto ausente por %.1fs consecutivos — "
                        "RULES=DANGER (face_lost)",
                        face_lost_frames / effective_fps,
                    )
            else:
                if (
                    face_lost_frames >= FACE_LOST_ALERT_FRAMES
                    and calibrator.is_calibrated
                ):
                    logger.info(
                        "Rosto recuperado após %.1fs ausente.",
                        face_lost_frames / effective_fps,
                    )
                face_lost_frames = 0
            update_eye_state_logging(
                feats_frame,
                calibrator,
                args.eye_provisional_threshold,
                args.log_eye_every_frame,
                last_eye_state,
            )
            cal_frame = calibrator.push(feats_frame)

            # Durante warm-up, cal_frame é None — apenas loga progresso
            if cal_frame is None:
                pct = int(calibrator.warmup_progress * 100)
                if pct != prev_warmup_pct and pct % 10 == 0:
                    logger.info(f"Warm-up: {pct}%")
                    prev_warmup_pct = pct

                if args.display:
                    _draw_hud(frame, feats_frame, calibrator, None, None)
                    cv2.imshow("SALTE TEV9", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c") and calibrator.force_calibrate():
                        logger.info("Calibração forçada.")
                continue

            # Primeira vez que a calibração termina: seta baseline PERCLOS
            if calibrator.is_calibrated and not perclos_baseline_set:
                b = calibrator.baseline
                if b is not None and b.is_valid:
                    window.set_perclos_baseline(b.ear_mean)
                    logger.info(
                        f"Calibrado. ear_mean={b.ear_mean:.3f} "
                        f"ear_std={b.ear_std:.3f} "
                        f"mar_mean={b.mar_mean:.3f} mar_std={b.mar_std:.3f}"
                    )
                perclos_baseline_set = True

            feats_dict = window.push(cal_frame)

            if feats_dict is not None:
                vec15 = build_feature_vector(feats_dict)

                iv = guardrails.validate_input(vec15)
                if not iv.valid:
                    logger.warning(f"Input rejeitado: {iv.reason}")
                    continue

                rules_result = rules.evaluate(vec15)
                mlp_label, mlp_conf = inference.predict(vec15)

                # Fusão: max (rules, mlp) — nunca reduz alerta
                final_label = max(rules_result["label"], mlp_label)
                voted = guardrails.temporal_vote(final_label)

                predictions += 1
                logger.info(
                    "[%d] MLP=%d (%.3f) | RULES=%s %s | FINAL=%s (voted=%d)",
                    predictions,
                    mlp_label, mlp_conf,
                    rules_result["level"],
                    rules_result["alerts"] or "-",
                    "DANGER" if final_label == 1 else "SAFE",
                    voted,
                )

                if args.display:
                    _draw_hud(frame, feats_frame, calibrator, feats_dict, {
                        "label": final_label,
                        "mlp_conf": mlp_conf,
                        "rules_level": rules_result["level"],
                    })

            if args.display:
                if feats_dict is None:
                    _draw_hud(frame, feats_frame, calibrator, None, None)
                cv2.imshow("SALTE TEV9", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and calibrator.force_calibrate():
                    logger.info("Calibração forçada.")

            # Health log a cada 10s
            now = time.monotonic()
            elapsed = now - last_log
            if elapsed > 10.0:
                fps_now = frames_in_window / elapsed
                logger.info(
                    f"health: {frames_seen}f total / {predictions}pred "
                    f"({fps_now:.1f} FPS na última janela)"
                )
                last_log = now
                frames_in_window = 0

    finally:
        cam.close()
        if args.display:
            cv2.destroyAllWindows()

    return 0


# --- HUD opcional ---------------------------------------------------------


def _draw_hud(frame, feats_frame, calibrator, feats_dict, pred) -> None:
    """Overlay simples no frame (só no modo --display)."""
    h, w = frame.shape[:2]

    txt_lines = []
    if not calibrator.is_calibrated:
        txt_lines.append(f"WARMUP {calibrator.warmup_progress * 100:.0f}%")
    else:
        b = calibrator.baseline
        txt_lines.append(f"CALIB ear={b.ear_mean:.2f}+-{b.ear_std:.3f}")

    if feats_frame.face_detected:
        txt_lines.append(
            f"EAR={feats_frame.ear_avg:.3f} MAR={feats_frame.mar:.3f}"
        )
    else:
        txt_lines.append("NO FACE")

    if pred is not None:
        state = "DANGER" if pred["label"] == 1 else "SAFE"
        color = (0, 0, 255) if pred["label"] == 1 else (0, 200, 0)
        cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
        cv2.putText(frame, f"{state}  p={pred['mlp_conf']:.2f}  "
                           f"rules={pred['rules_level']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    y = 60
    for line in txt_lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22


if __name__ == "__main__":
    sys.exit(main())
