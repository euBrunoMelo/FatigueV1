"""
SegmentedVideoRecorder — gravação contínua segmentada em disco.

Roda numa thread dedicada com fila bounded; o produtor (loop de inferência)
nunca bloqueia. Se a fila enche, o frame é descartado (com log) — perder
frame de vídeo é estritamente preferível a atrasar o pipeline de inferência.

Layout em disco:
    <root>/
        YYYY-MM-DD/
            YYYY-MM-DDTHH-MM-SS.mp4
            YYYY-MM-DDTHH-MM-SS.mp4
            ...

Rotação:
    - A cada `segment_sec` segundos (padrão 60s).
    - Ao virar o dia: novo subdiretório.

Codec:
    Tenta `mp4v` → `.mp4`. Se falhar (raro mas acontece em builds enxutos do
    OpenCV em ARM), cai para `XVID` → `.avi`. MJPG não é tentado: o arquivo
    fica ~10× maior e enche o NVMe rápido demais.

Limitação conhecida (v1):
    O FPS gravado no header é o nominal (`fps` passado no construtor), não o
    real. Se o pipeline cair abaixo do nominal sob carga, o playback parece
    acelerado. Para v2 considerar PTS reais via libav. Por enquanto a
    sidecar de eventos (event_logger.py) carrega timestamps wall-clock.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("SALTE.recorder")


_CODECS = [
    ("mp4v", ".mp4"),
    ("XVID", ".avi"),
]


class SegmentedVideoRecorder:
    """Gravador segmentado, thread-safe, com backpressure por drop."""

    def __init__(
        self,
        root_dir: Path,
        fps: int,
        frame_size: Tuple[int, int],
        segment_sec: float = 60.0,
        queue_max: int = 60,
        drop_log_every: int = 100,
    ) -> None:
        # frame_size segue convenção do cv2.VideoWriter: (width, height).
        self._root = Path(root_dir)
        self._fps = int(fps)
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        self._segment_sec = float(segment_sec)
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_max)
        self._drop_log_every = int(drop_log_every)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Estado do segmento corrente — só a thread worker escreve, leitores
        # externos (event_logger) consultam via current_segment() protegido por lock.
        self._state_lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._cur_path: Optional[Path] = None
        self._cur_started_at: Optional[float] = None  # time.time() wall-clock
        self._cur_day: Optional[str] = None
        self._cur_codec_ext: Optional[Tuple[str, str]] = None

        # Métricas internas
        self._dropped = 0
        self._written = 0

    # ---- API pública ----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="SALTE-recorder", daemon=True
        )
        self._thread.start()
        logger.info(
            "Recorder iniciado: root=%s fps=%d size=%dx%d segment=%.0fs",
            self._root, self._fps, self._frame_size[0], self._frame_size[1],
            self._segment_sec,
        )

    def enqueue(self, frame: np.ndarray) -> None:
        """Não bloqueia. Se a fila estiver cheia, descarta e conta o drop."""
        if self._thread is None or self._stop_event.is_set():
            return
        try:
            # copy() é necessário: o loop de inferência reusa o buffer do
            # CameraBackend e desenha HUD in-place quando --display está ativo.
            # Sem cópia o writer pega frame poluído ou já sobrescrito.
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            self._dropped += 1
            if self._dropped % self._drop_log_every == 0:
                logger.warning(
                    "Recorder: %d frames descartados (fila cheia, qsize=%d)",
                    self._dropped, self._queue.qsize(),
                )

    def current_segment(self) -> Optional[Tuple[Path, float]]:
        """(caminho_relativo_ao_root, wall_clock_em_que_o_segmento_abriu) ou None."""
        with self._state_lock:
            if self._cur_path is None or self._cur_started_at is None:
                return None
            try:
                rel = self._cur_path.relative_to(self._root)
            except ValueError:
                rel = self._cur_path
            return rel, self._cur_started_at

    def close(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        # Sentinela para destravar o worker se estiver em get() bloqueante.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("Recorder: thread não encerrou em %.1fs", timeout)
        self._thread = None
        self._close_writer_locked()
        logger.info(
            "Recorder finalizado. Escritos=%d, descartados=%d",
            self._written, self._dropped,
        )

    # ---- Worker ---------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._queue.get(timeout=0.5)
                except queue.Empty:
                    # Mesmo sem frame, checa rotação por tempo — evita ficar
                    # com um segmento aberto eternamente se a câmera travar.
                    self._rotate_if_needed()
                    continue
                if frame is None:  # sentinela de close()
                    break
                self._rotate_if_needed()
                self._ensure_writer_open()
                if self._writer is not None:
                    self._writer.write(frame)
                    self._written += 1
        except Exception:
            logger.exception("Recorder: erro fatal na thread worker")
        finally:
            with self._state_lock:
                self._close_writer_locked()

    # ---- Helpers internos ----------------------------------------------

    def _rotate_if_needed(self) -> None:
        now = time.time()
        with self._state_lock:
            if self._writer is None:
                return
            age = now - (self._cur_started_at or now)
            day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            day_changed = day != self._cur_day
            if age >= self._segment_sec or day_changed:
                self._close_writer_locked()

    def _ensure_writer_open(self) -> None:
        with self._state_lock:
            if self._writer is not None:
                return
            now = time.time()
            dt = datetime.fromtimestamp(now)
            day = dt.strftime("%Y-%m-%d")
            stamp = dt.strftime("%Y-%m-%dT%H-%M-%S")
            day_dir = self._root / day
            day_dir.mkdir(parents=True, exist_ok=True)

            for codec, ext in _CODECS:
                path = day_dir / f"{stamp}{ext}"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    str(path), fourcc, float(self._fps), self._frame_size
                )
                if writer.isOpened():
                    self._writer = writer
                    self._cur_path = path
                    self._cur_started_at = now
                    self._cur_day = day
                    self._cur_codec_ext = (codec, ext)
                    logger.info("Recorder: novo segmento %s (codec=%s)", path, codec)
                    return
                writer.release()
                logger.warning("Recorder: codec %s indisponível, tentando próximo", codec)

            logger.error(
                "Recorder: nenhum codec aceito por cv2.VideoWriter. Gravação inativa."
            )

    def _close_writer_locked(self) -> None:
        # Chamar SEMPRE com self._state_lock segurado.
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                logger.exception("Recorder: falha ao liberar writer")
        self._writer = None
        self._cur_path = None
        self._cur_started_at = None
        self._cur_day = None
        self._cur_codec_ext = None
