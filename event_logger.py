"""
EventLogger — JSONL de transições SAFE↔DANGER, com referência ao segmento de vídeo.

Por que só transições?
    O pipeline emite uma predição por janela (~15s, stride 2s). Salvar 1 linha
    por janela polui o arquivo (~30 linhas/min) e dificulta a correlação visual
    com o vídeo. Só transições mantém o arquivo curto e auditável:

        SAFE  ...
        SAFE  → DANGER   ← linha gravada
        DANGER ...
        DANGER → SAFE    ← linha gravada

Layout em disco:
    <root>/
        YYYY-MM-DD.jsonl   ← uma linha por transição, append-only

Formato de cada linha (JSON):
    {
      "ts": "2026-05-13T08:30:42.310",
      "frame_idx": 12873,
      "transition": "safe_to_danger" | "danger_to_safe",
      "voted_label": 0|1,
      "mlp_label": 0|1,
      "mlp_conf": 0.71,
      "rules_level": "warning" | "danger" | "normal",
      "alerts": ["long_blink_pct", ...],
      "video_file": "2026-05-13/2026-05-13T08-30-00.mp4",   // pode ser null
      "video_offset_s": 42.310                                // pode ser null
    }

`video_file` e `video_offset_s` ficam `null` quando o recorder não está ativo
ou ainda não abriu nenhum segmento.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SALTE.events")


class EventLogger:
    """Loga transições SAFE↔DANGER em arquivos JSONL por dia."""

    def __init__(self, root_dir: Path, recorder: Optional[Any] = None) -> None:
        # `recorder` é um SegmentedVideoRecorder (ou qualquer objeto com
        # current_segment() -> Optional[(Path, float)]). Mantido como Any
        # para não criar dependência forte de import.
        self._root = Path(root_dir)
        self._recorder = recorder
        self._lock = threading.Lock()
        self._fp = None  # type: Optional[Any]
        self._cur_day: Optional[str] = None
        self._last_voted: Optional[int] = None

    # ---- API pública ----------------------------------------------------

    def open(self) -> None:
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            self._ensure_file_open_locked()
        logger.info("EventLogger iniciado: root=%s", self._root)

    def log_prediction(
        self,
        voted_label: int,
        frame_idx: int,
        mlp_label: int,
        mlp_conf: float,
        rules_level: str,
        alerts: Optional[List[str]],
    ) -> None:
        """
        Chamar a cada predição emitida pelo pipeline. Só grava se houver
        transição em relação à última predição registrada.
        """
        voted_label = int(voted_label)
        with self._lock:
            prev = self._last_voted
            self._last_voted = voted_label
            if prev is None:
                # Primeira predição da sessão: não considera transição.
                return
            if voted_label == prev:
                return

            transition = "safe_to_danger" if voted_label == 1 else "danger_to_safe"
            record = self._build_record(
                transition=transition,
                voted_label=voted_label,
                frame_idx=int(frame_idx),
                mlp_label=int(mlp_label),
                mlp_conf=float(mlp_conf),
                rules_level=str(rules_level),
                alerts=list(alerts) if alerts else [],
            )
            self._write_record_locked(record)

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.flush()
                    self._fp.close()
                except Exception:
                    logger.exception("EventLogger: erro ao fechar arquivo")
                self._fp = None
                self._cur_day = None
        logger.info("EventLogger finalizado.")

    # ---- Internos -------------------------------------------------------

    def _build_record(
        self,
        transition: str,
        voted_label: int,
        frame_idx: int,
        mlp_label: int,
        mlp_conf: float,
        rules_level: str,
        alerts: List[str],
    ) -> Dict[str, Any]:
        now = time.time()
        ts = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")

        video_file: Optional[str] = None
        video_offset_s: Optional[float] = None
        if self._recorder is not None:
            seg = self._recorder.current_segment()
            if seg is not None:
                rel_path, started_at = seg
                video_file = str(rel_path).replace("\\", "/")
                offset = now - float(started_at)
                # Clamp defensivo: se o relógio do sistema deu salto para
                # trás, não escreve offset negativo (cria fragilidade no
                # consumidor offline).
                video_offset_s = max(0.0, round(offset, 3))

        return {
            "ts": ts,
            "frame_idx": frame_idx,
            "transition": transition,
            "voted_label": voted_label,
            "mlp_label": mlp_label,
            "mlp_conf": round(mlp_conf, 4),
            "rules_level": rules_level,
            "alerts": alerts,
            "video_file": video_file,
            "video_offset_s": video_offset_s,
        }

    def _ensure_file_open_locked(self) -> None:
        # SEMPRE chamar com self._lock segurado.
        day = datetime.now().strftime("%Y-%m-%d")
        if self._fp is not None and self._cur_day == day:
            return
        # Virada de dia (ou primeira abertura): fecha e reabre.
        if self._fp is not None:
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                logger.exception("EventLogger: erro ao rotacionar arquivo")
        path = self._root / f"{day}.jsonl"
        # buffering=1 = line-buffered: cada linha vira disco assim que escreve '\n'.
        # Suficiente para auditoria; fsync seria exagero e custa I/O.
        self._fp = open(path, "a", encoding="utf-8", buffering=1)
        self._cur_day = day
        logger.info("EventLogger: arquivo aberto em %s", path)

    def _write_record_locked(self, record: Dict[str, Any]) -> None:
        # SEMPRE chamar com self._lock segurado.
        self._ensure_file_open_locked()
        if self._fp is None:
            return
        try:
            self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("EventLogger: falha ao escrever transição")
