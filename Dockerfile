# ============================================================================
# SALTE TEV8 / C37 — Raspberry Pi 5 (ARM64) + IMX500 (câmera comum)
# ============================================================================
# Build:   docker build -t salte-tev8 .
# Run:     docker compose up            (ver docker-compose.yml)
# ============================================================================

FROM debian:bookworm-slim

LABEL maintainer="Bruno Melo"
LABEL description="SALTE TEV8 Agentic V1 — detecção de fadiga em tempo real (Pi 5)"

# ----------------------------------------------------------------------------
# 1. Evita prompts interativos no apt
# ----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------------------------------------------------------
# 2. Adiciona o repositório da Raspberry Pi Foundation
#    libcamera e python3-picamera2 não estão no Debian upstream.
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg curl ca-certificates \
    && curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] \
        http://archive.raspberrypi.com/debian/ bookworm main" \
        > /etc/apt/sources.list.d/raspi.list \
    && apt-get update

# ----------------------------------------------------------------------------
# 3. Dependências de sistema
#    - python3-picamera2: traz picamera2 + libcamera + python3-libcamera
#    - python3-numpy:     picamera2/simplejpeg são linkados contra o numpy do apt
#    - libgl1, libglib2:  runtime do opencv-python-headless
#    - libatomic1:        onnxruntime em aarch64 às vezes precisa
# ----------------------------------------------------------------------------
RUN apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-numpy \
        python3-picamera2 \
        libgl1 \
        libglib2.0-0 \
        libatomic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# 4. Pacotes Python do PyPI
#    picamera2 e numpy já vieram do apt.
#    --break-system-packages é necessário no Bookworm (PEP 668).
#    --no-deps: não deixar o pip instalar numpy por cima do python3-numpy —
#    senão quebra simplejpeg com "numpy.dtype size changed ... binary incompat".
# ----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --break-system-packages \
        --no-deps "onnxruntime>=1.18.0" \
    && pip3 install --no-cache-dir --break-system-packages \
        --no-deps "opencv-python-headless>=4.8.0"

# ----------------------------------------------------------------------------
# 5. Usuário não-root nos grupos video/render (acesso a /dev/video* e /dev/dri/*)
#    GID 993 para render casa com o Raspberry Pi OS.
# ----------------------------------------------------------------------------
RUN groupadd -f -g 44 video && \
    groupadd -f -g 993 render && \
    useradd -m -s /bin/bash -G video,render salte

# ----------------------------------------------------------------------------
# 6. Aplicação — layout plano (sem pacote SALTE_INFERENCE)
# ----------------------------------------------------------------------------
WORKDIR /app

# Código do pipeline
COPY run_host.py             /app/
COPY feature_extractor_rt.py /app/
COPY subject_calibrator_rt.py /app/
COPY window_factory_rt.py    /app/
COPY salte_edge_runtime.py   /app/
COPY video_recorder.py       /app/
COPY event_logger.py         /app/

# Artefatos ONNX + config (modelo TEV8 e face mesh)
COPY best_model.onnx         /app/
COPY best_model.onnx.data    /app/
COPY inference_config.json   /app/
COPY blazeface_detector.onnx /app/
COPY face_mesh_landmark.onnx /app/

RUN mkdir -p /app/logs /app/recordings /app/events && chown -R salte:salte /app

# ----------------------------------------------------------------------------
# 7. Troca para usuário não-root
# ----------------------------------------------------------------------------
USER salte

# ----------------------------------------------------------------------------
# 8. Comando padrão — headless, picamera2 (IMX500 como câmera comum)
#    Override via `command:` no docker-compose.yml para --display, webcam etc.
# ----------------------------------------------------------------------------
CMD ["python3", "run_host.py", "--model-dir", "/app", "--picamera"]
