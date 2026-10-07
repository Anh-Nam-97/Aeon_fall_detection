# syntax=docker/dockerfile:1.7
# =============================================================================
# Aeon_fall_detection - production GPU image (RTX 4090 Ada / RTX 5090 Blackwell)
#
# CUDA 12.9 is the newest 12.x line and the first family that ships kernels
# for both sm_89 (Ada) and sm_120 (Blackwell). PyTorch wheels bundle their
# own CUDA libraries, so the runtime (not devel) image is sufficient: it
# provides the cuDNN/driver shims and keeps the image ~6 GB smaller.
# Host requirement: NVIDIA driver >= 575 + NVIDIA Container Toolkit.
# =============================================================================
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG RA_REF=main
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu129

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/data/hf_cache \
    YOLO_CONFIG_DIR=/data/ultralytics \
    AEON_MODE=torch_gpu \
    AEON_HOST=0.0.0.0 \
    AEON_PORT=7860 \
    AEON_ARTIFACTS_DIR=/data/models/relsgg-vits16plus

# Ubuntu 24.04 ships Python 3.12, the minimum RelateAnything supports.
# libgl1/libglib2.0-0: OpenCV runtime; ffmpeg: browser-playable video for
# Gradio; git: pip installs CLIP from GitHub; curl: HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git curl ffmpeg \
        libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade pip setuptools wheel

WORKDIR /app

# Torch first, from the CUDA 12.9 index, in its own layer: it is the largest
# and least frequently changing dependency, so it stays cached across
# application rebuilds.
RUN pip install "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"

# RelateAnything is cloned inside the image (pinned by RA_REF) instead of
# copied from the build context, so the image is reproducible from git.
RUN git clone https://github.com/Maelic/RelateAnything.git /app/RelateAnything \
    && git -C /app/RelateAnything checkout "${RA_REF}" \
    && pip install -e "/app/RelateAnything[hub,deploy]"

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install "git+https://github.com/ultralytics/CLIP.git" \
    && pip install --force-reinstall --no-deps "opencv-python-headless>=4.10" \
    && python -c "import torch, relsgg, ultralytics, gradio, onnxruntime, openvino, cv2; print('deps ok', torch.__version__)"

COPY pipeline.py app.py export_onnx.py generate_docx_report.py /app/
COPY docs /app/docs

# Non-root runtime user; /data is the single writable volume (HF cache,
# detector weights, exported models).
RUN useradd --create-home --uid 1000 aeon \
    && mkdir -p /data/hf_cache /data/ultralytics /data/models \
    && chown -R aeon:aeon /data /app
USER aeon
VOLUME ["/data"]

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${AEON_PORT}/" >/dev/null || exit 1

# --preload builds the engine before the port opens, so a broken GPU
# passthrough fails the container immediately instead of on first request.
CMD ["python", "app.py", "--preload"]
