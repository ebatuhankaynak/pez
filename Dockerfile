# Person -> meme transition detector.
# CUDA 12.8 runtime to match torch 2.10.0+cu128. Runs on GPU when the container
# is started with `--gpus all`; falls back to CPU automatically otherwise.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/opt/insightface \
    PATH=/opt/venv/bin:$PATH

# System deps: python + ffmpeg (needed by ffmpeg-python, decord, and split_clips.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (Ubuntu 24.04 python is PEP-668 externally-managed).
RUN python3 -m venv /opt/venv && pip install --upgrade pip

WORKDIR /app

# torch first — big, rarely-changing layer, so edits below don't re-download it.
COPY requirements-torch.txt /app/requirements-torch.txt
RUN pip install -r requirements-torch.txt

# Stage-1 (TransNetV2) + video I/O deps. TransNetV2 weights ship inside its wheel.
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Face-labeling stage deps + baked buffalo_l model (offline / reproducible).
COPY requirements-face.txt /app/requirements-face.txt
RUN pip install -r requirements-face.txt
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='/opt/insightface', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"

# Let the image run as an arbitrary host UID (docker run --user / compose `user:`)
# without permission errors on the model cache or the working dir.
RUN chmod -R a+rwX /opt/insightface /app

# Application code last (cheap layer to rebuild).
COPY detect_transitions.py relabel_faces.py split_clips.py segment_clips.py build_report.py build_verify_ui.py /app/

# `python` is the entrypoint; pick the script + args at `docker run` time.
ENTRYPOINT ["python"]
CMD ["detect_transitions.py", "--help"]
