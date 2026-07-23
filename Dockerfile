# Person -> meme transition detector.
# CUDA 12.8 runtime to match torch 2.10.0+cu128. Runs on GPU when the container
# is started with `--gpus all`; falls back to CPU automatically otherwise.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/opt/insightface \
    TORCH_HOME=/opt/torch \
    MINIMAX_DIR=/opt/minimax \
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

# Track 2 (caption inpainting) — LaMa engine, ALWAYS baked. rapidocr-onnxruntime and
# simple-lama-inpainting go in --no-deps: their transitive Pillow pin would force a
# no-zlib source build, and they'd pull a second onnxruntime/opencv that collides with
# the baked onnxruntime-gpu + opencv above. Their real pure-wheel deps are in
# requirements-inpaint.txt; onnxruntime-gpu/opencv/pillow/numpy are already present.
COPY requirements-inpaint.txt /app/requirements-inpaint.txt
RUN pip install -r requirements-inpaint.txt \
 && pip install --no-deps rapidocr-onnxruntime==1.4.4 simple-lama-inpainting==0.1.2
# Bake big-lama weights into TORCH_HOME (set in ENV, outside /app so the compose bind
# mount can't shadow it) so a fresh runner is offline and any host UID can read them.
RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama()" \
 && chmod -R a+rwX /opt/torch

# MiniMax engine (auto-default for textured bands) — OPT-IN, off by default to keep the
# image lean and reproducible. Enable with `MINIMAX=1 docker compose build` (passes
# --build-arg INCLUDE_MINIMAX=1). Vendored repo + weights land in MINIMAX_DIR=/opt/minimax
# (outside /app, so the compose bind mount doesn't hide them). When absent, --engine auto
# falls back to LaMa and --engine minimax errors with a clear rebuild hint.
# NOTE: pin the git SHA + HF revision below before treating this as reproducible.
ARG INCLUDE_MINIMAX=0
RUN if [ "$INCLUDE_MINIMAX" = "1" ]; then set -eux; \
      apt-get update && apt-get install -y --no-install-recommends git \
        && rm -rf /var/lib/apt/lists/*; \
      git clone --depth 1 https://github.com/zibojia/MiniMax-Remover.git "$MINIMAX_DIR"; \
      # Install ONLY what the headless band-crop path imports (see inpaint_text.load_minimax
      # + the vendored pipeline/transformer). The repo's requirements.txt is NOT used: it
      # pins torch==2.7.1 / numpy==1.26.4 / opencv==4.10 / Pillow==9.2 -- every one a
      # DOWNGRADE that would clobber the base CUDA stack (torch 2.10+cu128) or trigger a
      # from-source Pillow build (no zlib headers -> fail). diffusers 0.33.1 carries the Wan
      # VAE/pipeline the vendored code needs; it keeps the base torch/numpy/Pillow as-is. \
      pip install --no-cache-dir \
        diffusers==0.33.1 accelerate==0.30.1 einops==0.8.0 scipy \
        "huggingface_hub[cli]==0.32.4"; \
      huggingface-cli download zibojia/minimax-remover --local-dir "$MINIMAX_DIR/weights"; \
      chmod -R a+rwX "$MINIMAX_DIR"; \
    fi

# Application code + UI last (cheap layer to rebuild). In compose the whole repo is
# bind-mounted over /app anyway; baking these keeps a plain `docker run` self-contained.
COPY src /app/src
COPY build_report.py peznav.py peznav.css serve.py \
     app.html editor.html index.html /app/
COPY vendor /app/vendor

EXPOSE 8000

# `python` is the entrypoint; pick the script + args at `docker run` time
# (e.g. `serve.py 8000`, `src/detect_transitions.py`, `src/relabel_faces.py --qa`).
ENTRYPOINT ["python"]
CMD ["src/detect_transitions.py", "--help"]
