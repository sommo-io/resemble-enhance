# RunPod serverless worker for resemble-enhance (speech denoise + enhance).
# Build from the repo root so the local resemble_enhance package is used:
#   docker buildx build --platform linux/amd64 -f runpod/Dockerfile -t <image> .
#
# Everything RunPod-specific lives in runpod/; upstream files are untouched so the fork can
# keep merging resemble-ai/resemble-enhance cleanly.
ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    RE_MODEL_DIR=/models/enhancer_stage2 \
    PYTHONPATH=/opt/resemble-enhance:/opt/deepspeed_stub

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Keep the base image's torch/torchaudio; no deepspeed (stubbed) and no gradio/torchvision (web demo only).
COPY runpod/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY runpod/deepspeed_stub /opt/deepspeed_stub
COPY resemble_enhance /opt/resemble-enhance/resemble_enhance

# Bake weights (~700 MB) into the image so cold starts don't hit HuggingFace, then load them on
# CPU once: a build-time check that the inference import chain works with the deepspeed stub.
RUN python -c "from resemble_enhance.enhancer.download import download; download('${RE_MODEL_DIR}')" \
    && python -c "from resemble_enhance.enhancer.inference import load_enhancer; load_enhancer('${RE_MODEL_DIR}', 'cpu')"

WORKDIR /app
COPY runpod/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
