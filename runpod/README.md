# RunPod serverless worker

Queue-based RunPod Serverless worker for resemble-enhance: speech denoising and enhancement
(bandwidth extension to 44.1 kHz). Model weights are baked into the image.

Everything RunPod-specific lives in this directory (plus `.github/workflows/runpod-image.yml`).
Upstream files are untouched, so the fork keeps merging `resemble-ai/resemble-enhance` cleanly:

```bash
git remote add upstream https://github.com/resemble-ai/resemble-enhance.git
git pull upstream main
```

The image doesn't install deepspeed. resemble_enhance imports it at module level for training, but
inference never calls it, so `deepspeed_stub/` satisfies the imports. That drops the CUDA devel base
image and deepspeed's build toolchain. The image build loads the model once on CPU to check this.

## Image

Every push to `main` that touches `resemble_enhance/` or `runpod/` builds
`ghcr.io/sommo-io/resemble-enhance:latest` (and `:<commit sha>`) via GitHub Actions.
Pushing a tag like `v1.2.3` builds `:v1.2.3`, `:v1.2` and `:v1`; pin endpoints to one of those.

Build locally from the repo root:

```bash
docker buildx build --platform linux/amd64 -f runpod/Dockerfile -t <image> --push .
```

The base image is a build arg (`BASE_IMAGE`, default `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`).
PyTorch 2.7 / CUDA 12.8 supports Blackwell (RTX 50xx, RTX PRO 4500/6000 and their MIG slices) as well as
Ampere/Ada, so any GPU pool works. Images up to v1.0.0 were built on PyTorch 2.4 and don't run on Blackwell.
Hosts need a driver for CUDA 12.8+.

## Deploy

RunPod console → Serverless → New Endpoint → queue-based → container image above.
GPU: 16 GB+ (A4000 / L4 / A5000 / 4090). Container disk: 20 GB. Idle timeout ~5 s.
For outputs as links instead of base64 (recommended for audio > ~1 min), see [Output storage](#output-storage).

## Request

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/noisy.mp3", "mode": "enhance", "denoise": true}}'
```

| field | default | notes |
|---|---|---|
| `audio_url` / `audio_base64` | — | one is required |
| `mode` | `enhance` | `enhance`, `denoise`, or `both` |
| `solver` | `midpoint` | `midpoint`, `rk4`, `euler` |
| `nfe` | `64` | 1–128; lower is faster |
| `tau` | `0.5` | CFM prior temperature, 0–1 |
| `denoise` | `false` | denoise before enhancing (lambd 0.9 vs 0.1) |
| `lambd` | — | set lambd directly, overrides `denoise` |
| `output_format` | `flac` | `flac`, `wav`, `mp3` |
| `gcs_bucket` | `$GCS_BUCKET` | upload outputs to this GCS bucket |
| `gcs_prefix` | `$GCS_PREFIX` | object name prefix, e.g. `enhanced/` |

Response:

```json
{"format": "flac", "duration": 12.3, "sample_rate": 44100, "inference_seconds": 4.1,
 "enhanced": {"base64": "..."}, "denoised": {"base64": "..."}}
```

With a bucket configured, each audio field is a link instead of base64 (see below).

## Output storage

**Google Cloud Storage** (takes precedence). Set on the endpoint:

| env var | notes |
|---|---|
| `GCS_BUCKET` | default bucket; a request can override it with `gcs_bucket` |
| `GCS_PREFIX` | default object prefix; a request can override it with `gcs_prefix` |
| `GCS_SERVICE_ACCOUNT_JSON` | service account key, raw JSON or base64 of it. Needs `storage.objects.create` on the bucket (e.g. Storage Object Creator) |

Objects are named `<prefix><job id>-enhanced.<fmt>` / `-denoised`, and each audio field becomes
`{"gcs_uri": "gs://bucket/...", "url": "https://storage.googleapis.com/bucket/..."}`. The URL doesn't expire;
it opens for anyone only if the bucket grants `allUsers` Storage Object Viewer.

Credentials are read from env only, never from the request, so they don't end up in RunPod job logs.

**S3-compatible**: set `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY`; each audio
field becomes `{"url": "..."}`.

Decode the base64 output:

```bash
jq -r '.output.enhanced.base64' resp.json | base64 -d > enhanced.flac
```

## Local test (needs an NVIDIA GPU)

```bash
docker run --gpus all --rm <image> python -u /app/handler.py --test_input "$(cat runpod/test_input.json)"
```
