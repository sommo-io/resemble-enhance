"""RunPod serverless handler for resemble-enhance.

Input (job["input"]):
    audio_url        str   URL of the source audio (any format ffmpeg can read), or
    audio_base64     str   base64-encoded audio file bytes
    mode             str   "enhance" (default) | "denoise" | "both"
    solver           str   "midpoint" (default) | "rk4" | "euler"
    nfe              int   CFM function evaluations, 1-128 (default 64)
    tau              float CFM prior temperature, 0-1 (default 0.5)
    denoise          bool  denoise before enhancing (default false); sets lambd 0.9 vs 0.1
    lambd            float override lambd directly, 0-1
    output_format    str   "flac" (default) | "wav" | "mp3"
    gcs_bucket       str   upload outputs to this GCS bucket (overrides GCS_BUCKET env)
    gcs_prefix       str   object name prefix, e.g. "enhanced/2026/" (overrides GCS_PREFIX env)

Output:
    {"sample_rate": 44100, "duration": s, "format": ..., "enhanced": <audio>, "denoised": <audio>}
    <audio> is, in order of precedence:
      {"gcs_uri": "gs://...", "url": "https://..."}  when a GCS bucket is set (input or GCS_BUCKET)
      {"url": ...}                                   when S3 env vars are set (BUCKET_ENDPOINT_URL, ...)
      {"base64": ...}                                otherwise

GCS env vars:
    GCS_BUCKET                default bucket
    GCS_PREFIX                default object name prefix
    GCS_SERVICE_ACCOUNT_JSON  service account key, raw JSON or base64 of it; if unset,
                              Application Default Credentials are used

"url" is the plain https://storage.googleapis.com/<bucket>/<object> link; it only opens
if the bucket allows reads (public, or the caller has access).
"""

import base64
import binascii
import io
import json
import os
import tempfile
import time
import uuid

import requests
import runpod
import soundfile as sf
import torch
import torchaudio
from google.cloud import storage
from google.oauth2 import service_account
from runpod.serverless.utils import rp_upload

from resemble_enhance.enhancer.inference import denoise, enhance, load_enhancer

MODEL_DIR = os.environ.get("RE_MODEL_DIR", "/models/enhancer_stage2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORMATS = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "mp3": ("MP3", None)}

CONTENT_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}

# Load once per worker so only the first cold start pays for it.
load_enhancer(MODEL_DIR, DEVICE)

_gcs_client = None


def _gcs() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        raw = os.environ.get("GCS_SERVICE_ACCOUNT_JSON", "").strip()
        if raw:
            if not raw.startswith("{"):
                raw = base64.b64decode(raw).decode("utf-8")
            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info)
            _gcs_client = storage.Client(project=info.get("project_id"), credentials=creds)
        else:
            _gcs_client = storage.Client()
    return _gcs_client


def _upload_gcs(data: bytes, bucket: str, name: str, fmt: str) -> dict:
    blob = _gcs().bucket(bucket).blob(name)
    blob.upload_from_string(data, content_type=CONTENT_TYPES[fmt])
    return {"gcs_uri": f"gs://{bucket}/{name}", "url": blob.public_url}


def _fetch_audio(inp: dict) -> str:
    """Write the input audio to a temp file and return its path."""
    if inp.get("audio_url"):
        url = inp["audio_url"]
        suffix = os.path.splitext(url.split("?")[0])[1] or ".audio"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.content
    elif inp.get("audio_base64"):
        b64 = inp["audio_base64"]
        if "," in b64[:100]:  # strip data: URI prefix
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        suffix = ".audio"
    else:
        raise ValueError("Provide 'audio_url' or 'audio_base64'")

    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _encode(wav: torch.Tensor, sr: int, fmt: str, job_id: str, name: str, gcs: tuple) -> dict:
    container, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, wav.clamp(-1, 1).numpy(), sr, format=container, subtype=subtype)
    data = buf.getvalue()

    bucket, prefix = gcs
    if bucket:
        try:
            return _upload_gcs(data, bucket, f"{prefix}{job_id}-{name}.{fmt}", fmt)
        except Exception as e:
            raise RuntimeError(f"GCS upload to gs://{bucket} failed: {e}") from e
    if os.environ.get("BUCKET_ENDPOINT_URL"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{name}.{fmt}")
            with open(path, "wb") as f:
                f.write(data)
            url = rp_upload.upload_file_to_bucket(f"{job_id}-{name}-{uuid.uuid4().hex[:8]}.{fmt}", path)
        return {"url": url}
    return {"base64": base64.b64encode(data).decode("ascii")}


def handler(job):
    inp = job.get("input") or {}
    job_id = job.get("id", "local")

    mode = inp.get("mode", "enhance").lower()
    solver = inp.get("solver", "midpoint").lower()
    nfe = int(inp.get("nfe", 64))
    tau = float(inp.get("tau", 0.5))
    lambd = float(inp["lambd"]) if "lambd" in inp else (0.9 if inp.get("denoise") else 0.1)
    fmt = inp.get("output_format", "flac").lower()
    gcs = (
        inp.get("gcs_bucket") or os.environ.get("GCS_BUCKET"),
        inp.get("gcs_prefix", os.environ.get("GCS_PREFIX", "")),
    )

    if mode not in ("enhance", "denoise", "both"):
        return {"error": f"mode must be enhance|denoise|both, got {mode!r}"}
    if fmt not in FORMATS:
        return {"error": f"output_format must be one of {list(FORMATS)}, got {fmt!r}"}

    try:
        path = _fetch_audio(inp)
    except (requests.RequestException, ValueError, binascii.Error) as e:
        return {"error": f"Could not read input audio: {e}"}

    try:
        dwav, sr = torchaudio.load(path)
    except Exception as e:
        return {"error": f"Could not decode input audio: {e}"}
    finally:
        os.remove(path)

    dwav = dwav.mean(dim=0)  # mono
    t0 = time.perf_counter()
    out = {"format": fmt, "duration": round(dwav.shape[-1] / sr, 3)}

    try:
        if mode in ("denoise", "both"):
            wav, new_sr = denoise(dwav, sr, DEVICE, run_dir=MODEL_DIR)
            out["denoised"] = _encode(wav.cpu(), new_sr, fmt, job_id, "denoised", gcs)
        if mode in ("enhance", "both"):
            wav, new_sr = enhance(
                dwav, sr, DEVICE, nfe=nfe, solver=solver, lambd=lambd, tau=tau, run_dir=MODEL_DIR
            )
            out["enhanced"] = _encode(wav.cpu(), new_sr, fmt, job_id, "enhanced", gcs)
    except AssertionError as e:  # upstream validates params with asserts
        return {"error": str(e)}
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out["sample_rate"] = new_sr
    out["inference_seconds"] = round(time.perf_counter() - t0, 3)
    return out


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
