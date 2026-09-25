"""Modal deployment of the resemble-enhance worker, with a RunPod-style queue API.

Runs the same runpod/handler.py as the RunPod image, so inputs and outputs are identical.

Deploy from the repo root:
    modal deploy modal_app/app.py

HTTP API (Modal proxy auth: send Modal-Key / Modal-Secret headers from a proxy auth token):
    POST /run            {"input": {...}, "webhook"?: "https://..."}  -> {"id": "fc-...", "status": "IN_QUEUE"}
    POST /runsync        same body; waits up to 90 s  -> finished body, or {"id", "status": "IN_PROGRESS"}
    GET  /status/{id}                                  -> {"id", "status", "output" | "error"}
    POST /cancel/{id}                                  -> {"id", "status": "CANCELLED"}
status is IN_QUEUE, IN_PROGRESS (a worker picked it up), COMPLETED, FAILED, CANCELLED or TIMED_OUT.

With "webhook", the final /status body is POSTed there for every terminal status (COMPLETED,
FAILED, CANCELLED, TIMED_OUT), signed with WEBHOOK_SECRET from the Modal secret "audio-webhook";
delivery is done by the `watch` function and retried for ~12 min (see _send_webhook).
Same contract as the demucs app.

GCS upload uses the same env vars as RunPod, all from Modal secrets (nothing deployment-specific is
kept in this public repo): "demucs-gcs" holds GCS_SERVICE_ACCOUNT_JSON, "resemble-enhance-config" holds
GCS_BUCKET and GCS_PREFIX.
"""

import time

import modal

# Tried in order when a container starts: L4 (~$0.80/h), then A10 (~$1.10/h, same 24 GB), then
# L40S (~$1.95/h, often free) instead of waiting minutes for L4/A10 capacity.
# Memory snapshots are per GPU type, so the first container on each type builds its own.
GPU = ["L4", "A10", "L40S"]
MODEL_DIR = "/models/enhancer_stage2"

app = modal.App("resemble-enhance")

# call id -> start time, written when a worker picks the job up, so /status can tell
# IN_QUEUE from IN_PROGRESS (Modal itself only knows "not finished yet").
started = modal.Dict.from_name("resemble-enhance-started", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch==2.7.1", "torchaudio==2.7.1")
    .pip_install_from_requirements("runpod/requirements.txt")
    .env(
        {
            "RE_MODEL_DIR": MODEL_DIR,
            # deepspeed is only imported (training code), never used: the stub satisfies it.
            "PYTHONPATH": "/opt/resemble-enhance:/opt/deepspeed_stub:/opt/worker",
        }
    )
    .add_local_dir("runpod/deepspeed_stub", "/opt/deepspeed_stub", copy=True, ignore=["**/__pycache__"])
    .add_local_dir(
        "resemble_enhance",
        "/opt/resemble-enhance/resemble_enhance",
        copy=True,
        ignore=["**/__pycache__", "model_repo"],
    )
    # Bake the weights (~700 MB) into the image so cold starts never download them.
    .run_commands(
        f"python -c \"from resemble_enhance.enhancer.download import download; download('{MODEL_DIR}')\""
    )
    .add_local_file("runpod/handler.py", "/opt/worker/handler.py")
)


@app.cls(
    gpu=GPU,
    # Reserved, not a cap: decode and mp3 encode need real cores. Peak RAM is ~5.8 GB for a short
    # clip (model + libraries) and ~6.5 GB for a 40-min file.
    cpu=2.0,
    memory=10240,
    image=image,
    secrets=[modal.Secret.from_name("demucs-gcs"), modal.Secret.from_name("resemble-enhance-config"),
             modal.Secret.from_name("audio-webhook")],
    # enhance on a 40-min file takes several minutes on an L4.
    timeout=1800,
    # Keep an idle container 2 min: requests often come in bursts (preview, then the full file),
    # and a cold start costs ~10-25 s. Idle time is billed (~$0.016/min).
    scaledown_window=120,
    # The workspace GPU limit (10 on Starter) is shared with the demucs app and split 5/5; idle
    # warm containers keep their slot, so one app could otherwise starve the other.
    max_containers=5,
    enable_memory_snapshot=True,
)
class Enhancer:
    # Memory snapshot: imports and the CPU model load are captured once, so new containers
    # restore them instead of redoing them. There's no GPU during this phase, so importing
    # handler loads the model with DEVICE == "cpu".
    @modal.enter(snap=True)
    def load_cpu(self):
        t0 = time.perf_counter()
        import handler

        self.handler = handler
        # One tiny CPU pass (denoise + enhance on 1 s of silence) so every lazy import happens now
        # and lands in the snapshot. Otherwise the first request does them; a cancel mid-import
        # leaves a module half-initialized and every later request in that container fails.
        import torch
        import torch._dynamo  # noqa: F401
        from resemble_enhance.enhancer.inference import denoise, enhance

        silence = torch.zeros(44100)
        denoise(silence, 44100, "cpu", run_dir=MODEL_DIR)
        enhance(silence, 44100, "cpu", nfe=1, run_dir=MODEL_DIR)
        # Same for the request path around the model: torchaudio's decode backend and the GCS client.
        import tempfile

        import soundfile as sf
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            sf.write(f.name, silence.numpy(), 44100, format="MP3")
            torchaudio.load(f.name)
        try:
            handler._gcs()
        except Exception as e:  # no credentials configured: uploads fall back to S3/base64
            print(f"GCS client not created: {e}")
        print(f"model loaded and warmed on cpu in {time.perf_counter() - t0:.2f}s")

    @modal.enter(snap=False)
    def load_gpu(self):
        t0 = time.perf_counter()
        import torch
        from resemble_enhance.enhancer import inference

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # load_enhancer is cached per (run_dir, device): take the snapshotted CPU model, move it,
        # and make denoise()/enhance() use it instead of loading a second copy for "cuda".
        model = inference.load_enhancer(MODEL_DIR, "cpu").to(device)
        inference.load_enhancer = lambda run_dir, device: model
        # handler.DEVICE was computed at import, before the GPU was attached.
        self.handler.DEVICE = device
        print(f"model moved to {device} in {time.perf_counter() - t0:.2f}s")

    @modal.method()
    def warmup(self, hold_seconds: float = 0) -> dict:
        """Start a container (building a memory snapshot if its worker type has none) without
        touching storage. deploy.sh calls it in parallel after each deploy; hold_seconds keeps the
        container busy so the parallel calls land on separate containers."""
        import os

        import torch

        time.sleep(hold_seconds)
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        return {"task": os.environ.get("MODAL_TASK_ID"), "device": gpu}

    @modal.method()
    def enhance(self, job_id: str, inp: dict, submitted_at: float) -> dict:
        import resource

        import torch

        started_at = time.time()
        started[modal.current_function_call_id()] = started_at
        torch.cuda.reset_peak_memory_stats()
        try:
            out = self.handler.handler({"id": job_id, "input": inp})
        except (AttributeError, ImportError) as e:
            if "partially initialized module" in str(e) or isinstance(e, ImportError):
                # Module state is corrupt (e.g. an import interrupted by a cancel): stop taking
                # inputs so Modal replaces this container instead of failing every job on it.
                modal.experimental.stop_fetching_inputs()
            raise
        out["queue_seconds"] = round(started_at - submitted_at, 3)
        out["total_seconds"] = round(time.time() - started_at, 3)
        out["peak_ram_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() // 2**20
        return out


def _status_body(call_id: str, out) -> dict:
    """The /status response for a finished call; webhooks send the same body."""
    if isinstance(out, dict) and "error" in out:
        return {"id": call_id, "status": "FAILED", "error": out["error"]}
    return {"id": call_id, "status": "COMPLETED", "output": out}


def _error_status(call_id: str, e: Exception) -> dict:
    """Map an exception from FunctionCall.get to a RunPod-style terminal status."""
    if isinstance(e, modal.exception.FunctionTimeoutError):
        status = "TIMED_OUT"
    elif "cancelled" in str(e).lower():  # a cancelled call surfaces as RemoteError("Function call was cancelled ...")
        status = "CANCELLED"
    else:
        status = "FAILED"
    return {"id": call_id, "status": status, "error": f"{type(e).__name__}: {e}"}


def _send_webhook(url: str, body: dict) -> None:
    """POST body to url, signed with WEBHOOK_SECRET; 5 attempts over ~12 min, never raises.

    Headers: X-Webhook-Timestamp: <unix seconds>
             X-Webhook-Signature: v1=<hex HMAC-SHA256(secret, f"{timestamp}.{raw body}")>
    """
    import hashlib
    import hmac
    import json
    import os

    import requests

    raw = json.dumps(body, separators=(",", ":")).encode()
    for attempt, delay in enumerate((0, 5, 30, 120, 600)):
        time.sleep(delay)
        ts = str(int(time.time()))
        sig = hmac.new(os.environ["WEBHOOK_SECRET"].encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
        headers = {"Content-Type": "application/json", "X-Webhook-Timestamp": ts, "X-Webhook-Signature": f"v1={sig}"}
        try:
            r = requests.post(url, data=raw, headers=headers, timeout=10)
            if r.status_code < 300:
                return
            print(f"webhook attempt {attempt + 1}: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"webhook attempt {attempt + 1}: {e}")
    print(f"webhook to {url} failed; the result is still available via /status")


web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "requests")


@app.function(image=web_image, secrets=[modal.Secret.from_name("audio-webhook")], timeout=6 * 3600)
@modal.concurrent(max_inputs=1000)
async def watch(call_id: str, webhook: str) -> None:
    """Wait for a job to reach ANY terminal state and deliver the webhook.

    Runs outside the GPU worker, so timeouts, crashes and cancellations still produce a
    webhook (TIMED_OUT / FAILED / CANCELLED), like RunPod's platform-side webhooks.
    Only waits, so one small CPU container serves many jobs at once.
    """
    import asyncio

    try:
        body = _status_body(call_id, await modal.FunctionCall.from_id(call_id).get.aio())
    except Exception as e:
        body = _error_status(call_id, e)
    await asyncio.to_thread(_send_webhook, webhook, body)


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)  # requests only enqueue/poll/wait, so one container serves many
@modal.asgi_app(requires_proxy_auth=True)
def api():
    import uuid

    from fastapi import FastAPI, HTTPException

    web = FastAPI()

    async def spawn(body: dict):
        inp = body.get("input")
        if not isinstance(inp, dict):
            raise HTTPException(400, "body must be {\"input\": {...}, \"webhook\"?: \"https://...\"}")
        webhook = body.get("webhook")
        if webhook is not None and not (isinstance(webhook, str) and webhook.startswith("https://")):
            raise HTTPException(400, "webhook must be an https:// URL")
        call = await Enhancer().enhance.spawn.aio(uuid.uuid4().hex, inp, time.time())
        if webhook:
            await watch.spawn.aio(call.object_id, webhook)
        return call

    @web.post("/run")
    async def run(body: dict):
        call = await spawn(body)
        return {"id": call.object_id, "status": "IN_QUEUE"}

    @web.post("/runsync")
    async def runsync(body: dict):
        """Waits up to 90 s; if the job isn't done by then, returns IN_PROGRESS and the id to poll."""
        call = await spawn(body)
        try:
            out = await call.get.aio(timeout=90)
        except TimeoutError:  # builtin: not finished within 90 s (a job timeout is FunctionTimeoutError)
            running = await started.contains.aio(call.object_id)
            return {"id": call.object_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except Exception as e:
            return _error_status(call.object_id, e)
        return _status_body(call.object_id, out)

    def check_id(call_id: str) -> None:
        if not call_id.startswith("fc-"):  # Modal raises ExecutionError, not NotFound, for these
            raise HTTPException(404, "job not found")

    @web.get("/status/{call_id}")
    async def status(call_id: str):
        check_id(call_id)
        try:
            out = await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except TimeoutError:  # builtin: not finished yet (a job timeout is FunctionTimeoutError, handled below)
            running = await started.contains.aio(call_id)
            return {"id": call_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        except Exception as e:  # the call itself raised (crash, timeout, cancel)
            return _error_status(call_id, e)
        return _status_body(call_id, out)

    @web.post("/cancel/{call_id}")
    async def cancel(call_id: str):
        check_id(call_id)
        try:
            await modal.FunctionCall.from_id(call_id).cancel.aio()
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        return {"id": call_id, "status": "CANCELLED"}

    return web
