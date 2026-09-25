# Modal deployment

Runs the same worker as RunPod (`runpod/handler.py`, same input/output) on Modal, with a RunPod-style
queue API. Upstream files stay untouched, like `runpod/`.

```bash
modal_app/deploy.sh     # modal deploy + pre-build memory snapshots (needs modal CLI and uv)
```

`deploy.sh` runs `modal deploy modal_app/app.py`, then `warmup.py`, which starts 5 containers (= max_containers) in
parallel through the `warmup` method (no storage writes). Modal keeps 2-3 memory snapshots per GPU
type and builds each lazily on the first container of that worker type (+30-40 s for that request),
so this moves that cost off real requests. Coverage isn't guaranteed, and a warm-up costs ~5 GPU-minutes.

Keep this directory out of a folder named `modal` (it would shadow the `modal` package).

## API

`modal deploy` prints the URL (`https://<workspace>--resemble-enhance-api.modal.run`). It is protected by
Modal proxy auth: send the
`Modal-Key` / `Modal-Secret` headers from a proxy auth token (Modal dashboard → Settings → Proxy Auth Tokens).

| request | response |
|---|---|
| `POST /run` `{"input": {...}}` | `{"id": "fc-...", "status": "IN_QUEUE"}` |
| `GET /status/{id}` | `{"id", "status": "IN_PROGRESS" \| "COMPLETED" \| "FAILED", "output" \| "error"}` |
| `POST /cancel/{id}` | `{"id", "status": "CANCELLED"}` |

`input` is exactly the RunPod input (see `runpod/README.md`). `output` additionally has `queue_seconds`,
`total_seconds`, `peak_ram_mb` and `peak_gpu_mb`.

From Python (no HTTP auth needed):

```python
import time, modal
enhancer = modal.Cls.from_name("resemble-enhance", "Enhancer")()
out = enhancer.enhance.remote("job-id", {"audio_url": "...", "mode": "enhance", "denoise": True}, time.time())
```

## Setup

- GPU L4, falling back to A10 when no L4 is free; 2 CPU, 10 GB RAM reserved (peak ~6.5 GB on a 40-min file), timeout 30 min, scale to zero after 2 min idle, max 5 containers (workspace GPU limit shared with demucs).
- Memory snapshot: imports and the CPU model load are snapshotted; a new container restores it and only
  moves the model to the GPU (~1 s). `load_enhancer` is swapped for the restored model so nothing reloads.
- GCS output is configured only through Modal secrets, so no bucket names live in this public repo:
  `demucs-gcs` (`GCS_SERVICE_ACCOUNT_JSON`) and `resemble-enhance-config` (`GCS_BUCKET`, `GCS_PREFIX`).
  Create the latter with `modal secret create resemble-enhance-config GCS_BUCKET=<bucket> GCS_PREFIX=<prefix>`.
