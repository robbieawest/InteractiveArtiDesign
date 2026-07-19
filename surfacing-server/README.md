# Surfacing job server

A local FastAPI sidecar that runs surfacing methods on the sketch. The Vite
app reaches it through the `/api` proxy in `vite.config.ts`.

One-time setup:

```bash
cd surfacing-server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

After that, `npm run dev` starts and stops it together with Vite (the
`surfacingServer` plugin in `vite.config.ts`). To run it by hand instead —
e.g. to watch its logs separately or restart it after editing an adapter:

```bash
.venv/bin/uvicorn server:app --port 8801
```

(If it's already running when Vite starts, the spawned copy exits on the
taken port and the proxy talks to yours.)

The app works fine without it — the Surfacer panel just reports the server
as offline.

## Protocol

- `GET  /api/health` → `{ status, methods }`
- `POST /api/jobs` with `{ method, sketch, options }` → `{ jobId }`
- `GET  /api/jobs/{id}` → `{ status: pending|running|done|error, progress, message, error }`
- `GET  /api/jobs/{id}/result` → binary glTF (`.glb`)

`sketch` is built by `src/surfacing/client.ts`: world-space stroke
centerlines with part ids, plus the part and joint (screw) tables, so
articulation-aware methods get the full picture and baselines ignore what
they don't need.

## Adapters

One surfacing method = one adapter in `adapters/` (register it in
`adapters/__init__.py`). Adapters run in a background thread per job; slow
methods should call `report(progress, message)` as they go.

Real methods (VNS etc.) should live in their own repos/environments and be
invoked by their adapter as a subprocess (e.g. `conda run -n vns python ...`)
— this server's env stays torch-free so method dependencies never conflict.
The included `bbox` adapter is a stand-in that boxes each part, to exercise
the whole pipeline end to end.
