# clip-warp-worker

RunPod serverless worker that renders a video clip from a text description.

The picture is produced frame by frame: each next frame is the previous one,
nudged by an affine transform and redrawn with img2img. That gives the
recognisable flowing motion — and also the constraint the worker is built
around, see **Shots** below.

Engine: Stable Diffusion 1.5 + LCM-LoRA (both OpenRAIL-M, commercially usable).
Weights are baked into the image at build time, so a cold start loads modules
rather than downloading gigabytes.

## Input

```jsonc
{
  "input": {
    "prompt": "neon city street at night, rain, reflections",
    "style": "flow",            // flow | drift | push | orbit
    "seconds": 60,              // 1..300
    "fps": 12,                  // 8..15
    "shot_seconds": 8,          // 4..15, ignored when `shots` is given
    "size": 512,                // 512 | 640
    "seed": 12345,              // optional
    "mark": true,               // AI-provenance metadata, default true
    "shots": [                  // optional prompt schedule
      {"prompt": "verse in the city", "seconds": 20},
      {"prompt": "chorus in space",   "seconds": 40}
    ],
    "upload": {                 // optional; without it the mp4 rides inline
      "url": "https://…/clip/upload",
      "gen_id": 42, "expires": 1757000000, "sig": "…"
    }
  }
}
```

## Output

With `upload` — a receipt (the file is POSTed to the given URL):

```json
{"stored": true, "bytes": 8123456, "sha256": "…", "seconds": 60.0, "shots": 8,
 "gen_seconds": 145.2, "sec_per_frame": 0.2, "fps": 12, "n_frames": 720}
```

Without `upload` — the same metrics plus `mp4_b64`. Inline delivery is capped
(`INLINE_MAX_BYTES`, default 6 MiB): base64 inflates bytes by a third, and a
long clip does not fit a serverless result payload. That is what `upload` is for.

On failure: `{"error": "…"}`. The caller is expected to treat any error as a
full refund — nothing is delivered.

## Shots — why a long clip is cut into scenes

The zoom of the warp loop **accumulates**. At a modest 0.6% per frame the camera
doubles the image in nine seconds; over three minutes it would end up inside a
tiny detail. So the worker restarts the composition (a fresh txt2img frame) at
every shot boundary, and each shot gets its own colour anchor. Without this a
long clip is not merely worse — it is unusable.

`shots` lets the caller place those boundaries and give each stretch its own
prompt. The plan is **truncated to the requested clip length**, never extended.

## Notes that cost time to learn

* **Frames are streamed to ffmpeg as they are produced.** Accumulating them and
  encoding at the end works in a test and dies at the target: 2160 frames of
  512×512 are ~1.7 GB of live bytes.
* **In img2img what matters is `steps × strength`.** The pipeline runs only a
  `strength` fraction of the steps, so lowering strength without raising steps
  starves the model instead of stabilising it.
* **Colour drifts** across an img2img chain; each shot is re-anchored to the
  LAB statistics of its own first frame.
* **torch 2.6+ is required.** `transformers` 5.x silently disables its torch
  integration on older torch, and what then fails is `diffusers`, with
  `name 'nn' is not defined` — an error that names neither the culprit nor the
  versions.

## Build

CI builds and pushes to GHCR on every push to `main`
(`ghcr.io/bandidas1/timbrica-clip-warp-worker:latest`).

## Local run without Docker

`local_smoke.py` in the parent project runs `handler()` directly with a stub for
the `runpod` module — useful for checking the loop and the shot schedule on a
desktop GPU.
