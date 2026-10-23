# JevVision Live — Real-Time Webcam Object Detection + Jev AI Identification

Point your webcam at the world: objects are detected and boxed **live, in the browser**,
and every few seconds a frame is sent to **Jev** (TypeSafe AI) to verify and identify
what is really in the scene.

![stack](https://img.shields.io/badge/python-stdlib%20only-green) ![frontend](https://img.shields.io/badge/frontend-TensorFlow.js%20COCO--SSD-red) ![ai](https://img.shields.io/badge/AI-Jev%20(TypeSafe)-orange)

## How it works

```
Webcam ──► Browser (COCO-SSD, TensorFlow.js) ──► bounding boxes @ full FPS (nothing uploaded)
              │
              └─ every 2.5 s: one JPEG frame ──► Flask-free Python backend ──► Jev vision model
                                                     POST /api/identify        verify + identify
                                                     ◄── objects, confidence, scene summary
```

- **Detection is 100% local** — COCO-SSD runs in the browser via TensorFlow.js, so boxes are real-time.
- **Jev does the identification** — it verifies the local detector's guesses, drops false positives,
  adds objects the lightweight detector missed, and describes the scene in one line.
- **No Jev key? No problem.** Without `JEV_API_KEY` the app still detects in real time and the
  Jev panel mirrors the local labels. Add the key and Jev takes over identification.

## Run it

**Zero dependencies** — the backend is pure Python standard library.

```bash
git clone https://github.com/Khelendrameena/jev-vision-live.git
cd jev-vision-live
python app.py
```

Open **http://localhost:8000**, click **Start Camera**, allow webcam access.

> Webcam requires a secure context — `localhost` counts, so local runs work over plain HTTP.

### Enable Jev identification

Windows (PowerShell):

```powershell
$env:JEV_API_KEY = "your-key"
$env:JEV_MODEL = "your-vision-model"   # optional
python app.py
```

Linux / macOS:

```bash
export JEV_API_KEY="your-key"
python app.py
```

| Variable      | Default                  | Purpose                                  |
|---------------|--------------------------|------------------------------------------|
| `JEV_API_KEY` | —                        | TypeSafe / Jev API key                   |
| `JEV_API_URL` | `https://api.typesafe.ai` | Jev API base URL (OpenAI-compatible)    |
| `JEV_MODEL`   | `jev-vision`             | Vision-capable Jev model name            |
| `PORT`        | `8000`                   | Server port                              |

## Features

- Live bounding boxes with confidence labels (80 COCO classes: people, vehicles, animals, electronics…)
- Jev panel — scene summary, verified object chips with confidence and short notes
- Live event log — "person entered the frame" / "laptop left the frame" with timestamps
- Stats — detector FPS, objects in frame, total detections, Jev call count + latency
- Adjustable Jev interval (1.5–6 s), manual "Ask Jev Now" snapshot, loader + status badges
  (no invisible progress — every subsystem shows its state)

## Project structure

```
app.py            # stdlib-only backend: static server + /api/health + /api/identify (Jev client)
static/index.html # webcam UI, COCO-SSD detection loop, Jev panel, event log
tests/smoke_test.py
```

## Honest notes

- Detection quality = COCO-SSD (MobileNet v2) — fast and local, not state-of-the-art;
  Jev's verification layer exists precisely to catch its mistakes.
- Frames sent to Jev are downscaled to 640 px wide JPEGs; detection frames never leave the machine.
- The Jev call path (auth, timeout, error handling, JSON parsing) is implemented and unit-tested
  against the fallback; live Jev responses depend on your key/model entitlements.
