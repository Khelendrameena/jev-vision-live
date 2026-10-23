# JevVision Live — Real-Time Webcam Object Detection + Jev AI Decision Layer

Point your webcam at the world: objects are detected and boxed **live, in the browser**,
and every few seconds **Jev** (TypeSafe AI's decision model) judges the detector's output —
is that person really there? Is a detection likely false? Did the detector miss something?

![stack](https://img.shields.io/badge/python-stdlib%20only-green) ![frontend](https://img.shields.io/badge/frontend-TensorFlow.js%20COCO--SSD-red) ![ai](https://img.shields.io/badge/AI-Jev%20decision%20model%20(TypeSafe)-orange)

## How it works

```
Webcam ──► Browser (COCO-SSD, TensorFlow.js) ──► bounding boxes @ full FPS
              │                                     (video never leaves the machine)
              └─ every 2.5 s: detection STATE (names + confidences, no pixels)
                        │
                        ▼
              Python backend ──► Jev  POST /v1/decide
                                 { state, questions: [typed probability questions] }
                                 ◄── calibrated probabilities (~100 ms)
```

- **Detection is 100% local** — COCO-SSD runs in the browser via TensorFlow.js, so boxes are real-time.
- **Jev is the judge, not the eyes** — Jev is a *decision* model: it takes structured state + typed
  questions and returns calibrated probabilities. It never sees your video. Every few seconds it scores:
  *Person actually present · False-positive risk · Missed-object risk · Worth flagging for review.*
  This is the same integration pattern as [BugScope AI](https://github.com/Khelendrameena/bugscope-ai).
- **No Jev key? No problem.** Without `JEV_API_KEY` the app still detects in real time; add the key
  and the Jev Decision Layer panel switches on.

## Run it

**Zero dependencies** — the backend is pure Python standard library.

```bash
git clone https://github.com/Khelendrameena/jev-vision-live.git
cd jev-vision-live
python app.py
```

Open **http://localhost:8000**, click **Start Camera**, allow webcam access.

> Webcam requires a secure context — `localhost` counts, so local runs work over plain HTTP.

### Enable the Jev decision layer

Windows (PowerShell):

```powershell
$env:JEV_API_KEY = "your-key"
python app.py
```

Linux / macOS:

```bash
export JEV_API_KEY="your-key"
python app.py
```

| Variable      | Default                                | Purpose                          |
|---------------|----------------------------------------|----------------------------------|
| `JEV_API_KEY` | —                                      | TypeSafe / Jev API key           |
| `JEV_API_URL` | `https://api.typesafe.ai/v1/decide`    | Jev decide endpoint (full URL)   |
| `PORT`        | `8000`                                 | Server port                      |

## Features

- Live bounding boxes with confidence labels (80 COCO classes: people, vehicles, animals, electronics…)
- Jev Decision Layer — probability bars + verdict line, refreshed every few seconds, with call latency
- Live event log — "person entered the frame" / "laptop left the frame" with timestamps
- Stats — detector FPS, objects in frame, total detections, Jev call count
- Adjustable Jev interval (1.5–6 s), manual "Ask Jev Now", loader + status badges
  (no invisible progress — every subsystem shows its state)

## Project structure

```
app.py              # stdlib-only backend: static server + /api/health + /api/identify (Jev /v1/decide client)
static/index.html   # webcam UI, COCO-SSD detection loop, Jev panel, event log
tests/smoke_test.py
```

## Honest notes

- Detection quality = COCO-SSD (MobileNet v2) — fast and local, not state-of-the-art;
  Jev's judgement layer exists precisely to quantify how much to trust it on any given frame.
- Privacy by design: only object names and confidence numbers are sent to Jev — never images or video.
- The Jev call path (auth, timeout, error handling, answer parsing) is implemented against the
  `/v1/decide` contract and verified end-to-end against a local stub of that contract;
  live Jev responses depend on your key's entitlements.
