# JevVision Live — Real-Time Webcam Object Detection (YOLOv8) + Jev AI Decision Layer

Point your webcam at the world: **YOLOv8** detects and boxes objects live from your webcam feed,
and every few seconds **Jev** (TypeSafe AI's decision model) judges the detector's output —
is that person really there? Is a detection likely false? Did YOLO miss something?

![stack](https://img.shields.io/badge/detection-YOLOv8%20(ultralytics)-red) ![backend](https://img.shields.io/badge/backend-Python%20stdlib%20server-green) ![ai](https://img.shields.io/badge/AI-Jev%20decision%20model%20(TypeSafe)-orange)

## How it works

```
Webcam ──► Browser (smooth video preview; grabs a frame ~8x/sec)
              │
              └─ POST /api/detect ──► Python backend ──► YOLOv8 (ultralytics)
                                          ◄── bounding boxes + confidences
              Browser draws the boxes over the video.
              Every few seconds the detection STATE (names + confidences) also goes:
                        POST /api/identify ──► Jev  POST /v1/decide
                                 { state, questions: [typed probability questions] }
                                 ◄── calibrated probabilities (~100 ms)
```

- **YOLO does the seeing** — state-of-the-art detection (80 COCO classes), far stronger than
  in-browser mini-models. On CPU expect ~5–12 detections/sec; the video preview itself stays smooth.
- **Jev is the judge, not the eyes** — Jev is a *decision* model: it takes structured state + typed
  questions and returns calibrated probabilities. Every few seconds it scores:
  *Person actually present · False-positive risk · Missed-object risk · Worth flagging for review.*
  Same integration pattern as [BugScope AI](https://github.com/Khelendrameena/bugscope-ai).
- **No Jev key? No problem.** Without `JEV_API_KEY`, YOLO detection still works fully;
  add the key and the Jev Decision Layer panel switches on.

## Run it

```bash
git clone https://github.com/Khelendrameena/jev-vision-live.git
cd jev-vision-live
pip install -r requirements.txt     # ultralytics (pulls PyTorch — a few minutes, one time)
python app.py
```

Open **http://localhost:8000**, click **Start Camera**, allow webcam access.
The first detection downloads the YOLOv8n weights (~6 MB) automatically.

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

| Variable      | Default                                | Purpose                                   |
|---------------|----------------------------------------|-------------------------------------------|
| `JEV_API_KEY` | —                                      | TypeSafe / Jev API key                    |
| `JEV_API_URL` | `https://api.typesafe.ai/v1/decide`    | Jev decide endpoint (full URL)            |
| `YOLO_MODEL`  | `yolov8n.pt`                           | Any ultralytics YOLO weights (n/s/m/l/x)  |
| `YOLO_CONF`   | `0.35`                                 | Detection confidence threshold            |
| `PORT`        | `8000`                                 | Server port                               |

Want sharper detection and have a GPU or patience? `YOLO_MODEL=yolov8s.pt` (or `m`) trades speed for accuracy.

## Features

- Live YOLOv8 bounding boxes with confidence labels over the webcam feed
- Jev Decision Layer — probability bars + verdict line, refreshed every few seconds, with call latency
- Live event log — "person entered the frame" / "laptop left the frame" with timestamps
- Stats — detection rate (FPS), objects in frame, total detections, Jev call count
- Adjustable Jev interval (1.5–6 s), manual "Ask Jev Now", status badges for camera / YOLO / Jev
  (no invisible progress — every subsystem shows its state)

## Project structure

```
app.py              # stdlib HTTP server + /api/detect (YOLO) + /api/identify (Jev /v1/decide) + /api/health
static/index.html   # webcam UI, frame-grab loop, box overlay, Jev panel, event log
requirements.txt    # ultralytics + pillow
tests/smoke_test.py
```

## Honest notes

- Detection rate is CPU-bound (YOLOv8n on a typical laptop CPU ≈ 5–12 fps of *detections*;
  the preview video is unaffected). A CUDA GPU raises this dramatically.
- Frames go from your browser to your own local backend only — nothing is uploaded anywhere
  except the detection *state* (object names + numbers) sent to Jev. No images leave the machine.
- The Jev call path (auth, timeout, error handling, answer parsing) is implemented against the
  `/v1/decide` contract and verified end-to-end against a local stub of that contract;
  live Jev responses depend on your key's entitlements.
