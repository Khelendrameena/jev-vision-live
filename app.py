#!/usr/bin/env python3
"""
JevVision Live — real-time webcam object detection + Jev AI decision layer.

Architecture
------------
Browser (static/index.html)
  - getUserMedia webcam stream
  - TensorFlow.js COCO-SSD runs IN the browser -> bounding boxes at full frame rate.
    No video ever leaves the machine for detection.
  - Every `interval` seconds it POSTs the current detection *state*
    (objects, confidences, counts) to /api/identify.

Backend (this file, Python standard library only — no pip installs needed)
  - Serves the page and static assets.
  - POST /api/identify -> asks Jev (TypeSafe AI's decision model) typed
    probability questions about the detection state, e.g. "how likely is a
    person actually present?", "how likely is one of these detections a
    false positive?", "how likely did the local detector miss an object?".
    Jev returns calibrated probabilities in ~100 ms — it is a decision
    model, not a text/vision generator, so it judges the detector's output
    instead of looking at pixels. Same integration contract as BugScope AI.
  - If JEV_API_KEY is not set, the endpoint answers with
    status="not_configured" and echoes the local detections, so the app is
    fully usable for real-time detection out of the box.

Configuration (environment variables)
-------------------------------------
  JEV_API_KEY   Your TypeSafe / Jev API key            (required for Jev layer)
  JEV_API_URL   Full Jev decide endpoint               (default: https://api.typesafe.ai/v1/decide)
  PORT          Port to serve on                       (default: 8000)

Run:  python app.py   ->  open http://localhost:8000
"""

import json
import os
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(ROOT, "static")
PORT = int(os.environ.get("PORT", "8000"))

JEV_API_KEY = os.environ.get("JEV_API_KEY", "").strip()
JEV_API_URL = os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/decide")
JEV_TIMEOUT = 10  # seconds — Jev is a ~100ms decision model; never freeze the live UI

# Typed questions Jev answers over the detection state (same pattern as BugScope AI).
QUESTIONS = [
    {
        "id": "person_present",
        "label": "Person actually present",
        "question": "How likely is it that a person is actually present in this frame, given the local detector's reported objects and confidences?",
        "type": "probability",
    },
    {
        "id": "false_positive",
        "label": "False-positive risk",
        "question": "How likely is it that at least one of the reported detections is a false positive (an object that is not really there)?",
        "type": "probability",
    },
    {
        "id": "missed_object",
        "label": "Missed-object risk",
        "question": "How likely is it that the local detector missed at least one real object that is present in this frame?",
        "type": "probability",
    },
    {
        "id": "alert_worthy",
        "label": "Worth flagging for review",
        "question": "How likely is it that this frame shows activity worth flagging for a human to review (unusual, crowded, or important)?",
        "type": "probability",
    },
]


def build_state(detections: list, stats: dict) -> dict:
    """The 'state' Jev reasons over — detector output, not pixels."""
    classes = [d.get("class", "object") for d in detections]
    return {
        "detector": "coco-ssd-mobilenet-v2 (in-browser)",
        "objects": [
            {"class": d.get("class", "object"),
             "confidence": round(float(d.get("score", 0)), 3)}
            for d in detections
        ],
        "object_count": len(detections),
        "unique_classes": sorted(set(classes)),
        "person_reported": "person" in classes,
        "detector_fps": stats.get("fps"),
    }


def call_jev(detections: list, stats: dict) -> list:
    """Ask Jev the typed questions; return [{id, label, probability}]."""
    payload = {"state": build_state(detections, stats), "questions": QUESTIONS}
    req = urllib.request.Request(
        JEV_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JEV_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=JEV_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answers = data.get("answers", {})
    return [
        {"id": q["id"], "label": q["label"],
         "probability": round(float(answers.get(q["id"], 0)), 3)}
        for q in QUESTIONS
    ]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def log_message(self, fmt, *args):  # keep the console clean
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def send_json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            return self.send_json(200, {
                "ok": True,
                "jev_configured": bool(JEV_API_KEY),
                "jev_mode": "decide (typed probability questions)" if JEV_API_KEY else None,
            })
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/identify":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:  # state payloads are tiny — 1 MB is generous
                return self.send_json(413, {"error": "payload too large"})
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self.send_json(400, {"error": "bad request body"})

        detections = req.get("detections", [])
        stats = req.get("stats", {})
        started = time.time()
        local_objects = [
            {"name": d.get("class", "object"),
             "confidence": round(float(d.get("score", 0)), 2)}
            for d in detections
        ]

        if not JEV_API_KEY:
            return self.send_json(200, {
                "status": "not_configured",
                "jev_configured": False,
                "answers": [],
                "objects": local_objects,
                "latency_ms": int((time.time() - started) * 1000),
                "hint": "Set JEV_API_KEY to enable the Jev decision layer.",
            })

        try:
            answers = call_jev(detections, stats)
            return self.send_json(200, {
                "status": "ok",
                "jev_configured": True,
                "answers": answers,
                "objects": local_objects,
                "latency_ms": int((time.time() - started) * 1000),
            })
        except urllib.error.HTTPError as e:
            return self.send_json(200, {
                "status": "error", "jev_configured": True,
                "answers": [], "objects": local_objects,
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"Jev API returned HTTP {e.code}. Check JEV_API_URL / key.",
            })
        except Exception as e:  # timeout, DNS, malformed reply, ...
            return self.send_json(200, {
                "status": "error", "jev_configured": True,
                "answers": [], "objects": local_objects,
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"Jev call failed: {type(e).__name__}",
            })


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"JevVision Live -> http://localhost:{PORT}")
    print(f"Jev decision layer: {'ON (' + JEV_API_URL + ')' if JEV_API_KEY else 'OFF — set JEV_API_KEY to enable (local detection still works)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
