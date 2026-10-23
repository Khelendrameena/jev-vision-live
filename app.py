#!/usr/bin/env python3
"""
JevVision Live — real-time webcam object detection + Jev AI identification.

Architecture
------------
Browser (static/index.html)
  - getUserMedia webcam stream
  - TensorFlow.js COCO-SSD runs IN the browser -> bounding boxes at full frame rate
  - Every `interval` seconds it grabs one frame (JPEG) and POSTs it to /api/identify
    together with the local detections.

Backend (this file, Python standard library only — no pip installs needed)
  - Serves the page and static assets
  - POST /api/identify -> forwards the frame to the Jev vision model
    (OpenAI-compatible chat-completions endpoint) and returns Jev's
    identification: confirmed objects, objects the local detector missed,
    and a one-line scene summary.
  - If JEV_API_KEY is not set, the endpoint answers with
    status="not_configured" and echoes the local detections, so the app is
    fully usable for real-time detection out of the box.

Configuration (environment variables)
-------------------------------------
  JEV_API_KEY   Your TypeSafe / Jev API key            (required for Jev layer)
  JEV_API_URL   Base URL of the Jev API                (default: https://api.typesafe.ai)
  JEV_MODEL     Vision-capable Jev model name          (default: jev-vision)
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
JEV_API_URL = os.environ.get("JEV_API_URL", "https://api.typesafe.ai").rstrip("/")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-vision")
JEV_TIMEOUT = 20  # seconds — a slow Jev must never freeze the live UI

IDENTIFY_PROMPT = """You are Jev, a precise vision identification model.
Look at this webcam frame and identify the objects in it.
A lightweight local detector (COCO-SSD) reported these candidate objects: {candidates}.
Verify them against the image: drop false positives, add anything important it missed.

Reply with ONLY a JSON object in exactly this shape, no markdown, no extra text:
{{
  "objects": [
    {{"name": "person", "confidence": 0.95, "note": "sitting at a desk"}}
  ],
  "scene": "one short sentence describing the scene"
}}
Rules: confidence is 0..1. note is at most 5 words. List at most 10 objects.
If the frame is empty/unclear, return an empty objects list and say so in scene."""


def call_jev(image_data_uri: str, detections: list) -> dict:
    """Send one frame to Jev and return its parsed identification."""
    candidates = ", ".join(
        f"{d.get('class', '?')} ({round(float(d.get('score', 0)) * 100)}%)"
        for d in detections
    ) or "none"
    payload = {
        "model": JEV_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IDENTIFY_PROMPT.format(candidates=candidates)},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }
    req = urllib.request.Request(
        f"{JEV_API_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JEV_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=JEV_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"].strip()
    # Models sometimes wrap JSON in code fences — strip them defensively.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    parsed = json.loads(text)
    return {
        "objects": parsed.get("objects", []),
        "scene": parsed.get("scene", ""),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC, **kwargs)

    def log_message(self, fmt, *args):  # keep the console clean
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    # -- helpers ---------------------------------------------------------
    def send_json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        if self.path == "/api/health":
            return self.send_json(200, {
                "ok": True,
                "jev_configured": bool(JEV_API_KEY),
                "jev_model": JEV_MODEL if JEV_API_KEY else None,
            })
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/identify":
            return self.send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 8_000_000:  # ~8 MB guard — frames are ~100-300 KB
                return self.send_json(413, {"error": "frame too large"})
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self.send_json(400, {"error": "bad request body"})

        image = req.get("image", "")
        detections = req.get("detections", [])
        started = time.time()

        if not JEV_API_KEY:
            # Fully usable fallback: identification = local detector output.
            return self.send_json(200, {
                "status": "not_configured",
                "jev_configured": False,
                "objects": [
                    {"name": d.get("class", "object"),
                     "confidence": round(float(d.get("score", 0)), 2),
                     "note": "local detector"}
                    for d in detections
                ],
                "scene": "",
                "latency_ms": int((time.time() - started) * 1000),
                "hint": "Set JEV_API_KEY to enable Jev identification.",
            })

        try:
            result = call_jev(image, detections)
            return self.send_json(200, {
                "status": "ok",
                "jev_configured": True,
                "objects": result["objects"],
                "scene": result["scene"],
                "latency_ms": int((time.time() - started) * 1000),
            })
        except urllib.error.HTTPError as e:
            return self.send_json(200, {
                "status": "error", "jev_configured": True,
                "objects": [], "scene": "",
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"Jev API returned HTTP {e.code}. Check JEV_API_URL / JEV_MODEL / key.",
            })
        except Exception as e:  # timeout, DNS, bad JSON from model, ...
            return self.send_json(200, {
                "status": "error", "jev_configured": True,
                "objects": [], "scene": "",
                "latency_ms": int((time.time() - started) * 1000),
                "error": f"Jev call failed: {type(e).__name__}",
            })


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"JevVision Live -> http://localhost:{PORT}")
    print(f"Jev identification: {'ON (model: ' + JEV_MODEL + ')' if JEV_API_KEY else 'OFF — set JEV_API_KEY to enable (local detection still works)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
