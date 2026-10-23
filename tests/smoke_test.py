#!/usr/bin/env python3
"""Smoke tests for JevVision Live — run: python tests/smoke_test.py (no network, no camera)."""
import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.pop("JEV_API_KEY", None)  # force fallback mode for the test

import app as server


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:8137{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:8137{path}", timeout=5) as r:
        return r.status, r.read()


def main():
    server.PORT = 8137
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 8137), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    status, body = get("/api/health")
    health = json.loads(body)
    assert status == 200 and health["ok"] and health["jev_configured"] is False, health

    out = post("/api/identify", {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "detections": [{"class": "person", "score": 0.9}, {"class": "dog", "score": 0.7}],
    })
    assert out["status"] == "not_configured", out
    assert [o["name"] for o in out["objects"]] == ["person", "dog"], out
    assert out["objects"][0]["confidence"] == 0.9, out

    status, body = get("/")
    assert status == 200 and b"JEV" in body and b"coco-ssd" in body.lower() or b"cocoSsd" in body, status

    httpd.shutdown()
    print("smoke tests: 3/3 passed")


if __name__ == "__main__":
    main()
