#!/usr/bin/env python3
"""Smoke tests for JevVision Live — run: python tests/smoke_test.py (no network, no camera)."""
import json
import os
import sys
import threading
import urllib.request
import urllib.error

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
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # e.g. 503 when YOLO isn't installed — body is still JSON
        return json.loads(e.read())


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
        "detections": [{"class": "person", "score": 0.9}, {"class": "dog", "score": 0.7}],
        "stats": {"fps": 24},
    })
    assert out["status"] == "not_configured", out
    assert [o["name"] for o in out["objects"]] == ["person", "dog"], out
    assert out["objects"][0]["confidence"] == 0.9, out

    status, body = get("/")
    assert status == 200 and b"YOLO" in body and b"/api/detect" in body, status

    out = post("/api/detect", {"image": "data:image/jpeg;base64,AAAA"})
    assert out.get("status") in ("unavailable", "error", "ok"), out

    httpd.shutdown()
    print("smoke tests: 4/4 passed")


if __name__ == "__main__":
    main()
