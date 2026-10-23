#!/usr/bin/env python3
"""Local web UI for FASHN VTON v1.5.

    python web/app.py

Open the printed URL, upload a person photo and a garment photo, and generate
a try-on image. Uses ./weights by default.
"""

from __future__ import annotations

import argparse
import base64
import os
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fashn-mpl"))

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
INDEX_PATH = WEB_DIR / "index.html"
SAMPLE_DIR = ROOT / "examples" / "data"
SAMPLES = {
    "/samples/model.webp": SAMPLE_DIR / "model.webp",
    "/samples/garment.webp": SAMPLE_DIR / "garment.webp",
}

MAX_IMAGE_BYTES = 20 * 1024 * 1024
CATEGORIES = ("tops", "bottoms", "one-pieces")
PHOTO_TYPES = ("model", "flat-lay")

_state_lock = threading.Lock()
_infer_lock = threading.Lock()
_state = {
    "pipeline": None,
    "loading": True,
    "error": None,
    "device": None,
    "busy": False,
    "weights_dir": None,
}


def _load_pipeline(weights_dir: Path, device: Optional[str]) -> None:
    from fashn_vton import TryOnPipeline

    try:
        pipeline = TryOnPipeline(weights_dir=str(weights_dir), device=device)
        with _state_lock:
            _state["pipeline"] = pipeline
            _state["device"] = str(pipeline.device)
            _state["error"] = None
    except Exception as exc:
        traceback.print_exc()
        with _state_lock:
            _state["error"] = str(exc)
    finally:
        with _state_lock:
            _state["loading"] = False
        print("Model ready." if _state["error"] is None else f"Model failed to load: {_state['error']}", flush=True)


def _open_rgb(data: bytes, label: str) -> Image.Image:
    if not data:
        raise ValueError(f"Choose a {label} image.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"The {label} image is larger than 20 MB.")
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read the {label} image.") from exc
    return image.convert("RGB")


def _int_field(value: str, name: str, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number.") from exc
    if number < low or number > high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return number


def _float_field(value: str, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if number < low or number > high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return number


def _generate(fields: dict) -> dict:
    with _state_lock:
        pipeline = _state["pipeline"]
        loading = _state["loading"]
        error = _state["error"]
    if loading:
        raise RuntimeError("The model is still loading. Wait a moment and try again.")
    if error:
        raise RuntimeError(error)
    if pipeline is None:
        raise RuntimeError("The model is not loaded.")

    person = _open_rgb(fields["person"], "person")
    garment = _open_rgb(fields["garment"], "garment")
    category = fields["category"]
    photo_type = fields["garment_photo_type"]
    if category not in CATEGORIES:
        raise ValueError("Category must be tops, bottoms, or one-pieces.")
    if photo_type not in PHOTO_TYPES:
        raise ValueError("Garment photo type must be model or flat-lay.")

    num_samples = _int_field(fields["num_samples"], "Samples", 1, 4)
    num_timesteps = _int_field(fields["num_timesteps"], "Steps", 1, 50)
    seed = _int_field(fields["seed"], "Seed", 0, 2**31 - 1)
    guidance_scale = _float_field(fields["guidance_scale"], "Guidance", 1.0, 5.0)
    segmentation_free = fields["segmentation_free"] == "true"

    if not _infer_lock.acquire(blocking=False):
        raise RuntimeError("A generation is already running.")
    with _state_lock:
        _state["busy"] = True
    try:
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )
        encoded = []
        for image in result.images:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
        return {"images": encoded}
    finally:
        with _state_lock:
            _state["busy"] = False
        _infer_lock.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "FashnTryOn/1.5"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self._send_json(200, self._status())
            return
        sample = SAMPLES.get(path)
        if sample is not None and sample.is_file():
            self._send(200, sample.read_bytes(), "image/webp")
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/generate":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            fields = self._read_form()
            payload = _generate(fields)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"Generation failed: {exc}"})
            return
        self._send_json(200, payload)

    def _status(self) -> dict:
        with _state_lock:
            return {
                "loading": _state["loading"],
                "ready": _state["pipeline"] is not None,
                "error": _state["error"],
                "device": _state["device"],
                "busy": _state["busy"],
                "weights_dir": _state["weights_dir"],
            }

    def _read_form(self) -> dict:
        from cgi import FieldStorage

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Missing form data.")
        if length > MAX_IMAGE_BYTES * 2 + 1024 * 1024:
            raise ValueError("Upload is too large.")
        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(length),
            },
        )

        def text(name: str, default: str = "") -> str:
            if name not in form:
                return default
            value = form[name].value
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            return str(value)

        def blob(name: str) -> bytes:
            if name not in form or not getattr(form[name], "filename", ""):
                return b""
            item = form[name]
            data = item.file.read() if getattr(item, "file", None) else item.value
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data

        return {
            "person": blob("person"),
            "garment": blob("garment"),
            "category": text("category", "tops"),
            "garment_photo_type": text("garment_photo_type", "model"),
            "num_samples": text("num_samples", "1"),
            "num_timesteps": text("num_timesteps", "30"),
            "guidance_scale": text("guidance_scale", "1.5"),
            "seed": text("seed", "42"),
            "segmentation_free": text("segmentation_free", "true"),
        }

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        import json

        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description="FASHN VTON v1.5 web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--weights-dir", default=str(ROOT / "weights"))
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).resolve()
    with _state_lock:
        _state["weights_dir"] = str(weights_dir)

    threading.Thread(target=_load_pipeline, args=(weights_dir, args.device), daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}", flush=True)
    print(f"Weights: {weights_dir}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
