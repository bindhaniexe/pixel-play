#!/usr/bin/env python3
"""Pixel Play studio: local static server plus a key-safe paint API.

The browser sends prompts only. Model credentials are read from environment
variables here and are attached to outbound requests in this process; they are
never stored, logged, or echoed back. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

from providers import GRID_SIZES, REPRESENTATIONS, ProviderError, provider_from_env

WEB = Path(__file__).resolve().parent / "web"
MAX_BODY = 16_384
MAX_PROMPT = 2_000

# Windows registry lookups can mislabel .js; modules must load as JavaScript.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

ProviderFactory = Callable[[], Any]


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "PixelPlay/1"
    web_root: Path = WEB
    provider_factory: ProviderFactory = lambda: None

    # -- plumbing ----------------------------------------------------------

    def log_message(self, *_args: Any) -> None:
        """Silence access logs: URLs may contain prompts-in-progress."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; worker-src 'self'; style-src 'self'")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        return CONTENT_TYPES.get(suffix, mimetypes.guess_type(path)[0] or "application/octet-stream")

    def local(self) -> bool:
        """Accept only this machine's own tab on this server's port."""
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        if self.headers.get("Host") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{host}" for host in hosts}

    def reply_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def reply_error(self, status: int, message: str) -> None:
        self.reply_json(status, {"error": message})

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:
        if not self.local():
            return self.send_error(403)
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            return self.status_route()
        if path.startswith("/api/"):
            return self.reply_error(404, "unknown endpoint")
        return self.static_route(path)

    def do_POST(self) -> None:
        if not self.local():
            return self.send_error(403)
        if self.path.split("?", 1)[0] != "/api/paint":
            return self.reply_error(404, "unknown endpoint")
        return self.paint_route()

    # -- handlers ----------------------------------------------------------

    def status_route(self) -> None:
        provider = self.provider_factory()
        if provider is None:
            self.reply_json(200, {"configured": False})
        else:
            self.reply_json(
                200,
                {"configured": True, "provider": provider.name, "model": provider.model},
            )

    def paint_route(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self.reply_error(400, "bad request")
        if not 0 < length <= MAX_BODY:
            return self.reply_error(413, "request too large")
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError):
            return self.reply_error(400, "body must be JSON")

        prompt = request.get("prompt") if isinstance(request, Mapping) else None
        representation = request.get("representation", "palette") if isinstance(request, Mapping) else "palette"
        size = request.get("size", 16) if isinstance(request, Mapping) else 16
        if not isinstance(prompt, str) or not prompt.strip():
            return self.reply_error(400, "a prompt is required")
        prompt = prompt.strip()
        if len(prompt) > MAX_PROMPT:
            return self.reply_error(400, "prompt is too long")
        if representation not in REPRESENTATIONS:
            return self.reply_error(400, "unknown representation")
        if not isinstance(size, int) or size not in GRID_SIZES:
            return self.reply_error(400, "unknown grid size")

        provider = self.provider_factory()
        if provider is None:
            return self.reply_error(
                503,
                "no model key is configured on the server (set JEV_API_KEY or OPENAI_API_KEY)",
            )
        started = time.monotonic()
        try:
            field = provider.generate(prompt, representation, size)
        except ProviderError as error:
            # Provider messages are written here; upstream bodies never pass through.
            return self.reply_error(502, str(error))
        self.reply_json(200, {"field": field, "ms": int((time.monotonic() - started) * 1000)})

    def static_route(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        target = (self.web_root / path.lstrip("/")).resolve()
        try:
            target.relative_to(self.web_root.resolve())
        except ValueError:
            return self.send_error(403)
        if not target.is_file():
            return self.send_error(404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(target)))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(port: int, web_root: Path = WEB, provider_factory: ProviderFactory | None = None) -> ThreadingHTTPServer:
    handler = type(
        "BoundStudioHandler",
        (StudioHandler,),
        {
            "web_root": Path(web_root),
            "provider_factory": staticmethod(provider_factory or (lambda: None)),
        },
    )
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles may be cp1252.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8791")))
    args = parser.parse_args()

    provider = provider_from_env()
    if provider is None:
        print("No model key in the environment; the studio will open without painting.", flush=True)
    else:
        print(f"Painter: {provider.name} ({provider.model})", flush=True)

    with build_server(args.port, provider_factory=lambda: provider) as server:
        print(f"Pixel Play -> http://127.0.0.1:{args.port}   (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
