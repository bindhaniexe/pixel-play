"""Offline tests for the studio server: routes, hardening, key secrecy."""
from __future__ import annotations

import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import providers as P  # noqa: E402
import server as studio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class StubProvider:
    name = "stub"
    model = "stub-1"

    def __init__(self, explode: bool = False) -> None:
        self._explode = explode

    def generate(self, prompt: str, representation: str, size: int) -> dict:
        if self._explode:
            raise P.ProviderError("the model rejected the request (HTTP 401)")
        cells = [
            [[60.0, 70.0, 90.0, 1.0]]
            for _ in range(size * size)
        ]
        return P.make_field(size, size, cells, {"provider": self.name, "model": self.model})


class StudioServerTests(unittest.TestCase):
    def start(self, provider=StubProvider()) -> None:
        factory = (lambda: provider) if provider is not None else (lambda: None)
        self.server = studio.build_server(0, web_root=ROOT / "web", provider_factory=factory)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, dict(response.getheaders()), payload

    # -- static ------------------------------------------------------------

    def test_serves_the_studio(self) -> None:
        self.start()
        status, headers, body = self.call("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Pixel Play", body)
        self.assertIn("no-store", headers.get("Cache-Control", ""))

    def test_modules_load_as_javascript(self) -> None:
        self.start()
        for name in ("app.js", "paint.js", "paint.worker.js"):
            status, headers, _ = self.call("GET", f"/{name}")
            self.assertEqual(status, 200, name)
            self.assertIn("text/javascript", headers.get("Content-Type", ""), name)

    def test_unknown_files_and_api_paths_404(self) -> None:
        self.start()
        self.assertEqual(self.call("GET", "/missing.js")[0], 404)
        self.assertEqual(self.call("GET", "/api/nope")[0], 404)

    # -- hardening ---------------------------------------------------------

    def test_foreign_origins_are_refused(self) -> None:
        self.start()
        status, _, _ = self.call(
            "GET", "/", headers={"Origin": "https://evil.example"}
        )
        self.assertEqual(status, 403)
        status, _, _ = self.call(
            "POST",
            "/api/paint",
            body=b"{}",
            headers={"Origin": "https://evil.example", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)

    def test_oversized_bodies_are_refused(self) -> None:
        self.start()
        status, _, _ = self.call(
            "POST",
            "/api/paint",
            body=b"x" * (studio.MAX_BODY + 1),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 413)

    def test_bad_input_is_rejected(self) -> None:
        self.start()
        cases = (
            (b"not json", 400),
            (json.dumps({}).encode(), 400),
            (json.dumps({"prompt": "  "}).encode(), 400),
            (json.dumps({"prompt": "hi", "representation": "fresco"}).encode(), 400),
            (json.dumps({"prompt": "hi", "size": 13}).encode(), 400),
            (json.dumps({"prompt": "x" * (studio.MAX_PROMPT + 1)}).encode(), 400),
        )
        for body, expected in cases:
            status, _, _ = self.call(
                "POST", "/api/paint", body=body, headers={"Content-Type": "application/json"}
            )
            self.assertEqual(status, expected, body[:40])

    # -- api ---------------------------------------------------------------

    def test_status_reports_shape_without_secrets(self) -> None:
        self.start(StubProvider())
        status, _, body = self.call("GET", "/api/status")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"configured": True, "provider": "stub", "model": "stub-1"})

    def test_status_when_no_key_is_configured(self) -> None:
        self.start(provider=None)
        status, _, body = self.call("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"configured": False})

    def test_paint_returns_a_field(self) -> None:
        self.start()
        request = json.dumps(
            {"prompt": "a lighthouse", "representation": "hsl", "size": 8}
        ).encode()
        status, _, body = self.call(
            "POST", "/api/paint", body=request, headers={"Content-Type": "application/json"}
        )
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["field"]["cells"]), 64)
        self.assertIn("ms", payload)

    def test_paint_without_key_explains_itself(self) -> None:
        self.start(provider=None)
        request = json.dumps({"prompt": "a lighthouse"}).encode()
        status, _, body = self.call(
            "POST", "/api/paint", body=request, headers={"Content-Type": "application/json"}
        )
        self.assertEqual(status, 503)
        self.assertIn("key", json.loads(body)["error"])

    def test_upstream_failures_never_echo_secrets(self) -> None:
        self.start(StubProvider(explode=True))
        request = json.dumps({"prompt": "a lighthouse"}).encode()
        status, _, body = self.call(
            "POST", "/api/paint", body=request, headers={"Content-Type": "application/json"}
        )
        payload = json.loads(body)
        self.assertEqual(status, 502)
        self.assertIn("401", payload["error"])


if __name__ == "__main__":
    unittest.main()
