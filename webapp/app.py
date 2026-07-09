"""Minimal web app for phone-level pronunciation diagnosis."""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_minimal_phone_demo import run_demo  # noqa: E402


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, object], status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return
    content = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def diagnose(audio_path: Path, text: str, utterance_id: str, speaker_id: str) -> list[dict[str, object]]:
    args = argparse.Namespace(
        audio=audio_path,
        text=text,
        output=ROOT / "reports" / "web_demo_latest.csv",
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        native_language="Mandarin",
        speaker_age=0.0,
        duration_ms=None,
        threshold=0.77,
        oov_phone="SPN",
        lexicon=ROOT / "speechocean762" / "resource" / "lexicon.txt",
        phones=ROOT / "phones.csv",
        model=ROOT / "models" / "combined" / "acceptance_error_extra_trees.joblib",
        threshold_predictions=ROOT / "reports" / "combined_acceptance_error_fusion_predictions.csv",
    )
    return run_demo(args)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "PronunciationDemo/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            file_response(self, STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/").replace("/", "\\")
            file_response(self, STATIC_DIR / rel)
            return
        if parsed.path == "/api/prompts":
            json_response(
                self,
                {
                    "prompts": [
                        {"label": "ability", "text": "ability"},
                        {"label": "very well", "text": "very well"},
                        {"label": "three red leaves", "text": "three red leaves"},
                        {"label": "she sees the blue bird", "text": "she sees the blue bird"},
                    ]
                },
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/diagnose":
            self.send_error(404)
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            json_response(self, {"error": "Expected multipart/form-data."}, status=400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        text = str(form.getfirst("text", "")).strip()
        utterance_id = str(form.getfirst("utterance_id", "web_demo")).strip() or "web_demo"
        speaker_id = str(form.getfirst("speaker_id", "web_user")).strip() or "web_user"
        audio_item = form["audio"] if "audio" in form else None

        if not text:
            json_response(self, {"error": "Text is required."}, status=400)
            return
        if audio_item is None or not getattr(audio_item, "file", None):
            json_response(self, {"error": "WAV audio is required."}, status=400)
            return

        with tempfile.NamedTemporaryFile(prefix="pron_demo_", suffix=".wav", delete=False) as tmp:
            tmp.write(audio_item.file.read())
            tmp_path = Path(tmp.name)

        try:
            rows = diagnose(tmp_path, text, utterance_id, speaker_id)
        except Exception as exc:  # Keep the demo server responsive with a clear API error.
            json_response(self, {"error": str(exc)}, status=500)
            return
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

        correct = sum(1 for row in rows if int(row["prediction"]) == 1)
        json_response(
            self,
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "text": text,
                "n_phones": len(rows),
                "n_acceptable": correct,
                "n_needs_review": len(rows) - correct,
                "rows": rows,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Pronunciation demo running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
