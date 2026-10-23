"""Standalone Qwen3-ASR streaming inference server.

Runs the official Qwen3-ASR streaming algorithm (vLLM backend) inside an
isolated process/environment and exposes a minimal HTTP contract so the
main mosslive runtime can use it as a local ASR without sharing dependencies
(notably NumPy 2.x and vLLM).

Run as::

    python /app/qwen3_asr_server.py \
        --model /models/main --host 0.0.0.0 --port 8000

HTTP contract (JSON):

- ``POST /v1/session`` -> ``{"session_id": str}``
- ``POST /v1/session/<sid>/stream`` with ``{"audio_b64": str,
  "is_final": bool}`` -> ``{"text": str, "language": str | None}``.
  ``audio_b64`` is base64 PCM s16le mono 16 kHz; ``text`` is the
  cumulative transcript for the session.
- ``POST /v1/session/<sid>/reset`` -> ``{}``
- ``DELETE /v1/session/<sid>`` -> ``{}``
- ``GET /healthz`` -> ``{"status": "ok"}``
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _StreamingSession:
    """One Qwen3-ASR streaming state with its cumulative transcript."""

    def __init__(self, state: Any) -> None:
        """Wrap one backend-owned streaming state object."""
        self.state = state


class _ASRService:
    """Serialize all model calls behind one lock; own session states."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load the vLLM-backed Qwen3-ASR model and build streaming params."""
        from qwen_asr import Qwen3ASRModel

        self._asr = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
        )
        self._streaming_kwargs = {
            "unfixed_chunk_num": args.unfixed_chunk_num,
            "unfixed_token_num": args.unfixed_token_num,
            "chunk_size_sec": args.chunk_size_sec,
        }
        self._sessions: dict[str, _StreamingSession] = {}
        self._lock = threading.Lock()

    def create_session(self) -> str:
        """Create one streaming session and return its identity."""
        session_id = uuid.uuid4().hex
        with self._lock:
            state = self._asr.init_streaming_state(**self._streaming_kwargs)
            self._sessions[session_id] = _StreamingSession(state)
        return session_id

    def stream(self, session_id: str, pcm: bytes, is_final: bool) -> dict[str, Any]:
        """Feed one PCM chunk; return the cumulative transcript.

        Parameters
        ----------
        session_id:
            Existing streaming session.
        pcm:
            PCM s16le mono 16 kHz bytes, may be empty.
        is_final:
            Flush hint: the backend finishes the buffered tail audio while
            keeping the accumulated transcript for continuation.

        Returns
        -------
        dict
            ``text`` (cumulative transcript) and ``language`` when known.
        """
        import numpy as np

        with self._lock:
            session = self._sessions[session_id]
            state = session.state
            if pcm:
                samples = (
                    np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                )
                self._asr.streaming_transcribe(samples, state)
            if is_final:
                self._asr.finish_streaming_transcribe(state)
            return {
                "text": state.text or "",
                "language": getattr(state, "language", None),
            }

    def reset(self, session_id: str) -> None:
        """Replace one session's streaming state with a fresh one."""
        with self._lock:
            state = self._asr.init_streaming_state(**self._streaming_kwargs)
            self._sessions[session_id].state = state

    def drop(self, session_id: str) -> None:
        """Remove one streaming session."""
        with self._lock:
            self._sessions.pop(session_id, None)


def _build_handler(service: _ASRService) -> type[BaseHTTPRequestHandler]:
    """Create the request handler bound to one ASR service."""

    class Handler(BaseHTTPRequestHandler):
        """Minimal JSON-over-HTTP handler for the streaming ASR contract."""

        server_version = "Qwen3ASRStreaming/1.0"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            """Silence per-request stderr logging."""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            """Serve the health probe."""
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming
            """Serve session creation, streaming, and reset."""
            try:
                if self.path == "/v1/session":
                    self._send_json(200, {"session_id": service.create_session()})
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[:2] == ["v1", "session"]:
                    session_id, action = parts[2], None
                elif len(parts) == 4 and parts[:2] == ["v1", "session"]:
                    session_id, action = parts[2], parts[3]
                else:
                    self._send_json(404, {"error": "not found"})
                    return
                if action == "stream":
                    payload = self._read_json()
                    pcm = base64.b64decode(payload.get("audio_b64") or b"")
                    result = service.stream(
                        session_id,
                        pcm,
                        bool(payload.get("is_final")),
                    )
                    self._send_json(200, result)
                    return
                if action == "reset":
                    service.reset(session_id)
                    self._send_json(200, {})
                    return
                self._send_json(404, {"error": "not found"})
            except KeyError:
                self._send_json(404, {"error": "unknown session"})
            except Exception as exc:  # keep the server alive on model errors
                self._send_json(500, {"error": str(exc)})

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler naming
            """Drop one streaming session."""
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["v1", "session"]:
                service.drop(parts[2])
                self._send_json(200, {})
                return
            self._send_json(404, {"error": "not found"})

    return Handler


def main() -> None:
    """Parse arguments and serve the streaming ASR HTTP endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3-ASR path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    # Half a second is what a conversation needs: the segmenter confirms text
    # only across two successive partials, so the chunk is paid twice before a
    # clause can be cut, and proactive turn-taking waits on the same clock.
    parser.add_argument("--chunk-size-sec", type=float, default=0.5)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    args = parser.parse_args()

    service = _ASRService(args)
    server = ThreadingHTTPServer((args.host, args.port), _build_handler(service))
    print(f"qwen3-asr streaming server on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
