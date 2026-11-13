"""Standalone Qwen3-ASR streaming inference server.

Runs the official Qwen3-ASR streaming algorithm (vLLM backend) inside an
isolated process/environment and exposes a minimal HTTP contract so the
main mosslive runtime can use it as a local ASR without sharing dependencies
(notably NumPy 2.x and vLLM).

Run as::

    python /app/qwen3_asr_server.py \
        --model /models/main --host 0.0.0.0 --port 8000

HTTP contract (JSON):

- ``POST /v1/session`` with optional ``{"context": str}`` ->
  ``{"session_id": str}``; a non-empty context is fixed for the session
  (kept across reset) and echoed as ``"context_chars": int``. The same body
  accepts optional ``"language": "Chinese"`` (a Qwen language name), echoed
  unchanged in the create response and retained across reset. Omit language
  for automatic detection; invalid language values return HTTP 400.
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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TypedDict


# Canonical names from qwen-asr 0.0.6 inference.utils.SUPPORTED_LANGUAGES.
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
)


@dataclass(frozen=True)
class _CreateSessionRequest:
    context: str = ""
    language: str | None = None
    canonical_language: str | None = None


def _parse_create_session_request(
    payload: object, max_context_chars: int
) -> _CreateSessionRequest:
    """Validate client fields before any model state is allocated."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    context = payload.get("context", "")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    if len(context) > max_context_chars:
        raise ValueError("context too long")
    if "language" not in payload:
        return _CreateSessionRequest(context=context)
    language = payload["language"]
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    # Match qwen-asr's normalize_language_name; ISO codes are not supported.
    stripped = language.strip()
    canonical = stripped[:1].upper() + stripped[1:].lower()
    if canonical not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    return _CreateSessionRequest(context, language, canonical)


class _SessionIdentity(TypedDict):
    session_id: str


class _CreateSessionResponse(_SessionIdentity, total=False):
    context_chars: int
    language: str


class _StreamingSession:
    """One Qwen3-ASR streaming state with its cumulative transcript."""

    def __init__(
        self, state: Any, context: str = "", language: str | None = None
    ) -> None:
        """Wrap backend state and the fixed session context and language."""
        self.state = state
        self.context = context
        self.language = language


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
        self.max_context_chars = args.max_context_chars
        self._sessions: dict[str, _StreamingSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self, context: str = "", language: str | None = None
    ) -> str:
        """Create a session with fixed context and optional forced language.

        Empty context and no language keep the original automatic-detection
        prompt, so callers that send no body see no change.
        """
        session_id = uuid.uuid4().hex
        with self._lock:
            state = self._asr.init_streaming_state(
                context=context, language=language, **self._streaming_kwargs
            )
            self._sessions[session_id] = _StreamingSession(state, context, language)
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
            session = self._sessions[session_id]
            session.state = self._asr.init_streaming_state(
                context=session.context,
                language=session.language,
                **self._streaming_kwargs,
            )

    def drop(self, session_id: str) -> None:
        """Remove one streaming session."""
        with self._lock:
            self._sessions.pop(session_id, None)


def _build_handler(service: _ASRService) -> type[BaseHTTPRequestHandler]:
    """Create the request handler bound to one ASR service."""

    class Handler(BaseHTTPRequestHandler):
        """Minimal JSON-over-HTTP handler for the streaming ASR contract."""

        server_version = "Qwen3ASRStreaming/1.0"

        def _send_json(
            self, status: int, payload: dict[str, Any] | _CreateSessionResponse
        ) -> None:
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
                    try:
                        request = _parse_create_session_request(
                            self._read_json(), service.max_context_chars
                        )
                    except ValueError as exc:
                        self._send_json(400, {"error": str(exc)})
                        return
                    reply: _CreateSessionResponse = {
                        "session_id": service.create_session(
                            request.context, request.canonical_language
                        )
                    }
                    if request.context:
                        # Lets a client tell this server from one that drops context.
                        reply["context_chars"] = len(request.context)
                    if request.language is not None:
                        reply["language"] = request.language
                    self._send_json(200, reply)
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
    # Context shares max_model_len with the audio and the transcript prefix.
    parser.add_argument("--max-context-chars", type=int, default=1000)
    args = parser.parse_args()

    service = _ASRService(args)
    server = ThreadingHTTPServer((args.host, args.port), _build_handler(service))
    print(f"qwen3-asr streaming server on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
