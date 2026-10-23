# MOSS-Live Qwen3-ASR streaming engine

This branch starts at upstream vLLM `v0.14.0`, commit
`b17039bccc17b94a2ea107272ad7bc93508708df`.
It adds the standalone ASR HTTP wrapper and an image recipe. The Dockerfile
reuses the official vLLM 0.14.0 binary image; changes to vLLM core source in
this fork are **not** compiled by this recipe.

## Files and provenance

- `mosslive_services/qwen3_asr_server.py`: the existing MOSS-Live standalone
  service, imported from the supplied source bundle with only the invocation
  example updated; runtime behavior is unchanged.
  Source SHA-256: `6f04d80e251538c2d42c8d1a5e27d76efe83b0798d8bd85fb0db46a331fdcdc5`.
  The bundle did not provide a verifiable original service Git commit.
  The new fork commit records this imported file, not its earlier history.
- `requirements/mosslive-asr.txt`: direct dependencies, including
  `qwen-asr[vllm]==0.0.6` and `qwen-omni-utils==0.0.9`.
- `docker/mosslive-qwen3-asr/Dockerfile`: official v0.14.0 base image,
  pinned by image-index digest. The recipe constrains installed engine/GPU
  packages during dependency resolution and checks package consistency.

The official image index digest is
`sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd`.
Its Linux amd64 manifest is
`sha256:48a03c91eaa04fb7e71c77121a85e661d62de1b2207edf591aea6a05d779f9ef`.
Direct dependencies are pinned. This is not a complete transitive lockfile;
capture the resolved environment and final image digest during an authorized
build before treating it as a reproducible release.

## Build configuration (not executed)

| Field | Value |
| --- | --- |
| Repository | `https://github.com/Xtaiyang/vllm.git` |
| Branch | `mosslive-qwen3-asr-v0.14.0` |
| Source revision | Use the full commit containing all files above |
| Build context | `.` (repository root) |
| Dockerfile | `docker/mosslive-qwen3-asr/Dockerfile` |
| Target architecture | `linux/amd64` for the intended NVIDIA GPU |
| Service port | `8000` |
| Readiness / liveness path | `/healthz` |

Use a builder that executes the specified repository Dockerfile. The generic
vLLM source-build template uses its own build recipe and does not automatically
execute this Dockerfile. This commit alone does not add platform UI support
for selecting a custom Dockerfile.

No image build, package installation in a container, GPU inference, or platform
deployment has been performed for this addition. Python syntax, CLI parsing,
HTTP contract wiring with a fake backend, and Dockerfile source paths can be
checked locally without loading model weights. Image dependency resolution,
GPU/driver compatibility, and recognition quality remain to be validated.

## Runtime configuration

Mount a complete local `Qwen/Qwen3-ASR-1.7B` snapshot at `/models/main`, or set
`MODEL_PATH` to the mounted model directory. Model weights are not copied into
the image. Runtime Hugging Face downloads are disabled by default.
The previously selected weight revision is
`7278e1e70fe206f11671096ffdd38061171dd6e5`; verify the mounted asset separately.

The image's default command is equivalent to:

```sh
exec python3 -u /app/qwen3_asr_server.py \
  --model "${MODEL_PATH:-/models/main}" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 4096 --max-new-tokens 64 \
  --chunk-size-sec 0.5 --unfixed-chunk-num 2 --unfixed-token-num 5
```

`ASR_PORT`, `ASR_GPU_MEMORY_UTILIZATION`, `ASR_MAX_MODEL_LEN`,
`ASR_MAX_NEW_TOKENS`, `ASR_CHUNK_SIZE_SEC`, `ASR_UNFIXED_CHUNK_NUM`, and
`ASR_UNFIXED_TOKEN_NUM` override the default command's parameters.
The reference allocation is one full 96 GB GPU, 4 CPU cores, 24 GiB container
memory, and 4 GiB shared memory. It is a starting configuration, not a measured
minimum. The 0.15 GPU utilization fraction applies to visible GPU memory;
recalculate it if using a GPU slice.

## HTTP contract

The server loads the model before binding its HTTP port.

| Endpoint | Request / result |
| --- | --- |
| `GET /healthz` | `{"status":"ok"}` after initialization |
| `POST /v1/session` | Returns `{"session_id":"..."}` |
| `POST /v1/session/{sid}/stream` | JSON `{"audio_b64":"...","is_final":false}`; returns cumulative `text` and `language` |
| `POST /v1/session/{sid}/reset` | Resets the session; returns `{}` |
| `DELETE /v1/session/{sid}` | Releases the session; returns `{}` |

Audio is base64-encoded signed 16-bit little-endian PCM, mono, 16 kHz, without
a WAV header. `is_final:true` flushes buffered tail audio. This is the
MOSS-Live session API, not the OpenAI transcription API or SSE token streaming.
Model calls are serialized by a global lock; clients must delete finished
sessions. This import preserves the existing handler and session behavior.
