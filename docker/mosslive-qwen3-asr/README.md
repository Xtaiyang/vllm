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
- `requirements/mosslive-asr-resolved.txt`: the 39 added or changed Python
  distributions recorded after successful temporary-container validation on
  2026-10-23. The Dockerfile uses this snapshot as additional pip constraints.
- `docker/mosslive-qwen3-asr/Dockerfile`: official v0.14.0 base image,
  pinned by image-index digest. The recipe constrains installed engine/GPU
  packages during dependency resolution and checks package consistency.

The official image index digest is
`sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd`.
Its Linux amd64 manifest is
`sha256:48a03c91eaa04fb7e71c77121a85e661d62de1b2207edf591aea6a05d779f9ef`.
The pinned base plus direct requirements, the resolved ASR delta snapshot,
and dynamically captured engine/GPU constraints describe the tested package
combination. The delta snapshot is not a standalone, cross-platform lockfile.
Capture the complete final package inventory and image digest during the
first authorized image build; GPU execution must be tested separately.

NumPy `2.2.6` and SoXR `1.1.0` are explicitly pinned to the recorded R75 ASR
environment. The `ASR_BASE_IMAGE` build argument can select a verified mirror
without changing the default official base image. For Linux amd64, the audited
mirror reference is:

```text
docker.m.daocloud.io/vllm/vllm-openai@sha256:48a03c91eaa04fb7e71c77121a85e661d62de1b2207edf591aea6a05d779f9ef
```

Pass that reference as `--build-arg ASR_BASE_IMAGE=...` when building on the
company server. The reference selects only Linux amd64; a different target
architecture requires its matching manifest or the official image index.
Use the build argument for an equivalent pinned base image, not an arbitrary
vLLM/PyTorch version; Dockerfile version checks and ABI constraints still apply.

## Base-image audit (2026-10-23)

The Linux amd64 mirror manifest above was pulled on the company server and
inspected in temporary containers with networking disabled, a read-only root
filesystem, and no GPU, published port, or model mount. This initial audit did not install
packages; a later disposable-container ASR install is recorded below.
Package metadata and `torch/version.py` report:

| Component | Actual base-image version | R75 reference |
| --- | --- | --- |
| Python | 3.12.12 | 3.12.13 |
| vLLM | 0.14.0 | 0.14.0 |
| PyTorch | 2.9.1+cu129 | 2.9.1+cu128 |
| torchaudio / torchvision | 2.9.1+cu129 / 0.24.1+cu129 | 2.9.1 / 0.24.1; CUDA suffixes not independently recorded |
| Triton | 3.5.1 | 3.5.1 |
| flashinfer-python | 0.5.3 | 0.5.3 |
| Transformers / Accelerate | 4.57.6 / 1.12.0 | 4.57.6 / 1.12.0 |
| NumPy | 2.2.6 | 2.2.6 |
| nvidia-cublas-cu12 | 12.9.1.4 | 12.8.4.1 |
| qwen-asr / qwen-omni-utils / SoXR | Not installed | 0.0.6 / 0.0.9 / 1.1.0; pinned for the ASR build |

The system CUDA toolkit environment is `12.9.1`; PyTorch's own build reports
CUDA `12.9`. This base is not an exact copy of the R75 conda environment.
Keep the engine/GPU packages together; replacing only PyTorch with cu128 can
break vLLM's compiled extensions. GPU execution remains untested.

## Complete ASR dependency validation (2026-10-23)

A fresh disposable container from the audited Linux amd64 cu129 manifest
successfully installed all 15 direct dependency pins. Validation used source
`8352b4b4d5d64978bb19484ccbbbe4ffcff8103c` plus the distro Blinker cleanup now
included in this Dockerfile. The 39 added or changed package versions were
captured in `requirements/mosslive-asr-resolved.txt` and are now also applied
as constraints; the initial successful installation produced this snapshot.
The Dockerfile's new index argument and snapshot wiring have been checked
statically, without a formal image build.

Two inherited system-package issues are handled before pip installation:

- Unused `python3-gi` exposes PyGObject metadata requiring absent pycairo,
  causing the original base's `pip check` to fail. Remove it and the unused
  software-properties helpers.
- Distro `python3-blinker` supplies Blinker 1.4 via distutils, which pip cannot
  uninstall when Flask 3.1.2 needs a newer version. Remove the distro package
  first; pip then installs the validated Blinker 1.9.0.

APT also removes the reverse dependencies `python3-oauthlib`,
`python3-lazr.restfulclient`, and `python3-launchpadlib`. Combined with the GI
helpers and Blinker, exactly seven APT packages were removed in validation.
`--no-auto-remove` preserves unrelated packages. Strict `pip check` runs after
cleanup and again after the ASR install; both passed.

Validation results:

- All 15 direct pins matched, including NumPy 2.2.6 and SoXR 1.1.0.
- All 26 protected engine/GPU packages retained their versions and distribution
  RECORD hashes. vLLM remained 0.14.0; torch and torchaudio remained
  2.9.1+cu129; torchvision remained 0.24.1+cu129.
- PyTorch/audio/tokenizer imports, Qwen ASR streaming API imports, the wrapper
  module, and wrapper CLI `--help` all returned exit code 0.
- The vLLM backend was imported explicitly, beyond Qwen ASR's top-level import
  which can suppress backend import failures; that check also passed.

The company server has no NVIDIA GPU or driver, so a missing `libcuda.so.1`
warning is expected. No model loading or GPU inference was tested. Python
`sox==1.5.0` imports with a warning that the system SoX executable is absent;
SoX CLI processing was not tested. No SoX references were found in the
installed qwen_asr or qwen_omni_utils Python sources, and this wrapper's input
is already raw PCM mono 16 kHz. The R75 Python patch version and CUDA flavor
still differ from this official cu129 base as shown above.

The disposable containers were removed after validation. The base image,
host packages, and existing services were not modified. No formal ASR image
was built, saved, pushed, imported, or deployed.

## Build-time Python package index

`ASR_PIP_INDEX_URL` defaults to `https://pypi.org/simple`. Its value is passed
to pip as the process environment variable `PIP_INDEX_URL`, including pip's
isolated build subprocesses. It is a build argument, not a persistent runtime
setting. Runtime Hugging Face offline settings do not prevent build-time pip
network access. Keep TLS certificate verification enabled.

The validated install used `https://mirrors.aliyun.com/pypi/simple`. On the
company server, public PyPI and the public mirror route were very slow. A
container-only mapping of `mirrors.aliyun.com` to `100.100.2.158` used the
reachable Aliyun internal route while retaining the valid HTTPS hostname and
certificate. This address was obtained from `mirrors.cloud.aliyuncs.com` on
that network; re-check its resolution and HTTPS reachability before a later
build. Do not hard-code the mapping into this Dockerfile or change host DNS.
Do not use the internal DNS name directly for HTTPS if its certificate does
not match. The public mirror needs no host mapping where it is already fast.

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

The following company-server command is an example only; it has not been
executed. Check out the full reviewed branch commit in a clean directory
first, then run from the repository root. Use the internal host mapping only
after confirming the network route as described above.

```sh
docker buildx build --platform linux/amd64 --load \
  --add-host mirrors.aliyun.com:100.100.2.158 \
  --build-arg ASR_BASE_IMAGE=docker.m.daocloud.io/vllm/vllm-openai@sha256:48a03c91eaa04fb7e71c77121a85e661d62de1b2207edf591aea6a05d779f9ef \
  --build-arg ASR_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  -f docker/mosslive-qwen3-asr/Dockerfile \
  -t "mosslive-qwen3-asr-vllm:0.14.0-$(git rev-parse --short=12 HEAD)" .
```

The dependency validation clears the known Python installation blockers and
makes a formal image build the next step. It does not replace that build or
GPU verification. After an authorized build, check the image's package
inventory, `pip check`, imports, and default entrypoint; record its image ID
and digest before an authorized registry push/import. Registry namespace and
push credentials still need to be supplied or verified. Model loading,
GPU/driver compatibility, streaming recognition quality, and platform routing
remain to be tested on the target GPU.

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
