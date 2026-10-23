# Build context: repository root.
# Upstream source base: v0.14.0 / b17039bccc17b94a2ea107272ad7bc93508708df.
# Reuse the released engine; this Dockerfile does not compile the fork's vLLM.
# The default pins the official multi-platform v0.14.0 image index.
# Override ASR_BASE_IMAGE to use a verified mirror of the same image digest.
ARG ASR_BASE_IMAGE=vllm/vllm-openai:v0.14.0@sha256:1d6866b87630d94f5e0cdae55ab5abb4ce0b03fcb84d9d10612f9d518d19d4fd
FROM ${ASR_BASE_IMAGE}

# Build-only package index; use HTTPS and keep certificate verification enabled.
ARG ASR_PIP_INDEX_URL=https://pypi.org/simple

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Remove unused distro GI and distutils Blinker before ASR pip installation.
# Keep unrelated packages; pip supplies the newer Blinker required by Flask.
RUN DEBIAN_FRONTEND=noninteractive apt-get remove --yes --no-auto-remove \
        python3-gi python3-software-properties software-properties-common python3-blinker \
    && python3 -m pip check

WORKDIR /app
COPY requirements/mosslive-asr.txt /tmp/mosslive-asr.txt
COPY requirements/mosslive-asr-resolved.txt /tmp/mosslive-asr-resolved.txt

# Prevent pip from replacing the engine or its installed GPU ABI packages.
# Pin the added/changed packages to the validated ASR dependency snapshot.
RUN python3 -c 'from importlib.metadata import distributions; names={"vllm", "torch", "torchaudio", "torchvision", "triton", "xformers", "vllm-flash-attn"}; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in distributions() if d.metadata["Name"].lower().replace("_", "-") in names or d.metadata["Name"].lower().replace("_", "-").startswith(("nvidia-", "flashinfer-")))))' > /tmp/mosslive-base-constraints.txt \
    && PIP_INDEX_URL="${ASR_PIP_INDEX_URL}" python3 -m pip install --constraint /tmp/mosslive-base-constraints.txt --constraint /tmp/mosslive-asr-resolved.txt --requirement /tmp/mosslive-asr.txt \
    && python3 -m pip check \
    && python3 -c 'import sys; from importlib.metadata import version; from packaging.version import Version; assert sys.version_info[:2] == (3, 12), sys.version; assert Version(version("vllm")).base_version == "0.14.0"; assert Version(version("torch")).base_version == "2.9.1"; assert version("qwen-asr") == "0.0.6"'

COPY mosslive_services/qwen3_asr_server.py /app/qwen3_asr_server.py

ENV ASR_PORT=8000 \
    ASR_GPU_MEMORY_UTILIZATION=0.15 \
    ASR_MAX_MODEL_LEN=4096 \
    ASR_MAX_NEW_TOKENS=64 \
    ASR_CHUNK_SIZE_SEC=0.5 \
    ASR_UNFIXED_CHUNK_NUM=2 \
    ASR_UNFIXED_TOKEN_NUM=5

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python3 -c 'import os, urllib.request; urllib.request.urlopen("http://127.0.0.1:" + os.environ.get("ASR_PORT", "8000") + "/healthz", timeout=3).read()'

# Clear the official image's OpenAI-server entrypoint for the ASR HTTP service.
ENTRYPOINT []
CMD ["sh", "-c", "exec python3 -u /app/qwen3_asr_server.py --model \"${MODEL_PATH:-/models/main}\" --host 0.0.0.0 --port \"${ASR_PORT:-8000}\" --gpu-memory-utilization \"$ASR_GPU_MEMORY_UTILIZATION\" --max-model-len \"$ASR_MAX_MODEL_LEN\" --max-new-tokens \"$ASR_MAX_NEW_TOKENS\" --chunk-size-sec \"$ASR_CHUNK_SIZE_SEC\" --unfixed-chunk-num \"$ASR_UNFIXED_CHUNK_NUM\" --unfixed-token-num \"$ASR_UNFIXED_TOKEN_NUM\""]
