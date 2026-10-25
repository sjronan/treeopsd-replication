# syntax=docker/dockerfile:1.7
#
# TreeOPSD replication image for DGX Station GB300 (aarch64, sm_103), its RTX PRO
# 6000 sidecar (sm_120), and DGX Spark GB10 (sm_121). Build natively on an arm64 host:
#
#   docker build -f env/Dockerfile -t treeopsd-repl:verl0.9.1-vllm0.32.0 .
#
# Base: the official vLLM v0.32.0 multi-arch image, pinned by index digest.
#   index  : sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
#   arm64  : sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
#   created 2026-10-18; VLLM_BUILD_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be;
#   CUDA 13.0.2, Python 3.12, NCCL 2.30.7, TORCH_CUDA_ARCH_LIST "8.0 8.7 8.9 9.0 10.0 11.0 12.0"
# Fallback base (if v0.32.0 breaks verl's vLLM adapter, Phase-0 gate G1):
#   vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063
#   (nightly af7f9488, "0.30.1rc1"; arm64 sha256:c3bede3517c5c982e217c785fd10a8b8ef26e1641c169f5a8634c5b7aa71ca03)
# Second fallback: verl's own docker/Dockerfile.stable.vllm at the same verl commit,
#   built natively on aarch64 (its header pins vllm 0.20.2 for aarch64).
ARG BASE_IMAGE=vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
FROM --platform=linux/arm64 ${BASE_IMAGE}

# verl v0.9.1 (tag commit, 2026-09-20). Change only together with DESIGN.md section 2.
ARG VERL_REPO=https://github.com/verl-project/verl.git
ARG VERL_COMMIT=1876b06d0a3e4e71e06230be10af14492ca8a75b
# flash-attn has no aarch64 wheel; building it is optional (BUILD_FLASH_ATTN=1).
# Without it, run verl with actor.use_remove_padding=False and sdpa attention.
ARG BUILD_FLASH_ATTN=0
ARG FLASH_ATTN_VERSION=2.8.3

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    VLLM_USAGE_SOURCE=treeopsd-replication

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 1) Freeze what the base image ships so nothing below can silently replace
#    torch / vLLM / flashinfer / triton / transformers.
RUN python3 -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|vllm|flashinfer[-_a-z]*|triton|transformers|xformers|numpy|nvidia-[a-z0-9-]+)==' \
        > /etc/treeopsd-base-constraints.txt \
    && cat /etc/treeopsd-base-constraints.txt

# 2) verl at a pinned commit, installed WITHOUT dependency resolution. Founder decision
#    2026-10-21 16:45 PT (deviation D12): keep the base image's transformers 5.19.0 although
#    verl 1876b06 pins ">=5.5.3,!=5.6.0,<5.11"; vLLM 0.32 needs it. The base's own metadata is
#    inconsistent (torch 2.13.0+cu130 pins nccl 2.29.7 / cusparselt 0.9.1, base ships 2.30.7 /
#    0.8.1), so no resolver may re-resolve torch: (2a) packages that do not need torch, resolved
#    by uv under the base pins; (2b) torch-dependent packages with --no-deps plus their
#    non-torch requirements read from their metadata (env/nodeps_reqs.py); (2c) import check. Step 5b proves no base pin moved.
RUN git clone --filter=blob:none ${VERL_REPO} /opt/verl \
    && cd /opt/verl && git checkout ${VERL_COMMIT} \
    && test "$(git rev-parse HEAD)" = "${VERL_COMMIT}" \
    && python3 -m pip install --no-deps -e .
RUN uv pip install --system -c /etc/treeopsd-base-constraints.txt --override /etc/treeopsd-base-constraints.txt \
        codetiming datasets dill hydra-core pandas "pyarrow>=19.0.0" pybind11 pylatexenc \
        "ray[default]>=2.41.0" wandb "packaging>=20.0" tensorboard \
        psutil pyyaml safetensors huggingface_hub tqdm cloudpickle orjson importlib_metadata urllib3 requests
COPY env/nodeps_reqs.py /tmp/nodeps_reqs.py
RUN uv pip install --system --no-deps accelerate peft torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "liger-kernel>=0.8.2" \
    && python3 /tmp/nodeps_reqs.py accelerate peft torchdata tensordict liger-kernel > /etc/treeopsd-nodeps-reqs.txt \
    && cat /etc/treeopsd-nodeps-reqs.txt \
    && uv pip install --system -c /etc/treeopsd-base-constraints.txt --override /etc/treeopsd-base-constraints.txt -r /etc/treeopsd-nodeps-reqs.txt \
    && python3 -c "import accelerate, peft, torchdata, tensordict, liger_kernel, hydra, ray, datasets, wandb; print('2c imports ok', accelerate.__version__, peft.__version__, tensordict.__version__)"

# 3) Retriever + harness deps (exact flat search is torch.mm on GPU; faiss-cpu only to
#    reconstruct vectors from e5_Flat.index and to cross-check top-k ids).
RUN uv pip install --system -c /etc/treeopsd-base-constraints.txt --override /etc/treeopsd-base-constraints.txt \
        "faiss-cpu==1.12.0" "fastapi>=0.115" "uvicorn>=0.30" httpx requests aiohttp pytest

# 4) Optional flash-attn source build (long; arch list covers sm_100 PTX for sm_103).
RUN if [ "${BUILD_FLASH_ATTN}" = "1" ]; then \
        MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX" \
        python3 -m pip install --no-build-isolation -c /etc/treeopsd-base-constraints.txt \
            "flash-attn==${FLASH_ATTN_VERSION}"; \
    fi

# 5) This repo (reference math + verl extension). Patches to verl, if any, live in
#    env/patches/ and are applied to the pinned commit; the build fails if one does not apply.
COPY . /workspace/treeopsd
RUN set -e; for p in /workspace/treeopsd/env/patches/*.patch; do \
        [ -e "$p" ] || continue; git -C /opt/verl apply --check "$p" && git -C /opt/verl apply "$p"; \
    done
ENV PYTHONPATH=/workspace/treeopsd

# 5b) The base pins must be untouched after every install above.
RUN python3 -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|vllm|flashinfer[-_a-z]*|triton|transformers|xformers|numpy|nvidia-[a-z0-9-]+)==' \
        | diff /etc/treeopsd-base-constraints.txt - && echo "base pins unchanged"

# 6) Build-time self-checks (CPU only): math tests, imports, recorded versions.
RUN cd /workspace/treeopsd && python3 -m pytest -q env/smoke \
    && python3 -c "import verl, vllm, torch, transformers; print('verl', verl.__version__, 'vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)" \
    && python3 -m pip freeze > /etc/treeopsd-freeze.txt

WORKDIR /workspace/treeopsd
ENTRYPOINT []
CMD ["bash"]
