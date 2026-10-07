#!/usr/bin/env bash
# =============================================================================
# Aeon_fall_detection - environment initialisation.
#
# Creates the conda environment "Aeon_fall_detection", clones RelateAnything,
# installs every dependency and (optionally) exports the ONNX/OpenVINO models.
#
# Usage:
#   bash setup.sh                 # auto-detect GPU (CUDA build) or CPU build
#   TARGET=cpu bash setup.sh      # force CPU wheels (Intel i7-1260P edge box)
#   TARGET=gpu bash setup.sh      # force CUDA 12.9 wheels (RTX 4090/5090)
#   EXPORT_MODELS=1 bash setup.sh # also run export_onnx.py --openvino
#
# Python version: RelateAnything declares `requires-python >= 3.12` (it pins
# numpy 2.5, which has no 3.10 wheels). A 3.10 environment therefore cannot
# install it, so the default here is 3.12. PYTHON_VERSION may be overridden,
# but values below 3.12 are rejected up-front instead of failing deep inside
# pip with an unreadable resolver error.
# =============================================================================
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-Aeon_fall_detection}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TARGET="${TARGET:-auto}"
EXPORT_MODELS="${EXPORT_MODELS:-0}"
RA_REPO="https://github.com/Maelic/RelateAnything.git"
# torch 2.13 is the oldest release accepted by RelateAnything (~=2.13) and
# the newest one published for CUDA 12.9. CUDA 12.8+ is required for
# Blackwell (RTX 5090, sm_120); 12.9 also covers Ada (RTX 4090, sm_89).
TORCH_VERSION="2.13.0"
TORCHVISION_VERSION="0.28.0"
CUDA_INDEX="https://download.pytorch.org/whl/cu129"
CPU_INDEX="https://download.pytorch.org/whl/cpu"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
# Report the failing line: with `set -e` alone a failure deep in a long
# script is silent about *where* it happened.
trap 'die "command failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

# ---------------------------------------------------------------- checks ----
command -v git >/dev/null 2>&1 || die "git is required."
command -v conda >/dev/null 2>&1 || die "conda is required (Miniforge/Miniconda)."

py_minor="${PYTHON_VERSION#*.}"
if [[ "${PYTHON_VERSION%%.*}" -ne 3 || "${py_minor%%.*}" -lt 12 ]]; then
    die "PYTHON_VERSION=${PYTHON_VERSION}: RelateAnything requires Python >= 3.12."
fi

if [[ "${TARGET}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        TARGET="gpu"
    else
        TARGET="cpu"
    fi
fi
[[ "${TARGET}" == "gpu" || "${TARGET}" == "cpu" ]] || die "TARGET must be auto|gpu|cpu."
log "target=${TARGET} env=${ENV_NAME} python=${PYTHON_VERSION}"

# `conda activate` needs the shell hook in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ------------------------------------------------------------ conda env ----
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "conda env '${ENV_NAME}' already exists - reusing it."
else
    log "creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------- RelateAnything ----
if [[ -d RelateAnything/.git ]]; then
    log "RelateAnything already cloned - pulling latest changes."
    git -C RelateAnything pull --ff-only || warn "git pull failed; keeping current checkout."
else
    log "cloning ${RA_REPO}"
    git clone https://github.com/Maelic/RelateAnything.git
fi

# ---------------------------------------------------------------- torch ----
# PyTorch is installed FIRST from the explicit index: installing relsgg first
# would let pip pull the default (CUDA 13) wheel from PyPI, which needs a
# newer driver than many RTX 4090 hosts run and is wasted space on CPU boxes.
if [[ "${TARGET}" == "gpu" ]]; then
    log "installing torch ${TORCH_VERSION} (CUDA 12.9)"
    python -m pip install "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" --index-url "${CUDA_INDEX}"
else
    log "installing torch ${TORCH_VERSION} (CPU only)"
    python -m pip install "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" --index-url "${CPU_INDEX}"
fi

# ------------------------------------------------------- dependencies ----
log "installing RelateAnything (editable, with hub + deploy extras)"
python -m pip install -e "./RelateAnything[hub,deploy]"

log "installing application dependencies"
# The CPU onnxruntime wheel is used on both targets: on GPU servers the
# relation model runs in PyTorch, and onnxruntime-gpu cannot coexist with the
# onnxruntime wheel that the RelateAnything [deploy] extra already installs
# (both ship the same `onnxruntime` module).
python -m pip install -r requirements.txt
# YOLO-World encodes class prompts with CLIP; ultralytics would otherwise
# try to auto-install it on first use, which fails on offline edge devices.
python -m pip install "git+https://github.com/ultralytics/CLIP.git"

# relsgg pins opencv-python-headless while ultralytics pulls opencv-python;
# both provide the `cv2` module and the last one installed wins. Reinstalling
# the GUI build guarantees cv2.imshow works for local debugging.
python -m pip install --force-reinstall --no-deps "opencv-python>=4.10"

# ------------------------------------------------------------ verify ----
log "verifying imports"
python - <<'PY'
import importlib
import sys

modules = ["torch", "torchvision", "relsgg", "ultralytics", "gradio", "onnx",
           "onnxruntime", "openvino", "cv2", "docx", "numpy"]
failed = []
for name in modules:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:12s} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {exc}")
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
if failed:
    print("FAILED:\n  " + "\n  ".join(failed))
    sys.exit(1)
PY

# ------------------------------------------------------ optional export ----
if [[ "${EXPORT_MODELS}" == "1" ]]; then
    log "exporting RelateAnything to ONNX + OpenVINO"
    python export_onnx.py --openvino
fi

log "done. Next steps:"
cat <<EOF
  conda activate ${ENV_NAME}
  python export_onnx.py --openvino          # once, for ONNX_CPU mode
  python app.py --mode onnx_cpu             # edge (i7-1260P)
  python app.py --mode torch_gpu            # GPU server
  python generate_docx_report.py            # Vietnamese architecture .docx
EOF
