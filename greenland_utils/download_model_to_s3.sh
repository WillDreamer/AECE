#!/usr/bin/env bash
#
# Download a Hugging Face model and upload it to S3.
#
# Usage:
#   ./download_model_to_s3.sh <HF_REPO_ID> [S3_PREFIX]
#
# Examples:
#   ./download_model_to_s3.sh Qwen/Qwen2.5-7B-Instruct
#   ./download_model_to_s3.sh Qwen/Qwen2.5-7B-Instruct s3://whx-agent/Qwen3.5/
#
set -euo pipefail


# ---- Config -------------------------------------------------------------
MODEL_REPO="${1:-Qwen/Qwen3.5-35B-A3B-Base}"          # <-- exact HF repo id
S3_DEST="${2:-s3://whx-agent/Qwen3.5/Qwen3.5-35B-A3B-Base/}"  # target S3 prefix (trailing slash recommended)
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-greenland-dev}"    # AWS profile used for S3 access

# Local scratch dir (on /local which has the most space on a dev-dsk).
WORKDIR="${WORKDIR:-/local/home/whx/hf-downloads}"
LOCAL_DIR="${WORKDIR}/$(basename "${MODEL_REPO}")"

# Optional: set HF_TOKEN env var for gated/private repos.
export HF_HUB_ENABLE_HF_TRANSFER=1               # faster parallel downloads

# ---- Dependencies -------------------------------------------------------
# Python 3.7 on this box -> pin versions that still support it.
echo ">> Ensuring huggingface_hub + hf_transfer are installed..."
python3 -m pip install --user --quiet --upgrade \
    "huggingface_hub==0.16.4" "hf_transfer==0.1.4" 2>/dev/null || \
python3 -m pip install --user --quiet "huggingface_hub" "hf_transfer"

# ---- Download -----------------------------------------------------------
mkdir -p "${LOCAL_DIR}"
echo ">> Downloading ${MODEL_REPO} -> ${LOCAL_DIR}"
python3 - "$MODEL_REPO" "$LOCAL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, local_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo,
    local_dir=local_dir,
    local_dir_use_symlinks=False,   # store real files so S3 sync uploads content
    resume_download=True,
    # ignore_patterns=["*.gguf", "original/*"],  # uncomment to skip extra formats
)
print("Downloaded to:", path)
PY

# ---- Upload to S3 -------------------------------------------------------
echo ">> Uploading to ${S3_DEST} (profile: ${AWS_PROFILE_NAME})"
aws s3 sync "${LOCAL_DIR}/" "${S3_DEST}" --no-progress --profile "${AWS_PROFILE_NAME}"

# ---- Verify & clean -----------------------------------------------------
echo ">> Done. Contents in S3:"
aws s3 ls "${S3_DEST}" --profile "${AWS_PROFILE_NAME}"

# Free local disk after a successful upload.
echo ">> Removing local copy ${LOCAL_DIR}"
rm -rf "${LOCAL_DIR}"
