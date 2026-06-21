#!/bin/bash
# build_efa_image_mns.sh — build & push the EFA-enabled slime-greenland image.
# ─────────────────────────────────────────────────────────────────────────────
# Builds Dockerfile.slime-greenland-efa_mns:
#   FROM guangrli:slime-greenland (acct 241)  +  AWS EFA userspace + aws-ofi-nccl
# then pushes to YOUR ECR repo on acct 339, so multi-node NCCL uses EFA/RDMA
# instead of falling back to TCP sockets (~20x slower). See memory
# greenland-slime-image-efa-missing.
#
# CREDENTIAL SPLIT (discovered 2026-06-18):
#   * PULL base from 241 -> the per-job `greenland` profile (assumes the SDB's
#     821 greenland-access-* role) CAN read 241/guangrli. The `greenland-dev`
#     (339) role CANNOT (241 repo policy denies it). NOTE the 821 role is
#     EPHEMERAL — it dies with the SDB job; re-run reconnect-sdb.sh to refresh.
#   * PUSH to 339 -> the `greenland-dev` profile.
#
# Usage:
#   bash build_efa_image_mns.sh            # build + push
#   bash build_efa_image_mns.sh --preflight # only check creds/perms, no build
#   DEST_TAG=v1 bash build_efa_image_mns.sh # override pushed tag
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config (override via env) ──
REGION="${REGION:-ap-south-1}"
PULL_PROFILE="${PULL_PROFILE:-greenland}"          # 821 per-job role: can pull 241
PUSH_PROFILE="${PUSH_PROFILE:-greenland-dev}"      # 339: can push to your repo

SRC_ACCOUNT="${SRC_ACCOUNT:-241893993881}"
BASE_IMAGE="${BASE_IMAGE:-${SRC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/guangrli:slime-greenland}"

DEST_ACCOUNT="${DEST_ACCOUNT:-339712697413}"
DEST_REPO="${DEST_REPO:-whx/slime-greenland-efa}"
DEST_TAG="${DEST_TAG:-latest}"
DEST_REGISTRY="${DEST_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
DEST_IMAGE="${DEST_REGISTRY}/${DEST_REPO}:${DEST_TAG}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DOCKERFILE="${DOCKERFILE:-${SCRIPT_DIR}/Dockerfile.slime-greenland-efa_mns}"

# Build args (forwarded to the Dockerfile ARGs). The aws-ofi-nccl plugin is no
# longer compiled from source — the EFA installer ships it prebuilt — so only the
# installer version is parameterized.
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-latest}"

log() { echo "[build $(date -u +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── Preflight: fail fast on the things that actually break this build ──
preflight() {
  log "preflight: docker daemon"
  docker ps >/dev/null 2>&1 || die "docker daemon not accessible"

  log "preflight: Dockerfile exists -> ${DOCKERFILE}"
  [ -f "${DOCKERFILE}" ] || die "Dockerfile not found: ${DOCKERFILE}"

  log "preflight: push identity (${PUSH_PROFILE} -> acct ${DEST_ACCOUNT})"
  local push_acct
  push_acct="$(aws sts get-caller-identity --profile "${PUSH_PROFILE}" --query Account --output text 2>/dev/null)" \
    || die "cannot get identity for profile '${PUSH_PROFILE}' (mwinit?)"
  [ "${push_acct}" = "${DEST_ACCOUNT}" ] \
    || log "  WARN: ${PUSH_PROFILE} is acct ${push_acct}, expected ${DEST_ACCOUNT}"

  log "preflight: can ${PULL_PROFILE} pull base from ${SRC_ACCOUNT}? (manifest inspect, no download)"
  aws ecr get-login-password --region "${REGION}" --profile "${PULL_PROFILE}" 2>/dev/null \
    | docker login --username AWS --password-stdin "${SRC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null 2>&1 \
    || die "ECR login to ${SRC_ACCOUNT} failed for profile '${PULL_PROFILE}'"
  docker manifest inspect "${BASE_IMAGE}" >/dev/null 2>&1 \
    || die "profile '${PULL_PROFILE}' cannot pull ${BASE_IMAGE} (241 repo policy denies it). The 821 per-job role is ephemeral — run reconnect-sdb.sh, or get 241 to grant pull."
  log "  OK base image is pullable"
  log "preflight: PASSED"
}

# ── Ensure the destination repo exists (idempotent) ──
ensure_dest_repo() {
  log "ensuring dest repo ${DEST_REPO} on ${DEST_ACCOUNT}"
  aws ecr describe-repositories --repository-names "${DEST_REPO}" \
      --region "${REGION}" --profile "${PUSH_PROFILE}" >/dev/null 2>&1 \
    || { log "  creating ${DEST_REPO}"; \
         aws ecr create-repository --repository-name "${DEST_REPO}" \
             --region "${REGION}" --profile "${PUSH_PROFILE}" >/dev/null \
           || die "could not create repo ${DEST_REPO}"; }
}

main() {
  log "config:"
  log "  BASE  (pull via ${PULL_PROFILE}): ${BASE_IMAGE}"
  log "  DEST  (push via ${PUSH_PROFILE}): ${DEST_IMAGE}"
  log "  EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}"

  preflight
  [ "${1:-}" = "--preflight" ] && { log "--preflight only; stopping."; exit 0; }

  ensure_dest_repo

  # PULL base explicitly (login already done in preflight; layers download now).
  log "pulling base image (multi-GB, first time is slow)..."
  docker pull "${BASE_IMAGE}"

  # BUILD the derived image.
  log "building ${DEST_IMAGE} ..."
  DOCKER_BUILDKIT=1 docker build \
    -f "${DOCKERFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "EFA_INSTALLER_VERSION=${EFA_INSTALLER_VERSION}" \
    -t "${DEST_IMAGE}" \
    "${SCRIPT_DIR}"

  # PUSH to 339 (separate login — different registry/creds than pull).
  log "logging in to dest registry ${DEST_REGISTRY}"
  aws ecr get-login-password --region "${REGION}" --profile "${PUSH_PROFILE}" \
    | docker login --username AWS --password-stdin "${DEST_REGISTRY}" >/dev/null

  log "pushing ${DEST_IMAGE} ..."
  docker push "${DEST_IMAGE}"

  log "DONE. Run a 2-node job with:"
  log "  python greenland_cli_mns.py obx \\"
  log "    --script examples/math_reasoning/run_qwen35_4b_base_mns.sh \\"
  log "    --num-nodes 2 --image ${DEST_IMAGE}"
}

main "$@"
