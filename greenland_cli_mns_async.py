#!/usr/bin/env python3
"""
Greenland OBX job manager for the slime ASYNC RL training workflow (user: whx).

ASYNC variant of greenland_cli_mns.py. Identical bootstrap / staging / multinode
machinery, with two differences:
  * --script defaults to the async tau-bench run script
    (examples/tau-bench/run_qwen35_4b_tau_mns_async.sh).
  * --rollout-nodes is REQUIRED to be > 0. Async training (train_async.py) asserts
    `not args.colocate`, so colocate (single-pool) topology is invalid. This also
    is the whole point: disaggregated + train_async overlaps rollout and train so
    BOTH GPU pools stay busy, avoiding the Greenland GPU-idle (power<82W on P5EN)
    stuck-job watchdog that killed the sync disaggregated runs.

A submit does three things, fully automated (OBX has no SSH):
  1. Uploads the local slime checkout -> S3, so the container runs *this* code.
  2. Submits a P5EN batch job whose container bootstrap:
       a. assumes greenland-dev-role to reach s3://whx-agent,
       b. replaces /root/slime with the uploaded code,
       c. downloads model + data from S3 to local NVMe,
       d. points the run script at the local copies via env (ROOT_DIR/MODEL_ROOT/
          DATA_ROOT/SLIME_DIR/MEGATRON_DIR/RAY_TEMP_DIR),
       e. background-syncs checkpoints local -> S3 (and once more on exit),
       f. launches the run script.
  3. Prints the console URL.

Usage:
    # Submit the Qwen3.5-4B tau-bench ASYNC RL run (5 nodes = 1 train + 4 rollout)
    python greenland_cli_mns_async.py obx --num-nodes 5 --rollout-nodes 4 \\
        --stage-model Qwen3.5/Qwen3.5-4B-Base/ \\
        --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ --stage-data tau-bench/

    # List active jobs / inspect / terminate
    python greenland_cli_mns_async.py status
    python greenland_cli_mns_async.py describe <job-name-or-id>
    python greenland_cli_mns_async.py terminate <job-name-or-id>
"""

import base64
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml

# ── Defaults ──
GREENLAND_API = "https://prod.us-east-1.greenland.alloy.amazon.dev/v2"
CONFIG_PATH = Path(__file__).parent / "greenland_config.yaml"

DEFAULTS = {
    # Greenland / batch
    "region": "ap-south-1",
    # EFA-enabled derived image (base guangrli:slime-greenland + aws-ofi-nccl);
    # built/pushed via build_efa_image_mns.sh so multi-node NCCL uses EFA/RDMA
    # instead of falling back to TCP sockets. Override with --image if needed.
    # PINNED BY DIGEST, not :latest — a job submitted right after `docker push`
    # can pull a stale :latest (Greenland/ECS resolves the tag at schedule time);
    # one live run wasted that way. After any rebuild, repoint to the new digest.
    # 392e9980 = symlink + ABSOLUTE-path NCCL_NET_PLUGIN + LD_LIBRARY_PATH prepends
    # /opt/amazon/efa/lib (AWS libfabric w/ FABRIC_1.8 — the system one only has
    # 1.6, which made the plugin load then fail "FABRIC_1.8 not found" -> Socket).
    # dlopen-verified at build time. Prior broken digests: 80266b6b/7d6bb615/d0086f9b.
    "image": "339712697413.dkr.ecr.ap-south-1.amazonaws.com/whx/slime-greenland-efa@sha256:392e9980a2d4a3960d45262d4f1dde7eb481c5ae793e145f2b827003aa6e1e3c",
    "initiative_id": "RufusInSearch",
    "instance_type": "p5en.48xlarge",
    "role": "arn:aws:iam::339712697413:role/greenland-dev-role",
    "instance_count": 1,
    "job_prefix": "whx-",
    # Code staging (local -> S3 -> container)
    "local_slime_dir": "/home/whx/AECE/slime",
    "code_s3": "s3://whx-agent/code/slime",
    "container_slime_dir": "/root/slime",
    "container_megatron_dir": "/root/Megatron-LM",
    "source_profile": "greenland-dev",          # local profile that can write s3://whx-agent
    # Data staging (S3 <-> container NVMe)
    "model_s3": "s3://whx-agent",                # real MODEL_ROOT
    "data_s3": "s3://whx-agent/data",            # real DATA_ROOT
    "nvme_root": "/tmp/instance_storage/whx",    # mounted NVMe volume in container
    "output_prefix": "AECE/",                    # subtree of MODEL_ROOT synced back to S3
    # Inside the container, ECS task creds assume this to reach acct 339712697413.
    "greenland_dev_role": "arn:aws:iam::339712697413:role/greenland-dev-role",
    # Default S3 subpaths to stage for run_qwen35_4b_base.sh (relative to model_s3/data_s3).
    "stage_model": [
        "Qwen3.5/Qwen3.5-4B-Base/",
        "Qwen3.5/Qwen3.5-4B-Base_torch_dist/",
    ],
    "stage_data": [
        "dapo-math-17k/",
    ],
}


def load_config():
    for path in [CONFIG_PATH, Path.home() / ".greenland_config.yaml"]:
        if path.exists():
            try:
                with open(path) as f:
                    return {**DEFAULTS, **(yaml.safe_load(f) or {})}
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}", flush=True)
    return dict(DEFAULTS)


import shutil

_MIDWAY_COOKIE = str(Path.home() / ".midway" / "cookie")


def mcurl(url, data=None, method="POST"):
    """Midway-authenticated HTTP. Prefer `mcurl` if installed; otherwise fall
    back to plain `curl` with the Midway cookie jar (works wherever `mwinit`
    has produced ~/.midway/cookie, e.g. boxes without mcurl on PATH)."""
    if shutil.which("mcurl"):
        args = ["mcurl", "-s", "-X", method, "-H", "Content-Type: application/json", url]
    else:
        args = ["curl", "-s", "-L", "-b", _MIDWAY_COOKIE, "-c", _MIDWAY_COOKIE,
                "-X", method, "-H", "Content-Type: application/json", url]
    if data:
        args.extend(["-d", json.dumps(data)])
    result = subprocess.run(args, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"stdout": result.stdout, "stderr": result.stderr}


def ensure_auth():
    # Without mcurl we rely on the Midway cookie jar; if it's missing, mwinit.
    if not shutil.which("mcurl") and not Path(_MIDWAY_COOKIE).exists():
        print("No Midway cookie found, running mwinit...", flush=True)
        subprocess.run(["mwinit"])
        return
    resp = mcurl("https://midway-auth.amazon.com/api/session-status", method="GET")
    if isinstance(resp, dict) and not resp.get("authenticated", False):
        print("Not authenticated, running mwinit...", flush=True)
        subprocess.run(["mwinit"])


def _fmt_ts(raw_ts):
    if not raw_ts:
        return "N/A"
    if isinstance(raw_ts, (int, float)):
        if raw_ts > 1e12:
            raw_ts = raw_ts / 1000
        return datetime.fromtimestamp(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
    return str(raw_ts)[:19]


def _list_all_jobs(cfg, limited_fields=True, max_pages=50):
    """Paginate through all jobs (API returns ~100 per page, sorted oldest-first)."""
    all_jobs, token = [], None
    for _ in range(max_pages):
        params = {
            "InitiativeId": cfg["initiative_id"],
            "region": cfg["region"],
            "maxResults": 100,
            "queryWithLimitedFields": limited_fields,
        }
        if token:
            params["nextToken"] = token
        resp = mcurl(f"{GREENLAND_API}/listjobs", params)
        if not isinstance(resp, dict):
            break
        all_jobs.extend(resp.get("JobList", []))
        if not resp.get("HasMoreJobs"):
            break
        token = resp.get("NextToken")
    return all_jobs


# ═══════════════════════════════════════════════════════════════
# Bootstrap script (runs inside the container as the batch command)
# ═══════════════════════════════════════════════════════════════
# GPU keep-busy used DURING S3 staging. The Greenland stuck-job detector is a GPU
# electrical-IDLE watchdog (power < ~82W on P5EN); while the bootstrap is
# `aws s3 sync`-ing the model + (now ~26GB) data to NVMe, the GPUs sit at 0W and
# the detector can SIGKILL the job before training ever starts (job 63b76437 died
# at the workspace-bench data-staging step this way, exitCode=137). This tiny
# loop runs a small matmul on every visible GPU to hold power above the threshold
# during staging only; it is killed the instant staging finishes so it never
# competes with SGLang/Megatron for memory. Deliberately minimal + fully guarded:
# if torch is missing or there is no GPU it exits cleanly (staging proceeds either
# way). Small tensors (1024^2) + a sleep keep utilization low but power non-idle.
_GPU_KEEPBUSY_PY = r"""
import os, time
try:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(0)
    n = torch.cuda.device_count()
    streams = []
    mats = []
    for i in range(n):
        torch.cuda.set_device(i)
        a = torch.randn(1024, 1024, device=f"cuda:{i}")
        b = torch.randn(1024, 1024, device=f"cuda:{i}")
        mats.append((a, b))
    # Loop until the parent kills us (staging done). Touch every GPU each pass.
    while True:
        for i in range(n):
            torch.cuda.set_device(i)
            a, b = mats[i]
            c = a @ b
            _ = c.sum().item()  # force sync so the kernel actually runs
        time.sleep(0.2)
except SystemExit:
    pass
except Exception:
    # Never let keep-busy failure affect staging.
    pass
"""


def _build_bootstrap(cfg, script_rel, wandb_key, hf_token, stage_model, stage_data, extra_env,
                     rollout_nodes=0, user_sim_nodes=0):
    """Render the in-container bootstrap bash. Passed base64-encoded as the
    batch command so no pre-existing entrypoint file is required in the image.

    rollout_nodes > 0 selects DISAGGREGATED mode: the last `rollout_nodes` nodes
    serve rollout (SGLang) and the remaining nodes train. The bootstrap just
    computes ROLLOUT_NUM_GPUS / ACTOR_NUM_NODES and hands them to the run script;
    all nodes still join one Ray cluster (slime's placement group decides which
    physical GPUs become actor vs rollout). rollout_nodes == 0 keeps colocate.

    user_sim_nodes > 0 reserves that many of the rollout nodes to serve a SEPARATE
    user-simulation model; the bootstrap exports USER_SIM_NUM_GPUS (= user_sim_nodes
    * 8) and the run script splits the rollout pool into actor vs user_sim via a
    multi-model --sglang-config. user_sim GPUs are still part of ROLLOUT_NUM_GPUS
    (one Ray placement group); the split is purely a run-script / slime concern."""

    def _norm(p):
        return p.strip("/")

    stage_model_cmds = []
    for p in stage_model:
        p = _norm(p)
        stage_model_cmds.append(
            f'log "staging model {p}/"\n'
            f'mkdir -p "$MODEL_ROOT/{p}"\n'
            f'$AWS s3 sync "{cfg["model_s3"]}/{p}/" "$MODEL_ROOT/{p}/" --only-show-errors'
        )
    stage_data_cmds = []
    for p in stage_data:
        p = _norm(p)
        stage_data_cmds.append(
            f'log "staging data {p}/"\n'
            f'mkdir -p "$DATA_ROOT/{p}"\n'
            f'$AWS s3 sync "{cfg["data_s3"]}/{p}/" "$DATA_ROOT/{p}/" --only-show-errors'
        )

    # base64 the GPU keep-busy script so it embeds in the bootstrap without any
    # bash/f-string quoting hazards (decoded + run in the container during staging).
    gpu_keepbusy_b64 = base64.b64encode(_GPU_KEEPBUSY_PY.encode("utf-8")).decode("ascii")

    extra_env_lines = "\n".join(f'export {k}="{v}"' for k, v in extra_env.items())
    # Only export HF creds when present, so an unset token doesn't blank out
    # whatever the image already has configured.
    hf_lines = ""
    if hf_token:
        hf_lines = (f'export HF_TOKEN="{hf_token}"\n'
                    f'export HUGGING_FACE_HUB_TOKEN="{hf_token}"')
    out_prefix = _norm(cfg["output_prefix"]) + "/"

    return f"""#!/bin/bash
set -euo pipefail
log() {{ echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }}

# ── 1. Credentials for s3://whx-agent ──
# The container's native ECS task role (821 account) cannot touch whx-agent (339),
# but it CAN assume this job's declared CustomerRoleArn (greenland-dev-role @ 339),
# which has S3 access. So: base creds = ECS container creds, then assume the dev role.
# (Verified on an SDB with the same image: read+write to whx-agent works this way.)
# The ECS creds endpoint is exposed via AWS_CONTAINER_CREDENTIALS_RELATIVE_URI, which
# the entrypoint inherits from PID 1; re-export it from PID 1 in case it's missing.
if [ -z "${{AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-}}" ]; then
  export AWS_CONTAINER_CREDENTIALS_RELATIVE_URI="$(tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^AWS_CONTAINER_CREDENTIALS_RELATIVE_URI=//p' | head -1)"
fi
mkdir -p /root/.aws
cat > /root/.aws/config <<'AWSCFG'
[default]
region = {cfg["region"]}
credential_source = EcsContainer
role_arn = {cfg["greenland_dev_role"]}
AWSCFG
export AWS_CONFIG_FILE=/root/.aws/config

if ! command -v aws >/dev/null 2>&1; then
  log "aws CLI not found; installing via pip"
  pip install -q awscli || python -m pip install -q awscli
fi
AWS="aws"
log "RELATIVE_URI=${{AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-MISSING}}"
log "identity: $($AWS sts get-caller-identity --query Arn --output text 2>/dev/null || echo unknown)"

# ── 2. Replace {cfg["container_slime_dir"]} with the uploaded code ──
log "syncing code {cfg["code_s3"]} -> {cfg["container_slime_dir"]}"
mkdir -p {cfg["container_slime_dir"]}
$AWS s3 sync "{cfg["code_s3"]}/" "{cfg["container_slime_dir"]}/" --delete --only-show-errors
( cd {cfg["container_slime_dir"]} && pip install -e . --no-deps -q 2>/dev/null || true )

# ── 3. Download model + data from S3 to local NVMe ──
export ROOT_DIR={cfg["nvme_root"]}
export MODEL_ROOT={cfg["nvme_root"]}/model_root
export DATA_ROOT={cfg["nvme_root"]}/data_root
mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$ROOT_DIR/ray_temp"

# ── 3a. GPU keep-busy DURING staging (defeats the GPU-idle stuck-job detector) ──
# S3 staging (esp. the ~26GB workspace-bench data) leaves the GPUs at 0W; the
# Greenland watchdog (power < ~82W on P5EN) can SIGKILL the job before training
# starts (job 63b76437 died this way at the data-staging step). Run a tiny matmul
# loop on every GPU to hold power above the threshold, then kill it the instant
# staging completes so it never competes with SGLang/Megatron for memory. Fully
# guarded: no torch / no GPU -> it exits cleanly and staging proceeds regardless.
KEEPBUSY_PID=""
if command -v python3 >/dev/null 2>&1; then
  printf '%s' "{gpu_keepbusy_b64}" | base64 -d > /tmp/gpu_keepbusy.py 2>/dev/null || true
  ( python3 /tmp/gpu_keepbusy.py >/dev/null 2>&1 ) &
  KEEPBUSY_PID=$!
  log "started GPU keep-busy (pid $KEEPBUSY_PID) to avoid the idle-GPU stuck-job detector during staging"
fi
stop_keepbusy() {{
  if [ -n "${{KEEPBUSY_PID:-}}" ]; then
    kill "$KEEPBUSY_PID" 2>/dev/null || true
    wait "$KEEPBUSY_PID" 2>/dev/null || true
    log "stopped GPU keep-busy (staging complete)"
    KEEPBUSY_PID=""
  fi
}}
# Safety net: ensure keep-busy is reaped even if staging aborts early.
trap stop_keepbusy EXIT

{chr(10).join(stage_model_cmds)}
{chr(10).join(stage_data_cmds)}

# Staging done -> free the GPUs for training/inference.
stop_keepbusy

# ── 4. Background checkpoint sync (local outputs -> S3) ──
sync_outputs() {{ $AWS s3 sync "$MODEL_ROOT/{out_prefix}" "{cfg["model_s3"]}/{out_prefix}" --only-show-errors || true; }}
( while true; do sleep 300; sync_outputs; done ) &
SYNC_PID=$!
# This EXIT trap REPLACES the staging-time keep-busy trap (bash keeps only the
# last EXIT trap). It also calls stop_keepbusy as a final backstop — harmless
# no-op once staging already stopped it (KEEPBUSY_PID was cleared).
trap 'log "final checkpoint sync"; stop_keepbusy; sync_outputs; kill $SYNC_PID 2>/dev/null || true' EXIT

# ── 5. Env for the run script (overrides its dev-box defaults) ──
# Skip the script's dev-box LD_LIBRARY_PATH prepend (it breaks cuDNN in this image).
export SKIP_SYS_LDPATH=1
export SLIME_DIR={cfg["container_slime_dir"]}
export MEGATRON_DIR={cfg["container_megatron_dir"]}
export RAY_TEMP_DIR=$ROOT_DIR/ray_temp
# Checkpoint retention (--save-retain-interval): so the in-cluster prune
# in train.py can mirror its local deletion to S3, tell it the local->S3 root map.
export CKPT_LOCAL_MODEL_ROOT="$MODEL_ROOT"
export CKPT_S3_MODEL_ROOT="{cfg["model_s3"]}"
export WANDB_API_KEY="{wandb_key}"
{hf_lines}
{extra_env_lines}

# ── 5b. Multi-node Ray topology (AWS Batch multinode) ──
# The SAME bootstrap runs on every node. AWS Batch injects these on each node;
# on a single-node job they're absent -> NODE_INDEX=0, NUM_NODES=1, and this
# whole block is a no-op so single-node behaviour is byte-for-byte unchanged.
#   AWS_BATCH_JOB_NODE_INDEX                    this node's rank (0 = main)
#   AWS_BATCH_JOB_MAIN_NODE_INDEX               which rank is the main node
#   AWS_BATCH_JOB_NUM_NODES                     total nodes in the job
#   AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS  head IP, visible to all nodes
NODE_INDEX="${{AWS_BATCH_JOB_NODE_INDEX:-0}}"
MAIN_INDEX="${{AWS_BATCH_JOB_MAIN_NODE_INDEX:-0}}"
NUM_NODES="${{AWS_BATCH_JOB_NUM_NODES:-1}}"
if [ "$NUM_NODES" -gt 1 ]; then
  # Routable IP of THIS node: take the eth0 address, NOT `hostname -I | awk '{{print $1}}'`
  # (that can return the ECS link-local 169.254.x.x, which the other node can't reach).
  SELF_IP="$(ip -4 -o addr show eth0 2>/dev/null | awk '{{print $4}}' | cut -d/ -f1 | head -1)"
  [ -z "$SELF_IP" ] && SELF_IP="$(hostname -I | tr ' ' '\\n' | grep -vE '^(127\\.|169\\.254\\.)' | head -1)"
  # Head IP visible to all nodes. On WORKER nodes AWS Batch sets
  # AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS; on the MAIN node it is empty
  # (known Batch quirk), so the main node's head IP is just its own SELF_IP.
  if [ "$NODE_INDEX" = "$MAIN_INDEX" ]; then
    MAIN_IP="$SELF_IP"
  else
    MAIN_IP="${{AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-127.0.0.1}}"
  fi
  GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"; GPUS_PER_NODE="${{GPUS_PER_NODE:-8}}"
  # Pin NCCL/Gloo to the routable NIC. The ECS metadata link-local
  # (169.254.x.x / ecs-eth0) is NOT reachable from the other node, so without
  # this NCCL may pick it and cross-node collectives hang.
  export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-eth0}}"
  export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-eth0}}"
  log "multinode: node $NODE_INDEX/$NUM_NODES main_index=$MAIN_INDEX main_ip=$MAIN_IP self_ip=$SELF_IP gpus=$GPUS_PER_NODE"

  if [ "$NODE_INDEX" != "$MAIN_INDEX" ]; then
    # Worker node: join the head's Ray cluster and block. It does NOT run
    # train.py — the head drives the whole job; its actors get scheduled onto
    # these GPUs via Ray. When the main node exits, Greenland stops this child.
    #
    # HARDENING (cross-subnet / boot-skew visibility): a live 2-node run failed
    # because `ray start --address ... --block` printed "Ray runtime started"
    # even though the worker never actually reached the head's GCS (port 6379)
    # across a subnet boundary (head 10.2.199.x vs worker 10.2.3.x). The head
    # then sat at 8/16 GPUs and was killed by the stuck-job detector. So before
    # blocking we (1) log both subnets so a mismatch is obvious, (2) TCP-probe
    # head:6379 with retries (the head may still be booting; a cross-subnet
    # block shows as repeated failures), (3) start ray WITHOUT --block and
    # VERIFY this node actually joined the head's cluster, then (4) block until
    # the head's GCS disappears (= head exited -> Greenland stops this child).
    SELF_SUBNET_24="${{SELF_IP%.*}}"; HEAD_SUBNET_24="${{MAIN_IP%.*}}"
    log "worker $NODE_INDEX: self=$SELF_IP (/.24=$SELF_SUBNET_24) head=$MAIN_IP (/.24=$HEAD_SUBNET_24)"
    if [ "$SELF_SUBNET_24" != "$HEAD_SUBNET_24" ]; then
      log "worker $NODE_INDEX: WARNING — self and head are on DIFFERENT /24 subnets; if the Ray control-plane port 6379 is not open across subnets the join WILL fail (this was the cause of a prior 2-node failure)."
    fi
    # (2) Probe head GCS port 6379 until reachable (head boot + reachability).
    GCS_OK=0
    for attempt in $(seq 1 120); do
      if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); rc=s.connect_ex(('${{MAIN_IP}}',6379)); s.close(); sys.exit(0 if rc==0 else 1)" 2>/dev/null; then
        log "worker $NODE_INDEX: head GCS ${{MAIN_IP}}:6379 REACHABLE on attempt $attempt"
        GCS_OK=1; break
      fi
      [ $(( attempt % 6 )) -eq 1 ] && log "worker $NODE_INDEX: head ${{MAIN_IP}}:6379 not reachable yet (attempt $attempt/120) — head still booting or cross-subnet block; retrying"
      sleep 5
    done
    if [ "$GCS_OK" != "1" ]; then
      log "worker $NODE_INDEX: FATAL — head GCS ${{MAIN_IP}}:6379 unreachable after 600s. self=$SELF_IP head=$MAIN_IP. Almost certainly a CROSS-SUBNET control-plane block (port 6379). This node cannot join; the head will time out. Exiting so the failure is explicit."
      exit 1
    fi
    # (3) Join WITHOUT --block, then verify membership against the HEAD's GCS.
    ray stop --force 2>/dev/null || true
    ray start --address="${{MAIN_IP}}:6379" --num-gpus "${{GPUS_PER_NODE}}" \
      --node-ip-address "${{SELF_IP}}" --disable-usage-stats \
      --temp-dir "${{RAY_TEMP_DIR}}"
    sleep 8
    JOIN_GPUS=$(python3 -c "import ray; ray.init(address='${{MAIN_IP}}:6379', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU',0))); ray.shutdown()" 2>/dev/null || echo 0)
    if [ "${{JOIN_GPUS:-0}}" -ge "$(( 2 * GPUS_PER_NODE ))" ]; then
      log "worker $NODE_INDEX: JOIN VERIFIED — head cluster now reports ${{JOIN_GPUS}} GPUs (head+worker visible)"
    else
      log "worker $NODE_INDEX: WARNING — after ray start the head cluster reports only ${{JOIN_GPUS}} GPUs (expected >= $(( 2 * GPUS_PER_NODE ))). Possible split-brain (this node formed/own local cluster) or other workers not in yet."
    fi
    # (4) Block until the head's GCS goes away (head exited / job ending).
    log "worker $NODE_INDEX: joined; blocking until head GCS ${{MAIN_IP}}:6379 disappears"
    while python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); rc=s.connect_ex(('${{MAIN_IP}}',6379)); s.close(); sys.exit(0 if rc==0 else 1)" 2>/dev/null; do
      sleep 30
    done
    log "worker $NODE_INDEX: head GCS gone -> exiting (final checkpoint sync runs via EXIT trap)"
    exit 0
  fi

  # Main node: hand the run script the real head IP + cluster size. The script
  # reads ${{MASTER_ADDR}}/${{ACTOR_NUM_NODES}}/${{ROLLOUT_NUM_GPUS}} (env-gated,
  # defaults preserve single-node). slime's placement-group waiter blocks until
  # all workers join.
  export MASTER_ADDR="${{MAIN_IP}}"
  # Disaggregated (rollout_nodes>0): last ROLLOUT_NODES nodes -> rollout (SGLang),
  # the rest -> training (actor). slime sorts the placement group by node-IP, so
  # actor takes the first ACTOR_NUM_NODES*8 GPUs and rollout the trailing
  # ROLLOUT_NUM_GPUS — a clean node-boundary split. rollout_nodes==0 => colocate.
  ROLLOUT_NODES={rollout_nodes}
  USER_SIM_NODES={user_sim_nodes}
  if [ "$ROLLOUT_NODES" -gt 0 ]; then
    export ROLLOUT_NUM_GPUS=$(( ROLLOUT_NODES * GPUS_PER_NODE ))
    export ACTOR_NUM_NODES=$(( NUM_NODES - ROLLOUT_NODES ))
    # Reserve USER_SIM_NODES of the rollout nodes for a separate user-sim model.
    # USER_SIM_NUM_GPUS is a SUBSET of ROLLOUT_NUM_GPUS (the run script splits the
    # rollout pool into an actor model + a user_sim model via --sglang-config).
    export USER_SIM_NUM_GPUS=$(( USER_SIM_NODES * GPUS_PER_NODE ))
    log "main node (DISAGGREGATED): MASTER_ADDR=$MASTER_ADDR ACTOR_NUM_NODES=$ACTOR_NUM_NODES ROLLOUT_NUM_GPUS=$ROLLOUT_NUM_GPUS USER_SIM_NUM_GPUS=$USER_SIM_NUM_GPUS (of $NUM_NODES total nodes)"
  else
    export ACTOR_NUM_NODES="${{NUM_NODES}}"
    export USER_SIM_NUM_GPUS=0
    log "main node (COLOCATE): MASTER_ADDR=$MASTER_ADDR ACTOR_NUM_NODES=$ACTOR_NUM_NODES"
  fi
fi

# ── 6. Launch training (train.py is invoked by bare name -> needs this CWD) ──
cd {cfg["container_slime_dir"]}
log "launching {script_rel}"
bash {script_rel}
"""


# ═══════════════════════════════════════════════════════════════
# CLI group
# ═══════════════════════════════════════════════════════════════
@click.group()
def cli():
    """Greenland OBX manager for the slime training workflow."""
    pass


# ═══════════════════════════════════════════════════════════════
# OBX: submit slime training job
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.option("--script", "script",
              default="examples/tau-bench/run_qwen35_4b_tau_mns_async.sh",
              show_default=True,
              help="Run script path under the local slime dir. Defaults to the "
                   "async tau-bench run script.")
@click.option("--num-nodes", default=1, type=int, help="Total number of P5EN nodes.")
@click.option("--rollout-nodes", default=0, type=int,
              help="DISAGGREGATED split: how many of --num-nodes serve rollout (SGLang) "
                   "instead of training. REQUIRED > 0 for async (train_async.py asserts "
                   "not colocate). E.g. --num-nodes 5 --rollout-nodes 4 => 1 training node "
                   "(8 GPU) + 4 rollout nodes (32 GPU).")
@click.option("--user-sim-nodes", default=1, type=int,
              help="How many of the --rollout-nodes are reserved to serve a SEPARATE "
                   "user-simulation inference model (e.g. GLM-4.7-Flash) instead of the "
                   "actor. Default 1. Carved OUT of --rollout-nodes, so the actor rollout "
                   "pool = (rollout-nodes - user-sim-nodes) nodes. Exported to the run "
                   "script as USER_SIM_NUM_GPUS (= user_sim_nodes*8); the run script builds "
                   "the multi-model --sglang-config from it. Set 0 to disable (single-model "
                   "rollout, e.g. for the Bedrock user-sim path).")
@click.option("--job-name", default=None, help="Custom job name (prefix added automatically).")
@click.option("--image", default=None, help="Docker image override.")
@click.option("--wandb-key", default=None,
              help="W&B API key (defaults to local $WANDB_API_KEY).")
@click.option("--hf-token", default=None,
              help="HuggingFace token (defaults to local $HF_TOKEN).")
@click.option("--stage-model", multiple=True,
              help="S3 subpath under model_s3 to download (repeatable). "
                   "Defaults to the Qwen3.5-4B-Base inputs.")
@click.option("--stage-data", multiple=True,
              help="S3 subpath under data_s3 to download (repeatable). "
                   "Defaults to dapo-math-17k.")
@click.option("--env", "extra_env", multiple=True,
              help="Extra container env as KEY=VALUE (repeatable).")
@click.option("--no-upload", is_flag=True, default=False,
              help="Skip uploading local slime to S3 (use code already staged there).")
@click.option("--production/--no-production", "is_production", default=False,
              help="Submit as a PRODUCTION job (IsProduction=true). Default is non-production.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the job definition and bootstrap, do not submit.")
def obx(script, num_nodes, rollout_nodes, user_sim_nodes, job_name, image, wandb_key, hf_token, stage_model,
        stage_data, extra_env, no_upload, is_production, dry_run):
    """Submit an OBX slime ASYNC training job."""
    cfg = load_config()

    # ── Validate disaggregated split ──
    # ASYNC: rollout_nodes MUST be > 0 (train_async.py asserts not colocate, and
    # the whole point is to keep both pools busy via rollout/train overlap).
    if rollout_nodes <= 0:
        sys.exit("ERROR: async requires DISAGGREGATED mode — pass --rollout-nodes > 0 "
                 "(e.g. --num-nodes 5 --rollout-nodes 4). train_async.py asserts not colocate.")
    if user_sim_nodes < 0:
        sys.exit("ERROR: --user-sim-nodes must be >= 0")
    if rollout_nodes >= num_nodes:
        sys.exit(f"ERROR: --rollout-nodes ({rollout_nodes}) must be < --num-nodes "
                 f"({num_nodes}); need at least 1 training node.")
    # The user-sim model is carved out of the rollout nodes, so there must be at
    # least 1 rollout node left for the actor's own SGLang engines.
    if user_sim_nodes >= rollout_nodes:
        sys.exit(f"ERROR: --user-sim-nodes ({user_sim_nodes}) must be < --rollout-nodes "
                 f"({rollout_nodes}); need at least 1 rollout node for the actor model.")
    actor_rollout_nodes = rollout_nodes - user_sim_nodes
    print(f"Disaggregated: {num_nodes - rollout_nodes} training node(s) + "
          f"{actor_rollout_nodes} actor-rollout node(s) + "
          f"{user_sim_nodes} user-sim node(s) = {num_nodes} total.", flush=True)

    # ── Resolve the run script -> container-relative path ──
    local_slime = Path(cfg["local_slime_dir"]).resolve()
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = (local_slime / script).resolve()
    else:
        script_path = script_path.resolve()
    if not script_path.exists():
        sys.exit(f"ERROR: run script not found: {script_path}")
    try:
        script_rel = script_path.relative_to(local_slime).as_posix()
    except ValueError:
        sys.exit(f"ERROR: --script must live under {local_slime} (got {script_path})")

    wandb_key = wandb_key if wandb_key is not None else os.environ.get("WANDB_API_KEY", "")
    if hf_token is None:
        hf_token = (os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    stage_model = list(stage_model) or cfg["stage_model"]
    stage_data = list(stage_data) or cfg["stage_data"]

    env_map = {}
    for kv in extra_env:
        if "=" not in kv:
            sys.exit(f"ERROR: --env must be KEY=VALUE (got '{kv}')")
        k, v = kv.split("=", 1)
        env_map[k] = v

    # ── Job naming ──
    base = job_name or script_path.stem
    base = base.replace(".", "p")
    full_job_name = f"{cfg['job_prefix']}{base}-{random.randint(0, 999999)}"

    # ── Upload local slime -> S3 (so the container runs this exact code) ──
    if not no_upload:
        print(f"Uploading {local_slime} -> {cfg['code_s3']} (profile {cfg['source_profile']})",
              flush=True)
        up = subprocess.run(
            ["aws", "s3", "sync", f"{local_slime}/", f"{cfg['code_s3']}/",
             "--delete", "--no-progress", "--profile", cfg["source_profile"],
             # _bootstrap/ lives under code/slime/ and holds OTHER jobs' per-job
             # bootstraps (not present locally). Without this exclude, --delete
             # wipes them: a second concurrent submit's code-sync deleted job
             # 555418's bootstrap before its containers fetched it -> 404 ->
             # instant Fail. Exclude keeps --delete from touching the dir.
             "--exclude", "_bootstrap/*",
             "--exclude", ".git/*", "--exclude", "*/__pycache__/*",
             "--exclude", "*.pyc", "--exclude", "*.egg-info/*"],
        )
        if up.returncode != 0:
            sys.exit("ERROR: code upload failed; aborting submit.")
    else:
        print(f"--no-upload: using code already at {cfg['code_s3']}", flush=True)

    # ── Build the container command ──
    # AWS Batch caps container overrides (the command) at 8192 chars. The multi-node
    # bootstrap base64-encodes to >8KB, so we CANNOT inline it. Instead: upload the
    # bootstrap to S3 and have the command be a tiny loader (configure creds, download,
    # exec). The loader is fixed-size (~700 chars) regardless of bootstrap length.
    bootstrap = _build_bootstrap(cfg, script_rel, wandb_key, hf_token, stage_model, stage_data, env_map,
                                 rollout_nodes=rollout_nodes, user_sim_nodes=user_sim_nodes)
    bootstrap_s3 = f"{cfg['code_s3']}/_bootstrap/{full_job_name}.sh"
    if not no_upload or True:  # always upload the bootstrap (it's job-specific)
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as bf:
            bf.write(bootstrap)
            bootstrap_local = bf.name
        up_bs = subprocess.run(
            ["aws", "s3", "cp", bootstrap_local, bootstrap_s3,
             "--no-progress", "--profile", cfg["source_profile"]],
        )
        os.unlink(bootstrap_local)
        if up_bs.returncode != 0:
            sys.exit("ERROR: bootstrap upload failed; aborting submit.")
        print(f"Uploaded bootstrap -> {bootstrap_s3}", flush=True)

    # Tiny loader: configure cross-account creds (assume CustomerRoleArn), then pull
    # the real bootstrap from S3 and exec it. Kept well under the 8192-char Batch limit.
    loader = (
        '[ -z "${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-}" ] && '
        "export AWS_CONTAINER_CREDENTIALS_RELATIVE_URI=\"$(tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^AWS_CONTAINER_CREDENTIALS_RELATIVE_URI=//p' | head -1)\"; "
        "mkdir -p /root/.aws; "
        f"printf '[default]\\nregion = {cfg['region']}\\ncredential_source = EcsContainer\\nrole_arn = {cfg['greenland_dev_role']}\\n' > /root/.aws/config; "
        "export AWS_CONFIG_FILE=/root/.aws/config; "
        "command -v aws >/dev/null 2>&1 || pip install -q awscli || python -m pip install -q awscli; "
        f"for i in 1 2 3 4 5; do aws s3 cp {bootstrap_s3} /tmp/slime_bootstrap.sh && break || sleep 5; done; "
        "exec bash /tmp/slime_bootstrap.sh"
    )
    cmd = ["bash", "-lc", loader]

    node_range = "0" if num_nodes == 1 else f"0:{num_nodes - 1}"
    job_data = {
        "JobName": full_job_name,
        "Topology": "Zonal",
        "InitiativeId": cfg["initiative_id"],
        "InstanceType": cfg["instance_type"],
        "IsProduction": is_production,
        "InstanceCount": num_nodes,
        "Role": cfg["role"],
        "region": cfg["region"],
        "BatchJobDefinitionParameters": {
            "nodeProperties": {
                "mainNode": 0,
                "nodeRangeProperties": [{
                    "container": {
                        "command": cmd,
                        "image": image or cfg["image"],
                        "resourceRequirements": [
                            {"type": "VCPU", "value": "96"},
                            {"type": "MEMORY", "value": "1132416"},
                            {"type": "GPU", "value": "8"},
                        ],
                        "privileged": True,
                        "linuxParameters": {
                            "sharedMemorySize": 1073741824,
                            "devices": [
                                {"hostPath": "/dev/infiniband", "containerPath": "/dev/infiniband",
                                 "permissions": ["READ", "WRITE", "MKNOD"]},
                                {"hostPath": "/dev/fuse", "containerPath": "/dev/fuse",
                                 "permissions": ["READ", "WRITE", "MKNOD"]},
                            ],
                        },
                        "volumes": [{"host": {"sourcePath": "/tmp/instance_storage"}, "name": "nvme-volume"}],
                        "mountPoints": [{"containerPath": "/tmp/instance_storage", "sourceVolume": "nvme-volume", "readOnly": False}],
                    },
                    "targetNodes": node_range,
                }],
                "numNodes": num_nodes,
            },
            "type": "multinode",
        },
    }

    print(f"\nSubmitting OBX job: {full_job_name}", flush=True)
    print(f"  Script:  {script_rel}", flush=True)
    print(f"  Image:   {image or cfg['image']}", flush=True)
    print(f"  IsProduction: {is_production}", flush=True)
    if rollout_nodes > 0:
        actor_rollout_nodes = rollout_nodes - user_sim_nodes
        usim = (f" [of which {user_sim_nodes} user-sim ({user_sim_nodes * 8} GPU) "
                f"+ {actor_rollout_nodes} actor-rollout ({actor_rollout_nodes * 8} GPU)]"
                if user_sim_nodes > 0 else "")
        print(f"  Nodes:   {num_nodes} x {cfg['instance_type']} (8 GPUs each) "
              f"= {num_nodes - rollout_nodes} train ({(num_nodes - rollout_nodes) * 8} GPU) "
              f"+ {rollout_nodes} rollout ({rollout_nodes * 8} GPU){usim} [DISAGGREGATED]", flush=True)
    else:
        print(f"  Nodes:   {num_nodes} x {cfg['instance_type']} (8 GPUs each) [COLOCATE]", flush=True)
    print(f"  Model S3:{cfg['model_s3']}  stage={stage_model}", flush=True)
    print(f"  Data S3: {cfg['data_s3']}  stage={stage_data}", flush=True)
    print(f"  Output:  $MODEL_ROOT/{cfg['output_prefix']} -> {cfg['model_s3']}/{cfg['output_prefix']}", flush=True)
    print(f"  Creds:   WANDB_API_KEY={'set' if wandb_key else 'MISSING'}  "
          f"HF_TOKEN={'set' if hf_token else 'unset'}", flush=True)

    if dry_run:
        print("\n--- BOOTSTRAP ---\n" + bootstrap, flush=True)
        print("\n--- JOB DEFINITION ---\n" + json.dumps(job_data, indent=2), flush=True)
        print("\n(dry-run: not submitted)", flush=True)
        return

    ensure_auth()
    resp = mcurl(f"{GREENLAND_API}/submitjob", job_data)
    if isinstance(resp, dict) and "jobId" in resp:
        print(f"\n✅ OBX job submitted! Job ID: {resp['jobId']}", flush=True)
        print(f"   Console: https://console.harmony.a2z.com/greenland/job/details/{cfg['region']}/{cfg['initiative_id']}/{resp['jobId']}", flush=True)
    else:
        print(f"Response: {json.dumps(resp, indent=2)}", flush=True)


# ═══════════════════════════════════════════════════════════════
# STATUS: List jobs
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.option("--all-jobs", is_flag=True, help="Show all jobs, not just running.")
def status(all_jobs):
    """List Greenland jobs."""
    cfg = load_config()
    ensure_auth()

    jobs = _list_all_jobs(cfg, limited_fields=True)
    jobs.sort(key=lambda x: x.get("SubmissionTime", ""), reverse=True)

    if not all_jobs:
        jobs = [j for j in jobs if j.get("Status") in ("Running", "Starting", "Submitted", "Pending")]

    if not jobs:
        print("No active jobs found. Use --all-jobs to see terminated/completed jobs.", flush=True)
        return

    print(f"{'Status':<12} {'Type':<6} {'Job Name':<35} {'Job ID':<40} {'Submitted'}", flush=True)
    print("-" * 130, flush=True)
    for job in jobs[:20]:
        jtype = "SDB" if job.get("JobType") == "SDB" else "OBX"
        submitted = _fmt_ts(job.get("SubmissionTime"))
        print(f"{job.get('Status', 'N/A'):<12} {jtype:<6} {job.get('JobName', 'N/A'):<35} {job.get('JobId', 'N/A'):<40} {submitted}", flush=True)
        if job.get("Status") in ("Running", "Starting"):
            url = f"https://console.harmony.a2z.com/greenland/job/details/{cfg['region']}/{cfg['initiative_id']}/{job['JobId']}"
            print(f"{'':>12} └─ {url}", flush=True)


# ═══════════════════════════════════════════════════════════════
# DESCRIBE: Detailed job info
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.argument("job_identifier")
def describe(job_identifier):
    """Describe a job in detail by Job ID or Job Name."""
    cfg = load_config()
    ensure_auth()

    jobs = _list_all_jobs(cfg, limited_fields=False)

    job = None
    for j in jobs:
        if j.get("JobId") == job_identifier:
            job = j
            break
        if j.get("JobName") == job_identifier:
            if job is None or job.get("Status") in ("Terminated", "Failed", "Cancelled"):
                job = j
            if j.get("Status") in ("Running", "Pending", "Submitted", "Starting"):
                break

    if not job:
        print(f"No job found matching: {job_identifier}", flush=True)
        return

    jtype = "SDB" if job.get("JobType") == "SDB" else "OBX"
    container = {}
    node_props = job.get("BatchJobDefinitionParameters", {}).get("nodeProperties", {})
    for nr in node_props.get("nodeRangeProperties", []):
        container = nr.get("container", {})
        break

    print(f"{'Job Name:':<22} {job.get('JobName', 'N/A')}", flush=True)
    print(f"{'Job ID:':<22} {job.get('JobId', 'N/A')}", flush=True)
    print(f"{'Type:':<22} {jtype}", flush=True)
    print(f"{'Status:':<22} {job.get('Status', 'N/A')}", flush=True)
    print(f"{'Requestor:':<22} {job.get('Requestor', 'N/A')}", flush=True)
    print(f"{'Instance Type:':<22} {job.get('InstanceType', 'N/A')}", flush=True)
    print(f"{'Instance Count:':<22} {node_props.get('numNodes', job.get('InstanceCount', 'N/A'))}", flush=True)
    print(f"{'Image:':<22} {container.get('image', 'N/A')}", flush=True)
    print(f"{'Submitted:':<22} {_fmt_ts(job.get('SubmissionTime'))}", flush=True)
    print(f"{'Requested Start:':<22} {_fmt_ts(job.get('RequestedStartTime'))}", flush=True)
    print(f"{'Priority:':<22} {job.get('Priority', 'N/A')}", flush=True)
    print(f"{'Role:':<22} {', '.join(job.get('CustomerRoleArn', ['N/A']))}", flush=True)
    print(f"{'Region:':<22} {job.get('Region', cfg['region'])}", flush=True)
    print(f"{'Initiative:':<22} {job.get('InitiativeId', 'N/A')}", flush=True)
    if job.get("MainNodeInstanceId"):
        print(f"{'Main Node ID:':<22} {job['MainNodeInstanceId']}", flush=True)
    if job.get("NodesEniHostIP"):
        print(f"{'Node IPs:':<22} {', '.join(job['NodesEniHostIP'])}", flush=True)
    if job.get("JobRoleArn"):
        print(f"{'Job Role ARN:':<22} {job['JobRoleArn']}", flush=True)
    print(f"{'Console:':<22} https://console.harmony.a2z.com/greenland/job/details/{cfg['region']}/{cfg['initiative_id']}/{job['JobId']}", flush=True)


# ═══════════════════════════════════════════════════════════════
# TERMINATE: Kill a running/queued job
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.argument("job_identifier")
@click.option("--reason", default="terminated via CLI", help="Reason for termination.")
def terminate(job_identifier, reason):
    """Terminate a job by Job ID or Job Name."""
    cfg = load_config()
    ensure_auth()

    # Resolve job name to ID if needed
    job_id = job_identifier
    if not job_identifier.count("-") >= 4:  # not a UUID
        for j in _list_all_jobs(cfg, limited_fields=True):
            if j.get("JobName") == job_identifier and j.get("Status") in ("Running", "Starting", "Submitted", "Pending", "Received"):
                job_id = j["JobId"]
                break

    resp = mcurl(f"{GREENLAND_API}/terminatejob", {
        "JobId": job_id,
        "InitiativeId": cfg["initiative_id"],
        "region": cfg["region"],
        "Reason": reason,
    })

    if isinstance(resp, dict) and "jobId" in resp:
        print(f"✅ Terminated: {resp['jobId']}", flush=True)
    else:
        print(f"Response: {json.dumps(resp, indent=2)}", flush=True)


if __name__ == "__main__":
    cli()
