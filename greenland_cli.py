#!/usr/bin/env python3
"""
Greenland OBX job manager for the slime RL training workflow (user: whx).

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
    # Submit the Qwen3.5-4B math-reasoning RL run
    python greenland_cli.py obx --script examples/math_reasoning/run_qwen35_4b_base.sh

    # List active jobs / inspect / terminate
    python greenland_cli.py status
    python greenland_cli.py describe <job-name-or-id>
    python greenland_cli.py terminate <job-name-or-id>
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
    "image": "241893993881.dkr.ecr.ap-south-1.amazonaws.com/guangrli:slime-greenland",
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
def _build_bootstrap(cfg, script_rel, wandb_key, hf_token, stage_model, stage_data, extra_env):
    """Render the in-container bootstrap bash. Passed base64-encoded as the
    batch command so no pre-existing entrypoint file is required in the image."""

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

{chr(10).join(stage_model_cmds)}
{chr(10).join(stage_data_cmds)}

# ── 4. Background checkpoint sync (local outputs -> S3) ──
sync_outputs() {{ $AWS s3 sync "$MODEL_ROOT/{out_prefix}" "{cfg["model_s3"]}/{out_prefix}" --only-show-errors || true; }}
( while true; do sleep 300; sync_outputs; done ) &
SYNC_PID=$!
trap 'log "final checkpoint sync"; sync_outputs; kill $SYNC_PID 2>/dev/null || true' EXIT

# ── 5. Env for the run script (overrides its dev-box defaults) ──
# Skip the script's dev-box LD_LIBRARY_PATH prepend (it breaks cuDNN in this image).
export SKIP_SYS_LDPATH=1
export SLIME_DIR={cfg["container_slime_dir"]}
export MEGATRON_DIR={cfg["container_megatron_dir"]}
export RAY_TEMP_DIR=$ROOT_DIR/ray_temp
export WANDB_API_KEY="{wandb_key}"
{hf_lines}
{extra_env_lines}

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
@click.option("--script", "script", required=True,
              help="Run script path under the local slime dir "
                   "(e.g. examples/math_reasoning/run_qwen35_4b_base.sh).")
@click.option("--num-nodes", default=1, type=int, help="Number of P5EN nodes.")
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
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the job definition and bootstrap, do not submit.")
def obx(script, num_nodes, job_name, image, wandb_key, hf_token, stage_model, stage_data,
        extra_env, no_upload, dry_run):
    """Submit an OBX slime training job."""
    cfg = load_config()

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
             "--exclude", ".git/*", "--exclude", "*/__pycache__/*",
             "--exclude", "*.pyc", "--exclude", "*.egg-info/*"],
        )
        if up.returncode != 0:
            sys.exit("ERROR: code upload failed; aborting submit.")
    else:
        print(f"--no-upload: using code already at {cfg['code_s3']}", flush=True)

    # ── Build the container command ──
    bootstrap = _build_bootstrap(cfg, script_rel, wandb_key, hf_token, stage_model, stage_data, env_map)
    b64 = base64.b64encode(bootstrap.encode()).decode()
    cmd = ["bash", "-lc",
           f"echo {b64} | base64 -d > /tmp/slime_bootstrap.sh && exec bash /tmp/slime_bootstrap.sh"]

    node_range = "0" if num_nodes == 1 else f"0:{num_nodes - 1}"
    job_data = {
        "JobName": full_job_name,
        "Topology": "Zonal",
        "InitiativeId": cfg["initiative_id"],
        "InstanceType": cfg["instance_type"],
        "IsProduction": False,
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
    print(f"  Nodes:   {num_nodes} x {cfg['instance_type']} (8 GPUs each)", flush=True)
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
