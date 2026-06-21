#!/usr/bin/env bash
# One-shot Greenland sdb (re)connect.
#
# Run this whenever you either:
#   (a) created a NEW sdb job (the old one expired/ended), or
#   (b) just can't reach the existing sdb (tunnel dropped / port conflict).
#
# It AUTO-DISCOVERS the current Running job named "$JOB_NAME" (default whx-agent)
# via greenland-ssh's midway-authenticated listjobs, reads its instance id + IP,
# then delegates to the proven reconnect flow in the greenland-sdb-reconnect
# skill (reconnect.sh): kill stale forward -> point `greenland` profile at the
# job's role -> start SSM port-forward -> enable root login -> verify SSH.
#
# Because it discovers the job fresh every time, the SAME command handles both
# "new machine" (new instance/IP/role) and "dropped" (same instance) cases.
#
# Usage:
#   reconnect-sdb.sh                       # auto-discover whx-agent, port 1053
#   reconnect-sdb.sh --job-name foo        # different job name
#   reconnect-sdb.sh --port 1055           # different local port
#   reconnect-sdb.sh --instance i-... --ip 10.x.x.x   # skip discovery (manual override)
#
# Account is always 339712697413 (greenland-dev) / region ap-south-1 /
# initiative RufusInSearch. See memories greenland-sdb-reconnect-skill,
# greenland-sdb-reconnect-troubleshooting, greenland-sdb-file-sync.
set -euo pipefail

INITIATIVE="RufusInSearch"
REGION="ap-south-1"
GREENLAND_SSH_DIR="$HOME/src/greenland-ssh"
RECONNECT_SH="$HOME/.claude/skills/greenland-sdb-reconnect/reconnect.sh"
CFG="$HOME/.greenland_ssh.yaml"

# Default job name from ~/.greenland_ssh.yaml if present, else whx-agent.
JOB_NAME="$(grep -E '^[[:space:]]*job_name:' "$CFG" 2>/dev/null | head -1 \
            | sed -E 's/^[^:]*:[[:space:]]*"?([^"#]+)"?.*/\1/' | xargs || true)"
JOB_NAME="${JOB_NAME:-whx-agent}"
PORT=1053
INSTANCE=""
IP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)   JOB_NAME="$2"; shift 2;;
    --port)       PORT="$2"; shift 2;;
    --instance)   INSTANCE="$2"; shift 2;;
    --ip)         IP="$2"; shift 2;;
    --initiative) INITIATIVE="$2"; shift 2;;
    --region)     REGION="$2"; shift 2;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -x "$RECONNECT_SH" ]] || { echo "ERROR: skill reconnect.sh not found/executable at $RECONNECT_SH" >&2; exit 1; }

if [[ -z "$INSTANCE" || -z "$IP" ]]; then
  echo "=== Discovering Running job '$JOB_NAME' (initiative $INITIATIVE, $REGION) ==="
  # Emit "<instance> <ip>" for the most recent Running job with this name.
  DISCO="$(cd "$GREENLAND_SSH_DIR" && .venv/bin/python - "$JOB_NAME" "$INITIATIVE" "$REGION" <<'PY'
import sys
from greenland_ssh.__main__ import get_jobs
job_name, initiative, region = sys.argv[1:4]
jobs = get_jobs(initiative, region) or []
cand = [j for j in jobs if j.get("Status") == "Running" and j.get("JobName") == job_name]
cand.sort(key=lambda x: x.get("SubmissionTime", 0), reverse=True)
if not cand:
    sys.exit(0)
j = cand[0]
ips = j.get("NodesEniHostIP") or []
print(f'{j.get("MainNodeInstanceId","")} {ips[0] if ips else ""}', end="")
PY
)"
  INSTANCE="$(echo "$DISCO" | awk '{print $1}')"
  IP="$(echo "$DISCO" | awk '{print $2}')"
  if [[ -z "$INSTANCE" || -z "$IP" ]]; then
    echo "ERROR: no Running job named '$JOB_NAME' found in account 339712697413." >&2
    echo "       - If you just created it, wait a moment for it to reach Running and retry." >&2
    echo "       - If discovery itself failed, your midway may have expired: run 'mwinit' and retry." >&2
    echo "       - If the job was renamed, pass --job-name <name> (and update $CFG)." >&2
    exit 1
  fi
  echo "discovered: instance=$INSTANCE ip=$IP"
fi

echo "=== Delegating to skill reconnect.sh ==="
exec "$RECONNECT_SH" --instance "$INSTANCE" --ip "$IP" --port "$PORT" --job-name "$JOB_NAME" \
  --initiative "$INITIATIVE" --region "$REGION"
