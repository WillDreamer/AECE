#!/usr/bin/env python3
"""
Submit an OBX job and auto-retry on capacity failure.

WHY: The Greenland Batch v2 API (the one greenland_cli_mns.py speaks) exposes NO
capacity endpoint — only submit/describe/list/terminate. So we cannot query "free
P5EN nodes" up front. Instead we use the only reliable capacity signal that exists:
a job that can't get nodes dies fast with AWS Batch statusReason "Task failed to
start" (observed on job 5b26222a: a restart attempt failed-to-start ~49s after the
node it needed was gone). This wrapper:

  1. submits the job (delegates to `greenland_cli_mns.py obx ...`),
  2. polls its status,
  3. if it reaches Running/Starting -> SUCCESS, stop (hand off to normal monitoring),
  4. if it Fails/Terminates while still "too young to have really run" (i.e. a
     capacity / failed-to-start death, not a real crash) -> sleep RETRY_SLEEP and
     resubmit,
  5. repeat up to MAX_ATTEMPTS.

It deliberately does NOT retry a job that ran a while then failed — that's a real
application error (OOM, bug), not capacity, and resubmitting would just burn nodes.

Usage (pass the SAME args you'd give `obx`, after `--`):
  python3 submit_with_retry_mns.py -- \
      --script examples/math_reasoning/run_qwen35_4b_base_mns.sh \
      --num-nodes 6 --rollout-nodes 4

Tunables via env:
  RETRY_SLEEP_SEC   (default 300 = 5 min)   sleep between attempts
  MAX_ATTEMPTS      (default 12)            give up after this many submits
  WATCH_SEC         (default 600 = 10 min)  how long to watch a submit before
                                            declaring it "stably Running" = success
  POLL_SEC          (default 30)            status poll interval
  YOUNG_FAIL_SEC    (default 240 = 4 min)   a Failed/Terminated job younger than
                                            this (ActualExecutionDuration) is treated
                                            as a capacity failure -> retry; older =
                                            real crash -> STOP (no retry)
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = os.path.join(HERE, "greenland_cli_mns.py")

RETRY_SLEEP_SEC = int(os.environ.get("RETRY_SLEEP_SEC", "300"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "12"))
WATCH_SEC = int(os.environ.get("WATCH_SEC", "600"))
POLL_SEC = int(os.environ.get("POLL_SEC", "30"))
YOUNG_FAIL_SEC = int(os.environ.get("YOUNG_FAIL_SEC", "240"))

# Statuses (match greenland_cli_mns.py vocabulary)
LIVE_OK = ("Running",)               # truly scheduled + executing
PENDING = ("Received", "Starting", "Submitted", "Pending", "Queued")
DEAD = ("Failed", "Terminated", "Cancelled", "Completed", "Succeeded")


def _log(msg):
    print(f"[retry {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _obx_args():
    """Args after the literal `--` are forwarded verbatim to `obx`."""
    if "--" not in sys.argv:
        sys.exit("ERROR: pass obx args after `--`, e.g.\n"
                 "  python3 submit_with_retry_mns.py -- --script <s> --num-nodes 6 --rollout-nodes 4")
    return sys.argv[sys.argv.index("--") + 1:]


def submit(obx_args):
    """Run `greenland_cli_mns.py obx ...`, return the submitted JobId (or None)."""
    cmd = ["python3", CLI, "obx", *obx_args]
    _log(f"submitting: {' '.join(cmd)}")
    out = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    # CLI prints: "✅ OBX job submitted! Job ID: <uuid>"
    m = re.search(r"Job ID:\s*([0-9a-f-]{36})", out.stdout)
    if not m:
        m = re.search(r'"jobId":\s*"([0-9a-f-]{36})"', out.stdout)
    return m.group(1) if m else None


def job_record(job_id):
    """Fetch the raw job dict via the CLI's own helpers (status + duration)."""
    # Import lazily so a CLI syntax error surfaces clearly.
    sys.path.insert(0, HERE)
    from greenland_cli_mns import load_config, _list_all_jobs
    cfg = load_config()
    for j in _list_all_jobs(cfg, limited_fields=True):
        if j.get("JobId") == job_id:
            return j
    return None


def watch(job_id):
    """Poll until Running (return 'ok'), capacity-fail (return 'retry'),
    or real-fail (return 'stop'). Times out to 'ok' if it stays pending-but-alive
    past WATCH_SEC without dying (let normal monitoring take over)."""
    t0 = time.time()
    while time.time() - t0 < WATCH_SEC:
        rec = job_record(job_id)
        if rec is None:
            _log(f"{job_id[:8]}: not found yet, waiting...")
            time.sleep(POLL_SEC)
            continue
        status = rec.get("Status", "?")
        dur_min = rec.get("ActualExecutionDuration") or 0  # minutes
        _log(f"{job_id[:8]}: status={status} dur={dur_min:.1f}min")

        if status in LIVE_OK:
            _log(f"{job_id[:8]}: RUNNING — success, handing off to normal monitoring.")
            return "ok"
        if status in DEAD:
            dur_sec = float(dur_min) * 60.0
            if status in ("Completed", "Succeeded"):
                _log(f"{job_id[:8]}: {status} — done, no retry.")
                return "stop"
            if dur_sec <= YOUNG_FAIL_SEC:
                _log(f"{job_id[:8]}: {status} after only {dur_sec:.0f}s "
                     f"(<= {YOUNG_FAIL_SEC}s) -> treat as CAPACITY failure, will retry.")
                return "retry"
            _log(f"{job_id[:8]}: {status} after {dur_sec:.0f}s of real runtime "
                 f"-> REAL error (OOM/bug/preempted-mid-run), NOT retrying. Investigate logs.")
            return "stop"
        # else still pending/starting -> keep waiting
        time.sleep(POLL_SEC)

    _log(f"{job_id[:8]}: still alive (pending/running) after {WATCH_SEC}s watch — "
         f"assuming it got capacity; handing off.")
    return "ok"


def main():
    obx_args = _obx_args()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _log(f"=== attempt {attempt}/{MAX_ATTEMPTS} ===")
        job_id = submit(obx_args)
        if not job_id:
            _log("submit did not return a Job ID (submit error?). Sleeping then retrying.")
            time.sleep(RETRY_SLEEP_SEC)
            continue
        verdict = watch(job_id)
        if verdict == "ok":
            _log(f"DONE: job {job_id} is up. Monitor it normally.")
            print(job_id)
            return 0
        if verdict == "stop":
            _log(f"STOP: job {job_id} failed for a non-capacity reason. Not retrying.")
            print(job_id)
            return 2
        # verdict == "retry"
        if attempt < MAX_ATTEMPTS:
            _log(f"capacity failure; sleeping {RETRY_SLEEP_SEC}s before resubmit...")
            time.sleep(RETRY_SLEEP_SEC)
    _log(f"gave up after {MAX_ATTEMPTS} attempts (still no capacity).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
