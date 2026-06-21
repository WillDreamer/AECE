---
description: Greenland GPU cluster for training and inference on P5EN instances. Use when submitting batch (OBX) or interactive (SDB) jobs, building Docker images, monitoring job status, SSH into nodes, or troubleshooting GPU training/inference failures.
---

# Greenland Dev

## Quick Reference

```bash
# CLI location
python greenland/greenland_cli.py <command>

# Submit SFT training job
python greenland/greenland_cli.py obx --trainer sft --model Qwen/Qwen3-4B \
    --s3-data s3://xjlei-data/accordion-train/data/sft_train.jsonl

# Interactive debug session (SSH-able)
python greenland/greenland_cli.py sdb --duration 6

# Check status
python greenland/greenland_cli.py status
python greenland/greenland_cli.py describe <job-name-or-id>

# Terminate a job
python greenland/greenland_cli.py terminate <job-name-or-id>
python greenland/greenland_cli.py terminate <job-id> --reason "resubmitting"
```

## Job Types

### SDB (Interactive)
Creates a workspace you can SSH into for testing/debugging.
```bash
python greenland/greenland_cli.py sdb --duration 6 --job-name xjlei-test
```

### OBX (Batch)
Fire-and-forget training/inference jobs.
```bash
python greenland/greenland_cli.py obx --trainer sft --model Qwen/Qwen3-4B \
    --s3-data s3://xjlei-data/accordion-train/data/sft_train.jsonl \
    --job-name sft-qwen3-4b --num-nodes 1
```

## Infrastructure

- **Instance**: p5en.48xlarge (96 vCPU, 1132 GB RAM, 8x H200 GPUs)
- **Region**: ap-south-1
- **Initiative**: RufusInSearch
- **ECR Image**: 339712697413.dkr.ecr.ap-south-1.amazonaws.com/xjlei:rq-obx-vllm
- **S3 Data**: s3://xjlei-data/accordion-train/
- **AWS Profiles**: `greenland-dev` (ECR/Greenland API), `resdev` (S3 data)
- **Console**: https://console.harmony.a2z.com/greenland/job/details/ap-south-1/RufusInSearch/<job-id>

## Docker Image

```bash
bash greenland/build_image.sh rq-obx-vllm
```

## Entrypoint Flow (OBX training)

`entrypoint.sh <job_name> <trainer> <model> <s3_data>`:
1. Downloads training JSONL from S3 to NVMe
2. Auto-detects eval data: `_train.jsonl` -> `_eval.jsonl`
3. Selects config by model size: `configs/sft_4b.yaml`
4. Runs `torchrun --nproc_per_node=8 train_{trainer}.py`
5. Uploads checkpoints to `s3://xjlei-data/accordion-train/output/<job_name>/`

### Bedrock Access from OBX
OBX entrypoint processes inherit `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` from PID 1 automatically. To enable Bedrock calls:
```bash
mkdir -p ~/.aws
cat > ~/.aws/config << EOF
[default]
credential_source = EcsContainer
role_arn = arn:aws:iam::339712697413:role/greenland-dev-role
region = us-west-2
EOF
```

## Remote Execution (No Manual SSH)

### Step 1: Get node info
```bash
python greenland/greenland_cli.py describe <job-name>
# Note: Main Node ID (i-xxx), Node IPs (10.x.x.x), JobRoleArn
```

### Step 2: Set up greenland AWS profile
```bash
aws configure set --profile greenland role_arn <JobRoleArn>
aws configure set --profile greenland source_profile greenland-dev
aws configure set --profile greenland region ap-south-1
```

### Step 3: Start SSM port forward (background)
Key: use `AWS-StartPortForwardingSessionToRemoteHost` with the container IP, not `AWS-StartPortForwardingSession` to the instance.
```bash
nohup aws ssm start-session \
    --target "<instance-id>" \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters '{"portNumber":["22"],"localPortNumber":["2222"],"host":["<node-ip>"]}' \
    --profile greenland --region ap-south-1 > /tmp/ssm_pf.log 2>&1 &
for i in $(seq 1 15); do ss -tlnp | grep -q ':2222' && break; sleep 1; done
```

### Step 4: Enable root login (first time per SDB)
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 \
    greenland-user@localhost \
    'sudo sed -i "s/^permitrootlogin[[:space:]]\+no/PermitRootLogin yes/i" /etc/ssh/sshd_config && sudo passwd -d root && sudo bash -c "echo \"while IFS= read -r -d \\\"\\\" var; do export \\\"\\\$var\\\"; done < /proc/1/environ\" >> /root/.bashrc" && sudo kill -HUP $(pidof sshd)'
```

### Step 5: Run commands as root
```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 root@localhost '<command>'
```

### Step 6: Transfer files via SCP
```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P 2222 \
    local_file root@localhost:/workdir/remote_file
```

### Container Environment
- Conda: `/root/miniforge3/bin/conda`, env `rq`
- Activate: `source /root/miniforge3/etc/profile.d/conda.sh; conda activate rq; export LD_LIBRARY_PATH=/root/miniforge3/envs/rq/lib:$LD_LIBRARY_PATH`
- Working dir: `/workdir`
- S3 access to s3://xjlei-data from container. After login, run "sudo -i" then access S3. If failed, use SCP.

## Reading Job Logs (CloudWatch)

OBX job logs are in CloudWatch under account `821346262838`.

```bash
# Auto-refreshes via credential_process. Manual fallback:
ada credentials update --account 821346262838 --role Greenland-Console-Access-All --provider conduit --profile greenland-console --once

# Find log streams
aws logs describe-log-streams \
    --log-group-name /aws/batch/job \
    --log-stream-name-prefix "xjlei-<job-name>" \
    --region ap-south-1 --profile greenland-console

# Read logs (filter S3 progress noise, read from end for crash/error)
aws logs get-log-events \
    --log-group-name /aws/batch/job \
    --log-stream-name "<log-stream-name>" \
    --region ap-south-1 --profile greenland-console \
    --limit 10000 --query 'events[*].message' --output text \
    | tr '\t' '\n' | grep -v 'Completed.*GiB.*remaining' | grep -v 'Completed.*MiB.*remaining'
```

## Common Issues

| Issue | Fix |
|-------|-----|
| Auth expired | Run `mwinit` before API calls |
| S3 AccessDenied | Use `--profile greenland-dev` or `resdev` |
| SDB not starting | Check P5EN capacity on Greenland console |
| SSM TargetNotConnected | Use `JobRoleArn` (821346262838), NOT `CustomerRoleArn` |
| NCCL issues | Set `NCCL_SOCKET_IFNAME=eth0`, mount InfiniBand devices |
| OOM on GPU | Enable flash attention, gradient checkpointing, reduce batch |
| `--job-name` double-prefixed | CLI auto-prepends username. Don't include `xjlei-` in value |
| CloudWatch logs truncated | Filter `grep -v 'Completed.*GiB.*remaining'` and paginate with `--next-token` |
| `GLIBCXX_3.4.32 not found` | Set `LD_LIBRARY_PATH=/root/miniforge3/envs/rq/lib:$LD_LIBRARY_PATH` |
| SSH Permission denied | Forward to container IP via `AWS-StartPortForwardingSessionToRemoteHost`, not instance port 22 |
| vLLM FlashInfer crash | CUDA 12.2 too old — set `VLLM_DISABLE_FUSED_ALLREDUCE_RMS=1` |
