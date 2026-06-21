# 把本地代码 sync 到 Greenland sdb（push: local → sdb）

把 cloud-desktop 上新的目录（例：`/home/whx/slime`）同步更新到 sdb job 上旧的同名目录（例：`/root/slime`）。
这是 `greenland-sdb-file-sync` 拉取方向的反向：用 `rsync` 交换 src/dst 往上推。

- **sdb** = Greenland 上的 GPU 训练 job，经 `greenland-ssh` 起的 SSM 端口转发访问，本地 `ssh -p <port> root@localhost` 连。
- 注意：`/home/whx` 是 `/local/home/whx` 的软链接，两路径同一份。
- sdb 上 `/root`、`/tmp/instance_storage` 都**随 job 生命周期清空，不持久**；job 重启后需重新 sync。

---

## 流程总览

| 步骤 | 做什么 | 关键命令 |
|------|--------|----------|
| 1 | 连上 sdb / 确认转发端口 | `greenland-ssh` → `ss -tlnp \| grep session-manager` |
| 2 | 摸清两端状态（版本/大小/rsync） | `git rev-parse HEAD`、`du -sh`、`command -v rsync` |
| 3 | dry-run 预览要改/要删什么 | `rsync -azn --delete --itemize-changes ...` |
| 4 | 实跑推送 | `rsync -az --delete --no-owner --no-group ...` |
| 5 | 验证收敛 | 再跑一次步骤 3 的 dry-run，应无文件传输行 |
| 6 | 远程重装可编辑包（按需） | `cd /root/slime && pip install -e . --no-deps` |

---

## 步骤 1 — 连接 / 确认转发端口

在**自己的非沙箱 shell** 里跑（agent 凭证沙箱里 `greenland-ssh` 写不了 `~/.aws/config`，起不了可用转发）：

```bash
greenland-ssh                              # 发现 running job + 起 SSM 转发 + 开 root 免密
ss -tlnp | grep session-manager            # 看 LISTEN 127.0.0.1:<port>，默认 1053（被占顺延）
```

统一 ssh 前缀（下文用 `$SSH`，把 `<port>` 换成上面查到的端口）：

```bash
SSH="ssh -p <port> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@localhost"
$SSH 'hostname'                            # 验证 root 登录通
```

> 换新 job 时直接跑 `greenland-ssh`，**不要**手动 `aws ssm start-session`——手动会跳过它写新 JobRoleArn 和开 root 登录的步骤。

---

## 步骤 2 — 摸清两端状态

```bash
# 本地（源）
git -C /home/whx/slime rev-parse HEAD       # 新版本 commit
git -C /home/whx/slime status -s            # 看有没有未跟踪/未提交的文件（rsync 会一并推）
du -sh /home/whx/slime /home/whx/slime/.git

# 远程（目标）
$SSH 'git -C /root/slime rev-parse HEAD;    # 旧版本 commit
      du -sh /root/slime /root/slime/.git;  # 远程 .git 常 ~308M
      command -v rsync || echo NO_RSYNC'    # 确认远程有 rsync
```

---

## 步骤 3 — dry-run 预览（**先看再推**）

`-n` 只预览不改动。重点看 `*deleting` 行（`--delete` 会删的远程文件）确认都是该删的过时文件：

```bash
rsync -azn --delete --no-owner --no-group --itemize-changes \
  -e "ssh -p <port> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  /home/whx/slime/  root@localhost:/root/slime/
```

排除项说明：
- `.git/` — 远程那份常 ~308M、本地很小，别互相覆盖（只同步工作区代码）。
- `*.egg-info/` — 远程 `pip install -e .` 生成的，保留。
- `__pycache__/` `*.pyc` — 编译缓存，不必传。

---

## 步骤 4 — 实跑推送

去掉 `-n`。`--no-owner --no-group` 让远程保持 `root:root`（不然会 chown 成本地 uid）。
源目录结尾的 `/` 表示“推目录内容进目标目录”，别漏：

```bash
rsync -az --delete --no-owner --no-group --itemize-changes --stats \
  -e "ssh -p <port> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  /home/whx/slime/  root@localhost:/root/slime/
```

`--stats` 末尾会报：transferred / created / deleted 文件数（实测 slime ~28M，<1s）。

---

## 步骤 5 — 验证收敛

重跑步骤 3 的 dry-run，**输出里没有任何文件传输/删除行**即代表远程工作区已与本地完全一致：

```bash
rsync -azn --delete --no-owner --no-group --itemize-changes \
  -e "ssh -p <port> -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  /home/whx/slime/  root@localhost:/root/slime/ \
  | grep -E '^(<|>|\*)' || echo "(no pending changes — 已同步)"

# 可选抽查
$SSH 'stat -c%s /root/slime/README.md'      # 和本地 stat -c%s /home/whx/slime/README.md 对比
```

---

## 步骤 6 — 远程重装可编辑包（按需）

只同步了工作区文件，远程 `.git` 仍指向旧 commit；代码已是最新。要让安装生效：

```bash
$SSH 'cd /root/slime && git pull 2>/dev/null; pip install -e . --no-deps'
```

---

## 一键脚本片段

```bash
PORT=1053                                   # 改成 ss 查到的端口
SRC=/home/whx/slime/                        # 注意结尾 /
DST=/root/slime/
RSH="ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
EXC=(--exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/')

# 预览
rsync -azn --delete --no-owner --no-group --itemize-changes -e "$RSH" "${EXC[@]}" "$SRC" "root@localhost:$DST"
# 实跑
rsync -az  --delete --no-owner --no-group --itemize-changes --stats -e "$RSH" "${EXC[@]}" "$SRC" "root@localhost:$DST"
# 验证
rsync -azn --delete --no-owner --no-group --itemize-changes -e "$RSH" "${EXC[@]}" "$SRC" "root@localhost:$DST" \
  | grep -E '^(<|>|\*)' || echo "(no pending changes — 已同步)"
```

---

## 排错

- **隧道掉线**（`ss` 里没了 session-manager / ssh `Connection refused`）：SSM 会话会超时。回自己的 shell 重跑 `greenland-ssh`。
- **`TargetNotConnected: i-... is not connected`**：实例当前没连上 SSM（job 没跑/重启中），本地无解，等实例上线。
- **agent 凭证沙箱里**：`$AWS_CONFIG_FILE` 被指到只读目录，`greenland-ssh` 写 `[profile greenland]` 会 `Permission denied` → 转发起不来。重连请在普通 shell 里做。
- **`--delete` 误删**：永远先 `-n` dry-run 看 `*deleting` 行；被排除的项（`.git/`、`*.egg-info/`）默认不会被 `--delete` 删。
