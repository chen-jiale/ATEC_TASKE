# Task E 本地开发、Git 同步与 SSH 测试 SOP

## 1. 工作模式

本项目采用以下工作流：

```text
本地目录：唯一代码源与本地单元测试
        ↓ Git push
服务器持久盘 bare Git 仓库：代码中转
        ↓ Git pull
服务器工作目录：Isaac Sim、训练和闭环评测
        ↓ rsync
本地 remote-results：日志、视频和实验记录
```

基本规则：

- 不直接在服务器修改源码。
- 本地和服务器通过 Git commit SHA 确认代码一致。
- 数据集、日志、视频和训练 checkpoint 不进入 Git。
- 每次远程实验记录代码 SHA、策略权重 SHA256、环境和退出码。
- 本地不安装 Isaac Sim；本地负责代码与 CPU 单元测试，服务器负责 Isaac、相机、物理、训练和闭环评测。

## 2. 固定路径与名称

本文默认使用以下值：

| 用途 | 值 |
| --- | --- |
| 本地项目 | `/home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone` |
| SSH 别名 | `atec-gpu` |
| 服务器 bare Git 仓库 | `/root/gpufree-data/repos/task-e.git` |
| 服务器运行副本 | `/root/gpufree-data/task-e-work/ATEC_TaskE_Standalone` |
| 服务器实验结果 | `/root/gpufree-data/task-e-results` |
| 本地下载结果 | `remote-results/` |

如果实际地址不同，请统一替换，不要混用旧的 `/root/autodl-tmp/` 工作副本。

## 3. 一次性初始化

### 3.1 在本地配置 SSH 别名

以下操作在**本地终端**完成。

编辑 SSH 配置：

```bash
nano ~/.ssh/config
```

加入：

```sshconfig
Host atec-gpu
    HostName <云服务器地址>
    User root
    Port <SSH端口>
    IdentityFile <本地私钥路径>
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

测试连接：

```bash
ssh atec-gpu 'hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

预期看到服务器主机名和 RTX 4090。不要把私钥内容提交到 Git 或发送给他人。

### 3.2 完善本地 `.gitignore`

以下操作在**本地终端**完成。

这一步是编辑本地项目的 `.gitignore`，告诉 Git 哪些文件不需要提交和上传。不需要删除任何文件；忽略规则不会影响本地运行。

进入项目并打开文件：

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone
nano .gitignore
```

保留文件原来的内容，检查是否已经包含以下规则。缺少的行补到文件末尾，已有的行不需要重复添加：

```gitignore
# Generated runtime artifacts
output/
logs/
remote-results/
datasets/atec_task_e/
scripts/act/runs/
*.mp4
*.hdf5
```

例如，原文件如果已经有 `logs/` 和 `scripts/act/runs/`，只需要补上 `output/`、`remote-results/`、`datasets/atec_task_e/`、`*.mp4` 和 `*.hdf5`。原来的数据集子目录规则可以保留，即使它们已被新的整目录规则覆盖。

编辑后，按 `Ctrl+O`，再按回车保存；最后按 `Ctrl+X` 退出。可以运行 `cat .gitignore` 查看保存后的内容。

机器人资产、场景资产和自带的 `atec_robot_model/baseline/act/policy.pt` 应保留在 Git 中：不要为这些文件添加忽略规则，也不要添加 `*.pt` 这种忽略所有权重的规则。以后新训练产生的 checkpoint 放在 `scripts/act/runs/` 下或服务器结果目录中，避免将实验权重提交到 Git。

完成后继续 3.3。注意：`.gitignore` 对已经被 Git 跟踪的文件不会自动取消跟踪；本 SOP 在首次提交前配置规则，因此此时不需要额外移除文件。

### 3.3 初始化本地 Git 仓库

以下操作在**本地终端**完成。

先创建 Git 仓库：

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone
git init -b main
```

看到“已初始化空的 Git 仓库于 …/.git/”就表示成功。`.git` 是 Git 自动创建的隐藏目录，用于保存版本记录，不需要手动创建或修改权限。

如果按照旧版文档先运行了 `chmod u+w .git`，出现“无法访问 '.git': 没有那个文件或目录”，原因是初始化前该目录尚不存在。只要后面的 `git init -b main` 已经成功，就可以继续，不需要重跑 `chmod` 或重新初始化。

接下来只为当前项目配置提交身份。将下面的占位符替换为自己的名字和邮箱，保留双引号，不保留尖括号。这是提交记录中的作者信息，不是 SSH 用户名或密码：

```bash
git config user.name "你的名字"
git config user.email "你的邮箱"
```

检查配置是否正确：

```bash
git config --get user.name
git config --get user.email
```

两条命令应分别输出刚填写的名字和邮箱。

例如，名字为 Chen 时，应输出 `Chen`，而不是 `<Chen>`。邮箱同样不应包含 `<` 和 `>`。如果误把尖括号一起填入，重新运行上述两条 `git config` 命令，填写不带尖括号的值即可覆盖旧配置，无需重新初始化仓库。在首次提交前改好即可。

检查准备提交的内容：

```bash
git status --short
```

首次提交前，文件名前的 `??` 表示“尚未纳入 Git 跟踪”，属于正常现象。`.gitignore`、源码、文档和 `atec_robot_model/` 等资产目录应当出现在列表中。

例如，看到 `?? .gitignore`、`?? README.md`、`?? atec_robot_model/`、`?? docs/`、`?? scripts/` 和 `?? source/` 都是正常的。此时 Git 会将未跟踪目录折叠为一行，因此还需要下一步暂存后展开清单，检查目录中的具体文件。`git add .` 只在本地准备提交内容，不会上传到服务器；执行后状态通常从 `??` 变为 `A`（新增文件）。

先确认列表中没有需要忽略的运行产物，再将文件加入暂存区（准备提交的文件集合），并展开检查文件清单：

```bash
git add .
git diff --cached --name-only
```

清单中不应包含 `output/`、`logs/`、`datasets/atec_task_e/`、`remote-results/`、`scripts/act/runs/` 下的文件，或 `.hdf5`、`.mp4` 文件；应包含自带的 `atec_robot_model/baseline/act/policy.pt`。如果清单不符合预期，先检查忽略规则，不要继续提交。

确认后建立原始基线提交：

```bash
git commit -m "Import Task E handoff baseline"
git rev-parse HEAD
git status --short
```

首次提交包含模型和资产，耗时较长是正常的。`git rev-parse HEAD` 会输出本次提交的完整 SHA（版本标识）；最后的 `git status --short` 没有输出，表示工作目录没有待提交的更改。完成后继续 3.4，在服务器创建中转仓库。

### 3.4 在服务器持久盘创建 bare Git 仓库

以下操作在**SSH 终端**完成。

```bash
ssh atec-gpu
mkdir -p /root/gpufree-data/repos
git init --bare /root/gpufree-data/repos/task-e.git
exit
```

### 3.5 从本地首次推送

以下操作在**本地终端**完成。

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone

git remote add gpu \
  atec-gpu:/root/gpufree-data/repos/task-e.git

git push -u gpu main
```

如果推送很慢，先看终端最后一行。例如：

```text
压缩对象中: 100% (141/141), 完成.
写入对象中: 7% (11/154), 13.66 MiB | 66.00 KiB/s
```

这表示本地压缩已完成，当前正在向服务器传输，显示的传输速率约为 66 KiB/s。`7% (11/154)` 是对象数量进度，不是总字节进度，不能据此准确估算剩余时间。此时改用同一 SSH 连接上传压缩包不保证提速；如果服务器有更快的网页上传入口，可以选择 3.7 的压缩包流程。

若决定切换方式，可在正在推送的终端按 `Ctrl+C` 停止。本地提交不会丢失；未成功的首次推送可能需要重新传输。以后重试 Git 推送只需再次执行 `git push -u gpu main`，不要重复执行 `git remote add gpu`，否则会提示远程名称已存在。选择压缩包方式则直接按照 3.7 操作。

设置服务器 bare 仓库的默认分支：

```bash
ssh atec-gpu \
  'git --git-dir=/root/gpufree-data/repos/task-e.git symbolic-ref HEAD refs/heads/main'
```

### 3.6 在服务器持久盘创建运行副本

以下操作在**SSH 终端**完成。

保留旧的 `/root/autodl-tmp/` 目录，不覆盖、不删除；新副本放在持久盘：

```bash
ssh atec-gpu

mkdir -p /root/gpufree-data/task-e-work
cd /root/gpufree-data/task-e-work

git clone \
  /root/gpufree-data/repos/task-e.git \
  ATEC_TaskE_Standalone

cd ATEC_TaskE_Standalone
git status --short
git rev-parse HEAD
```

重新绑定 editable install：

```bash
python -m pip install \
  --no-deps \
  --no-build-isolation \
  -e ./source/atec_rl_lab

export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -c 'from isaaclab.app import AppLauncher; print("APP LAUNCHER OK")'
```

以后服务器运行目录固定为：

```text
/root/gpufree-data/task-e-work/ATEC_TaskE_Standalone
```

### 3.7 可选：使用压缩包上传，替代远程 Git 同步

可以保留本地 Git 记录版本，将已提交的项目打包上传，不必完成 3.4–3.6 的远程 Git 仓库配置。下面的方法与第 4 节的 push/pull 流程二选一。

速度取决于瓶颈：`git add`、`git commit` 和推送时的 `Compressing objects` 主要是本地处理；推送的 `Writing objects` 才涉及传输。Git 本身也会压缩数据，所以压缩包经同一条 SSH 连接上传，不保证更快。模型权重、已压缩资产通常也难以进一步压缩。首次上传大资产可能较慢；后续只改代码时，Git 增量同步一般比反复上传整个压缩包更省流量。

#### 第一步：在本地生成压缩包

先按 3.3 完成本地提交。以后每次打包前，也要先提交希望上传的修改。以下命令只打包最新提交，不包含未提交或被忽略的文件，并保留已提交的机器人资产和基线权重：

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone
git status --short
```

预期没有输出；如有输出，先检查并提交需要上传的修改，再继续：

```bash
PACKAGE_SHA=$(git rev-parse HEAD)
PACKAGE_NAME="task-e-$PACKAGE_SHA.tar.gz"
mkdir -p remote-results/packages
git archive --format=tar.gz --prefix="task-e-$PACKAGE_SHA/" \
  --output="remote-results/packages/$PACKAGE_NAME" HEAD
ls -lh "remote-results/packages/$PACKAGE_NAME"
```

`ls -lh` 显示包的大小。包存放在已被忽略的目录中，不会加入下次 Git 提交。包名中的完整 SHA 标识对应的本地提交；包内不包含 `.git` 历史。以上变量用于后续命令，请在同一个本地终端继续操作。

#### 第二步：从本地上传到服务器持久盘

```bash
ssh atec-gpu 'mkdir -p /root/gpufree-data/task-e-packages'
rsync -avP "remote-results/packages/$PACKAGE_NAME" \
  atec-gpu:/root/gpufree-data/task-e-packages/
```

`rsync` 使用 SSH 连接，显示传输进度并保留未传完的文件。连接中断后，在变量仍有效的同一个终端重新运行上传命令即可利用已有文件继续传输。两端都需要安装 `rsync`。若无法使用，可改用以下命令，但 `scp` 中断后通常需要重新传输：

```bash
scp "remote-results/packages/$PACKAGE_NAME" \
  atec-gpu:/root/gpufree-data/task-e-packages/
```

也可以使用服务器提供的网页文件上传入口或 SFTP 软件，将同一个压缩包上传到 `/root/gpufree-data/task-e-packages/`。是否更快取决于该入口的实际带宽。

#### 第三步：核对上传内容并解压到独立目录

在刚才的本地终端执行：

```bash
sha256sum "remote-results/packages/$PACKAGE_NAME"
ssh atec-gpu "sha256sum /root/gpufree-data/task-e-packages/$PACKAGE_NAME"
```

两次输出开头的哈希值应完全相同；不同则不要解压，先重新上传。确认一致后，继续在本地终端执行下面的远程解压命令：

```bash
ssh atec-gpu "mkdir -p /root/gpufree-data/task-e-releases && \
  tar --keep-old-files -xzf /root/gpufree-data/task-e-packages/$PACKAGE_NAME \
  -C /root/gpufree-data/task-e-releases"
echo "/root/gpufree-data/task-e-releases/task-e-$PACKAGE_SHA"
```

最后一条输出本次版本的服务器运行目录，请记下完整路径。每个提交解压到独立目录，避免覆盖旧版本和残留已删除源码。若提示文件已存在，先确认该版本是否已经解压过，不要盲目覆盖。

#### 第四步：在服务器使用这个版本

登录服务器，进入上一步输出的完整目录（将下面的占位路径替换成实际路径）：

```bash
ssh atec-gpu
cd /root/gpufree-data/task-e-releases/task-e-实际的完整SHA
```

然后执行 3.6 中从 `python -m pip install` 开始的安装和导入检查命令，每次切换到新的解压目录都重新绑定 editable install。

压缩包副本没有 `.git`，因此不要在其中执行第 4 节的 `git pull` 或 `git rev-parse`。第 5 节记录实验时，需要将 `git rev-parse` 替换为压缩包文件名中的提交 SHA，将 Git 状态记录替换为“压缩包部署，无 Git 工作区”的说明；运行目录也应使用本次解压目录。本地后续修改继续在原项目中进行，提交后重复本节打包上传步骤。

## 4. 每次修改代码后的同步 SOP

### 4.1 本地修改前检查

以下操作在**本地终端**完成。

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone
git status --short
git log -3 --oneline
```

工作区应为空。随后在本地修改代码并运行不依赖 Isaac Sim 的单元测试。

### 4.2 本地检查改动

以下操作在**本地终端**完成。

```bash
git status --short
git diff --stat
git diff
```

确认没有数据集、日志、视频、密码、私钥或其他敏感文件。

### 4.3 本地提交并推送

以下操作在**本地终端**完成。

```bash
git add -A
git commit -m "<本次改动说明>"
git push gpu main
git rev-parse HEAD
```

记录输出的完整 commit SHA。

如果希望 Codex 执行本地 Git 提交，需要在当次任务中明确说明“请提交这些修改”；默认情况下 Codex 只修改和测试，不自动提交。

### 4.4 服务器拉取并验证版本

以下操作在**SSH 终端**完成。

```bash
ssh atec-gpu
cd /root/gpufree-data/task-e-work/ATEC_TaskE_Standalone
git status --short
```

`git status --short` 必须为空。如果出现源码修改，先不要拉取，执行以下命令并保存输出：

```bash
git diff
git status --short
```

工作区为空时：

```bash
git pull --ff-only
git rev-parse HEAD
git status --short
```

本地和服务器的 `git rev-parse HEAD` 完全相同，并且服务器 `git status` 为空，即可证明 Git 管理的代码、配置、资产和基线权重一致。

editable install 通常不需要每次重装。只有 `setup.py`、`pyproject.toml`、包名、目录结构或依赖发生变化时才重新执行安装。

## 5. 每次远程实验的记录 SOP

### 5.1 创建独立实验目录

以下操作在**SSH 终端**完成。

```bash
cd /root/gpufree-data/task-e-work/ATEC_TaskE_Standalone

export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$(git rev-parse --short HEAD)"
RESULT_DIR="/root/gpufree-data/task-e-results/$RUN_ID"

mkdir -p "$RESULT_DIR"
```

记录代码、环境和策略权重：

```bash
git rev-parse HEAD > "$RESULT_DIR/git_commit.txt"
git status --porcelain > "$RESULT_DIR/git_status.txt"

python --version > "$RESULT_DIR/python_version.txt"
python -m pip freeze > "$RESULT_DIR/pip_freeze.txt"
nvidia-smi > "$RESULT_DIR/nvidia_smi.txt"

sha256sum \
  atec_robot_model/baseline/act/policy.pt \
  > "$RESULT_DIR/policy.sha256"
```

`git_status.txt` 应为空。

### 5.2 运行实验并记录退出码

以下操作在**SSH 终端**完成。以下示例运行 500 步基线录像：

```bash
set -o pipefail

timeout --signal=INT --kill-after=30s 600s \
  python -u scripts/play_atec_task.py \
    --task ATEC-TaskE-Piper \
    --num_envs 1 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 500 \
  2>&1 | tee "$RESULT_DIR/run.log"

RUN_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$RUN_EXIT" \
  | tee "$RESULT_DIR/exit_code.txt"
```

保存视频：

```bash
mkdir -p "$RESULT_DIR/videos"

find logs/videos \
  -type f \
  -name '*.mp4' \
  -not -path '*/.ipynb_checkpoints/*' \
  -exec cp -a {} "$RESULT_DIR/videos/" \;
```

生成结果校验文件：

```bash
(
  cd "$RESULT_DIR"

  find . \
    -type f \
    ! -name SHA256SUMS \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
)
```

查看本次实验编号和文件：

```bash
echo "$RUN_ID"
find "$RESULT_DIR" -maxdepth 2 -type f -printf '%p %s bytes\n'
```

## 6. 把实验结果拉回本地

### 6.1 下载指定实验

以下操作在**本地终端**完成。

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone
mkdir -p remote-results

RUN_ID="<服务器输出的RUN_ID>"

rsync -av --progress \
  "atec-gpu:/root/gpufree-data/task-e-results/$RUN_ID/" \
  "remote-results/$RUN_ID/"
```

验证传输完整性：

```bash
(
  cd "remote-results/$RUN_ID"
  sha256sum -c SHA256SUMS
)
```

全部显示 `OK`，说明日志、视频和环境记录下载完整。

查看结果：

```bash
cat "remote-results/$RUN_ID/git_commit.txt"
cat "remote-results/$RUN_ID/exit_code.txt"
tail -n 200 "remote-results/$RUN_ID/run.log"
find "remote-results/$RUN_ID/videos" -type f -ls
```

## 7. 日常操作速查

### 7.1 本地修改、提交和推送

在**本地终端**执行：

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone
git status --short

# 修改代码并运行本地测试

git diff
git add -A
git commit -m "<本次改动说明>"
git push gpu main
git rev-parse HEAD
```

### 7.2 服务器更新代码并运行

在**SSH 终端**执行：

```bash
cd /root/gpufree-data/task-e-work/ATEC_TaskE_Standalone
git status --short
git pull --ff-only
git rev-parse HEAD

# 创建 RUN_ID 并运行远程实验
```

### 7.3 本地下载实验结果

在**本地终端**执行：

```bash
cd /home/chen/CUHK/ROSE5810/ATEC_TaskE_Standalone

RUN_ID="<服务器输出的RUN_ID>"

rsync -av --progress \
  "atec-gpu:/root/gpufree-data/task-e-results/$RUN_ID/" \
  "remote-results/$RUN_ID/"

(
  cd "remote-results/$RUN_ID"
  sha256sum -c SHA256SUMS
)
```

## 8. 异常处理

### 8.1 服务器工作区不是干净状态

不要执行 `git reset --hard`，也不要强行覆盖。先在服务器运行：

```bash
cd /root/gpufree-data/task-e-work/ATEC_TaskE_Standalone
git status --short
git diff > /root/gpufree-data/task-e-results/remote_uncommitted.patch
```

保存补丁后再分析改动来源。

### 8.2 `git pull --ff-only` 失败

停止操作并检查：

```bash
git status --short
git log --oneline --decorate --graph -10
```

不要使用 `git reset --hard` 或强制推送处理未知分叉。

### 8.3 `scripts/check_handoff.py` 在修改源码后报哈希错误

原始 `SHA256SUMS` 记录的是交接快照。修改被记录的源码后，校验失败属于预期行为，不代表代码必然错误。

开发阶段使用以下信息记录版本：

- Git commit SHA；
- `git status --porcelain`；
- 策略权重 SHA256；
- 每次实验的 `SHA256SUMS`。

后续可以将交接检查改为分别校验不可变资产和开发源码。

## 9. 最终交付

开发期间的新 checkpoint 保存在服务器结果目录中，不进入 Git。确定最终权重后：

1. 将选定 checkpoint 下载到本地；
2. 记录其 SHA256；
3. 放入正式交付目录；
4. 更新 `demo/solution.py` 或权重路径；
5. 提交最终代码和权重；
6. 用最终 commit SHA 在服务器全新进程中复现；
7. 保存最终视频、评测记录、依赖快照和启动命令。
