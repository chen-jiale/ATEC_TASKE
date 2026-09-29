# Task E 云实例启动清单

建议镜像：IsaacSim5.1 + IsaacLab-2.3.2 pip 安装版。
项目明确标注 Lab 2.3.2；Sim 5.1/Python 3.11 是候选配套组合，仍需场景实测。

## 开机前

- 确认单卡 RTX 4090 24GB、至少 64GB RAM、足够的持久磁盘空间。
- 查看镜像说明，记录已有 Python/Conda 环境的激活命令、Isaac Lab 路径、远程桌面入口。
- 向平台确认 Vulkan/RTX 相机渲染、驱动兼容性，以及 SSH 是否可用。
- 确认计费起点、关机操作、关机后的存储费、释放实例后的数据保留规则。
- 确认系统盘和数据盘路径与容量，不要套用其他平台的路径。
- 本地 VS Code 安装 Microsoft 的 Remote - SSH 扩展。
- 不在本地安装 Isaac Sim，也不预先下载任意 CUDA/PyTorch wheel。

## 上传

本地终端操作，将大写占位符替换为平台给出的实际值：

```bash
ssh -p SSH_PORT USER@HOST
```

首次核对服务器指纹。在服务器持久目录下创建工作目录后，从本地上传压缩包：

```bash
scp -P SSH_PORT /tmp/ATEC_TaskE_Standalone_upload.tar.gz USER@HOST:REMOTE_WORKDIR/
```

服务器终端：

```bash
cd REMOTE_WORKDIR
tar -xzf ATEC_TaskE_Standalone_upload.tar.gz
cd ATEC_TaskE_Standalone
```

## 确认镜像已有环境

先按镜像说明激活已有 Isaac 环境，再执行：

```bash
nvidia-smi
free -h
df -h .
which python
python --version
python -m pip show isaacsim isaaclab torch torchvision
python -c 'import torch; print("torch:", torch.__version__, "CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
python scripts/check_handoff.py
```

pip show 找不到 Isaac Lab 时，先核对镜像的源码安装方式和环境，不要直接重装。
CUDA 可用不代表 Vulkan 或相机渲染可用。

## 安装项目

确认环境正确后，保存安装前版本，并约束已有 torch/torchvision/numpy，避免安装项目时无意替换这些核心包：

```bash
mkdir -p logs/cloud_setup
python -m pip freeze > logs/cloud_setup/before.txt
python -c 'from importlib.metadata import version; from pathlib import Path; Path("logs/cloud_setup/core-constraints.txt").write_text("\n".join(f"{name}=={version(name)}" for name in ("torch", "torchvision", "numpy")) + "\n")'
python -m pip install --dry-run -r requirements.txt -c logs/cloud_setup/core-constraints.txt
```

检查 dry-run 结果，无核心依赖冲突后再执行：

```bash
python -m pip install -r requirements.txt -c logs/cloud_setup/core-constraints.txt
python -m pip install --no-deps --no-build-isolation -e ./source/atec_rl_lab
export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pip check
python -m pip freeze > logs/cloud_setup/after.txt
```

若提示缺少构建工具，先检查 setuptools、wheel、toml；若依赖冲突，保存错误并逐项处理，不要通过整体升级解决。

## 先验收，再开始长任务

有远程桌面时，在该桌面的终端运行：

```bash
python scripts/view_task_e.py --num_envs 1 --enable_cameras
```

此脚本持续运行，查看场景后用 Ctrl+C 停止。无桌面时可加 --headless，但仅启动成功不能证明相机画面正常。

后台进行一次有限步数的基线录像测试：

```bash
python scripts/play_atec_task.py --task ATEC-TaskE-Piper --num_envs 1 --headless --enable_cameras --video --video_length 500
```

500 步为 10 秒仿真时间，只验收启动、画面、动作、录像，不代表完整任务成功；程序可能因环境终止而提前结束。视频输出位于 logs/videos/ATEC-TaskE-Piper/play。

验收要求：资产正常、相机画面正常、机器人动作输出正常、视频可播放、没有显存不足或渲染设备错误。首次启动可能需要下载资源和编译着色器，保留日志判断进度。

## 停机与保存

- 长任务使用 tmux；关闭 SSH/VS Code 不会停止服务器计费。
- 结束后在平台控制台关机，并确认状态与计费规则。
- 权重、配置、日志与视频单独备份；镜像包含哪些磁盘以算力自由说明为准。
- 保存通过相机测试的环境镜像，记录软件版本和宿主驱动版本。
- 当前采集器有成功终止丢弃问题，修复前不要直接启动大批量三物体采集。
