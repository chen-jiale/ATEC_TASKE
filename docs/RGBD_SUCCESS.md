# RGB-D 自主策略验收

## 已完成的实测

2026-09-29，通过本地 `ssh ATEC_GPU` 在 RTX 4090 上运行 Isaac Sim 5.1 / Isaac Lab headless 评测。

- 云端工程：`/root/gpufree-data/ATEC_TaskE_Standalone`。
- 来源：`/root/gpufree-share/ATEC_TaskE_Standalone_upload.tar.gz`，已解压到数据盘。
- 抓取顺序：香蕉 → 芥末瓶 → 糖盒。
- 策略：RGB-D 几何定位、自主 FK/解析雅可比/阻尼 IK、闭环抓放状态机；不加载 ACT 权重。
- 首次完整成功：`logs/rgbd_clearance42`，种子 42，得分 **18/18**，仿真耗时 **63.74 秒**。
- 同策略复跑：`logs/rgbd_confirm42`，种子 42，再次 **18/18**，仿真耗时 **63.76 秒**，终止原因明确记录为 `basket_success`。
- 最终近景录像：`logs/rgbd_final_camera42`，种子 42，第三次 **18/18**，仿真耗时 **63.76 秒**，`basket_success=true`。
- 原生终止标志：`basket_success=true`；`time_out/illegal_contact/fall=false`。
- 浮点累计原值为 `17.999999597668648`，格式化显示为 18.00。

首次成功在自动重置前记录的三物体世界坐标：

| 物体 | X / m | Y / m | Z / m |
| --- | ---: | ---: | ---: |
| 糖盒 | 1.042265 | -0.325329 | 0.976148 |
| 芥末瓶 | 1.120442 | -0.318430 | 0.883475 |
| 香蕉 | 1.031122 | -0.384284 | 0.851905 |

三件物体同时满足原始篮子区域条件。首次日志的旧版 `termination_reason` 显示为 `terminated_unknown`；同一日志和 `result.json` 中捕获的原生 `basket_success=true` 才是具体终止原因。评测脚本已修复该展示问题。

## 策略与修复

`demo/solution.py` 导出 `demo/solution_rgbd.py` 的 `AlgSolution`。策略输入仅为标准 `proprio`、`video_rgb/video_depth` 和接口允许的累计分数。相机内外参和机器人运动学来自固定配置；仿真物体真值仅在评测脚本中写入独立诊断文件，不传入策略。

关键修复包括：放置后先上撤再回到安全姿态；糖盒夹爪沿短边闭合；芥末瓶使用固定可达倾角；完成抬升后才横移并提高瓶子搬运高度，避免碰倒糖盒；加入空抓有限重试与入篮分数门控。

解析雅可比在 100 组关节姿态上与有限差分对照，最大元素误差为 `3.07e-7`。数据盘中的 `env_cfg.py`、奖励和终止源码与原压缩包逐字节一致。未更改物理参数、物体分布、奖励或成功条件。

## 复现

在云端项目目录执行：

```bash
bash scripts/run_rgbd_headless.sh rgbd_new_run 42
```

每个运行名必须唯一。脚本保存 `run.log`、`result.json`、代码快照、RGB-D 诊断帧，以及策略外部 RGB 相机近景 `rgbd_success.mp4`。Python 环境路径为 `/opt/conda/envs/isaaclab/bin/python`。

原始默认 viewport 的远景成功录像已下载到本地 `output/rgbd_success_seed42/success.mp4`，同目录保留原始日志和结果。该文件为 H.264、1280×720、50 fps、3188 帧、63.76 秒，完整解码通过，云端与本地 SHA-256 相同：`91527386fe260d0c6d6d4905fd0418fb0884898559717fb6c7022c559aaccb79`。

**最终推荐观看：`output/rgbd_final/rgbd_success.mp4`。** 这是策略外部 RGB 相机的完整近景录像，H.264、640×480、50 fps、3190 帧、63.80 秒。完整解码与关键帧目视检查通过，云端与本地 SHA-256 均为 `53db1319dfb5be78cae757db923a55aaf797e14ad7296f3e2ee9d0e2a656cca4`。

同目录的 `result.json` 保存最终得分与重置前成功证据，`run.log` 保存终端输出，`source.tar.gz` 保存该次执行的策略和评测脚本，另有视频元数据、关键帧总览与末帧。录像按原生成功条件终止，不包含成功后额外静置测试。

这些结果证明已在该种子上完成整回合；不能视为覆盖随机位置的总体成功率。定位仍使用固定相机、颜色与空间区域先验，不是通用物体分割模型。
