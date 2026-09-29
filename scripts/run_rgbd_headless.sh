#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
run_name="${1:-rgbd_$(date +%Y%m%d_%H%M%S)}"
run_seed="${2:-42}"
run_dir="$project_dir/logs/$run_name"
mkdir -p "$run_dir"
if [[ -e "$run_dir/run.log" ]]; then
    printf 'Run already exists: %s\n' "$run_dir" >&2
    exit 2
fi
tar -czf "$run_dir/source.tar.gz" demo/solution.py demo/solution_rgbd.py \
    demo/piper_kinematics.py demo/rgbd_vbc_alore.py demo/rgbd/perception.py \
    scripts/play_atec_task.py scripts/run_rgbd_headless.sh
export PYTHONPATH="$project_dir/source/atec_rl_lab:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ATEC_FK_DIAGNOSTICS=1
export ATEC_CALIBRATION_OUTPUT="$run_dir"
exec /opt/conda/envs/isaaclab/bin/python -u scripts/play_atec_task.py \
    --headless --enable_cameras --video --camera_video --video_length 5000 \
    --seed "$run_seed" --output_dir "$run_dir" > "$run_dir/run.log" 2>&1
