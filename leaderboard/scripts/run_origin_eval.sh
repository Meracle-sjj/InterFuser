#!/bin/bash
# interfuser_origin — 干净基线 eval 脚本(5090 / CARLA 0.9.16)
# 模板来自现仓库 leaderboard/scripts/42_eval.sh,默认 checkpoint 改为 Phase3 最佳 model_20260121.tar。
set -euo pipefail

# ---- 激活专用环境(自包含,一条命令可跑)----
source /home/shijj/miniconda3/etc/profile.d/conda.sh
conda activate interfuser_origin

cd "$(dirname "$0")/../.."   # 回到 interfuser_origin 根

# ---- 推理 GPU(默认 4:余量大;可覆盖: CUDA_DEV=5 bash ... )----
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-4}

# ---- 模型 checkpoint(必须 export,agent 通过 os.environ 读)----
# 默认 Phase3 最佳(model_20260121.tar, DS68.4)。切回原版对照:
#   INTERFUSER_MODEL_PATH=/home/shijj/interfuser/leaderboard/team_code/interfuser.pth.tar CUDA_DEV=4 bash run_origin_eval.sh
export INTERFUSER_MODEL_PATH="${INTERFUSER_MODEL_PATH:-/home/shijj/interfuser/leaderboard/team_code/model_20260121.tar}"

# ---- CARLA / 路径 ----
export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
# PYTHONPATH:仓库内自带 timm 分叉放最前
export PYTHONPATH="interfuser:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/examples:${CARLA_ROOT}/PythonAPI/carla:${LEADERBOARD_ROOT}:${LEADERBOARD_ROOT}/team_code:${SCENARIO_RUNNER_ROOT}:."

# torch 2.10 / 5090 必需:带上 conda 的 nvidia 运行时库(cublas 等)
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  for p in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$p" ] && export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH}"
  done
fi

# ---- eval 参数 ----
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=${PORT:-2000}               # CARLA 服务器端口(可覆盖: PORT=2001 ...)
export TM_PORT=${TM_PORT:-2500}
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export ROUTES=${ROUTES:-leaderboard/data/training_routes/routes_town01_tiny3.xml}   # 可覆盖,如单路由加速
export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
export TEAM_AGENT=leaderboard/team_code/interfuser_agent.py
export TEAM_CONFIG=leaderboard/team_code/interfuser_config.py
export CHECKPOINT_ENDPOINT=results/origin_result.json
export SAVE_PATH=data/eval_origin      # 输出落 /data 大盘(origin 已软链)
export RECORD_PATH=""
export RESUME=False

export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=100000

echo "==== interfuser_origin eval ===="
echo "model : $INTERFUSER_MODEL_PATH"
echo "gpu   : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "route : $ROUTES"
echo "save  : $SAVE_PATH"
echo "================================"

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
  --scenarios=${SCENARIOS} \
  --routes=${ROUTES} \
  --repetitions=${REPETITIONS} \
  --track=${CHALLENGE_TRACK_CODENAME} \
  --checkpoint=${CHECKPOINT_ENDPOINT} \
  --agent=${TEAM_AGENT} \
  --agent-config=${TEAM_CONFIG} \
  --debug=${DEBUG_CHALLENGE} \
  --record=${RECORD_PATH} \
  --resume=${RESUME} \
  --port=${PORT} \
  --trafficManagerPort=${TM_PORT}
