#!/bin/bash

export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/examples
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
## Prefer the repo's vendored timm (with interfuser models) over site-packages
# Add the project-local package root so `import timm` resolves to ./interfuser/timm
export PYTHONPATH="interfuser:$PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:.

# Ensure CUDA/cuDNN component libraries from the active conda env are visible when launched via VirtualGL
if [ -n "$CONDA_PREFIX" ]; then
	# Base conda lib path
	export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
	# Add NVIDIA component libs shipped as Python wheels (cudnn, cublas, etc.)
	for p in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/*/lib; do
		if [ -d "$p" ]; then
			export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH}"
		fi
	done
fi

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port
export TM_PORT=2500 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs

# ========== 模型路径配置 ==========
# 修改这里可以快速切换不同的模型文件
# 可选值: interfuser.pth.tar, interfuser02.pth.tar 等
export INTERFUSER_MODEL_PATH="leaderboard/team_code/interfuser.pth.tar"

# ========== Python 性能优化 ==========
export PYTHONOPTIMIZE=1              # 启用 Python 优化
export PYTHONUNBUFFERED=1            # 禁用输出缓冲，减少卡顿
export MALLOC_TRIM_THRESHOLD_=100000 # 更积极的内存释放

# ========== VGL 和 OpenGL 稳定性优化 ==========
export VGL_READBACK=sync        # 同步回读，避免 OpenGL 上下文丢失
export VGL_LOGO=0               # 禁用 VGL logo
export VGL_SPOIL=0              # 禁用 spoiling，提高稳定性
export __GL_SYNC_TO_VBLANK=0    # 禁用垂直同步
export __GL_MaxFramesAllowed=1   # 限制帧缓冲队列

# ========== Interfuser 显示设置 ==========
# export INTERFUSER_HEADLESS=1    # 禁用 pygame 窗口显示，避免 segfault

export ROUTES=leaderboard/data/training_routes/routes_town01_tiny.xml
#export ROUTES=leaderboard/data/42routes/42routes.xml
#export ROUTES=leaderboard/data/training_routes/routes_town02_long.xml
export TEAM_AGENT=leaderboard/team_code/interfuser_agent.py # agent
export TEAM_CONFIG=leaderboard/team_code/interfuser_config.py # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/interfuser_result.json # results file
#export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
export SAVE_PATH=data/expert # path for saving episodes while evaluating
# When True, the evaluator won't change the current map.
# Set to False so it can automatically load the required town for each route.
export RESUME=False


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
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
