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

# ========== GPU 配置 ==========
export CUDA_VISIBLE_DEVICES=1        # 使用 GPU 1（第二块 GPU）

# ========== 模型路径配置 ==========
# 修改这里可以快速切换不同的模型文件
# 可选值: interfuser.pth.tar, interfuser02.pth.tar 等
export INTERFUSER_MODEL_PATH="leaderboard/team_code/interfuser02.pth.tar"

# ========== Python 性能优化 ==========
export PYTHONOPTIMIZE=1              # 启用 Python 优化
export PYTHONUNBUFFERED=1            # 禁用输出缓冲，减少卡顿
export MALLOC_TRIM_THRESHOLD_=100000 # 更积极的内存释放

# ========== 无头模式（禁用可视化窗口但保持渲染逻辑）==========
export INTERFUSER_HEADLESS=1         # 禁用 pygame 窗口，但保留所有渲染计算
export PYGAME_HIDE_SUPPORT_PROMPT=1  # 隐藏 pygame 提示

# ========== OpenGL 和渲染禁用（防止 segmentation fault）==========
export SDL_VIDEODRIVER=dummy         # 使用虚拟视频驱动，禁用 X11/Wayland
export SDL_AUDIODRIVER=dummy         # 禁用音频驱动
export LIBGL_ALWAYS_INDIRECT=1       # 禁用直接渲染（可能导致 segfault）
export DISPLAY=                      # 清除 DISPLAY 防止 X11 连接
export PYOPENGL_PLATFORM=osmesa      # 使用 Mesa（软件渲染）代替硬件 OpenGL

export ROUTES=leaderboard/data/42routes/42routes.xml
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


# ========== 错误日志和调试 ==========
set -o pipefail  # 如果管道中任何命令失败，整个管道失败

# 捕获 segmentation fault 并输出更详细的日志
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

# 检查退出码
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] 脚本以退出码 $EXIT_CODE 退出"
    if [ $EXIT_CODE -eq 139 ]; then
        echo "[ERROR] 检测到 Segmentation Fault (core dumped)"
        echo "[HINT] 可能的原因："
        echo "  1. OpenGL/GPU 驱动不兼容"
        echo "  2. CARLA 渲染模块出现问题"
        echo "  3. 内存不足"
    fi
fi

exit $EXIT_CODE
