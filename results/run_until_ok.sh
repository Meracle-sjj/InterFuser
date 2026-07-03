#!/bin/bash
# 自动重试跑 eval:segfault/CARLA崩了就重启重来,直到车辆真的在行驶。最多 5 轮。
source /home/shijj/miniconda3/etc/profile.d/conda.sh
conda activate interfuser_origin
PY=/data1/shijj/conda_envs/interfuser_origin/bin/python
CARLA=/home/shijj/interfuser_origin/carla
ROOT=/home/shijj/interfuser_origin
LOG=$ROOT/results/eval_tm.log
PORT=2155; TM=2255; GPU=5

kill_carla() { pkill -9 -f "CarlaUE4.sh --world-port=$PORT" 2>/dev/null; pkill -9 -f "CarlaUE4-Linux-Shipping CarlaUE4 --world-port=$PORT" 2>/dev/null; sleep 4; }
kill_eval()  { pkill -9 -f leaderboard_evaluator 2>/dev/null; sleep 2; }

start_carla() {
  kill_carla; rm -f /dev/shm/carla_* 2>/dev/null
  cd "$CARLA"
  setsid bash -c "./CarlaUE4.sh --world-port=$PORT -graphicsadapter=$GPU -quality-level=Low -RenderOffScreen" > "$ROOT/results/carla_retry.log" 2>&1 &
  for i in $(seq 1 30); do
    $PY -c "import carla; c=carla.Client('127.0.0.1',$PORT); c.set_timeout(2.0); c.get_server_version()" 2>/dev/null && return 0
    sleep 4
  done
  return 1
}

for attempt in $(seq 1 5); do
  echo "===== ATTEMPT $attempt ====="
  kill_eval
  start_carla || { echo "carla failed to start, retry"; continue; }
  echo "carla up, launching eval..."
  cd "$ROOT"; rm -f "$LOG"
  ROUTES=leaderboard/data/training_routes/routes_town01_tiny3.xml \
  INTERFUSER_LANE_CENTER_STEER_GAIN=0.5 \
  INTERFUSER_TM_HYBRID_PHYSICS=0 \
  INTERFUSER_BG_VEHICLES=20 \
  PORT=$PORT TM_PORT=$TM CUDA_DEV=0 \
  setsid bash leaderboard/scripts/run_origin_eval.sh > "$LOG" 2>&1 &
  # 等它加载+跑,检查是否在行驶
  for w in $(seq 1 20); do
    sleep 6
    NEWDIR=$(ls -td $ROOT/data/eval_origin/routes_town01_tiny3_* 2>/dev/null | head -1)
    CSV="$NEWDIR/control.csv"
    if [ -f "$CSV" ]; then
      ROWS=$(wc -l < "$CSV" 2>/dev/null)
      SPD=$($PY -c "import csv; r=list(csv.reader(open('$CSV'))); print(round(sum(float(x[1]) for x in r[1:min(40,len(r))] if x[1] not in ('','nan'))/max(1,min(39,len(r)-1)),2))" 2>/dev/null)
      echo "  rows=$ROWS early_speed=$SPD"
      if [ "${ROWS:-0}" -gt 30 ]; then
        echo "DRIVING OK (attempt $attempt)"; echo "$NEWDIR" > $ROOT/results/last_good_rundir.txt; exit 0
      fi
    fi
    # 如果 segfault 了,跳出等待
    grep -q "Segmentation fault" "$LOG" 2>/dev/null && { echo "  segfaulted, will retry"; break; }
    pgrep -f leaderboard_evaluator >/dev/null 2>&1 || { echo "  eval exited early"; break; }
  done
  echo "attempt $attempt did not drive cleanly; restarting carla"
done
echo "FAILED after 5 attempts"