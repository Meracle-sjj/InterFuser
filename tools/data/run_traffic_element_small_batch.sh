#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 PROFILED_TRAFFIC_LIGHT_ROUTE.xml PROFILED_HARD_NEGATIVE_ROUTE.xml" >&2
  exit 64
fi

cd "$(dirname "$0")/../.."

PORT=2400
TM_PORT=8400
GPU=2
MAX_ROUTES=4
MAX_FRAMES=2000
MAX_BYTES=$((2 * 1024 * 1024 * 1024))
PYTHON_BIN=/data1/shijj/conda_envs/interfuser_origin/bin/python
CARLA_ROOT=${CARLA_ROOT:-$PWD/carla}
RUN_ID=${BATCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
DATA_ROOT="data/traffic_element_small_batch/${RUN_ID}"
RESULT_ROOT="results/traffic_element_small_batch/${RUN_ID}"
RUN_LOG="${RESULT_ROOT}/run.log"
STATUS_FILE="${RESULT_ROOT}/route_status.tsv"

PROFILED_TRAFFIC_ROUTE=$1
PROFILED_HARD_NEGATIVE_ROUTE=$2

for route_file in "${PROFILED_TRAFFIC_ROUTE}" "${PROFILED_HARD_NEGATIVE_ROUTE}"; do
  if [ ! -f "${route_file}" ]; then
    echo "route file not found: ${route_file}" >&2
    exit 66
  fi
done

export PYTHONPATH="$PWD/interfuser:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/examples:${CARLA_ROOT}/PythonAPI/carla:$PWD/leaderboard:$PWD/leaderboard/team_code:$PWD/scenario_runner:$PWD"
PHASE_SCHEMA=$("${PYTHON_BIN}" -c 'from traffic_element_labels import SCHEMA_VERSION; print(SCHEMA_VERSION)')
VIEW_SCHEMA=$("${PYTHON_BIN}" -c 'from traffic_element_projection import IMAGE_SCHEMA_VERSION; print(IMAGE_SCHEMA_VERSION)')
[ "${PHASE_SCHEMA}" = 2 ] || { echo "expected phase schema 2" >&2; exit 65; }
[ "${VIEW_SCHEMA}" = 3 ] || { echo "expected evidence schema 3" >&2; exit 65; }
"${PYTHON_BIN}" tools/data/check_leaderboard_stop_target_geometry.py --help \
  >/dev/null || { echo "geometry parity tool unavailable" >&2; exit 65; }

if [ "${TRAFFIC_BATCH_VALIDATE_ONLY:-0}" = 1 ]; then
  echo "traffic batch validation passed: phase=${PHASE_SCHEMA} evidence=${VIEW_SCHEMA}"
  exit 0
fi

if [ -e "${DATA_ROOT}" ] || [ -e "${RESULT_ROOT}" ]; then
  echo "refusing to reuse existing batch run: ${RUN_ID}" >&2
  exit 73
fi
if [ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]; then
  echo "CARLA runtime not executable: ${CARLA_ROOT}/CarlaUE4.sh" >&2
  exit 66
fi

mkdir -p "${DATA_ROOT}" "${RESULT_ROOT}/logs" "${RESULT_ROOT}/checkpoints" \
  "${RESULT_ROOT}/audits" "${RESULT_ROOT}/geometry" "${RESULT_ROOT}/routes"
: > "${RUN_LOG}"
printf 'route\tbackground\tevaluator_exit\tgeometry_exit\tphase2_audit_exit\tview_audit_exit\tacceptance_exit\ttotal_frames\ttotal_bytes\n' \
  > "${STATUS_FILE}"

FIXED_ROUTE13="${RESULT_ROOT}/routes/route_13_Town03_Opt.xml"
FIXED_ROUTE36="${RESULT_ROOT}/routes/route_36_Town03_Opt.xml"
if ! "${PYTHON_BIN}" - leaderboard/data/42routes/42routes.xml \
  "${RESULT_ROOT}/routes" <<'PY'
import copy
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
source_root = ET.parse(source).getroot()
for route_id in ("13", "36"):
    source_route = next(
        (route for route in source_root.findall("route") if route.get("id") == route_id),
        None,
    )
    if source_route is None:
        raise SystemExit(f"route {route_id} is absent from {source}")
    route = copy.deepcopy(source_route)
    for weather in list(route.findall("weather")):
        route.remove(weather)
    town = route.attrib["town"].split("/")[-1]
    town = town if town.endswith("_Opt") else town + "_Opt"
    route.set("town", town)
    root = ET.Element("routes")
    root.append(route)
    ET.indent(root, space="  ")
    target = output_dir / f"route_{int(route_id):02d}_{town}.xml"
    ET.ElementTree(root).write(target, encoding="UTF-8", xml_declaration=True)
    print(target)
PY
then
  exit 65
fi

ROUTE_FILES=(
  "${FIXED_ROUTE13}"
  "${FIXED_ROUTE36}"
  "${PROFILED_TRAFFIC_ROUTE}"
  "${PROFILED_HARD_NEGATIVE_ROUTE}"
)
BACKGROUND_COUNTS=(0 0 20 20)
if [ "${#ROUTE_FILES[@]}" -ne "${MAX_ROUTES}" ]; then
  echo "internal error: bounded batch must contain exactly four routes" >&2
  exit 70
fi

export CUDA_VISIBLE_DEVICES=${GPU}
export CHALLENGE_TRACK_CODENAME=SENSORS
export PYTHONUNBUFFERED=1
export INTERFUSER_REUSE_CURRENT_WORLD=1
export INTERFUSER_BG_WARMUP_TICKS=40
export WEATHER_ID=0

count_frames() {
  find "${DATA_ROOT}" -path '*/traffic_element_views/*.json' -type f 2>/dev/null \
    | wc -l | tr -d ' '
}

count_bytes() {
  du -sb "${DATA_ROOT}" 2>/dev/null | awk '{print $1}'
}

port_2000_pids() {
  ps -eo pid=,args= \
    | awk '/CarlaUE4/ && /--world-port=2000/ && !/awk/ {print $1}' \
    | sort -n | tr '\n' ',' | sed 's/,$//'
}

kill_evaluator_2400() {
  local pids
  pids=$(ps -eo pid=,args= | awk \
    '/leaderboard_evaluator.py/ && (/--port=2400/ || /--port 2400/) && !/awk/ {print $1}')
  if [ -n "${pids}" ]; then
    kill ${pids} 2>/dev/null || true
    sleep 2
    kill -9 ${pids} 2>/dev/null || true
  fi
}

kill_carla_2400() {
  local pids
  pids=$(ps -eo pid=,args= \
    | awk '/CarlaUE4/ && /--world-port=2400/ && !/awk/ {print $1}')
  if [ -n "${pids}" ]; then
    kill ${pids} 2>/dev/null || true
    sleep 3
    kill -9 ${pids} 2>/dev/null || true
  fi
}

cleanup() {
  kill_evaluator_2400
  kill_carla_2400
}
trap cleanup EXIT INT TERM

start_carla() {
  local carla_log="${RESULT_ROOT}/logs/carla_2400.log"
  kill_evaluator_2400
  kill_carla_2400
  "${CARLA_ROOT}/CarlaUE4.sh" --world-port=${PORT} -quality-level=Low -RenderOffScreen \
    > "${carla_log}" 2>&1 &
  echo "$!" > "${RESULT_ROOT}/logs/carla_2400.pid"

  local attempt
  for attempt in $(seq 1 45); do
    if "${PYTHON_BIN}" -c \
      "import carla; c=carla.Client('127.0.0.1', ${PORT}); c.set_timeout(2.0); print(c.get_world().get_map().name)" \
      >> "${RUN_LOG}" 2>&1; then
      echo "[CARLA_READY] attempt=${attempt}" >> "${RUN_LOG}"
      return 0
    fi
    sleep 2
  done
  echo "[CARLA_START_FAILED]" >> "${RUN_LOG}"
  tail -80 "${carla_log}" >> "${RUN_LOG}" 2>/dev/null || true
  return 1
}

route_map() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
import xml.etree.ElementTree as ET

route = ET.parse(sys.argv[1]).getroot().find("route")
if route is None:
    raise SystemExit("route XML has no route element")
town = route.attrib["town"].split("/")[-1]
print(town if town.endswith("_Opt") else town + "_Opt")
PY
}

preload_route_map() {
  local route_file=$1
  local town
  town=$(route_map "${route_file}") || return 1
  "${PYTHON_BIN}" - "${town}" "${PORT}" >> "${RUN_LOG}" 2>&1 <<'PY'
import sys
import time

import carla

town = sys.argv[1]
port = int(sys.argv[2])
client = carla.Client("127.0.0.1", port)
client.set_timeout(60.0)
started = time.time()
world = client.load_world(town)
print(
    "[PRELOAD] requested={} loaded={} wall_s={:.3f}".format(
        town, world.get_map().name, time.time() - started
    ),
    flush=True,
)
PY
}

check_route_geometry() {
  local route_file=$1
  local town output
  town=$(route_map "${route_file}") || return 1
  output="${RESULT_ROOT}/geometry/${town}.json"
  if [ -f "${output}" ]; then
    return 0
  fi
  "${PYTHON_BIN}" tools/data/check_leaderboard_stop_target_geometry.py \
    --host 127.0.0.1 --port "${PORT}" --output "${output}" "${town}" \
    >> "${RUN_LOG}" 2>&1
}

validate_acceptance() {
  local phase_audit=$1
  local view_audit=$2
  "${PYTHON_BIN}" - "${phase_audit}" "${view_audit}" <<'PY'
import json
from pathlib import Path
import sys

phase = json.loads(Path(sys.argv[1]).read_text())
view = json.loads(Path(sys.argv[2]).read_text())
phase_fields = {
    "valid_stop_targets",
    "unknown_stop_targets",
    "unknown_reasons",
    "primary_stop_target_frames",
    "forbidden_stop_occurrences",
}
view_fields = {
    "projected_stop_boundaries",
    "projected_stop_corridors",
    "painted_line_candidates",
    "verified_painted_lines",
    "lidar_available_targets",
    "lidar_unknown_targets",
    "error_frames",
    "forbidden_stop_occurrences",
}
missing = sorted(phase_fields - set(phase)) + sorted(view_fields - set(view))
if missing:
    raise SystemExit("acceptance fields missing: {}".format(missing))
if phase["forbidden_stop_occurrences"] or view["forbidden_stop_occurrences"]:
    raise SystemExit("forbidden stop occurrences must be zero")
if view["error_frames"]:
    raise SystemExit("evidence error_frames must be zero")
target_count = phase["valid_stop_targets"] + phase["unknown_stop_targets"]
lidar_count = view["lidar_available_targets"] + view["lidar_unknown_targets"]
if lidar_count != target_count:
    raise SystemExit(
        "lidar target evidence mismatch: phase={} evidence={}".format(
            target_count, lidar_count
        )
    )
if phase["unknown_stop_targets"] and not phase["unknown_reasons"]:
    raise SystemExit("unknown stop targets require reason counts")
if phase["valid_stop_targets"] and (
    not view["projected_stop_boundaries"]
    or not view["projected_stop_corridors"]
):
    raise SystemExit("valid targets require projected camera boundary and corridor evidence")
PY
}

monitor_caps() {
  local process_group=$1
  local marker=$2
  while kill -0 "${process_group}" 2>/dev/null; do
    sleep 10
    local frames bytes
    frames=$(count_frames)
    bytes=$(count_bytes)
    if [ "${frames}" -ge "${MAX_FRAMES}" ] || [ "${bytes}" -ge "${MAX_BYTES}" ]; then
      printf 'frames=%s bytes=%s\n' "${frames}" "${bytes}" > "${marker}"
      kill -INT -- "-${process_group}" 2>/dev/null || true
      return 0
    fi
  done
}

run_route() {
  local route_file=$1
  local background=$2
  local key
  key=$(basename "${route_file}" .xml)
  local output_root="${DATA_ROOT}/${key}_bg${background}"
  local route_log="${RESULT_ROOT}/logs/${key}_bg${background}.log"
  local checkpoint="${RESULT_ROOT}/checkpoints/${key}_bg${background}.json"
  local cap_marker="${RESULT_ROOT}/${key}_cap_reached.txt"

  echo "[ROUTE_START] key=${key} background=${background}" >> "${RUN_LOG}"
  if ! preload_route_map "${route_file}"; then
    printf '%s\t%s\t97\t-\t-\t-\t-\t%s\t%s\n' "${key}" "${background}" \
      "$(count_frames)" "$(count_bytes)" >> "${STATUS_FILE}"
    return 97
  fi

  local geometry_exit=0
  check_route_geometry "${route_file}" || geometry_exit=$?
  if [ "${geometry_exit}" -ne 0 ]; then
    printf '%s\t%s\t-\t%s\t-\t-\t-\t%s\t%s\n' \
      "${key}" "${background}" "${geometry_exit}" \
      "$(count_frames)" "$(count_bytes)" >> "${STATUS_FILE}"
    return 96
  fi

  mkdir -p "${output_root}"
  setsid timeout --signal=INT --kill-after=30s 600s env \
    INTERFUSER_BG_VEHICLES="${background}" \
    SAVE_PATH="${output_root}" \
    ROUTES="${route_file}" \
    "${PYTHON_BIN}" leaderboard/leaderboard/leaderboard_evaluator.py \
      --host=127.0.0.1 \
      --port=${PORT} \
      --trafficManagerPort=${TM_PORT} \
      --trafficManagerSeed=0 \
      --carlaProviderSeed=2000 \
      --debug=0 \
      --routes="${route_file}" \
      --scenarios=leaderboard/data/42routes/42scenarios.json \
      --repetitions=1 \
      --agent=leaderboard/team_code/interfuser_collector_complete.py \
      --agent-config=leaderboard/team_code/interfuser_config.py \
      --track=SENSORS \
      --resume=False \
      --timeout=600 \
      --checkpoint="${checkpoint}" \
      > "${route_log}" 2>&1 &
  local evaluator_pid=$!
  monitor_caps "${evaluator_pid}" "${cap_marker}" &
  local monitor_pid=$!

  local evaluator_exit
  wait "${evaluator_pid}"
  evaluator_exit=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  if [ -f "${cap_marker}" ]; then
    evaluator_exit=90
  fi
  kill_evaluator_2400

  local phase2_exit=0
  local view_exit=0
  local acceptance_exit=0
  local phase_audit="${RESULT_ROOT}/audits/${key}_bg${background}_phase2.json"
  local view_audit="${RESULT_ROOT}/audits/${key}_bg${background}_evidence.json"
  "${PYTHON_BIN}" tools/data/audit_traffic_element_labels.py "${output_root}" \
    > "${phase_audit}" \
    2> "${RESULT_ROOT}/audits/${key}_bg${background}_phase2.err" \
    || phase2_exit=$?
  "${PYTHON_BIN}" tools/data/audit_traffic_element_views.py "${output_root}" \
    > "${view_audit}" \
    2> "${RESULT_ROOT}/audits/${key}_bg${background}_evidence.err" \
    || view_exit=$?
  if [ "${phase2_exit}" -eq 0 ] && [ "${view_exit}" -eq 0 ]; then
    validate_acceptance "${phase_audit}" "${view_audit}" \
      >> "${RUN_LOG}" 2>&1 || acceptance_exit=$?
  else
    acceptance_exit=1
  fi

  local frames bytes
  frames=$(count_frames)
  bytes=$(count_bytes)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${key}" "${background}" "${evaluator_exit}" "${geometry_exit}" \
    "${phase2_exit}" "${view_exit}" "${acceptance_exit}" \
    "${frames}" "${bytes}" >> "${STATUS_FILE}"
  echo "[ROUTE_END] key=${key} evaluator=${evaluator_exit} geometry=${geometry_exit} phase2=${phase2_exit} evidence=${view_exit} acceptance=${acceptance_exit} frames=${frames} bytes=${bytes}" \
    >> "${RUN_LOG}"

  if [ "${evaluator_exit}" -ne 0 ] || [ "${geometry_exit}" -ne 0 ] \
    || [ "${phase2_exit}" -ne 0 ] || [ "${view_exit}" -ne 0 ] \
    || [ "${acceptance_exit}" -ne 0 ]; then
    return 1
  fi
  return 0
}

PORT_2000_BASELINE=$(port_2000_pids)
echo "[BATCH_START] run_id=${RUN_ID} port2000_pids=${PORT_2000_BASELINE}" >> "${RUN_LOG}"
if ! start_carla; then
  exit 97
fi

overall_exit=0
for index in "${!ROUTE_FILES[@]}"; do
  frames=$(count_frames)
  bytes=$(count_bytes)
  if [ "${frames}" -ge "${MAX_FRAMES}" ] || [ "${bytes}" -ge "${MAX_BYTES}" ]; then
    echo "[BATCH_CAP_BEFORE_ROUTE] index=${index} frames=${frames} bytes=${bytes}" \
      >> "${RUN_LOG}"
    overall_exit=90
    break
  fi
  if ! run_route "${ROUTE_FILES[$index]}" "${BACKGROUND_COUNTS[$index]}"; then
    overall_exit=$?
    [ "${overall_exit}" -eq 0 ] && overall_exit=1
    break
  fi
done

cleanup
trap - EXIT INT TERM
PORT_2000_FINAL=$(port_2000_pids)
if [ "${PORT_2000_FINAL}" != "${PORT_2000_BASELINE}" ]; then
  echo "[PORT_2000_CHANGED] before=${PORT_2000_BASELINE} after=${PORT_2000_FINAL}" \
    >> "${RUN_LOG}"
  overall_exit=98
fi
echo "[BATCH_END] exit=${overall_exit} frames=$(count_frames) bytes=$(count_bytes) port2000_pids=${PORT_2000_FINAL}" \
  >> "${RUN_LOG}"
cat "${STATUS_FILE}"
exit "${overall_exit}"
