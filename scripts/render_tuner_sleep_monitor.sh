#!/usr/bin/env bash
set -euo pipefail

ROOT="/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/render_tuner"
LOG="/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/logs/render_tuner_sleep_monitor.log"
STATE="/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/runtime/full20_scene_20260404/logs/render_tuner_sleep_monitor_state.json"

count=0
last=""
if ls "$ROOT"/iter_*/obj_*_yaw*/reviews/*_trace.json >/dev/null 2>&1; then
  last=$(ls -t "$ROOT"/iter_*/obj_*_yaw*/reviews/*_trace.json | head -n 1)
fi

printf '{"status":"running","sleep_count":0,"last_trace":"%s"}\n' "$last" > "$STATE"
echo "[$(date '+%F %T')] monitor-start last_trace=$last" >> "$LOG"

while [ $count -lt 50 ]; do
  count=$((count + 1))
  sleep 30

  new=""
  if ls "$ROOT"/iter_*/obj_*_yaw*/reviews/*_trace.json >/dev/null 2>&1; then
    new=$(ls -t "$ROOT"/iter_*/obj_*_yaw*/reviews/*_trace.json | head -n 1)
  fi

  gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')
  echo "[$(date '+%F %T')] sleep#$count latest=$new gpu=$gpu" >> "$LOG"
  printf '{"status":"running","sleep_count":%d,"last_trace":"%s","gpu":"%s"}\n' "$count" "$new" "$gpu" > "$STATE"

  if [ -n "$new" ] && [ "$new" != "$last" ]; then
    echo "[$(date '+%F %T')] new-trace-detected sleep#$count trace=$new" >> "$LOG"
    printf '{"status":"new_trace_detected","sleep_count":%d,"last_trace":"%s"}\n' "$count" "$new" > "$STATE"
    exit 0
  fi

  last="$new"
done

echo "[$(date '+%F %T')] monitor-finished sleep_count=$count" >> "$LOG"
printf '{"status":"finished_50_sleeps","sleep_count":%d,"last_trace":"%s"}\n' "$count" "$last" > "$STATE"
