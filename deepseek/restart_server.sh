#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-8888}"
LOG_FILE="$SCRIPT_DIR/deepseek_server.out"
SERVER_FILE="$SCRIPT_DIR/deepseek_responses_api_server.py"

old_pids="$(lsof -ti tcp:"$PORT" || true)"
if [[ -n "$old_pids" ]]; then
  echo "停止端口 $PORT 上的旧服务: $old_pids"
  kill $old_pids
  sleep 1
fi

echo "启动 DeepSeek Responses 服务: http://localhost:$PORT"
cd "$ROOT_DIR"
PORT="$PORT" nohup python3 "$SERVER_FILE" > "$LOG_FILE" 2>&1 &
new_pid="$!"

for _ in {1..20}; do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "服务已启动，PID: $new_pid"
    echo "健康检查: http://localhost:$PORT/health"
    echo "日志文件: $LOG_FILE"
    exit 0
  fi
  sleep 0.5
done

echo "服务启动失败，请查看日志: $LOG_FILE" >&2
exit 1
