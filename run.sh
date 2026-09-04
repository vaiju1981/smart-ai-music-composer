#!/usr/bin/env bash
# Start/stop/status the local Phase 1 services:
#
#   valkey   — job queue broker (valkey-server, falls back to redis-server)
#   worker   — RQ worker that runs the composition pipeline (saimc-jobs)
#   api      — FastAPI backend that creates/queries jobs and serves artifacts
#
# The render-service (Node) is a one-shot CLI invoked by the worker's
# sheet stage, so there is nothing to daemonize for it.
#
# Usage: run.sh {start|stop|restart|status} [service...]
#   run.sh start            start everything
#   run.sh start worker api start only some services
#   run.sh status
#
# PID files and logs live under var/run/; job artifacts under var/jobs/.

set -u

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

RUN_DIR="$REPO_ROOT/var/run"
mkdir -p "$RUN_DIR"

VALKEY_URL="${SAIMC_VALKEY_URL:-valkey://127.0.0.1:6379/0}"
VALKEY_PORT="${SAIMC_VALKEY_PORT:-6379}"
API_HOST="${SAIMC_API_HOST:-127.0.0.1}"
API_PORT="${SAIMC_API_PORT:-8000}"
API_PORT_CHECK="$(printf '%s' "$API_PORT" | tr -cd '0-9')"
[ -n "$API_PORT_CHECK" ] && API_PORT="$API_PORT_CHECK"

export SAIMC_JOBS_DIR="${SAIMC_JOBS_DIR:-$REPO_ROOT/var/jobs}"
export SAIMC_VALKEY_URL="$VALKEY_URL"
# Prefer the repo's audited LGPL build over any GPL ffmpeg on PATH —
# the render audit gate would (correctly) reject the latter.
if [ -z "${SAIMC_RENDER_FFMPEG:-}" ] && [ -x "$REPO_ROOT/dist/ffmpeg/8.1.2/ffmpeg" ]; then
    export SAIMC_RENDER_FFMPEG="$REPO_ROOT/dist/ffmpeg/8.1.2/ffmpeg"
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
WORKER_BIN="${WORKER_BIN:-$REPO_ROOT/.venv/bin/saimc-jobs}"
UVICORN_BIN="${UVICORN_BIN:-$REPO_ROOT/.venv/bin/uvicorn}"

find_broker_server() {
    # Prefer valkey-server; redis-server speaks the same RESP protocol and
    # works with RQ as well.
    for candidate in valkey-server redis-server; do
        command -v "$candidate" >/dev/null 2>&1 && { printf '%s' "$candidate"; return 0; }
    done
    return 1
}

broker_ping() {
    # Ping the broker through the venv's redis client so we honour $VALKEY_URL
    # (redis-py rejects the valkey:// scheme; redis_url() rewrites it).
    "$PYTHON_BIN" - <<PY 2>/dev/null
import redis
from saimc.jobs.worker import redis_url
try:
    redis.Redis.from_url(redis_url("$VALKEY_URL"), socket_connect_timeout=1.0).ping()
except Exception:
    raise SystemExit(1)
PY
}

pid_alive() {
    local pidfile="$1"
    [ -f "$pidfile" ] || return 1
    local pid
    pid="$(cat "$pidfile")"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_one() {
    local name="$1" pidfile="$RUN_DIR/$2.pid" logfile="$RUN_DIR/$2.log"
    if pid_alive "$pidfile"; then
        echo "$name already running (pid $(cat "$pidfile"))"
        return 0
    fi
    rm -f "$pidfile"
    shift 2
    nohup "$@" >>"$logfile" 2>&1 &
    local pid=$!
    echo "$pid" >"$pidfile"
    # Give the process a moment to fail fast (missing binary, port in use).
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "$name failed to start — see $logfile"
        rm -f "$pidfile"
        return 1
    fi
    echo "$name started (pid $pid, log $logfile)"
}

stop_one() {
    local name="$1" pidfile="$RUN_DIR/$2.pid"
    if ! pid_alive "$pidfile"; then
        rm -f "$pidfile"
        echo "$name not running"
        return 0
    fi
    local pid
    pid="$(cat "$pidfile")"
    kill "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null
        echo "$name force-stopped (pid $pid)"
    else
        echo "$name stopped (pid $pid)"
    fi
    rm -f "$pidfile"
}

status_one() {
    local name="$1" pidfile="$RUN_DIR/$2.pid"
    if pid_alive "$pidfile"; then
        echo "$name: running (pid $(cat "$pidfile"))"
        return 0
    fi
    echo "$name: stopped"
    return 1
}

start_valkey() {
    if broker_ping; then
        echo "valkey already reachable at $VALKEY_URL"
        return 0
    fi
    local server
    server="$(find_broker_server)" || {
        echo "no valkey-server/redis-server found — install one, e.g.: brew install valkey"
        return 1
    }
    mkdir -p "$SAIMC_JOBS_DIR"
    start_one valkey valkey "$server" --port "$VALKEY_PORT" --daemonize no --save "" --appendonly no
    for _ in 1 2 3 4 5; do
        broker_ping && return 0
        sleep 0.5
    done
    echo "valkey did not become ready — see $RUN_DIR/valkey.log"
    return 1
}

start_worker() {
    [ -x "$WORKER_BIN" ] || { echo "worker binary missing: $WORKER_BIN (create the venv first)"; return 1; }
    start_one worker worker "$WORKER_BIN" --jobs-root "$SAIMC_JOBS_DIR" --valkey-url "$VALKEY_URL"
}

start_api() {
    [ -x "$UVICORN_BIN" ] || { echo "uvicorn missing: $UVICORN_BIN"; return 1; }
    start_one api api "$UVICORN_BIN" \
        saimc.jobs.api:create_app --factory \
        --host "$API_HOST" --port "$API_PORT"
}

ALL_SERVICES="valkey worker api"

normalize_services() {
    if [ $# -eq 0 ]; then
        printf '%s\n' $ALL_SERVICES
    else
        for s in "$@"; do
            case "$s" in
                valkey|worker|api) printf '%s\n' "$s" ;;
                *) echo "unknown service: $s (choose from: $ALL_SERVICES)"; return 1 ;;
            esac
        done
    fi
}

cmd_start() {
    local services
    services="$(normalize_services "$@")" || return 1
    local rc=0
    # shellcheck disable=SC2178
    for s in $services; do
        case "$s" in
            valkey) start_valkey || rc=1 ;;
            worker) start_worker || rc=1 ;;
            api)    start_api || rc=1 ;;
        esac
    done
    if [ "$rc" -eq 0 ]; then
        echo "API listening on http://$API_HOST:$API_PORT"
    fi
    return "$rc"
}

cmd_stop() {
    local services
    services="$(normalize_services "$@")" || return 1
    # Stop in reverse dependency order.
    local reversed="api worker valkey"
    for s in $reversed; do
        printf '%s\n' "$services" | grep -qx "$s" || continue
        stop_one "$s" "$s"
    done
}

cmd_status() {
    local rc=0
    for s in $ALL_SERVICES; do
        status_one "$s" "$s" || rc=1
    done
    if broker_ping; then
        echo "broker: reachable at $VALKEY_URL"
    else
        echo "broker: unreachable at $VALKEY_URL"
        rc=1
    fi
    return "$rc"
}

cmd_restart() {
    cmd_stop "$@"
    cmd_start "$@"
}

case "${1:-}" in
    start) shift; cmd_start "$@" ;;
    stop) shift; cmd_stop "$@" ;;
    restart) shift; cmd_restart "$@" ;;
    status) cmd_status ;;
    *)
        echo "usage: run.sh {start|stop|restart|status} [valkey|worker|api ...]" >&2
        exit 2
        ;;
esac