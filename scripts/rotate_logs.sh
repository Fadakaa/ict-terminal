#!/bin/bash
# Rotate ICT terminal backend logs
# Keeps 3 rotated copies, rotates when > 1MB

LOGDIR="/Users/getmoney/dealfinder/ict-terminal/logs"

rotate_log() {
    local logfile="$1"
    local max_size="${2:-1048576}"  # 1MB default

    if [ ! -f "$logfile" ]; then
        return
    fi

    local size=$(stat -f%z "$logfile" 2>/dev/null || echo 0)
    if [ "$size" -gt "$max_size" ]; then
        # Rotate: .3 -> delete, .2 -> .3, .1 -> .2, current -> .1
        rm -f "${logfile}.3"
        [ -f "${logfile}.2" ] && mv "${logfile}.2" "${logfile}.3"
        [ -f "${logfile}.1" ] && mv "${logfile}.1" "${logfile}.2"
        cp "$logfile" "${logfile}.1"
        : > "$logfile"  # Truncate in place (keeps file handle)
        echo "[$(date)] Log rotated: $logfile" >> "$logfile"
    fi
}

rotate_log "$LOGDIR/server.log" 1048576       # 1MB
rotate_log "$LOGDIR/server_error.log" 524288   # 512KB
