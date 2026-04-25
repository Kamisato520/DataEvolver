#!/bin/bash
# Wrapper script to start Blender MCP server
# Run via: tmux new-session -d -s blender-mcp 'bash /home/wuwenzhuo/blender-mcp/run_blender_mcp.sh'

BLENDER=/home/wuwenzhuo/blender-4.24/blender
SCRIPT=/home/wuwenzhuo/blender-mcp/start_blender_mcp_server.py
LOGFILE=/home/wuwenzhuo/blender-mcp/blender_mcp.log

echo "[$(date)] Starting Blender MCP server..." | tee "$LOGFILE"
$BLENDER -b --python "$SCRIPT" >> "$LOGFILE" 2>&1
echo "[$(date)] Blender exited with code $?" | tee -a "$LOGFILE"
