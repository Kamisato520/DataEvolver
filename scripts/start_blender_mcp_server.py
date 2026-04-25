"""
Headless Blender startup script for BlenderMCP.
Usage: /home/wuwenzhuo/blender-4.24/blender -b --python start_blender_mcp_server.py

Blender 4.2 in background mode may not honor bpy.app.timers for keep-alive.
We use a non-daemon thread + threading.Event to keep the process alive
while still allowing the Blender event loop (and timer-based callbacks) to run.
"""
import bpy
import sys
import os
import threading
import time

# Add the addon directory to sys.path so we can import addon.py
addon_dir = os.path.dirname(os.path.abspath(__file__))
if addon_dir not in sys.path:
    sys.path.insert(0, addon_dir)

# Import the addon and register everything
import addon
addon.register()

# Create and start the server
port = int(os.environ.get("BLENDER_MCP_PORT", "9876"))
server = addon.BlenderMCPServer(host='localhost', port=port)
server.start()

# Make the server thread non-daemon so Blender won't exit while it's running
if server.server_thread and server.server_thread.daemon:
    # Can't change daemon after start, but we track it
    pass

# Store server reference
bpy.types.blendermcp_server = server
bpy.context.scene.blendermcp_server_running = True

print(f"[BlenderMCP] Server started on localhost:{port}")

# Keep the process alive via a non-daemon watchdog thread
# This thread just sleeps; its existence (non-daemon) prevents Blender from exiting.
# Meanwhile, Blender's event loop remains free to process bpy.app.timers callbacks
# (which the addon uses to execute commands in the main thread).
_stop_event = threading.Event()

def watchdog():
    """Non-daemon thread that keeps the process alive."""
    while not _stop_event.is_set():
        time.sleep(1)
    print("[BlenderMCP] Watchdog exiting.")

_watchdog_thread = threading.Thread(target=watchdog, name="BlenderMCP-Watchdog")
_watchdog_thread.daemon = False
_watchdog_thread.start()

print("[BlenderMCP] Watchdog thread started. Blender will stay running.")
