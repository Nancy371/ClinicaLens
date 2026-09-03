"""Agent 包入口，支持 python -m agent 运行。"""

from agent.server import run_server
import os

port = int(os.environ.get("PORT", "7860"))
run_server(host="0.0.0.0", port=port)