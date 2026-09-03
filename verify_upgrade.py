"""验证 server.py 修改和服务可启动性"""
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
import json

results = []


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# 1. 语法检查
print("=== 1. 语法检查 ===")
try:
    server_path = Path("agent/server.py")
    compile(server_path.read_text(encoding="utf-8"), str(server_path), "exec")
    results.append(("server.py 语法", "PASS"))
    print("server.py 语法检查通过")
except (OSError, SyntaxError) as e:
    results.append(("server.py 语法", f"FAIL: {e}"))
    print(f"server.py 语法错误: {e}")

# 2. 导入检查
print("\n=== 2. 导入检查 ===")
try:
    from agent.server import create_app, handle_test, run_server
    results.append(("server.py 导入", "PASS"))
    print("server.py 导入成功")
except Exception as e:
    results.append(("server.py 导入", f"FAIL: {e}"))
    print(f"server.py 导入失败: {e}")

# 3. contestServiceToken 代码检查
print("\n=== 3. contestServiceToken 处理代码 ===")
with open("agent/server.py", "r", encoding="utf-8") as f:
    code = f.read()
checks = [
    ("contestServiceToken" in code, "包含 contestServiceToken 读取"),
    ("SERVICE_TRAIN_TOKEN" in code, "包含 SERVICE_TRAIN_TOKEN 设置"),
    ("os.environ" in code, "包含 os.environ 操作"),
]
for ok, desc in checks:
    status = "PASS" if ok else "FAIL"
    results.append((desc, status))
    print(f"  {desc}: {status}")

# 4. 服务启动测试
print("\n=== 4. 服务启动测试 ===")
try:
    port = find_free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent.agent"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    time.sleep(6)
    
    # 检查进程是否存活
    if proc.poll() is None:
        results.append(("服务启动", "PASS"))
        print("服务启动成功，进程运行中")
        
        # 健康检查
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                ok = data.get("status") == "ok"
                results.append(("健康检查", "PASS" if ok else "FAIL"))
                print(f"健康检查: {data}")
        except Exception as e:
            results.append(("健康检查", f"FAIL: {e}"))
            print(f"健康检查失败: {e}")
        
        # 终止
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
    else:
        stdout, stderr = proc.communicate()
        results.append(("服务启动", f"FAIL: 进程已退出 code={proc.returncode}"))
        print(f"服务启动失败，退出码: {proc.returncode}")
        if stderr:
            print(f"stderr: {stderr[:500]}")
except Exception as e:
    results.append(("服务启动", f"FAIL: {e}"))
    print(f"服务启动异常: {e}")

# 输出汇总
print("\n=== 结果汇总 ===")
all_pass = True
for name, status in results:
    is_pass = status == "PASS"
    if not is_pass:
        all_pass = False
    print(f"  {name}: {status}")

print(f"\n总体结果: {'ALL PASS' if all_pass else 'HAS FAILURES'}")

with open("test_output.txt", "w", encoding="utf-8") as f:
    for name, status in results:
        f.write(f"{name}: {status}\n")
    f.write(f"总体: {'ALL_PASS' if all_pass else 'HAS_FAILURES'}\n")
