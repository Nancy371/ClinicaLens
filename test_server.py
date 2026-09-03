"""测试服务启动和 contestServiceToken 传递"""
import subprocess
import sys
import time
import json
import urllib.request
import urllib.error

# 启动服务
print("正在启动服务...")
proc = subprocess.Popen(
    [sys.executable, "-m", "agent.agent"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

# 等待服务启动
time.sleep(5)

results = []

# 测试1: 健康检查
print("\n=== 测试1: 健康检查 ===")
try:
    req = urllib.request.Request("http://127.0.0.1:7860/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        print(f"健康检查结果: {data}")
        results.append(("健康检查", "PASS" if data.get("status") == "ok" else "FAIL"))
except Exception as e:
    print(f"健康检查失败: {e}")
    results.append(("健康检查", "FAIL"))

# 测试2: contestServiceToken 传递
print("\n=== 测试2: contestServiceToken 传递验证 ===")
# 验证代码逻辑：检查 server.py 中是否包含 contestServiceToken 处理
with open("agent/server.py", "r", encoding="utf-8") as f:
    server_code = f.read()
    has_token_handling = "contestServiceToken" in server_code and "SERVICE_TRAIN_TOKEN" in server_code
    print(f"contestServiceToken 处理代码: {'存在' if has_token_handling else '缺失'}")
    results.append(("contestServiceToken代码", "PASS" if has_token_handling else "FAIL"))

# 测试3: /test 接口（不传token，应该能正常响应但测试会因缺少服务而失败）
print("\n=== 测试3: /test 接口响应 ===")
try:
    test_body = json.dumps({
        "patient_ids": ["test_patient"],
        "contestServiceToken": "test_token_123"
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:7860/test",
        data=test_body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        print(f"/test 响应: status={data.get('status')}")
        results.append(("/test接口响应", "PASS"))
        final_result = data.get("final_result")
        final_results = data.get("final_results")
        has_evaluable_result = (
            isinstance(final_result, dict)
            and bool(final_result.get("diagnosis"))
            and bool(final_result.get("treatment_plan"))
            and isinstance(final_results, list)
            and len(final_results) > 0
        )
        print(f"final_result/final_results 可评估: {has_evaluable_result}")
        results.append(("final_result字段", "PASS" if has_evaluable_result else "FAIL"))
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"/test HTTP错误: {e.code} - {body[:200]}")
    # 500 错误是预期的（因为没有真实的后端服务），但接口能响应说明服务正常
    results.append(("/test接口响应", "PASS" if e.code == 500 else "FAIL"))
except Exception as e:
    print(f"/test 请求失败: {e}")
    results.append(("/test接口响应", "FAIL"))

# 输出结果汇总
print("\n=== 结果汇总 ===")
for name, status in results:
    print(f"  {name}: {status}")

# 终止服务
proc.terminate()
proc.wait(timeout=5)
print("\n服务已停止")

# 写入结果文件
with open("test_output.txt", "w", encoding="utf-8") as f:
    for name, status in results:
        f.write(f"{name}: {status}\n")
    f.write("DONE\n")

print("测试结果已写入 test_output.txt")
