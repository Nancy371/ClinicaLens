"""
本地测试和批量评估示例脚本。

使用方法：
    python test.py

该脚本会：
1. 启动 Agent 服务
2. 调用 POST /test 生成 final_results.jsonl
3. 调用 batch_evaluation 进行批量评估
4. 输出测试和评估结果

测试患者可以在 config.yaml 的 test 中配置。
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time

import yaml

from agent import MyDoctorAgent


def setup_logging():
    """配置日志。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


async def run_test_and_evaluate():
    """运行测试并进行批量评估。"""
    setup_logging()
    logger = logging.getLogger(__name__)

    # 加载配置
    config = load_config()

    # 检查环境变量
    required_env = ["SERVICE_BASE_URL", "SERVICE_TRAIN_TOKEN", "MODEL_API_KEY", "TEAM_ID"]
    missing_env = [key for key in required_env if not os.environ.get(key)]
    if missing_env:
        logger.error(f"缺少必要的环境变量: {missing_env}")
        sys.exit(1)

    # 创建 Agent 实例
    agent = MyDoctorAgent(config)

    # 运行测试
    logger.info("开始测试...")
    try:
        run_info = await agent.run_test()
        logger.info("测试完成！")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

    # 查找测试结果文件
    latest_test_dir = run_info.get("test_dir", "") if isinstance(run_info, dict) else ""
    results_file = run_info.get("results_file", "") if isinstance(run_info, dict) else ""

    if not os.path.exists(results_file):
        logger.error(f"未找到测试结果文件: {results_file}")
        return

    # 读取并显示测试结果
    logger.info(f"测试结果文件: {results_file}")
    with open(results_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    logger.info(f"测试病例数: {len(results)}")
    for result in results:
        patient_id = result.get("patient_id", "unknown")
        diagnosis = result.get("diagnosis", [])
        conversation_rounds = result.get("conversation_rounds", 0)
        logger.info(
            f"  患者 {patient_id}: 诊断={diagnosis}, "
            f"问诊轮数={conversation_rounds}"
        )

    # 批量评估
    logger.info("开始批量评估...")
    try:
        report = await agent.actions.batch_evaluation(latest_test_dir)
        logger.info("批量评估完成！")
        logger.info(f"评估报告: {json.dumps(report, ensure_ascii=False, indent=2)}")
    except Exception as e:
        logger.error(f"批量评估失败: {e}")


async def main():
    """主流程。"""
    await run_test_and_evaluate()


if __name__ == "__main__":
    asyncio.run(main())
