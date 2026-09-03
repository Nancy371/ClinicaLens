"""
本地训练入口脚本。

使用方法：
    python train.py

训练患者可以在 config.yaml 的 train 中配置。
训练结果会保存到 output_dir 指定的目录。
"""

import asyncio
import json
import logging
import os
import sys
import warnings

import yaml

# Windows 下 Python 3.14 默认 ProactorEventLoop 存在偶发 getaddrinfo failed 问题，
# 使用 SelectorEventLoopPolicy 以获得更稳定的 DNS 解析行为
if sys.platform == "win32":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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


async def main():
    """训练主流程。"""
    setup_logging()
    logger = logging.getLogger(__name__)

    # 加载配置
    config = load_config()
    logger.info("配置加载完成: sections=%s", sorted(config.keys()))

    # 检查环境变量
    required_env = ["SERVICE_BASE_URL", "SERVICE_TRAIN_TOKEN", "MODEL_API_KEY", "TEAM_ID"]
    missing_env = [key for key in required_env if not os.environ.get(key)]
    if missing_env:
        logger.error(f"缺少必要的环境变量: {missing_env}")
        logger.error("请设置以下环境变量：")
        logger.error("  export SERVICE_BASE_URL=https://baconroot-hospital-service.ms.show")
        logger.error("  export SERVICE_TRAIN_TOKEN=<your-train-service-token>")
        logger.error("  export MODEL_API_KEY=<your-model-api-key>")
        logger.error("  export TEAM_ID=<your-team-id>")
        sys.exit(1)

    # 创建 Agent 实例
    agent = MyDoctorAgent(config)

    # 运行训练
    logger.info("开始训练...")
    try:
        result = await agent.run_train()
        logger.info(
            "训练完成: summary=%s",
            json.dumps(result.get("summary", {}), ensure_ascii=False),
        )
        if result.get("summary_file"):
            logger.info("训练报告: %s", result["summary_file"])
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
