"""
LLM 客户端模块。

支持 OpenAI 兼容 API（通义千问、DeepSeek、OpenAI 等），
提供统一的异步调用接口，用于驱动问诊、检查、诊断等决策。
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """异步 LLM 客户端，兼容 OpenAI Chat Completions API。

    支持的环境变量：
    - MODEL_API_KEY: API 密钥（必需）
    - MODEL_BASE_URL: API 基础 URL（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
    - MODEL_NAME: 模型名称（默认 qwen-plus）

    配置项（config.yaml llm 节）：
    - api_key: API 密钥（优先使用环境变量）
    - base_url: API 基础 URL
    - model_name: 模型名称
    - temperature: 生成温度
    - max_tokens: 最大生成 token 数
    - max_retries: 最大重试次数
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 LLM 客户端。

        Args:
            config: 配置字典，包含 llm 节
        """
        llm_config = config.get("llm", {})

        # 优先使用环境变量，其次使用配置文件
        self.api_key = os.environ.get("MODEL_API_KEY", "") or llm_config.get("api_key", "")
        self.base_url = (
            os.environ.get("MODEL_BASE_URL", "")
            or llm_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        ).rstrip("/")
        self.model_name = (
            os.environ.get("MODEL_NAME", "")
            or llm_config.get("model_name", "qwen-plus")
        )

        self.temperature = llm_config.get("temperature", 0.7)
        self.max_tokens = llm_config.get("max_tokens", 2048)
        self.max_retries = llm_config.get("max_retries", 3)
        self.retry_base_delay = float(llm_config.get("retry_base_delay", 1.0) or 1.0)
        self.request_timeout = float(llm_config.get("request_timeout", 120.0) or 120.0)

        self._client: Optional[httpx.AsyncClient] = None
        self.last_call_metadata: Dict[str, Any] = {}

        if not self.api_key:
            logger.warning("[LLM] MODEL_API_KEY 未设置，LLM 调用将失败")

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.request_timeout,
            )
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """调用 Chat Completions API。

        Args:
            messages: 消息列表，每条包含 role 和 content
            temperature: 生成温度（覆盖默认值）
            max_tokens: 最大 token 数（覆盖默认值）
            **kwargs: 传递给 API 的额外参数

        Returns:
            模型生成的文本

        Raises:
            Exception: API 调用失败
        """
        client = await self._get_client()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs,
        }

        last_error = None
        for attempt in range(self.max_retries):
            started = time.monotonic()
            metadata: Dict[str, Any] = {
                "model": self.model_name,
                "model_invoked": True,
                "attempt_index": attempt + 1,
                "max_retries": self.max_retries,
                "http_status": None,
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "finish_reason": None,
                "raw_response_present": False,
                "response_chars": 0,
                "exception_type": "",
            }
            try:
                response = await client.post("/chat/completions", json=payload)
                metadata["http_status"] = response.status_code
                response.raise_for_status()
                data = response.json()

                choice = (data.get("choices") or [{}])[0]
                content = ((choice.get("message") or {}).get("content")) or ""
                usage = data.get("usage", {})
                metadata["finish_reason"] = choice.get("finish_reason")
                metadata["input_tokens"] = usage.get("prompt_tokens")
                metadata["output_tokens"] = usage.get("completion_tokens")
                metadata["total_tokens"] = usage.get("total_tokens")
                metadata["raw_response_present"] = bool(content)
                metadata["response_chars"] = len(content)
                metadata["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                self.last_call_metadata = metadata
                logger.debug(
                    f"[LLM] 调用成功, tokens: prompt={usage.get('prompt_tokens', '?')}, "
                    f"completion={usage.get('completion_tokens', '?')}"
                )
                return content

            except httpx.HTTPStatusError as e:
                last_error = e
                metadata["http_status"] = e.response.status_code
                metadata["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                metadata["exception_type"] = type(e).__name__
                self.last_call_metadata = metadata
                logger.warning(
                    f"[LLM] HTTP 错误 (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e.response.status_code} - {e.response.text[:200]}"
                )
                if e.response.status_code == 429:
                    # 速率限制，等待后重试
                    await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                elif e.response.status_code >= 500:
                    # 服务端错误，重试
                    await asyncio.sleep(self.retry_base_delay)
                else:
                    # 客户端错误（4xx），不重试
                    break
            except Exception as e:
                last_error = e
                metadata["latency_ms"] = round((time.monotonic() - started) * 1000, 3)
                metadata["exception_type"] = type(e).__name__
                self.last_call_metadata = metadata
                logger.warning(f"[LLM] 调用失败 (attempt {attempt + 1}/{self.max_retries}): {e}")
                await asyncio.sleep(self.retry_base_delay)

        raise Exception(f"LLM 调用失败，已重试 {self.max_retries} 次: {last_error}")

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """调用 Chat API 并解析 JSON 输出。

        在 prompt 中要求 JSON 输出，并尝试从响应中提取 JSON。

        Args:
            messages: 消息列表
            temperature: 生成温度
            **kwargs: 额外参数

        Returns:
            解析后的 JSON 字典
        """
        # 添加 JSON 输出提示
        augmented_messages = list(messages)
        last_msg = augmented_messages[-1]
        if "json" not in last_msg["content"].lower():
            augmented_messages[-1] = {
                "role": last_msg["role"],
                "content": last_msg["content"] + "\n\n请以 JSON 格式输出结果。",
            }

        response = await self.chat(augmented_messages, temperature=temperature, **kwargs)
        return self._parse_json_response(response)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """从 LLM 响应中提取 JSON。

        支持以下格式：
        - 纯 JSON
        - ```json ... ``` 代码块
        - 混合文本中的 JSON 对象

        Args:
            response: LLM 响应文本

        Returns:
            解析后的字典
        """
        # 尝试直接解析
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { ... } 对象
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"[LLM] 无法解析 JSON 响应: {response[:200]}...")
        return {"raw_response": response}

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
