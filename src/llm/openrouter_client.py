"""OpenRouter client for accessing open-source models via API"""

import httpx
import json
import time
from typing import List, Dict, Optional, AsyncIterator
from src.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import log_llm_metrics

logger = get_logger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.main_model
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        self.client = httpx.AsyncClient(timeout=timeout)

        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

    async def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None, stream: bool = False):
        """Chat completion"""
        prompt_length = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"LLM chat: model={self.model}, messages={len(messages)}, prompt_len={prompt_length}, stream={stream}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Subagents RAG"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": temperature or settings.temperature,
            "stream": stream
        }

        if stream:
            return self._stream_chat(headers, payload)

        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            duration_ms = (time.time() - start_time) * 1000
            log_llm_metrics(logger, self.model, duration_ms, usage.get("prompt_tokens"),
                          usage.get("completion_tokens"), response_length=len(content))
            return content
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"LLM error: {str(e)}, duration={duration_ms:.0f}ms")
            raise

    async def _stream_chat(self, headers: dict, payload: dict) -> AsyncIterator[str]:
        """Stream chat completion"""
        try:
            async with self.client.stream("POST", f"{self.base_url}/chat/completions",
                                        headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield "\n\n[Stream error. Please try again.]"

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


def get_llm(model_type: str = "main") -> OpenRouterClient:
    """Factory function to get LLM client"""
    model = settings.main_model if model_type == "main" else settings.router_model
    return OpenRouterClient(model=model)
