"""Unified LLM interface: mock provider + Azure OpenAI GPT-4o."""

import json
import os
from typing import Optional

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


class LLMRouter:
    def __init__(self, provider: str = "mock", model: str = "gpt-4o", **kwargs):
        self.provider = provider
        self.model = model
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self._client = None

    def _get_azure_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_ENDPOINT"],
                api_key=os.environ["AZURE_API_KEY"],
                api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
            )
        return self._client

    def generate(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        mock_response: Optional[dict] = None,
    ) -> str:
        if self.provider == "mock":
            return self._mock_generate(system, user, mock_response)
        if self.provider == "azure_openai":
            return self._azure_generate(system, user, json_mode)
        raise NotImplementedError(f"Provider {self.provider} not implemented")

    def _mock_generate(self, system, user, mock_response):
        LOGGER.info("LLM mock generate (%d+%d chars)", len(system), len(user))
        if mock_response is not None:
            return json.dumps(mock_response)
        return json.dumps({"mock": True})

    def _azure_generate(self, system: str, user: str, json_mode: bool) -> str:
        client = self._get_azure_client()
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        LOGGER.info("Azure OpenAI call: model=%s json_mode=%s", self.model, json_mode)
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        LOGGER.info("Azure OpenAI response: %d chars, usage=%s", len(text), resp.usage)
        return text


def get_router(profile: str = "azure_gpt4o") -> LLMRouter:
    if profile == "mock":
        return LLMRouter(provider="mock")
    if profile in ("azure_gpt4o", "azure_openai"):
        return LLMRouter(provider="azure_openai", model="gpt-4o")
    return LLMRouter(provider="mock")
