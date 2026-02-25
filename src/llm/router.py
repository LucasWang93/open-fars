"""Unified LLM interface with mock provider for testing."""

import json
import random
from typing import Optional

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


class LLMRouter:
    def __init__(self, provider: str = "mock", **kwargs):
        self.provider = provider
        self.kwargs = kwargs

    def generate(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict] = None,
        mock_response: Optional[dict] = None,
    ) -> str:
        if self.provider == "mock":
            return self._mock_generate(system, user, json_schema, mock_response)
        raise NotImplementedError(f"Provider {self.provider} not implemented yet")

    def _mock_generate(
        self,
        system: str,
        user: str,
        json_schema: Optional[dict],
        mock_response: Optional[dict],
    ) -> str:
        LOGGER.info("LLM mock generate (system=%d chars, user=%d chars)", len(system), len(user))
        if mock_response is not None:
            return json.dumps(mock_response)
        return json.dumps({"mock": True, "message": "mock response"})


def get_router(profile: str = "mock") -> LLMRouter:
    return LLMRouter(provider=profile)
