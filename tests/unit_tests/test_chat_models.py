"""Test chat model integration."""

from typing import Type

from langchain_opperai.chat_models import ChatOpperAI
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatOpperAIUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatOpperAI]:
        return ChatOpperAI

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "api_key": "test",  # Add API key for testing
            "task_name": "test_chat",
            "model_name": "anthropic/claude-3.5-sonnet",
            "instructions": "You are a helpful test assistant. Provide clear, concise responses.",
        }
