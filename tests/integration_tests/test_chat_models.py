"""Test ChatOpperAI chat model."""

from typing import Type

from langchain_opperai.chat_models import ChatOpperAI
from langchain_tests.integration_tests import ChatModelIntegrationTests


class TestChatOpperAIIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatOpperAI]:
        return ChatOpperAI

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        # Note: Integration tests will use real OPPER_API_KEY from environment
        return {
            "task_name": "integration_test_chat",
            "model_name": "anthropic/claude-3.5-sonnet", 
            "instructions": "You are a helpful test assistant for integration testing. Provide clear, concise responses.",
        }
