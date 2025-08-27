"""Integration tests for Opper chat models using LangChain standard tests."""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_opper.chat_models import OpperChatModel


class TestOpperChatModelIntegration(ChatModelUnitTests):
    """Test OpperChatModel integration using LangChain standard tests."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return the chat model class to test."""
        return OpperChatModel

    @pytest.fixture
    def model(self) -> BaseChatModel:
        """Create a chat model instance for testing."""
        # Skip if no API key is available
        api_key = os.environ.get("OPPER_API_KEY")
        if not api_key:
            pytest.skip("OPPER_API_KEY not set")
        
        return OpperChatModel(
            task_name="test_chat",
            model_name="anthropic/claude-3.5-sonnet",
            instructions="You are a helpful test assistant. Be concise.",
        )
