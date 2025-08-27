"""Unit tests for Opper chat models."""

import os
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain_opper.chat_models import OpperChatModel


class ResponseSchema(BaseModel):
    """Test response schema."""
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")


class TestOpperChatModel:
    """Test OpperChatModel."""

    def test_init_with_api_key_env(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper") as mock_opper:
                model = OpperChatModel()
                mock_opper.assert_called_once_with(http_bearer="test-key")
                assert model.task_name == "chat"
                assert model.model_name == "anthropic/claude-3.5-sonnet"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPPER_API_KEY environment variable is required"):
                OpperChatModel()

    def test_llm_type(self):
        """Test _llm_type property."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper"):
                model = OpperChatModel()
                assert model._llm_type == "opper"

    def test_with_structured_output(self):
        """Test with_structured_output method."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper") as mock_opper:
                mock_client = mock_opper.return_value
                model = OpperChatModel(opper_client=mock_client)
                structured_model = model.with_structured_output(ResponseSchema)
                
                assert structured_model._structured_schema == ResponseSchema
                assert structured_model._output_parser is not None

    def test_with_structured_output_invalid_schema(self):
        """Test with_structured_output with invalid schema raises error."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper"):
                model = OpperChatModel()
                
                with pytest.raises(ValueError, match="Schema must be a Pydantic BaseModel class"):
                    model.with_structured_output({"not": "a pydantic model"})

    def test_prepare_input_for_opper_single_message(self):
        """Test _prepare_input_for_opper with single message."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper"):
                model = OpperChatModel()
                messages = [HumanMessage(content="Hello")]
                
                input_data = model._prepare_input_for_opper(messages)
                
                assert input_data == {"input": "Hello"}

    def test_prepare_input_for_opper_multiple_messages(self):
        """Test _prepare_input_for_opper with multiple messages."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper"):
                model = OpperChatModel()
                messages = [
                    HumanMessage(content="Hi"),
                    HumanMessage(content="How are you?")
                ]
                
                input_data = model._prepare_input_for_opper(messages)
                
                assert input_data["input"] == "How are you?"
                assert "conversation_history" in input_data
                assert len(input_data["conversation_history"]) == 1
                assert input_data["conversation_history"][0]["content"] == "Hi"

    def test_extract_text_response(self):
        """Test _extract_text_response method."""
        with patch.dict(os.environ, {"OPPER_API_KEY": "test-key"}):
            with patch("langchain_opper.chat_models.base.Opper"):
                model = OpperChatModel()
                
                # Test with response field
                mock_result = Mock()
                mock_result.json_payload = {"response": "Test response"}
                
                text = model._extract_text_response(mock_result)
                assert text == "Test response"
                
                # Test with answer field
                mock_result.json_payload = {"answer": "Test answer"}
                text = model._extract_text_response(mock_result)
                assert text == "Test answer"
                
                # Test fallback
                mock_result.json_payload = {"other": "Test content with more than 10 characters"}
                text = model._extract_text_response(mock_result)
                assert text == "Test content with more than 10 characters"
