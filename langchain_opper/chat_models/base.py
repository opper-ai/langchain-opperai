"""Opper chat model implementation for LangChain."""

import os
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from opperai import Opper
else:
    try:
        from opperai import Opper
    except ImportError:
        Opper = Any  # Fallback for testing


class OpperChatModel(BaseChatModel):
    """Opper chat model that leverages LangChain's native structured output patterns.
    
    This integration provides:
    - LangChain's standard with_structured_output() pattern
    - Simplified schema handling
    - Integration with LangChain state management
    - Opper's tracing and metrics capabilities
    
    Setup:
        Install ``langchain-opper`` and set environment variable ``OPPER_API_KEY``.

        .. code-block:: bash

            pip install langchain-opper
            export OPPER_API_KEY="your-api-key"

    Key init args — completion params:
        task_name: Name for the Opper task
        model_name: Model to use with Opper
        instructions: Instructions for the model

    Key init args — client params:
        opper_client: Opper client instance
        parent_span_id: Parent span ID for tracing

    Instantiate:
        .. code-block:: python

            from langchain_opper import OpperChatModel

            llm = OpperChatModel(
                task_name="chat",
                model_name="anthropic/claude-3.5-sonnet",
                instructions="You are a helpful AI assistant.",
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant"),
                ("human", "What is the capital of France?"),
            ]
            llm.invoke(messages)

        .. code-block:: none

            AIMessage(content='The capital of France is Paris.')

    Structured output:
        .. code-block:: python

            from pydantic import BaseModel

            class Joke(BaseModel):
                setup: str
                punchline: str

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: none

            Joke(setup='Why don't cats play poker in the jungle?', punchline='Too many cheetahs!')

    """
    
    opper_client: Optional[Any] = None
    task_name: str = Field(default="chat", description="Name for the Opper task")
    model_name: str = Field(default="anthropic/claude-3.5-sonnet", description="Model to use with Opper")
    instructions: str = Field(
        default="You are a helpful AI assistant. Provide clear, structured responses.",
        description="Instructions for the model"
    )
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID for tracing")
    provider_ref: Optional["OpperProvider"] = Field(default=None, description="Reference to provider for dynamic trace access")
    
    # LangChain integration
    _output_parser: Optional[PydanticOutputParser] = None
    _structured_schema: Optional[Type[BaseModel]] = None
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.opper_client is None:
            api_key = os.getenv("OPPER_API_KEY")
            if not api_key:
                raise ValueError("OPPER_API_KEY environment variable is required")
            self.opper_client = Opper(http_bearer=api_key)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this model."""
        return "opper"
    
    def _get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID, preferring provider's current trace over static parent_span_id."""
        if self.provider_ref and self.provider_ref.current_trace_id:
            return self.provider_ref.current_trace_id
        return self.parent_span_id
    
    def _debug_trace_info(self) -> Dict[str, Any]:
        """Get debug information about trace state."""
        return {
            "model_static_parent": self.parent_span_id,
            "provider_current_trace": self.provider_ref.current_trace_id if self.provider_ref else None,
            "effective_trace_id": self._get_current_trace_id(),
            "has_provider_ref": self.provider_ref is not None
        }
    
    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        **kwargs: Any,
    ) -> "OpperChatModel":
        """Implement LangChain's standard with_structured_output method for Opper.
        
        This method leverages Opper's native output_schema support directly.
        """
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise ValueError("Schema must be a Pydantic BaseModel class")
        
        # Create new instance with structured output configuration
        new_instance = self.__class__(
            opper_client=self.opper_client,
            task_name=self.task_name,
            model_name=self.model_name,
            instructions=self.instructions,
            parent_span_id=self.parent_span_id,
            provider_ref=self.provider_ref,
            **kwargs
        )
        
        # Set up native structured output support
        new_instance._structured_schema = schema
        new_instance._output_parser = PydanticOutputParser(pydantic_object=schema)
        
        return new_instance
    
    def _prepare_input_for_opper(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Prepare input for Opper call as structured dictionary.
        
        This method provides a simple, flexible approach that lets the client
        control the input format while following Opper's best practices.
        Always returns a structured dict input for Opper.
        """
        if not messages:
            return {"input": ""}
        
        # Get the last message and check for additional kwargs
        last_message = messages[-1]
        
        # Build structured input dictionary
        input_data = {
            "input": last_message.content  # Primary input content
        }
        
        # Add conversation history for multi-message scenarios
        if len(messages) > 1:
            input_data["conversation_history"] = [
                {
                    "role": "user" if hasattr(msg, '__class__') and msg.__class__.__name__ == "HumanMessage"
                           else "assistant" if hasattr(msg, '__class__') and msg.__class__.__name__ == "AIMessage"
                           else "system",
                    "content": msg.content,
                    **(msg.additional_kwargs if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs else {})
                }
                for msg in messages[:-1]  # All messages except the last one
            ]
        
        # Add additional kwargs from the last message as structured context
        if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
            input_data["context"] = last_message.additional_kwargs
        
        return input_data
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using Opper with structured output support."""
        
        try:
            # Prepare input using LangChain conventions
            input_data = self._prepare_input_for_opper(messages)
            
            # Use Opper's native structured output if schema is specified
            output_schema = self._structured_schema
            
            # Make the Opper call with native schema support
            result = self.opper_client.call(
                name=self.task_name,
                instructions=self.instructions,
                input=input_data,
                output_schema=output_schema,  # Direct Opper schema support
                model=self.model_name,
                parent_span_id=self._get_current_trace_id(),  # Use dynamic trace ID
                **kwargs
            )
            
            # Handle structured vs unstructured responses
            if self._structured_schema and self._output_parser:
                # For structured output, return the parsed Pydantic object
                try:
                    # Opper returns structured data directly in json_payload
                    structured_data = result.json_payload
                    
                    # Validate and create Pydantic instance
                    parsed_output = self._structured_schema(**structured_data)
                    
                    # Create AI message with structured content
                    ai_message = AIMessage(
                        content=str(parsed_output),
                        additional_kwargs={
                            "parsed": parsed_output,
                            "span_id": getattr(result, 'span_id', None),
                            "structured": True,
                            **structured_data
                        }
                    )
                    
                except Exception as e:
                    # Fallback to text parsing if direct structured parsing fails
                    text_content = self._extract_text_response(result)
                    parsed_output = self._output_parser.parse(text_content)
                    
                    ai_message = AIMessage(
                        content=str(parsed_output),
                        additional_kwargs={
                            "parsed": parsed_output,
                            "span_id": getattr(result, 'span_id', None),
                            "structured": True,
                            "fallback_parsed": True,
                            "raw_content": text_content
                        }
                    )
            else:
                # For unstructured output, extract text response
                text_content = self._extract_text_response(result)
                
                ai_message = AIMessage(
                    content=text_content,
                    additional_kwargs={
                        "span_id": getattr(result, 'span_id', None),
                        "structured": False,
                        **(result.json_payload if hasattr(result, 'json_payload') else {})
                    }
                )
            
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise ValueError(f"Error calling Opper with structured output: {str(e)}")
    
    def _extract_text_response(self, opper_result: Any) -> str:
        """Extract text response from Opper result."""
        if hasattr(opper_result, 'json_payload'):
            payload = opper_result.json_payload
            
            # Common response fields in order of preference
            for field in ["response", "answer", "output", "result", "content", "message"]:
                if field in payload and isinstance(payload[field], str):
                    return payload[field]
            
            # Fallback to first substantial string field
            for value in payload.values():
                if isinstance(value, str) and len(value) > 10:
                    return value
        
        return str(opper_result) if opper_result else "No response generated"
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate."""
        return self._generate(messages, stop, run_manager, **kwargs)


# Forward reference resolution - import after class definition
def _rebuild_model():
    """Rebuild the model to resolve forward references."""
    try:
        from langchain_opper.llms import OpperProvider  # noqa: F401
        OpperChatModel.model_rebuild()
    except ImportError:
        # OpperProvider not yet available, will be resolved later
        pass

# Call rebuild when module is imported
_rebuild_model()
