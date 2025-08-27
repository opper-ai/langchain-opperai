"""Opper provider for creating chat models and managing traces."""

import os
from typing import Any, Optional, Type

from opperai import Opper
from pydantic import BaseModel


class OpperProvider:
    """Provider that leverages LangGraph's native patterns effectively.
    
    Features:
    - Simple architecture using LangGraph conventions
    - Direct integration with Opper's native structured output
    - State-driven configuration
    - Seamless tool calling integration

    Setup:
        Install ``langchain-opper`` and set environment variable ``OPPER_API_KEY``.

        .. code-block:: bash

            pip install langchain-opper
            export OPPER_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_opper import OpperProvider

            provider = OpperProvider()

    Create chat model:
        .. code-block:: python

            chat_model = provider.create_chat_model(
                task_name="chat",
                model_name="anthropic/claude-3.5-sonnet",
                instructions="You are a helpful AI assistant.",
            )

    Create structured model:
        .. code-block:: python

            from pydantic import BaseModel

            class Response(BaseModel):
                answer: str
                confidence: float

            structured_model = provider.create_structured_model(
                task_name="structured_chat",
                instructions="Provide structured responses.",
                output_schema=Response,
            )

    With tracing:
        .. code-block:: python

            provider.start_trace("conversation", "User wants help with X")
            
            # Models will now use this trace
            result = chat_model.invoke("Help me with X")
            
            provider.end_trace("Provided help successfully")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the provider."""
        self.api_key = api_key or os.getenv("OPPER_API_KEY")
        if not self.api_key:
            raise ValueError("OPPER_API_KEY must be provided or set as environment variable")
        
        self.client = Opper(http_bearer=self.api_key)
        self.current_trace_id = None
    
    def create_chat_model(
        self,
        task_name: str = "chat",
        model_name: str = "anthropic/claude-3.5-sonnet",
        instructions: str = "You are a helpful AI assistant. Provide clear, structured responses.",
    ) -> "OpperChatModel":
        """Create a new Opper chat model."""
        from langchain_opper.chat_models import OpperChatModel
        
        return OpperChatModel(
            opper_client=self.client,
            task_name=task_name,
            model_name=model_name,
            instructions=instructions,
            parent_span_id=self.current_trace_id,
            provider_ref=self,  # Pass provider reference for dynamic trace access
        )
    
    def create_structured_model(
        self,
        task_name: str,
        instructions: str,
        output_schema: Type[BaseModel],
        model_name: str = "anthropic/claude-3.5-sonnet",
    ) -> "OpperChatModel":
        """Create a model with structured output using LangChain's native pattern.
        
        This method creates a model that leverages both Opper's native structured
        output and LangChain's with_structured_output() pattern.
        """
        base_model = self.create_chat_model(
            task_name=task_name,
            model_name=model_name,
            instructions=instructions,
        )
        
        return base_model.with_structured_output(schema=output_schema)
    
    def start_trace(self, name: str, input_data: Any = None) -> str:
        """Start a new trace for tracking the workflow.
        
        Creates a parent span that will contain all subsequent model calls.
        All models created by this provider will use this span as their parent.
        """
        span = self.client.spans.create(
            name=name,
            input=str(input_data) if input_data else None
        )
        self.current_trace_id = span.id
        return span.id
    
    def end_trace(self, output_data: Any = None):
        """End the current trace."""
        if self.current_trace_id:
            self.client.spans.update(
                span_id=self.current_trace_id,
                output=str(output_data) if output_data else None
            )
            self.current_trace_id = None
    
    def add_metric(self, span_id: str, dimension: str, value: float, comment: str = ""):
        """Add a metric to a span."""
        self.client.span_metrics.create_metric(
            span_id=span_id,
            dimension=dimension,
            value=value,
            comment=comment
        )


# Rebuild OpperChatModel after OpperProvider is defined
try:
    from langchain_opper.chat_models import OpperChatModel
    OpperChatModel.model_rebuild()
except ImportError:
    # OpperChatModel not yet available
    pass
