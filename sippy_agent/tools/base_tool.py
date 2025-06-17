"""
Base classes and interfaces for Sippy Agent tools.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


class SippyToolInput(BaseModel):
    """Base input schema for Sippy tools."""
    pass


class SippyBaseTool(BaseTool, ABC):
    """Base class for all Sippy Agent tools."""
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    args_schema: Type[BaseModel] = SippyToolInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """Execute the tool with the given arguments."""
        pass
    
    async def _arun(self, **kwargs: Any) -> str:
        """Async version of _run. Default implementation calls _run."""
        return self._run(**kwargs)


class ExampleTool(SippyBaseTool):
    """Example tool to demonstrate the structure."""
    
    name: str = "example_tool"
    description: str = "An example tool that echoes back the input"
    
    class ExampleInput(SippyToolInput):
        message: str = Field(description="Message to echo back")
    
    args_schema: Type[BaseModel] = ExampleInput
    
    def _run(self, message: str) -> str:
        """Echo back the input message."""
        return f"Echo: {message}"
