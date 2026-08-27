from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class RouteType(str, Enum):
    RAG = "RAG"
    MCP = "MCP"
    RAG_MCP_REASONING = "RAG_MCP_REASONING"
    MCP_ACTION = "MCP_ACTION"
    DATA_AGENT = "DATA_AGENT"
    GENERAL_LLM = "GENERAL_LLM"

class RouteDecision(BaseModel):
    user_query: str
    route: RouteType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    extracted_entities: Optional[Dict[str, Any]] = Field(default_factory=dict)