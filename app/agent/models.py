from pydantic import BaseModel
from typing import Optional, List


class AgentRequest(BaseModel):
    query: str
    context: Optional[dict] = None  # Carries pending action confirmation or session data if present


class AgentResponse(BaseModel):
    user_query: str
    route_used: str
    response_text: str
    requires_action_confirmation: bool = False
    pending_action: Optional[dict] = None
    execution_steps: Optional[List[str]] = None