"""Request and response models for POST /agent/query."""

from typing import Literal

from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    """User query submitted to the agent."""

    query: str = Field(..., min_length=1, description="Non-empty user query string")


class AgentQueryAccepted(BaseModel):
    """Successful acceptance of a query."""

    status: Literal["ok"]
    agent_run_id: str = Field(..., description="Unique run identifier for SSE stream")


class AgentQueryRejected(BaseModel):
    """Rejection due to invalid input."""

    status: Literal["rejected"]
    agent_run_id: str = Field(..., description="Unique run identifier for error stream")
    reason: str = Field(..., description="Reason for rejection (e.g., 'invalid query')")
