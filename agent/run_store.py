"""In-memory registry of agent runs and their event queues."""

import time
import uuid
from asyncio import Queue
from dataclasses import dataclass
from typing import Optional


@dataclass
class Run:
    """Single agent run."""

    id: str
    query: str
    created_at: float
    status: str
    events: Queue
    result: Optional[dict] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    in_progress_sent_at: Optional[float] = None


class RunStore:
    """In-memory registry of active runs."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    async def create(self, query: str) -> Run:
        """Generate a unique agent_run_id, enqueue in_progress, and return the Run.

        The in_progress event is queued before this method returns.
        in_progress_sent_at is recorded for the 5-second hold in Story 2.
        """
        run_id = uuid.uuid4().hex
        now = time.time()

        run = Run(
            id=run_id,
            query=query,
            created_at=now,
            status="pending",
            events=Queue(),
            in_progress_sent_at=now,
        )

        self._runs[run_id] = run

        await run.events.put(
            {
                "event": "in_progress",
                "data": {"run_id": run_id},
            }
        )

        return run

    async def get(self, run_id: str) -> Optional[Run]:
        """Retrieve a Run by agent_run_id, or None if it is missing."""
        return self._runs.get(run_id)

    async def enqueue_event(self, run_id: str, event_type: str, data: dict) -> None:
        """Append an event to a run's queue for later SSE delivery.

        Full scheduling behavior is Story 2. Story 1 only appends the event
        when the run exists.
        """
        run = self._runs.get(run_id)
        if run:
            await run.events.put({"event": event_type, "data": data})

    async def mark_failed(self, run_id: str, reason: str) -> None:
        """Mark a rejected run as failed and enqueue the failed event.

        create() has already queued in_progress. Both events are then waiting
        on run.events for SSE delivery.
        """
        run = self._runs.get(run_id)
        if run:
            run.status = "failed"
            await run.events.put(
                {
                    "event": "failed",
                    "data": {"run_id": run_id, "error": reason},
                }
            )

    async def cleanup_expired(self, ttl_seconds: int = 3600) -> None:
        """Delete runs older than ttl_seconds.

        Placeholder in Story 1. Story 2 implements TTL cleanup.
        """
        return None
