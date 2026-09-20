from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from agent_reflex.security import sanitize
from agent_reflex.types import DecisionRequest


class ContextSelector(Protocol):
    def select(self, request: DecisionRequest) -> DecisionRequest: ...


class SafeContextSelector:
    """Redacts common secrets and bounds context size before probabilistic providers.

    It does not make untrusted text trustworthy. Providers still receive trusted and
    untrusted context separately and hard policy must remain outside model control.
    """

    def __init__(self, *, max_string: int = 32_768, max_items: int = 256, max_depth: int = 8):
        self.max_string = max_string
        self.max_items = max_items
        self.max_depth = max_depth

    def select(self, request: DecisionRequest) -> DecisionRequest:
        kwargs = dict(max_string=self.max_string, max_items=self.max_items, max_depth=self.max_depth)
        return replace(
            request,
            state=sanitize(request.state, **kwargs),
            metadata=sanitize(request.metadata, **kwargs),
            trusted_context=sanitize(request.trusted_context, **kwargs),
            untrusted_context=sanitize(request.untrusted_context, **kwargs),
        )
