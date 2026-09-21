from __future__ import annotations

from dataclasses import replace

from agent_reflex.calibration import TemperatureCalibrator
from agent_reflex.providers.base import DecisionProvider, capabilities
from agent_reflex.types import Decision, DecisionRequest


class CalibratedProvider:
    def __init__(self, provider: DecisionProvider, calibrator: TemperatureCalibrator, name: str | None = None):
        self.provider = provider
        self.calibrator = calibrator
        self.name = name or f"{provider.name}:calibrated"
        self.capabilities = capabilities(provider)

    def decide(self, request: DecisionRequest) -> Decision | None:
        d = self.provider.decide(request)
        if d is None:
            return None
        raw = d.confidence
        d = replace(d, confidence=self.calibrator.transform(raw), provider=self.name)
        d.metadata = {**d.metadata, "raw_confidence": raw, "temperature": self.calibrator.temperature}
        return d
