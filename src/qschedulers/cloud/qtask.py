from dataclasses import dataclass
from typing import Any


@dataclass
class QuantumTask:
    id: int
    circuit: Any
    arrival_time: float = 0.0
    priority: int = 0
    estimated_duration: float = None

    def __post_init__(self):
        if self.estimated_duration is None:
            # Estimate duration based on circuit complexity
            # A simple heuristic: use circuit depth as a base estimate
            # More sophisticated estimates could consider gate types and connectivity
            try:
                self.estimated_duration = float(self.circuit.depth()) * 0.001  # assuming 1ms per layer of depth
            except Exception:
                self.estimated_duration = 1.0  # default duration if estimation fails