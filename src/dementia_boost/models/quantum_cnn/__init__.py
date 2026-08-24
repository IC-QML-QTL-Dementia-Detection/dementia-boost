"""Quantum neural network components and variational circuits for dementia detection.

This subpackage exposes the PennyLane variational quantum circuit (Ansatz),
device resolution utilities, and the `QuantumClassifierHead` implementing
the Dressed Quantum Network (DQN).
"""

from .circuit import (
    DEFAULT_QUANTUM_DEVICE,
    FALLBACK_QUANTUM_DEVICE,
    create_quantum_layer,
    resolve_quantum_device,
)
from .heads import QuantumClassifierHead

__all__ = [
    "DEFAULT_QUANTUM_DEVICE",
    "FALLBACK_QUANTUM_DEVICE",
    "QuantumClassifierHead",
    "create_quantum_layer",
    "resolve_quantum_device",
]
