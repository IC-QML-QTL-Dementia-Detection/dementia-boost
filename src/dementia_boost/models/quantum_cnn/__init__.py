"""Quantum neural network components and variational circuits for dementia detection.

This subpackage exposes the PennyLane variational quantum circuit (Ansatz)
and the `QuantumClassifierHead` implementing the Dressed Quantum Network (DQN).
"""

from .circuit import create_quantum_layer
from .heads import QuantumClassifierHead

__all__ = [
    "QuantumClassifierHead",
    "create_quantum_layer",
]
