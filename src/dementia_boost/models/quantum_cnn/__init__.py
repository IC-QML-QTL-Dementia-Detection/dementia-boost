"""Quantum neural network components and variational circuits for dementia detection.

This subpackage exposes two independent, interchangeable Dressed Quantum
Network (DQN) implementations sharing the same Bhowmik et al. (2025) angle
embedding and ansatz formulation:

- PennyLane: `PennylaneQuantumClassifierHead`, backed by `pennylane_circuit.py` and
  executed through a `TorchLayer`.
- Qiskit v2.x: `QiskitQuantumClassifierHead`, backed by `qiskit_circuit.py`
  and executed through Qiskit Primitives V2 on an injectable estimator
  (`qiskit_runner.py`) inside a custom autograd function with SPSA gradients.
"""

from .pennylane_circuit import (
    DEFAULT_QUANTUM_DEVICE,
    FALLBACK_QUANTUM_DEVICE,
    create_quantum_layer,
    resolve_quantum_device,
)
from .pennylane_heads import PennylaneQuantumClassifierHead
from .qiskit_circuit import build_qiskit_ansatz
from .qiskit_heads import QiskitQuantumClassifierHead
from .qiskit_layer import create_qiskit_quantum_layer
from .qiskit_runner import QiskitExpectationRunner, resolve_qiskit_estimator

__all__ = [
    "DEFAULT_QUANTUM_DEVICE",
    "FALLBACK_QUANTUM_DEVICE",
    "PennylaneQuantumClassifierHead",
    "QiskitExpectationRunner",
    "QiskitQuantumClassifierHead",
    "build_qiskit_ansatz",
    "create_qiskit_quantum_layer",
    "create_quantum_layer",
    "resolve_qiskit_estimator",
    "resolve_quantum_device",
]
