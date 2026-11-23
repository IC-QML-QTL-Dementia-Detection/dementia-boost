"""Quantum neural network components and variational circuits for dementia detection.

This subpackage holds the Dressed Quantum Network (DQN) head on Qiskit v2.x:
`QiskitQuantumClassifierHead`, backed by `qiskit_circuit.py` (a registered
embedding from `qiskit_embeddings.py` followed by layers of a registered
variational block from `qiskit_blocks.py`) and executed through Qiskit
Primitives V2 on an injectable estimator (`qiskit_runner.py`) inside a custom
autograd function with SPSA gradients.
"""

from .qiskit_blocks import BLOCKS, DEFAULT_BLOCK, get_block
from .qiskit_circuit import build_qiskit_ansatz
from .qiskit_embeddings import DEFAULT_EMBEDDING, EMBEDDINGS, get_embedding
from .qiskit_heads import QiskitQuantumClassifierHead
from .qiskit_layer import create_qiskit_quantum_layer
from .qiskit_runner import QiskitExpectationRunner, resolve_qiskit_estimator

__all__ = [
    "BLOCKS",
    "DEFAULT_BLOCK",
    "DEFAULT_EMBEDDING",
    "EMBEDDINGS",
    "QiskitExpectationRunner",
    "QiskitQuantumClassifierHead",
    "build_qiskit_ansatz",
    "create_qiskit_quantum_layer",
    "get_block",
    "get_embedding",
    "resolve_qiskit_estimator",
]
