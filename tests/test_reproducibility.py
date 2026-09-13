"""Unit tests for deterministic seed locking across all RNG backends.

This module validates that `set_seed` enforces bitwise-identical outputs for
Python's ``random``, NumPy, and PyTorch CPU/CUDA RNGs across repeated identical
seed invocations, and that cuDNN backend flags are correctly configured for
strict determinism.

Regression coverage
-------------------
- Silent drift in dataset splits caused by un-seeded Python ``random`` calls.
- Divergent weight initialization when PyTorch's RNG is not reset identically.
- cuDNN non-determinism from benchmark mode remaining enabled after seeding.
"""

import random

import numpy as np
import pytest
import torch

from dementia_boost.core.reproducibility import set_seed

_SEED_A: int = 42
_SEED_B: int = 99  # must differ from _SEED_A


class TestSetSeedPythonRandom:
    """Validates that set_seed reproducibly controls Python's built-in RNG."""

    def test_identical_seeds_produce_identical_values(self) -> None:
        """Same seed → same sequence from ``random.random()``."""
        set_seed(_SEED_A)
        first_run = [random.random() for _ in range(10)]

        set_seed(_SEED_A)
        second_run = [random.random() for _ in range(10)]

        assert first_run == second_run, (
            "random.random() must produce identical sequences for identical seeds."
        )

    def test_different_seeds_produce_different_values(self) -> None:
        """Different seeds → different sequences (guards against no-op seeding)."""
        set_seed(_SEED_A)
        run_a = [random.random() for _ in range(10)]

        set_seed(_SEED_B)
        run_b = [random.random() for _ in range(10)]

        assert run_a != run_b, (
            "random.random() must produce divergent sequences for different seeds."
        )


class TestSetSeedNumpy:
    """Validates that set_seed reproducibly controls NumPy's RNG."""

    def test_identical_seeds_produce_identical_arrays(self) -> None:
        """Same seed → bitwise-identical arrays from ``np.random.rand``."""
        set_seed(_SEED_A)
        first_run = np.random.rand(20)

        set_seed(_SEED_A)
        second_run = np.random.rand(20)

        np.testing.assert_array_equal(
            first_run,
            second_run,
            err_msg="np.random.rand must produce identical arrays for identical seeds.",
        )

    def test_different_seeds_produce_different_arrays(self) -> None:
        """Different seeds → different arrays (guards against no-op seeding)."""
        set_seed(_SEED_A)
        run_a = np.random.rand(20)

        set_seed(_SEED_B)
        run_b = np.random.rand(20)

        assert not np.array_equal(run_a, run_b), (
            "np.random.rand must produce divergent arrays for different seeds."
        )


class TestSetSeedTorch:
    """Validates that set_seed reproducibly controls PyTorch CPU (and CUDA) RNGs."""

    def test_identical_seeds_produce_identical_cpu_tensors(self) -> None:
        """Same seed → bitwise-identical tensors from ``torch.randn`` on CPU."""
        set_seed(_SEED_A)
        first_run = torch.randn(50)

        set_seed(_SEED_A)
        second_run = torch.randn(50)

        assert torch.equal(first_run, second_run), (
            "torch.randn (CPU) must produce identical tensors for identical seeds."
        )

    def test_different_seeds_produce_different_cpu_tensors(self) -> None:
        """Different seeds → different tensors (guards against no-op seeding)."""
        set_seed(_SEED_A)
        run_a = torch.randn(50)

        set_seed(_SEED_B)
        run_b = torch.randn(50)

        assert not torch.equal(run_a, run_b), (
            "torch.randn (CPU) must produce divergent tensors for different seeds."
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA device not available; skipping CUDA RNG test.",
    )
    def test_identical_seeds_produce_identical_cuda_tensors(self) -> None:
        """Same seed → bitwise-identical tensors from ``torch.randn`` on CUDA."""
        device = torch.device("cuda")

        set_seed(_SEED_A)
        first_run = torch.randn(50, device=device)

        set_seed(_SEED_A)
        second_run = torch.randn(50, device=device)

        assert torch.equal(first_run, second_run), (
            "torch.randn (CUDA) must produce identical tensors for identical seeds."
        )


class TestSetSeedCuDNNDeterminism:
    """Validates that set_seed configures cuDNN for strict deterministic execution."""

    def test_cudnn_deterministic_flag_is_enabled(self) -> None:
        """``torch.backends.cudnn.deterministic`` must be True after seeding."""
        set_seed(_SEED_A)
        assert torch.backends.cudnn.deterministic is True, (
            "cudnn.deterministic must be True after set_seed() to prevent "
            "non-deterministic cuDNN kernel selection."
        )

    def test_cudnn_benchmark_flag_is_disabled(self) -> None:
        """``torch.backends.cudnn.benchmark`` must be False after seeding."""
        set_seed(_SEED_A)
        assert torch.backends.cudnn.benchmark is False, (
            "cudnn.benchmark must be False after set_seed() to prevent "
            "algorithm auto-tuning that introduces run-to-run variance."
        )
