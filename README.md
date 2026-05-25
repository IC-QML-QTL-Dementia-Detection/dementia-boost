# Quantum Transfer Learning to Boost Dementia Detection

> TODO: more details

> TODO: better explanation

> TODO: add images

> TODO: describe next steps after base paper reproduction

## Overview

This project aims to reproduce and extend research on applying Quantum Transfer
Learning (QTL) to the classification of Dementia from raw 3D/2D NIfTI brain MRI scans.

The core hypothesis is that a Dressed Quantum Neural Network (DQNN) used as a
transfer-learning classification head can significantly outperform and stabilize
a "weak" classical Convolutional Neural Network (CNN) baseline in low-data
medical imaging scenarios.

## Project Goals

- **Phase 1: The Classical Baseline**: Establish a reproducible classical CNN
  (LeNet-based) to prove the baseline instability. _(Completed)_

- **Phase 2: Classical Transfer Learning**: Freeze the baseline feature
  extractor and train a new classical dense head to establish a
  standard transfer learning benchmark. _(TODO)_

- **Phase 3: Quantum Transfer Learning**: Replace the classical head with a
  parameterized Quantum Circuit (using PennyLane/Qiskit) and compare
  the statistical variance and AUC against the classical models. _(TODO)_

## System Architecture

This project enforces strict Software Engineering and MLOps patterns,
avoiding "messy notebook" syndrome:

- **`core/`**: Centralized reproducibility (seed locking across Python,
  NumPy, PyTorch, and cuDNN).
- **`data/`**: ETL pipelines with custom `MinMaxNormalize` for raw NIfTI
  intensity squashing.
- **`models/`**: Decoupled, Dependency-Injected PyTorch architectures
  (`LeNetFeatureExtractor`, `ClassicalClassifierHead`).
- **`training/`**: Isolated `BaselineTrainer` and `ModelEvaluator` separating
  PyTorch execution graphs from I/O and state management.
- **`telemetry/`**: Stateless DTO-driven (`dataclass`) metric calculation and
  visualization using `scikit-learn`, `matplotlib`, and `seaborn`.

## Phase 1 Results: The Classical Baseline

The classical baseline has been fully implemented and evaluated across multiple
random seeds to measure statistical variance.

**Key Finding:** As hypothesized by the foundational paper, the baseline model
is highly unstable.

Depending on the random initialization seed, the model's accuracy swings wildly
(from guessing the majority class at ~20% up to capturing spatial features at ~80%).

- This high variance acts as a "weak baseline."
- The primary metric for success in Phase 3 will not just be a higher peak AUC,
  but a **drastic reduction in statistical variance** across runs, proving the
  Quantum model's ability to reliably generalize.

## Quickstart

_(To be populated with `uv` instructions and CLI run commands)_
