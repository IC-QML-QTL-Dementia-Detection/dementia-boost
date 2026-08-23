# Quantum Transfer Learning to Boost Dementia Detection

## Overview

This project reproduces and extends the foundational research on applying **Quantum Transfer Learning (QTL)** to the classification of Dementia from structural brain MRI scans (OASIS-II dataset), based on Bhowmik et al. (2025).

The core hypothesis is that a **Dressed Quantum Neural Network (DQNN)**, composed of a classical pre-net for dimensionality reduction, a Parameterized Quantum Circuit (PQC / Ansatz) for feature processing in Hilbert space, and a classical post-net for logit output, can substantially boost performance and reduce statistical variance when transferring features from a "weak" or suboptimal classical Convolutional Neural Network (CNN) backbone.

---

## Project Goals

The research follows a multi-stage execution schedule (see `docs/qtl_research_goals_activities.md` for the complete 12-month plan). We are currently concluding **Activity 1** and transitioning into **Activity 2**:

### Activity 1: Baseline Integration and Reproduction _(Current / Concluding)_

- **ETL & Data Pipeline**: Ingest and process the OASIS-II longitudinal MRI dataset across both raw NIfTI volumes and 2D image slices, enforcing strict patient-level splitting to prevent data leakage across visits.

- **Classical Baseline & Transfer Learning (CTL)**: Implement the LeNet-based classical CNN architecture to establish the baseline performance and fine-tune classical dense heads under multi-seed initializations.

- **Hybrid Quantum Architecture (QTL)**: Implement the Dressed Quantum Network using **Angle Embedding** and the foundational variational ansatz with PennyLane state-vector simulators (`lightning.qubit`/`default.qubit`).

- **Multi-Seed Benchmark**: Establish reference metrics across multiple random seeds to measure statistical variance and evaluate the weak-baseline hypothesis.

### Activity 2: Alternative Feature Maps Modeling _(Next Activity)_

- **Advanced Quantum Encodings**: Mathematically formulate and implement expressive quantum data encodings beyond angle embedding, such as the **$ZZ$-Feature Map**, higher-order Pauli feature maps, and custom entanglement topologies.

- **Cross-Platform State-Vector Simulation**: Prototype alternative Ansätze and encoding circuits in both Python (`PennyLane`, `Qiskit`) and Julia (`Yao.jl`) for high-performance state-vector simulation and gradient evaluation.

- **Expressibility & Class Separability**: Evaluate decision boundary geometry, trainability (mitigating barren plateaus), and noise resilience under simulated NISQ channels before physical IBM QPU execution.

---

## System Architecture

The codebase enforces strict modularity and MLOps practices, keeping neural network definitions, quantum execution graphs, ETL pipelines, and metric analysis decoupled:

- **`core/`**: Centralized determinism and seed locking across Python, NumPy, PyTorch, and cuDNN (`reproducibility.py`).

- **`data/`**: Data loading and ETL pipelines with patient-level leakage prevention, dynamic PIL/NIfTI loaders, and custom `MinMaxNormalize` transforms (`data_loader.py`, `data_processor.py`, `jpg_indexer.py`, `dataset.py`).

- **`models/`**: Dependency-injected architectures pairing a `LeNetFeatureExtractor` backbone with interchangeable heads: `ClassicalClassifierHead` (CTL) or `QuantumClassifierHead` (DQN / QTL) built via `builder.py`.

- **`training/`**: Isolated `BaselineTrainer` and `ModelEvaluator` separating training/optimization loops from model definitions and file I/O.

- **`telemetry/`**: Stateless DTO-driven metric computation (`MetricsAnalyzer`), JSON persistence, dual logging (`setup_logger`), and publication-ready visualizers (`MetricsVisualizer`).

For full technical specifications and detailed Mermaid architectural diagrams, see **[docs/architecture.md](docs/architecture.md)**.

---

## Activity 1 Results

> [!NOTE]
> Below are placeholders for empirical metrics obtained across multi-seed evaluations on the OASIS-II dataset. Recorded values will be filled in upon completion of full-dataset runs.

### Quantitative Comparison Across Paradigms

| Metric           | Classical Baseline (Mean ± Std) | Classical Transfer Learning (CTL) (Mean ± Std) | Quantum Transfer Learning (QTL) (Mean ± Std) | QTL $\Delta$ vs Baseline (%) | QTL $\Delta$ vs CTL (%) |
| :--------------- | :-----------------------------: | :--------------------------------------------: | :------------------------------------------: | :--------------------------: | :---------------------: |
| **Accuracy (%)** |        `[ PLACEHOLDER ]`        |               `[ PLACEHOLDER ]`                |              `[ PLACEHOLDER ]`               |      `[ PLACEHOLDER ]`       |    `[ PLACEHOLDER ]`    |
| **Precision**    |        `[ PLACEHOLDER ]`        |               `[ PLACEHOLDER ]`                |              `[ PLACEHOLDER ]`               |      `[ PLACEHOLDER ]`       |    `[ PLACEHOLDER ]`    |
| **Recall**       |        `[ PLACEHOLDER ]`        |               `[ PLACEHOLDER ]`                |              `[ PLACEHOLDER ]`               |      `[ PLACEHOLDER ]`       |    `[ PLACEHOLDER ]`    |
| **F1-Score**     |        `[ PLACEHOLDER ]`        |               `[ PLACEHOLDER ]`                |              `[ PLACEHOLDER ]`               |      `[ PLACEHOLDER ]`       |    `[ PLACEHOLDER ]`    |
| **AUC-ROC**      |        `[ PLACEHOLDER ]`        |               `[ PLACEHOLDER ]`                |              `[ PLACEHOLDER ]`               |      `[ PLACEHOLDER ]`       |    `[ PLACEHOLDER ]`    |

### Key Observations & Findings

- **Baseline Instability**: `[ PLACEHOLDER: Summarize baseline variance across random seeds ]`
- **Classical Transfer Learning**: `[ PLACEHOLDER: Summarize performance changes when fine-tuning dense layers ]`
- **Quantum Transfer Learning**: `[ PLACEHOLDER: Summarize impact of Dressed Quantum Network on recall and variance reduction ]`

---

## Quickstart

### 1. Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management:

```bash
# Create virtual environment and sync dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Data Indexing & ETL Pipeline

```bash
# Process raw 3D NIfTI/HDR files into 2D slice tensors
python scripts/nifti/etl_pipeline.py

# Or generate deterministic patient-split CSV index for JPG dataset
python scripts/jpg/run_jpg_indexing.py
```

### 3. Training Models

```bash
# Train classical baseline CNN across multiple random seeds
python scripts/nifti/train_baseline.py
python scripts/jpg/train_baseline_jpg.py

# Train Classical Transfer Learning (CTL) dense heads on optimal baseline
python scripts/nifti/train_tl_multiseed.py
python scripts/jpg/train_tl_multiseed_jpg.py

# Train Quantum Transfer Learning (QTL) Dressed Quantum Network
python scripts/qtl/train_qtl_multiseed_jpg.py
```

### 4. Evaluation & Telemetry Visualization

```bash
# Evaluate models and generate metrics JSON
python scripts/nifti/evaluate_baseline.py
python scripts/nifti/evaluate_tl.py

# Generate comparison and delta improvement report
python scripts/nifti/generate_improvement_report.py
python scripts/metrics/generate_improvement_report_jpg.py

# Plot metric distributions, ROC curves, and confusion matrices
python scripts/nifti/visualize_baselines.py
python scripts/nifti/visualize_tl.py
```

---

## References

1. **Bhowmik, S., Perciano, T., & Thapliyal, H. (2025).** _Quantum Transfer Learning to Boost Dementia Detection_. Proceedings of the Great Lakes Symposium on VLSI 2025, 849–853.
2. **Marcus, D. S. et al. (2007).** _Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults_. Journal of Computer Assisted Tomography, 31(6), 1498–1504.
3. Complete bibliography available in `docs/qtl_research_goals_activities.md`.
