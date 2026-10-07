# Quantum Transfer Learning to Boost Dementia Detection

## Overview

This project reproduces and extends the foundational research on applying **Quantum Transfer Learning (QTL)** to the classification of Dementia from structural brain MRI scans (OASIS-II dataset), based on Bhowmik et al. (2025).

The core hypothesis is that a **Dressed Quantum Neural Network (DQNN)**, composed of a classical pre-net for dimensionality reduction, a Parameterized Quantum Circuit (PQC / Ansatz) for feature processing in Hilbert space, and a classical post-net for logit output, can substantially boost performance and reduce statistical variance when transferring features from a "weak" or suboptimal classical Convolutional Neural Network (CNN) backbone.

---

## Project Goals

The research follows a multi-stage execution schedule (see `docs/qtl_research_goals_activities.md` for the complete 12-month plan). We are currently concluding **Activity 1** and transitioning into **Activity 2**:

### Activity 1: Baseline Integration and Reproduction _(Current / Concluding)_

- **ETL & Data Pipeline**: Ingest and process the OASIS-II longitudinal MRI dataset from raw NIfTI volumes into 2D slices, enforcing a deterministic patient-level split into train, validation, and test cohorts to prevent data leakage across visits.

- **Classical Baseline & Transfer Learning (CTL)**: Implement the LeNet-based classical CNN architecture to establish the baseline performance and fine-tune classical dense heads under multi-seed initializations.

- **Hybrid Quantum Architecture (QTL)**: Implement the Dressed Quantum Network using **Angle Embedding** and the foundational variational ansatz with PennyLane state-vector simulators (`lightning.qubit`/`default.qubit`). A Hadamard layer precedes the RZ embedding: the circuit as drawn in the paper keeps the state at $|0\ldots0\rangle$ and is a constant function (see `docs/architecture.md`, section 3.1).

- **Multi-Seed Benchmark**: Establish reference metrics across multiple random seeds to measure statistical variance and evaluate the weak-baseline hypothesis.

### Activity 2: Alternative Feature Maps Modeling _(Next Activity)_

- **Advanced Quantum Encodings**: Mathematically formulate and implement expressive quantum data encodings beyond angle embedding, such as the **$ZZ$-Feature Map**, higher-order Pauli feature maps, and custom entanglement topologies.

- **Cross-Platform State-Vector Simulation**: Prototype alternative Ansätze and encoding circuits in both Python (`PennyLane`, `Qiskit`) and Julia (`Yao.jl`) for high-performance state-vector simulation and gradient evaluation.

- **Expressibility & Class Separability**: Evaluate decision boundary geometry, trainability (mitigating barren plateaus), and noise resilience under simulated NISQ channels before physical IBM QPU execution.

---

## System Architecture

The codebase enforces strict modularity and MLOps practices, keeping neural network definitions, quantum execution graphs, ETL pipelines, and metric analysis decoupled:

- **`core/`**: Centralized determinism and seed locking across Python, NumPy, PyTorch, and cuDNN (`reproducibility.py`), the typed run specification with the hash IDs and label derived from it (`identity.py`), and the on-disk layout of every run's artifacts (`layout.py`).

- **`data/`**: Data loading and ETL pipelines with patient-level leakage prevention (canonical subject IDs, a deterministic stratified train/val/test split, a `split_manifest.json`, and an integrity check of the cohort files against it), NIfTI slice loaders, custom `MinMaxNormalize` transforms, and in-memory feature embedding caching for fast transfer learning (`subject_ids.py`, `split.py`, `split_manifest.py`, `split_guards.py`, `cohort_audit.py`, `data_loader.py`, `data_processor.py`, `dataset.py`, `embedding_cache.py`).

- **`models/`**: Dependency-injected architectures pairing a `LeNetFeatureExtractor` backbone with interchangeable heads: `ClassicalClassifierHead` (CTL), `QuantumClassifierHead` (DQN / QTL on PennyLane), or `QiskitQuantumClassifierHead` (the same DQN on Qiskit Primitives V2, with an injectable estimator and Aer as the default noiseless backend) built via `builder.py`.

- **`training/`**: Isolated `BaselineTrainer` and `ModelEvaluator` separating training/optimization loops from model definitions and file I/O, and `checkpoint_evaluation` to evaluate every checkpoint on the validation and test cohorts. The trainer only sees the train and validation loaders, records a per-epoch `TrainingHistory` and saves it as JSON, but never plots.

- **`telemetry/`**: Stateless DTO-driven metric computation (`MetricsAnalyzer`), JSON persistence of evaluation metrics and training histories, run discovery from the histories and the lookup table of runs (`run_listing.py`, `run_table.py`), model selection on validation metrics (`selection.py`), the comparative report across paradigms (`report.py`), and dual logging (`setup_logger`).

- **`viz/`**: Publication-ready plots (`MetricsVisualizer`) and the plotting steps that draw every configuration's figures and the loss curves from the saved histories (`run_plots.py`); the scripts in `scripts/viz` are thin callers.

For full technical specifications and detailed Mermaid architectural diagrams, see **[docs/architecture.md](docs/architecture.md)**.

---

## Activity 1 Results

> [!NOTE]
> Below are placeholders for empirical metrics obtained across multi-seed evaluations on the OASIS-II dataset. Recorded values will be filled in upon completion of full-dataset runs.

> [!WARNING]
> QTL runs produced before the Hadamard layer was added used a constant circuit, so they measure a bias-only classifier and must not be used. All quantum results are regenerated with the corrected circuit.

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
# Create the virtual environment and sync dependencies
uv sync
```

The environment is managed entirely by `uv`: there is no need to activate it. Run every script and tool through `uv run` (for example `uv run scripts/etl_pipeline.py` or `uv run pytest`).

### 2. Data Indexing & ETL Pipeline

```bash
# Process raw 3D NIfTI/HDR files into 2D slice tensors, split by patient into
# train / val / test, and write data/results/split_manifest.json
uv run scripts/etl_pipeline.py

# Check that no patient appears in more than one cohort (exits with 1 if one does)
uv run scripts/audit_cohorts.py
```

The split is settled by the constants at the top of `scripts/etl_pipeline.py`: the seed, the test ratio (default 0.3), the validation ratio (default 0.2, taken from the non-test subjects), and the manual overrides `MANUAL_TRAIN_IDS` and `MANUAL_TEST_IDS`, which pin subjects to a cohort before the ratios are applied (subjects pinned to train never enter validation). Use canonical subject IDs such as `OAS2_0001`; an unknown ID, a visit ID such as `OAS2_0001_MR2`, or an ID in both lists is rejected.

The ETL is safe to rerun. It builds the whole split in a staging directory and only then replaces the `.pt` files in `data/results/train`, `val`, and `test`, so an interrupted run leaves the previous split untouched, and the same arguments reproduce the same files. Other files in those directories are never touched. Subjects that cannot be used (no raw data, no row in the CSV, no readable 3D volume, or the `Converted` group) are listed with a reason in `split_manifest.json`. The data loader refuses to load cohorts that do not match the manifest, so a stale or hand-edited directory is never trained on silently.

> [!IMPORTANT]
> **Evaluation protocol.** Models are trained on `train` and monitored on `val`. Model selection (for example, which baseline backbone the transfer learning heads use) reads validation metrics only. The `test` cohort is never used to train or to choose; every saved checkpoint is evaluated on it once and the results are only reported.

### 3. Training Models

```bash
# Train classical baseline CNN across multiple random seeds
uv run scripts/training/train_baseline.py

# Evaluate the baselines on val and test. The validation metrics it writes
# (data/results/metrics/baseline/<config_id>/val_results.json) are what the next
# scripts read to select the backbone.
uv run scripts/metrics/evaluate_baseline.py

# Train Classical Transfer Learning (CTL) dense heads on the baseline selected on
# validation (best validation AUC-ROC, then F1, then lowest log loss)
uv run scripts/training/train_tl_multiseed.py

# Train Quantum Transfer Learning (QTL) Dressed Quantum Network
uv run scripts/training/train_qtl_multiseed.py

# Train the same Dressed Quantum Network on Qiskit (Aer state-vector, SPSA gradients)
uv run scripts/training/train_qiskit_qtl_multiseed.py
```

Every training script writes a checkpoint and a per-epoch history (loss, accuracy, learning rate, duration, and the full run specification) for each seed, and skips a run whose checkpoint already exists. The `val_loss` and `val_acc` fields of the history are measured on the validation cohort. The training scripts do not produce plots; see the next step.

#### Where the results go, and how to find a run

Runs are identified by hashes of their specification, not by names: `config_id` is shared by all seeds of one configuration (paradigm, qubits, layers, learning rates, backbone, split, and so on) and `run_id` identifies one seed of it. Everything lives under `data/results`:

```
checkpoints/<paradigm>/<config_id>/seed_<n>.pt
histories/<paradigm>/<config_id>/seed_<n>.json      history, with the full run specification
histories/<paradigm>/<config_id>/config.json         the configuration and its label
metrics/<paradigm>/<config_id>/{val,test}_results.json
plots/<paradigm>/<config_id>/
```

Changing any hyperparameter gives a new `config_id`, so a new configuration never overwrites or gets skipped in place of an old one. To see what a hash is, use the read-only lookup, which is rebuilt from the histories on every call:

```bash
uv run scripts/list_runs.py                    # one row per configuration (label, split, seeds)
uv run scripts/list_runs.py --runs             # one row per run (seed, run_id, finished, paths)
uv run scripts/list_runs.py --paradigm qtl     # only one paradigm
uv run scripts/list_runs.py --find 62753438    # full spec and files of a config or run ID prefix
uv run scripts/list_runs.py --runs --csv runs.csv
```

### 4. Evaluation & Telemetry Visualization

```bash
# Evaluate models and generate metrics JSON. For every configuration each script
# writes val_results.json and test_results.json to
# data/results/metrics/<paradigm>/<config_id>/. evaluate_baseline.py was already
# run in step 3.
uv run scripts/metrics/evaluate_baseline.py
uv run scripts/metrics/evaluate_tl.py
uv run scripts/metrics/evaluate_qtl.py
uv run scripts/metrics/evaluate_qiskit_qtl.py

# Generate the comparative report: mean and std across seeds on test, the run
# selected on validation next to it, and percentage changes between the means.
# Each paradigm contributes its only configuration; if a paradigm has several,
# name one with --config <paradigm>=<config_id>. Paradigms trained on different
# splits are refused. Written to data/results/metrics/comparative_report.json.
uv run scripts/metrics/generate_improvement_report.py

# Plot metric distributions, ROC curves, and confusion matrices for every
# configuration, to data/results/plots/<paradigm>/<config_id>/. The isolated ROC
# curve and confusion matrix show the run selected on validation.
uv run scripts/viz/visualize_baselines.py
uv run scripts/viz/visualize_tl.py
uv run scripts/viz/visualize_qtl.py
uv run scripts/viz/visualize_qiskit_qtl.py

# Plot loss curves from the saved training histories: per-run and distribution
# plots go to data/results/plots/<paradigm>/<config_id>/, and the comparison
# across configurations to data/results/plots/loss_comparison.png. Can run while
# sweeps are in progress.
uv run scripts/viz/visualize_loss.py
```

---

## References

1. **Bhowmik, S., Perciano, T., & Thapliyal, H. (2025).** _Quantum Transfer Learning to Boost Dementia Detection_. Proceedings of the Great Lakes Symposium on VLSI 2025, 849–853.
2. **Marcus, D. S. et al. (2007).** _Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults_. Journal of Computer Assisted Tomography, 31(6), 1498–1504.
3. Complete bibliography available in `docs/qtl_research_goals_activities.md`.
