# System Architecture: Dementia-Boost QTL Framework

## 1. Architectural Overview

The `dementia-boost` framework is designed around clean Software Engineering and MLOps principles, avoiding monolithic scripts and unstructured notebooks.

It enforces strict separation of concerns across data processing, modular neural/quantum architectures, decoupled training loops, and deterministic telemetry.

The system is partitioned into five main pillars:

```mermaid
flowchart TD
    subgraph Core ["Core Layer"]
        REP["Reproducibility Module<br/>(Deterministic Seed Locking across RNGs, PyTorch, cuDNN)"]
    end

    subgraph Data ["Data Engineering Layer"]
        ETL_NIFTI["OasisDataProcessor<br/>(3D NIfTI to 2D Slice Extraction & Patient Split)"]
        ETL_JPG["JpgDataIndexer<br/>(Regex Patient ID Indexing & Split)"]
        DS["OasisDataset / JpgOasisDataset<br/>(PyTorch Dataset Abstractions)"]
        DL["OasisDataLoader<br/>(Unified DataLoader & MinMax Normalization)"]
    end

    subgraph Models ["Model Architectures Layer"]
        FE["LeNetFeatureExtractor<br/>(Convolutional Spatial Backbone)"]
        CH["ClassicalClassifierHead<br/>(Linear Dense Head)"]
        QH["QuantumClassifierHead<br/>(Dressed Quantum Network - DQN)"]
        DC["DementiaClassifier<br/>(Dependency-Injected Orchestrator)"]
        BUILDER["Model Builder<br/>(Weight Loading, Freezing & Swapping)"]
    end

    subgraph Training ["Training & Inference Layer"]
        TR["BaselineTrainer<br/>(Epoch Loops, Loss, Checkpointing)"]
        EV["ModelEvaluator<br/>(Inference, Probabilities, Ground Truths)"]
    end

    subgraph Telemetry ["Telemetry & Reporting Layer"]
        LOG["Logger<br/>(Dual Console & Timestamped Logs)"]
        MET["MetricsAnalyzer<br/>(Accuracy, Precision, Recall, F1, AUC, Confusion Matrix)"]
        VIS["MetricsVisualizer<br/>(Seaborn Boxplots, ROC Curves, Heatmaps)"]
    end

    Core --> Data
    Core --> Models
    Core --> Training
    Data --> Training
    Models --> Training
    Training --> Telemetry
```

---

## 2. Component Breakdown

### 2.1 Core Layer (`src/dementia_boost/core/`)

- **`reproducibility.py`**: Provides `set_seed(seed: int)`, which enforces deterministic execution across Python's `random`, `numpy`, PyTorch CPU/CUDA engines, and locks `torch.backends.cudnn.deterministic = True` with `benchmark = False`.

### 2.2 Data Engineering Layer (`src/dementia_boost/data/`)

The data pipeline eliminates patient-level data leakage across longitudinal MRI sessions:

- **`data_processor.py` (`OasisDataProcessor`)**: Streams raw 3D NIfTI/HDR volumes, groups visits by unique `Subject ID`, isolates subjects into train/test cohorts, extracts the central 2D axial slice, and serializes processed tensors to disk (`.pt`).

- **`jpg_indexer.py` (`JpgDataIndexer`)**: Uses regular expressions to extract patient IDs from 2D image filenames (`oas2?_\d+`), enforces patient-level train/test isolation, and generates immutable CSV index files (`train_jpg_index.csv`, `test_jpg_index.csv`).

- **`dataset.py`**:

  - `OasisDataset`: Loads serialized `.pt` image-label pairs.
  - `JpgOasisDataset`: Dynamically loads JPGs from disk via PIL, converting them to grayscale float tensors.

- **`data_loader.py`**:
  - `OasisDataLoader`: Unified factory creating PyTorch `DataLoader` instances for both `nifti` and `jpg` modalities.
  - `MinMaxNormalize`: Custom transform performing per-sample dynamic range squashing into `[0.0, 1.0]`.

```mermaid
flowchart LR
    subgraph RawData ["Raw OASIS-II Data"]
        NIFTI_FILES["3D NIfTI / HDR Volumes"]
        JPG_FILES["2D Grayscale JPG Images"]
    end

    subgraph SplitLogic ["Patient-Level Leakage Prevention"]
        SUBJ_SPLIT["Group Scans by Subject ID<br/>(Manual Overrides + 70/30 Split)"]
    end

    subgraph Pipelines ["Processing Pipelines"]
        NIFTI_PIPE["OasisDataProcessor<br/>- Middle Axial Slice Extraction<br/>- Serialized .pt (Tensor, Label)"]
        JPG_PIPE["JpgDataIndexer<br/>- Regex ID Parsing<br/>- Immutable train/test CSV Index"]
    end

    subgraph Loaders ["DataLoader Factory"]
        TRANSFORMS["Transform Pipeline<br/>- Resize (128, 128)<br/>- MinMaxNormalize<br/>- Normalize Mean/Std"]
        LOADER["OasisDataLoader<br/>(Batching, Shuffling, Device Pinning)"]
    end

    NIFTI_FILES --> SUBJ_SPLIT
    JPG_FILES --> SUBJ_SPLIT
    SUBJ_SPLIT --> NIFTI_PIPE
    SUBJ_SPLIT --> JPG_PIPE
    NIFTI_PIPE --> TRANSFORMS
    JPG_PIPE --> TRANSFORMS
    TRANSFORMS --> LOADER
```

---

## 3. Model Architecture & Hybrid Dressed Quantum Network (DQN)

The model ecosystem uses **Dependency Injection** via the `DementiaClassifier` container. A common spatial backbone (`LeNetFeatureExtractor`) can be dynamically paired with either a classical dense head or a variational quantum head.

```mermaid
flowchart TD
    INPUT["Input MRI Tensor<br/>(Batch, 1, 128, 128)"]

    subgraph Backbone ["LeNetFeatureExtractor (Backbone)"]
        C1["Conv2D(1 -> 8, k=4, s=2) + ReLU + MaxPool(2x2, s=1)"]
        C2["Conv2D(8 -> 16, k=8, s=2) + ReLU + MaxPool(2x2, s=1)"]
        C3["Conv2D(16 -> 32, k=8, s=2) + ReLU + MaxPool(2x2, s=1)"]
        C4["Conv2D(32 -> 64, k=4, s=1) + ReLU"]
        FMAP["Output Feature Map<br/>(Batch, 64, 6, 6) = 2304 features"]
    end

    INPUT --> C1 --> C2 --> C3 --> C4 --> FMAP

    subgraph ClassicalHead ["Option A: ClassicalClassifierHead (CTL)"]
        FLAT_C["Flatten -> 2304"]
        D1["Linear(2304 -> 5) + Dropout(0.5) + ReLU"]
        D2["Linear(5 -> 1)"]
        SIG["Optional Sigmoid / Logit"]
        FLAT_C --> D1 --> D2 --> SIG
    end

    subgraph QuantumHead ["Option B: QuantumClassifierHead (DQN / QTL)"]
        FLAT_Q["Flatten -> 2304"]
        PRE["Pre-Net: Linear(2304 -> n_qubits)"]
        SCALE["Angle Scaling: tanh(x) * (pi / 2)"]

        subgraph VQC ["Variational Quantum Circuit (Ansatz)"]
            ENC["Angle Embedding: RZ(inputs) on all qubits"]
            subgraph Repetitions ["Layers: 1 to n_layers (default: 4)"]
                RZ1["RZ(weights[l, 0, i])"]
                CNOT["Entangling Ring: CNOT(i, (i+1)%n)"]
                RZ2["RZ(weights[l, 1, i])"]
                CRY["Controlled-RY(weights[l, 2, target], wires=[ctrl, target])"]
                RZ1 --> CNOT --> RZ2 --> CRY
            end
            MEAS["Expectation Values: <PauliZ> on all qubits"]
            ENC --> Repetitions --> MEAS
        end

        POST["Post-Net: Linear(n_qubits -> 1)"]

        FLAT_Q --> PRE --> SCALE --> ENC
        MEAS --> POST
    end

    FMAP -.-> ClassicalHead
    FMAP -.-> QuantumHead
```

### 3.1 Mathematical Formulation of the Dressed Quantum Network (DQN)

1. **Classical Feature Extraction**:
   $$z_{\text{raw}} = f_{\text{backbone}}(x) \in \mathbb{R}^{2304}$$

2. **Pre-Net Dimensionality Reduction & Angle Mapping**:
   $$\tilde{z} = \tanh(W_{\text{pre}} z_{\text{raw}} + b_{\text{pre}}) \cdot \frac{\pi}{2} \in \left[-\frac{\pi}{2}, \frac{\pi}{2}\right]^{n_{\text{qubits}}}$$

3. **Quantum Encoding & Parameterized Evolution**:
   $$|\psi(\tilde{z})\rangle = \bigotimes_{i=1}^{n_{\text{qubits}}} R_z(\tilde{z}_i) |0\rangle^{\otimes n_{\text{qubits}}}$$
   $$|\phi_\theta(\tilde{z})\rangle = \prod_{l=1}^{L} \left[ U_{\text{CRY}}(\theta_{l,2}) U_{R_z}(\theta_{l,1}) U_{\text{CNOT}} U_{R_z}(\theta_{l,0}) \right] |\psi(\tilde{z})\rangle$$

4. **Quantum Measurement (POVM)**:
   $$\langle Z_i \rangle = \langle \phi_\theta(\tilde{z}) | \sigma_z^{(i)} | \phi_\theta(\tilde{z}) \rangle \in [-1, 1]$$

5. **Post-Net Classification Logit**:
   $$\hat{y}_{\text{logit}} = W_{\text{post}} \begin{bmatrix} \langle Z_1 \rangle \\ \vdots \\ \langle Z_n \rangle \end{bmatrix} + b_{\text{post}}$$

---

## 4. Transfer Learning Pipeline

The project supports three training modes orchestrated through `builder.py`:

```mermaid
sequenceDiagram
    autonumber
    participant D as OASIS-II Dataset
    participant Base as Classical Baseline CNN
    participant CTL as Classical Transfer Learning (CTL)
    participant QTL as Quantum Transfer Learning (QTL)

    Note over Base: Step 1: Train End-to-End Baseline
    D->>Base: Train LeNet Feature Extractor + Classical Dense Head
    Base-->>Base: Evaluate variance & save weights (baseline_seed_*.pt)

    Note over CTL,QTL: Step 2: Transfer Learning (Backbone Frozen)
    Base->>CTL: Load backbone weights & Freeze parameters
    CTL->>CTL: Re-initialize Dense Head (Glorot Uniform) & Fine-tune

    Base->>QTL: Load backbone weights & Freeze parameters
    QTL->>QTL: Attach Dressed Quantum Network (Pre-Net + VQC + Post-Net)
    QTL->>QTL: Optimize Quantum + Classical Head Parameters
```

---

## 5. Telemetry, Metric Aggregation & Evaluation Flow

Training and evaluation lifecycles are decoupled from serialization and plotting:

```mermaid
flowchart LR
    subgraph Execution ["Model Execution"]
        TRAINER["BaselineTrainer<br/>(Loss, Optimization, StepLR)"]
        EVAL["ModelEvaluator<br/>(Batch Inference, Raw Probs, Labels)"]
    end

    subgraph MetricsDTO ["Stateless Metrics Engine"]
        ANALYZER["MetricsAnalyzer<br/>- Accuracy, Precision, Recall, F1, AUC<br/>- Confusion Matrix"]
        DTO_INDIV["EvaluationResult (DTO)"]
        DTO_AGGR["AggregateMetrics (DTO)<br/>(mean, std, min, max)"]
    end

    subgraph Storage ["Telemetry Storage"]
        JSON_STORE["JSON File<br/>(individual_runs + aggregated_stats)"]
        LOGS["Timestamped Logs<br/>(logs/YYYYMMDD_HHMMSS_*.log)"]
    end

    subgraph Visualization ["Telemetry Visualizer"]
        BOXPLOT["MetricsVisualizer.plot_metric_distributions()"]
        ROC_AGG["MetricsVisualizer.plot_comparative_roc()"]
        ROC_ISO["MetricsVisualizer.plot_isolated_roc()"]
        CONF_MAT["MetricsVisualizer.plot_confusion_matrix()"]
    end

    TRAINER --> EVAL
    EVAL --> ANALYZER
    ANALYZER --> DTO_INDIV --> JSON_STORE
    ANALYZER --> DTO_AGGR --> JSON_STORE
    TRAINER --> LOGS
    JSON_STORE --> BOXPLOT
    JSON_STORE --> ROC_AGG
    JSON_STORE --> ROC_ISO
    JSON_STORE --> CONF_MAT
```

---

## 6. Directory and Module Structure

```
dementia-boost/
├── docs/                                  # Research documentation and specifications
│   ├── architecture.md                    # System architecture and technical overview
│   ├── qtl_research_goals_activities.md   # 12-month research schedule and activities
│   └── qtl_to_boost_dementia_detection.md # Foundational paper (Bhowmik et al. 2025)
├── src/
│   └── dementia_boost/
│       ├── core/                          # Reproducibility & runtime utilities
│       │   └── reproducibility.py         # Deterministic seed locker
│       ├── data/                          # Data processing, indexing & loaders
│       │   ├── data_loader.py             # Unified DataLoader & normalization transforms
│       │   ├── data_processor.py          # NIfTI 3D/2D ETL & patient-split orchestrator
│       │   ├── dataset.py                 # OasisDataset & JpgOasisDataset classes
│       │   └── jpg_indexer.py             # Regex patient ID parser & CSV indexer
│       ├── models/                        # Neural & Quantum network architectures
│       │   ├── builder.py                 # Factory functions for CTL and QTL models
│       │   ├── classical_cnn/             # Classical CNN backbone and dense heads
│       │   │   ├── classifier.py          # DementiaClassifier orchestrator
│       │   │   ├── feature_extractor.py   # LeNetFeatureExtractor backbone
│       │   │   └── heads.py               # ClassicalClassifierHead
│       │   └── quantum_cnn/               # Quantum neural network components
│       │       ├── circuit.py             # PennyLane QNode and custom Ansatz definition
│       │       └── heads.py               # QuantumClassifierHead (DQN)
│       ├── training/                      # Training and inference lifecycle runners
│       │   ├── evaluator.py               # Weight loading and inference predictor
│       │   └── trainer.py                 # Training loop with validation & checkpointing
│       └── telemetry/                     # Metrics calculation, serialization & plotting
│           ├── logger.py                  # Standardized dual console/file logger
│           ├── metrics.py                 # MetricsAnalyzer and DTO definitions
│           └── visualizer.py              # Publication-ready Seaborn/Matplotlib plots
├── scripts/                               # CLI entry-points for training and evaluation
│   ├── cherrypicked_baseline/             # Reference baseline execution scripts
│   ├── jpg/                               # Training and evaluation scripts for JPG pipeline
│   ├── metrics/                           # Quantitative comparison and delta report generators
│   └── qtl/                               # Multi-seed QTL training scripts
└── tests/                                 # Unit and integration test suites
```
