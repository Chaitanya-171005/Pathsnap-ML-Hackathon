# PathSnap — ML Hackathon

**GPS Map Matching and Route Reconstruction using Machine Learning and Hidden Markov Models**

A machine learning project developed for the **UE24CS352A Machine Learning Hackathon**, Department of Computer Science and Engineering, PES University.

## Project Overview

PathSnap aims to reconstruct realistic vehicle routes from noisy GPS observations by matching geographical coordinates to road segments in a given road network.

GPS measurements can be inaccurate due to signal interference, urban obstacles, irregular sampling intervals and temporary signal loss. These inaccuracies make it difficult to determine the actual roads travelled by a vehicle.

The objective is to develop an end-to-end machine learning system that combines candidate road-segment classification with **Hidden Markov Models (HMMs) and Viterbi decoding** to reconstruct geographically coherent vehicle trajectories.

## Project Objectives

- Explore and preprocess GPS trajectory and road-network datasets.
- Generate candidate road segments for GPS observations.
- Engineer spatial, temporal and road-topology features.
- Train and compare at least five machine learning models.
- Select a model using cross-validation and experimental evidence.
- Perform Bayesian hyperparameter optimisation.
- Implement HMM-based route reconstruction using Viterbi decoding.
- Evaluate prediction accuracy, route continuity and computational efficiency.
- Visualise trajectories and predicted routes using Folium.
- Develop a distributed system with a FastAPI backend and an evaluator-facing user interface.

## Dataset

The dataset consists of GPS trajectories and associated road-network information.

Each trip contains four types of files:

| File | Description |
|------|-------------|
| `.track` | GPS observations containing longitude, latitude and timestamp |
| `.nodes` | Geographical coordinates of road-network nodes |
| `.arcs` | Connections between nodes representing directed road segments |
| `.route` | Reference sequence of road-segment IDs |

### Dataset Partitions

| Partition | Trips | Purpose |
|-----------|-------|---------|
| Training | 80 | Model training, cross-validation and hyperparameter optimisation |
| Student Testing | 10 | Development-stage model evaluation |
| Evaluator Testing | 10 | Reserved for final evaluation |

Evaluator-testing data will remain separate from model development to prevent data leakage.

## Proposed Methodology

The project will follow the pipeline:

```text
GPS Observations + Road Network
                |
                v
      Data Preprocessing
                |
                v
      Candidate Generation
                |
                v
       Feature Engineering
                |
                v
   Machine Learning Classifier
                |
                v
       HMM + Viterbi
                |
                v
       Route Reconstruction
                |
                v
      FastAPI Backend
                |
                v
     UI + Folium Visualisation
```

### Machine Learning Experimentation

The initial experimental plan includes models drawn from different algorithm families:

1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Decision Tree
4. Random Forest
5. Gradient Boosting

Models will be evaluated using appropriate classification, candidate-selection and route-level metrics. The final model will be selected based on reproducible experiments rather than predetermined assumptions.

### Research-Inspired Exploration

The team plans to investigate:

- Spatial and temporal feature engineering.
- Road-network connectivity and movement constraints.
- Adaptive candidate road-segment generation.
- GPS sampling gaps and their effect on route reconstruction.
- Classifier-only versus HMM-decoded route performance.
- Model interpretability and error analysis.

These are proposed research directions. Their effectiveness will be established through experiments on the permitted data partitions.

## Distributed Architecture

The final system will operate across two laptops.

**Laptop 1 — Backend and Machine Learning**

- Hosts the trained model and preprocessing components.
- Performs road-segment prediction and HMM/Viterbi decoding.
- Serves inference requests through FastAPI.
- Performs permitted server-side evaluation.

**Laptop 2 — User Interface**

- Displays available evaluation trips.
- Sends selected trip IDs to the backend.
- Displays predicted routes and evaluation results.
- Provides geographical visualisations using Folium.

Evaluator ground-truth labels will remain server-side and will not be supplied as inference features.

## Planned Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Hyperparameter Optimisation | Bayesian optimisation |
| Graph and Spatial Processing | NetworkX, spatial indexing libraries |
| Route Reconstruction | HMM, Viterbi |
| Backend | FastAPI |
| Frontend | Streamlit (proposed) |
| Map Visualisation | Folium |
| Testing | pytest |
| Version Control | Git and GitHub |

The technology stack may evolve based on implementation requirements and measured performance.

## Project Roadmap

- [x] Dataset exploration, validation and visualisation
- [x] Road-segment candidate generation and inference feature engineering
- [x] Reference-route alignment and qualified proxy labels
- [ ] Baseline model implementation
- [ ] Training and comparison of five ML models
- [ ] Bayesian hyperparameter optimisation
- [ ] HMM and Viterbi implementation
- [ ] Classifier-only versus decoded-route evaluation
- [ ] FastAPI backend development
- [ ] User interface development
- [ ] Two-laptop integration and testing
- [ ] Technical report and final submission

## Evaluation

Planned evaluation metrics include:

- Accuracy, Precision, Recall and Macro F1-score
- Observation-level Top-1 Accuracy
- Candidate Miss Rate
- Log Loss, where applicable
- Route Continuity
- Sequence-level Performance
- Training and Inference Time

All reported metrics and visualisations will be generated from actual experiments.

## Installation and Execution

PathSnap uses Python 3.13. Obtain the original organiser dataset separately and put the trip folders in `data/raw/train/` and `data/raw/student_test/`. Keep all four files for each trip in its own zero-padded trip directory.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.eda.run_eda --data-dir data/raw
python -m pytest -q
```

The EDA command writes CSV and Markdown reports to `reports/eda/` and PNG plots and Folium maps to `visualizations/eda/` by default. Use `--report-dir` and `--visual-dir` to change the output locations. Use `--gap-seconds 5 10 30 60`, `--speed-mps 55` and `--map-road-limit 12000` to set the diagnostic thresholds and visual road-network cap explicitly. The supplied raw files, local guideline PDFs and outputs carrying exact trip coordinates are Git-ignored. The maps use network-loaded assets and basemap tiles.

See the [Phase 1 handoff](docs/PHASE1_HANDOFF.md), [dataset schema](docs/DATASET_SCHEMA.md), [EDA findings](docs/EDA_FINDINGS.md) and [data contract](docs/DATA_CONTRACT.md).

### Phase 2A candidate generation

```bash
python -m src.candidates --partition train --trip-id 00000083 --radius-m 50 --top-k 8
python -m src.candidates.benchmark
python -m pytest -q
```

The export command writes two tables under `data/candidates/<partition>/<trip_id>/`. The observation table includes empty candidate groups. The candidate table uses directed arc IDs and inference-time features. See the [Phase 2A handoff](docs/PHASE2A_HANDOFF.md) for schemas, benchmark results, and the investigation of trip `00000054`.

### Phase 2B reference-route alignment

```bash
python -m src.alignment --partition train --plots
python -m src.alignment --partition student_test --policy reports/phase2b/train/policy.json
python -m pytest -q
```

Phase 2B compares nearest and sequence-aware alignment and applies qualification gates to derive observation-level proxy labels. Partition summaries are written to `reports/phase2b/` and observation-level files to the Git-ignored `data/alignments/`. See the [Phase 2B handoff](docs/PHASE2B_HANDOFF.md) for the feature and label join, results and limitations.

## Academic Integrity

All model training, experimentation, evaluation and implementation will be conducted independently by the team.

Relevant research papers will be acknowledged, and all experimental claims will be supported by reproducible results.

## Project Status

**Status: Phase 2B complete**

The repository includes EDA, directed road candidates, inference features and qualified observation-level proxy labels. The next phase is classifier training.

---

**PES University | Department of Computer Science and Engineering | Machine Learning Hackathon**
