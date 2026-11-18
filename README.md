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

- [ ] Dataset exploration and visualisation
- [ ] Data preprocessing and candidate generation
- [ ] Feature engineering and reference-route alignment
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

Installation steps, dependency files, training commands and deployment instructions will be documented as the implementation progresses.

The final repository will include instructions for reproducing experiments and running the application on two laptops.

## Academic Integrity

All model training, experimentation, evaluation and implementation will be conducted independently by the team.

Relevant research papers will be acknowledged, and all experimental claims will be supported by reproducible results.

## Project Status

**Status: Planning and Initial Development**

The current focus is dataset exploration, research review and establishing the initial processing pipeline.

---

**PES University | Department of Computer Science and Engineering | Machine Learning Hackathon**
