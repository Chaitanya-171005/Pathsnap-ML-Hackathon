"""Fixed baseline pipelines and inference over Phase 2A candidate records."""

from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


NUMERIC = ("distance_m", "arc_length_m", "distance_rank", "delta_t_s", "speed_mps",
           "heading_difference_deg", "projection_fraction", "from_in_degree", "from_out_degree",
           "to_in_degree", "to_out_degree")
FLAGS = ("time_step_valid", "speed_valid", "heading_difference_valid", "projection_valid")
FEATURES = NUMERIC + FLAGS
feature_values = attrgetter(*FEATURES)
SEED = 2026
MODELS = ("logistic_regression", "knn", "decision_tree", "random_forest", "gradient_boosting")


def feature_matrix(candidates):
    features = np.asarray([feature_values(row) for row in candidates], dtype=np.float32)
    return features.reshape(-1, len(FEATURES))


def build_model(name: str, workers: int = 1) -> Pipeline:
    estimators = {
        "logistic_regression": LogisticRegression(solver="lbfgs", max_iter=300, random_state=SEED),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree",
                                    leaf_size=60, metric="euclidean", n_jobs=workers),
        "decision_tree": DecisionTreeClassifier(max_depth=14, min_samples_leaf=30, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=48, max_depth=14, min_samples_leaf=20,
                                               max_features="sqrt", n_jobs=workers, random_state=SEED),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
            max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=False, random_state=SEED),
    }
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if name in ("logistic_regression", "knn"):
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), list(range(len(NUMERIC)))),
        ("validity", "passthrough", list(range(len(NUMERIC), len(FEATURES)))),
    ])
    return Pipeline([("features", preprocess), ("classifier", estimators[name])])


def positive_scores(model, matrix: np.ndarray, batch_size: int = 8192,
                    native_threads: int = 1) -> np.ndarray:
    if batch_size < 1 or native_threads < 1:
        raise ValueError("Prediction batch size and native thread count must be positive")
    [positive] = np.flatnonzero(model.classes_ == 1)
    result = np.empty(len(matrix), dtype=np.float64)
    with threadpool_limits(native_threads):
        for start in range(0, len(matrix), batch_size):
            stop = min(start + batch_size, len(matrix))
            result[start:stop] = model.predict_proba(matrix[start:stop])[:, positive]
    if not (np.isfinite(result).all() and ((0 <= result) & (result <= 1)).all()):
        raise ValueError("Invalid positive probability")
    return result


@dataclass(frozen=True, slots=True)
class Prediction:
    partition: str
    trip_id: str
    observation_id: int
    arc_id: int
    candidate_rank: int
    model_id: str
    probability: float
    prediction_rank: int


def rank_indices(groups, candidate_ranks, arc_ids, scores):
    """Rank per group by descending raw score, Phase 2A rank, then arc ID."""
    return np.lexsort((arc_ids, candidate_ranks, -scores, groups))


def predict_trip(model, model_id, observation_batches, batch_size=8192, native_threads=1):
    """Yield original observations with ranked predictions, including empty groups."""
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    pending = []
    count = 0

    def flush():
        scores = positive_scores(model, feature_matrix(c for _, rows in pending for c in rows),
                                 batch_size, native_threads)
        offset = 0
        for observation, rows in pending:
            group_scores = scores[offset:offset + len(rows)]
            offset += len(rows)
            order = rank_indices(np.zeros(len(rows)), [r.distance_rank for r in rows],
                                 [r.arc_id for r in rows], group_scores)
            yield observation, tuple(
                Prediction(row.partition, row.trip_id, row.observation_id, row.arc_id,
                           row.distance_rank, model_id, float(group_scores[j]), rank)
                for rank, j in enumerate(order, 1) for row in (rows[j],))

    for observation, rows in observation_batches:
        pending.append((observation, rows))
        count += max(1, len(rows))
        if count >= batch_size:
            yield from flush()
            pending.clear()
            count = 0
    if pending:
        yield from flush()


def load_model(path: Path):
    """Load a trusted local joblib artifact produced by the experiment."""
    bundle = joblib.load(path)
    if tuple(bundle["features"]) != FEATURES or bundle["model_id"] not in MODELS:
        raise ValueError("Artifact does not match the current feature contract")
    return bundle
