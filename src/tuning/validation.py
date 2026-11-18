"""Nested trip splits and fitting through the frozen Phase 3 pipeline."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.models.evaluation import make_folds, observation_metrics, ranked_targets
from src.models.inference import build_model, positive_scores


MODEL = "gradient_boosting"


@dataclass(frozen=True)
class Fold:
    fit: np.ndarray
    valid: np.ndarray


def grouped_folds(data, rows=None, n_splits=5):
    if data.partition != "train":
        raise ValueError("Tuning requires training trips")
    if rows is None and n_splits == 5:
        return [Fold(fit, valid) for fit, valid in make_folds(data)]
    if rows is None:
        rows = np.arange(len(data.y))
    if rows.ndim != 1 or len(rows) == 0 or np.any(rows[1:] <= rows[:-1]):
        raise ValueError("Fold scope requires unique, sorted rows")
    groups = data.trip_codes[rows]
    if not 2 <= n_splits <= len(np.unique(groups)):
        raise ValueError("Fold count exceeds the available trips")
    folds = []
    seen = np.zeros(len(rows), dtype=np.uint8)
    for fit, valid in GroupKFold(n_splits=n_splits).split(rows, groups=groups):
        if np.intersect1d(groups[fit], groups[valid]).size:
            raise ValueError("Trip overlap within a grouped fold")
        if any(len(np.unique(data.y[rows[index]])) != 2 for index in (fit, valid)):
            raise ValueError("Both candidate classes are required in every fold")
        seen[valid] += 1
        folds.append(Fold(rows[fit], rows[valid]))
    if not np.all(seen == 1):
        raise ValueError("Each row must be validated exactly once")
    return folds


def fit_model(data, params, rows, workers):
    if data.partition != "train":
        raise ValueError("Only training data may fit a pipeline")
    model = build_model(MODEL, workers)
    model.set_params(**{f"classifier__{name}": value for name, value in params.items()})
    with threadpool_limits(workers):
        model.fit(data.X[rows], data.y[rows])
    return model


def score_fold(data, params, fold, workers, batch_size):
    start = perf_counter()
    model = fit_model(data, params, fold.fit, workers)
    fitted = perf_counter()
    scores = positive_scores(model, data.X[fold.valid], batch_size, workers)
    return scores, fitted - start, perf_counter() - fitted


def trial_metrics(data, scores, rows):
    """Return additive sufficient statistics for exact pooled ranking and log loss."""
    trips, _, ranks, _ = ranked_targets(data, scores, rows)
    stats = observation_metrics(ranks)
    trip_ranks = [observation_metrics(ranks[trips == trip]) for trip in np.unique(trips)]
    probs = np.clip(scores, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    loss = np.where(data.y[rows] == 1, -np.log(probs), -np.log1p(-probs))
    return dict(**stats, trips=len(trip_ranks), candidate_rows=len(rows),
                top3_correct=int(np.count_nonzero(ranks <= 3)), reciprocal_rank_sum=float((1 / ranks).sum()),
                trip_macro_top1=float(np.mean([x["top1"] for x in trip_ranks])),
                trip_top1_sum=sum(x["top1"] for x in trip_ranks),
                log_loss=float(loss.mean()), log_loss_sum=float(loss.sum()))


def pool_trial(folds):
    eligible = sum(row["eligible"] for row in folds)
    correct = sum(row["correct"] for row in folds)
    trips = sum(row["trips"] for row in folds)
    candidates = sum(row["candidate_rows"] for row in folds)
    return dict(eligible=eligible, correct=correct, trips=trips, candidate_rows=candidates,
                top1=correct / eligible, top3=sum(r["top3_correct"] for r in folds) / eligible,
                mrr=sum(r["reciprocal_rank_sum"] for r in folds) / eligible,
                trip_macro_top1=sum(r["trip_top1_sum"] for r in folds) / trips,
                log_loss=sum(r["log_loss_sum"] for r in folds) / candidates)
