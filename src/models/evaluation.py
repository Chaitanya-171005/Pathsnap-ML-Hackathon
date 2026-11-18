"""Trip-grouped folds and separate observation/candidate metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .inference import FEATURES, rank_indices


def make_folds(dataset):
    if dataset.partition != "train":
        raise ValueError("Cross-validation accepts only the training partition")
    folds = list(GroupKFold(n_splits=5).split(dataset.X, dataset.y, dataset.trip_codes))
    seen = np.zeros(len(dataset.y), dtype=np.int8)
    for train, validation in folds:
        if np.intersect1d(dataset.trip_codes[train], dataset.trip_codes[validation]).size:
            raise ValueError("Cross-validation trip overlap")
        if any(len(np.unique(dataset.y[rows])) != 2 for rows in (train, validation)):
            raise ValueError("Both classes must be present in each fitting and validation set")
        seen[validation] += 1
    if not np.all(seen == 1) or len(np.unique(dataset.trip_codes)) != len(dataset.trips):
        raise ValueError("Incomplete fold or training trip coverage")
    return folds


def ranked_targets(dataset, scores, rows=None):
    if rows is None:
        rows = slice(None)
    trips, obs, y = dataset.trip_codes[rows], dataset.observation_ids[rows], dataset.y[rows]
    group_ids = (trips.astype(np.int64) << 32) + obs
    ranks = dataset.X[rows, FEATURES.index("distance_rank")]
    order = rank_indices(group_ids, ranks, dataset.arc_ids[rows], np.asarray(scores))
    ordered_groups = group_ids[order]
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]) if len(order) else np.empty(0, int)
    sizes = np.diff(np.r_[starts, len(order)])
    positions = np.flatnonzero(y[order] == 1)
    if len(positions) != len(starts) or not np.array_equal(ordered_groups[positions], ordered_groups[starts]):
        raise ValueError("Ranking requires one positive per observation")
    return trips[order[starts]], obs[order[starts]], positions - starts + 1, sizes


def observation_metrics(target_ranks):
    correct = int((target_ranks == 1).sum())
    return dict(eligible=len(target_ranks), correct=correct,
                top1=float(np.mean(target_ranks == 1)) if len(target_ranks) else None,
                top3=float(np.mean(target_ranks <= 3)) if len(target_ranks) else None,
                mrr=float(np.mean(1 / target_ranks)) if len(target_ranks) else None)


def binary_metrics(y, scores):
    chosen = scores > 0.5  # Matches sklearn's lower-class tie at p=0.5.
    positive = y == 1
    tp, fp = int((chosen & positive).sum()), int((chosen & ~positive).sum())
    fn, tn = int((~chosen & positive).sum()), int((~chosen & ~positive).sum())

    def ratio(n, d):
        return n / d if d else 0.0

    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return dict(candidate_rows=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn), f1=f1,
                macro_f1=(f1 + ratio(2 * tn, 2 * tn + fp + fn)) / 2,
                log_loss=float(log_loss(y, scores, labels=[0, 1])),
                roc_auc=float(roc_auc_score(y, scores)) if tp + fn and tn + fp else None,
                average_precision=float(average_precision_score(y, scores)) if tp + fn else None)


def evaluate(dataset, scores, rows=None, *, probabilities=True):
    trips, obs, ranks, sizes = ranked_targets(dataset, scores, rows)
    per_trip = [dict(trip_id=dataset.trips[code], **observation_metrics(ranks[trips == code]))
                for code in np.unique(trips)]
    metrics = observation_metrics(ranks)
    metrics["trip_macro_top1"] = float(np.mean([r["top1"] for r in per_trip]))
    if probabilities:
        metrics.update(binary_metrics(dataset.y if rows is None else dataset.y[rows], scores))
    return metrics, per_trip, (trips, obs, ranks, sizes)


def selection_key(row):
    return -row["top1"], -row["trip_macro_top1"], row["log_loss"], row["model"]
