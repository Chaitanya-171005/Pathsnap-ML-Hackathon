"""Seeded TPE search with a replayable trial journal."""

from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.special import ndtr, ndtri

from src.models.__main__ import peak_mib
from src.models.inference import SEED

from .artifacts import canonical, fingerprint, read_record, save_record
from .validation import pool_trial, score_fold, trial_metrics


SPACE = {
    "learning_rate": ("log", 0.02, 0.20),
    "max_iter": ("int", 80, 300),
    "max_leaf_nodes": ("int", 15, 63),
    "max_depth": ("choice", None, 4, 6, 8, 12),
    "min_samples_leaf": ("int", 10, 80),
    "l2_regularization": ("log", 0.001, 10.0),
}
SAMPLER = dict(method="factorized_tree_structured_parzen_estimator", startup_trials=5,
               good_fraction=0.25, candidates=48, prior_weight=1, min_bandwidth=0.035)


def validate_params(params):
    if set(params) != set(SPACE):
        raise ValueError("Hyperparameter keys differ from the search space")
    for name, (kind, *bounds) in SPACE.items():
        value = params[name]
        if kind == "choice":
            valid = (value is None or type(value) is int) and value in bounds
        elif kind == "int":
            valid = type(value) is int and bounds[0] <= value <= bounds[1]
        else:
            valid = type(value) in (int, float) and bounds[0] <= value <= bounds[1]
        if not valid:
            raise ValueError(f"Invalid {name}: {value!r}")


def trial_key(params, metrics):
    return -metrics["correct"], -metrics["trip_macro_top1"], metrics["log_loss"], canonical(params)


def _normalise(value, kind, lo, hi):
    if kind == "log":
        return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (value - lo) / (hi - lo)


def _decode(values, kind, lo, hi):
    if kind == "log":
        return np.clip(np.exp(np.log(lo) + values * np.log(hi / lo)), lo, hi).tolist()
    return np.rint(lo + values * (hi - lo)).astype(int).tolist()


def _parzen(values):
    centers = np.sort(values)
    edges = np.diff(np.r_[0.0, centers, 1.0])
    widths = np.maximum(np.maximum(edges[:-1], edges[1:]), SAMPLER["min_bandwidth"])
    lower = ndtr(-centers / widths)
    mass = ndtr((1.0 - centers) / widths) - lower
    return centers, widths, lower, mass


def _density(x, kernels):
    centers, widths, _, mass = kernels
    z = (x[:, None] - centers) / widths
    kernel_density = np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * widths * mass)
    prior = SAMPLER["prior_weight"]
    return (prior + kernel_density.sum(axis=1)) / (prior + len(centers))


def _real_samples(rng, good, count):
    centers, widths, lower, mass = good
    component = rng.integers(0, len(centers) + SAMPLER["prior_weight"], size=count)
    x = rng.random(count)
    use = component < len(centers)
    ix = component[use]
    x[use] = centers[ix] + widths[ix] * ndtri(lower[ix] + mass[ix] * x[use])
    return np.clip(x, 0, 1)


def propose(study, number, history, baseline, seed=SEED):
    if number == 0:
        return dict(baseline)
    salt = int.from_bytes(sha256(study.encode()).digest()[:8], "big")
    rng = np.random.default_rng(np.random.SeedSequence([seed, salt, number]))
    completed = sorted((row for row in history if row["state"] == "COMPLETE"),
                       key=lambda row: trial_key(row["params"], row["metrics"]))
    used = {canonical(row["params"]) for row in history}
    if len(completed) < SAMPLER["startup_trials"]:
        while True:
            config = {}
            for name, (kind, *bounds) in SPACE.items():
                config[name] = bounds[int(rng.integers(len(bounds)))] if kind == "choice" else (
                    int(rng.integers(bounds[0], bounds[1] + 1)) if kind == "int" else
                    _decode(rng.random(1), kind, *bounds)[0])
            if canonical(config) not in used:
                return config

    cut = max(1, int(np.ceil(len(completed) * SAMPLER["good_fraction"])))
    good, bad = completed[:cut], completed[cut:]
    count = SAMPLER["candidates"]
    candidates = {}
    log_ratio = np.zeros(count)
    for name, (kind, *bounds) in SPACE.items():
        if kind == "choice":
            probabilities = []
            for partition in (good, bad):
                counts = np.full(len(bounds), SAMPLER["prior_weight"], dtype=float)
                for trial in partition:
                    counts[bounds.index(trial["params"][name])] += 1
                probabilities.append(counts / counts.sum())
            p, q = probabilities
            choices = rng.choice(len(bounds), size=count, p=p)
            candidates[name] = [bounds[i] for i in choices]
            log_ratio += np.log(p[choices]) - np.log(q[choices])
        else:
            kernels = []
            for partition in (good, bad):
                kernels.append(_parzen([_normalise(t["params"][name], kind, *bounds) for t in partition]))
            x = _real_samples(rng, kernels[0], count)
            candidates[name] = _decode(x, kind, *bounds)
            decoded_x = _normalise(np.asarray(candidates[name]), kind, *bounds)
            log_ratio += np.log(_density(decoded_x, kernels[0])) - np.log(_density(decoded_x, kernels[1]))
    for i in np.argsort(-log_ratio, kind="stable"):
        config = {name: values[i] for name, values in candidates.items()}
        if canonical(config) not in used:
            return config
    raise ValueError("TPE exhausted the sampled unique configurations")


def search_study(data, name, folds, baseline, budget, args, checkpoint_root, manifest_id):
    scope = fingerprint(dict(manifest=manifest_id, study=name))
    history = []
    for number in range(budget):
        params = propose(name, number, history, baseline, args.seed)
        validate_params(params)
        contract = fingerprint(dict(scope=scope, trial=number, params=params))
        path = checkpoint_root / name / f"{number:04d}.json"
        if path.exists():
            trial = read_record(path, contract)
            if (trial["params"] != params or trial["number"] != number or trial["study"] != name
                    or trial["state"] not in {"COMPLETE", "FAIL"}):
                raise ValueError(f"Invalid trial record: {path}")
            if trial["state"] == "FAIL":
                raise ValueError(f"Study contains a recorded failed trial: {path}")
            if len(trial["folds"]) != len(folds) or pool_trial(trial["folds"]) != trial["metrics"]:
                raise ValueError(f"Inconsistent trial metrics: {path}")
            history.append(trial)
            continue

        trial = dict(study=name, number=number, params=params, folds=[])
        print(f"{name} trial {number + 1}/{budget} {canonical(params)}", flush=True)
        start = perf_counter()
        try:
            for index, fold in enumerate(folds, 1):
                fold_path = path.parent / "folds" / f"{number:04d}-{index:02d}.json"
                fold_contract = fingerprint(dict(trial=contract, fold=index))
                if fold_path.exists():
                    row = read_record(fold_path, fold_contract)
                else:
                    scores, fit_s, predict_s = score_fold(data, params, fold, args.workers, args.batch_size)
                    row = dict(**trial_metrics(data, scores, fold.valid), fold=index, fit_s=fit_s,
                               predict_s=predict_s, rss_mib=peak_mib())
                    save_record(fold_path, fold_contract, row)
                if (row["fold"] != index or row["candidate_rows"] != len(fold.valid)
                        or row["eligible"] != int(data.y[fold.valid].sum())):
                    raise ValueError(f"Invalid validation counts: {fold_path}")
                trial["folds"].append(row)
                print(f"  {name} trial {number + 1} fold {index}/{len(folds)} "
                      f"Top-1={row['top1']:.4%}, fit={row['fit_s']:.2f}s", flush=True)
        except (FloatingPointError, MemoryError, ValueError, RuntimeError) as exc:
            trial.update(state="FAIL", error=f"{type(exc).__name__}: {exc}", wall_s=perf_counter() - start)
            save_record(path, contract, trial)
            raise RuntimeError(f"{name} trial {number + 1} failed. See {path}.") from exc
        trial.update(state="COMPLETE", metrics=pool_trial(trial["folds"]), wall_s=perf_counter() - start)
        save_record(path, contract, trial)
        history.append(trial)
    return history, min(history, key=lambda row: trial_key(row["params"], row["metrics"]))
