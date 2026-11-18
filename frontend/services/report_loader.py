"""Read only allowlisted aggregate reports, preserving string identifiers."""
import json
from functools import lru_cache
from pathlib import Path
import pandas as pd

REPORT_ROOT = Path(__file__).resolve().parents[2] / "reports"
LABELS = {
    "logistic_regression": "Logistic Regression", "knn": "KNN", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "gradient_boosting": "Histogram Gradient Boosting",
}
TABLES = {
    "comparison": "phase3/model_comparison.csv", "student": "phase3/student_test_metrics.csv",
    "coverage": "phase3/coverage_summary.csv", "per_trip": "phase3/per_trip_metrics.csv",
    "nearest": "phase3/nearest_baseline.csv", "runtime": "phase3/training_runtime.csv",
    "phase4": "phase4/baseline_vs_tuned.csv", "phase4_runtime": "phase4/training_runtime.csv",
    "trials": "phase4/optimization_trials.csv", "phase4_trips": "phase4/per_trip_comparison.csv",
}


class ReportLoader:
    def __init__(self, root=REPORT_ROOT):
        self.root = Path(root)

    def table(self, name):
        path = self.root / TABLES[name]
        if not path.is_file():
            return pd.DataFrame()
        # mtime-keyed cache refreshes after reports arrive without restarting the app.
        return _csv(str(path), path.stat().st_mtime_ns).copy()

    def selection(self, phase=3):
        if phase not in (3, 4):
            raise ValueError("Only Phase 3 and Phase 4 aggregate reports are permitted")
        path = self.root / f"phase{phase}/model_selection.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    @property
    def phase4_available(self):
        return not self.table("phase4").empty and bool(self.selection(4))

    @property
    def default_model(self):
        return self.selection().get("selected", "gradient_boosting")

    def comparison_data(self, metric):
        if metric not in {"top1", "top3", "f1", "log_loss", "trip_macro_top1", "mrr"}:
            raise ValueError("Unsupported comparison metric")
        frames = []
        for name, label in (("comparison", "Training OOF"), ("student", "Student testing")):
            table = self.table(name)
            if not table.empty and {"model", metric} <= set(table.columns):
                frame = table[["model", metric]].rename(columns={metric: "value"}).copy()
                frame["partition"] = label
                frame["label"] = frame.model.map(LABELS).fillna(frame.model)
                frame["selected"] = frame.model == self.default_model
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@lru_cache(maxsize=32)
def _csv(path, mtime):
    return pd.read_csv(path, dtype={"trip_id": "string"})
