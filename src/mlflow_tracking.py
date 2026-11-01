"""MLflow experiment tracking wrapper (local file store at mlruns/)."""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MLRUNS_DIR


def init_tracking(experiment_name: str = MLFLOW_EXPERIMENT) -> None:
    """Initialize MLflow with a local file store and set the experiment."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


class RunLogger:
    """Context manager for logging a single training run to MLflow."""

    def __init__(
        self,
        run_name: str | None = None,
        tags: dict | None = None,
    ) -> None:
        self.run_name = run_name
        self.tags = tags or {}
        self.run = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run is not None:
            mlflow.end_run()
        return False

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def _log_model(self, log_fn, model, artifact_path: str) -> None:
        """MLflow 3.x prefers `name=`; fall back to `artifact_path=` for older versions."""
        try:
            log_fn(model, name=artifact_path)
        except TypeError:
            log_fn(model, artifact_path=artifact_path)

    def log_model_sklearn(self, model, artifact_path: str = "model") -> None:
        self._log_model(mlflow.sklearn.log_model, model, artifact_path)

    def log_model_xgboost(self, model, artifact_path: str = "model") -> None:
        self._log_model(mlflow.xgboost.log_model, model, artifact_path)

    def tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)


def get_client():
    """Return an MLflow tracking client for querying runs."""
    return mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
