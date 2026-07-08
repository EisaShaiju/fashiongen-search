"""
mlflow_utils.py
===============
Tiny MLflow wrapper used across training/eval. Logs params, metrics and
artifacts to the configured tracking server (or local ./mlruns). Degrades to a
no-op if MLflow isn't installed, so scripts never hard-fail on logging.
"""
from __future__ import annotations

import os
from contextlib import contextmanager


class _NoOpRun:
    def log_metrics(self, *a, **k): pass
    def log_params(self, *a, **k): pass
    def log_artifact(self, *a, **k): pass
    def set_tag(self, *a, **k): pass


class MlflowRun:
    """Context manager: `with MlflowRun("exp", params=cfg) as run: ...`"""

    def __init__(self, experiment: str, params: dict | None = None,
                 tracking_uri: str | None = None, run_name: str | None = None):
        self.experiment = experiment
        self.params = params or {}
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.run_name = run_name
        self._mlflow = None
        self._active = None

    def __enter__(self):
        try:
            import mlflow
        except Exception:
            return _NoOpRun()
        self._mlflow = mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        self._active = mlflow.start_run(run_name=self.run_name)
        # only log flat, primitive params
        flat = {k: v for k, v in self.params.items()
                if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(flat)
        return self

    def log_metrics(self, metrics: dict, step: int | None = None):
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict):
        if self._mlflow:
            self._mlflow.log_params(params)

    def log_artifact(self, path: str):
        if self._mlflow:
            self._mlflow.log_artifact(path)

    def set_tag(self, k, v):
        if self._mlflow:
            self._mlflow.set_tag(k, v)

    def __exit__(self, *exc):
        if self._mlflow and self._active:
            self._mlflow.end_run()
        return False
