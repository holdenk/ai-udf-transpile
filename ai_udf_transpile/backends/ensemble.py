# SPDX-License-Identifier: Apache-2.0
"""Ensemble backend: run multiple AI backends, collect candidates, pick the best.

When ``backend="ensemble"`` the worker runs each configured sub-backend
(claude, codex, cursor, ...) on the same UDF.  Every non-declining result is
collected on ``self.candidates`` so the caller (demo, benchmark, worker) can
inspect agreement/divergence across models.  The first successful result is
returned so the normal worker pipeline (Hypothesis verify, mark_success)
proceeds without change.

The "choose best based on benchmark" step is a separate concern -- it needs
Spark to run actual queries, which ``Backend.run(job, sandbox)`` does not
have.  See ``ai_udf_transpile.benchmark.choose_best`` for that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ai_udf_transpile import conf
from ai_udf_transpile.backends import get_backend
from ai_udf_transpile.backends.base import BackendDecline, BackendError
from ai_udf_transpile.sandbox import write_sandbox
from ai_udf_transpile.targets import TranspileJob, TranspileResult

logger = logging.getLogger(__name__)


class EnsembleBackend:
    name = "ensemble"

    def __init__(self, spark: Any = None, *, backends: Optional[list[str]] = None):
        self.spark = spark
        if backends is not None:
            self._backend_names = backends
        else:
            raw = conf.get_value(conf.ENSEMBLE_BACKENDS, spark, conf.DEFAULTS[conf.ENSEMBLE_BACKENDS])
            self._backend_names = [b.strip().lower() for b in raw.split(",") if b.strip()]
        # Lazy: don't construct sub-backends until run() so a missing binary
        # doesn't crash at enable() time.
        self.candidates: list[tuple[str, str, TranspileResult]] = []

    def _sub_backends(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for name in self._backend_names:
            try:
                out.append((name, get_backend(name, self.spark)))
            except Exception:
                logger.debug("ensemble: sub-backend %s unavailable", name, exc_info=True)
        return out

    def reported_model(self) -> Optional[str]:
        return None

    def run(self, job: TranspileJob, sandbox: Path) -> TranspileResult:
        self.candidates = []
        errors: list[str] = []
        for name, backend in self._sub_backends():
            sub = sandbox / f"sub_{name}"
            write_sandbox(sub, job, inputs=conf.get_value(conf.AI_REPRESENTATION, self.spark))
            try:
                result = backend.run(job, sub)
            except BackendDecline as exc:
                logger.debug("ensemble: %s declined: %s", name, exc)
                continue
            except BackendError as exc:
                errors.append(f"{name}: {exc}")
                logger.debug("ensemble: %s errored: %s", name, exc)
                continue
            model = result.model or f"{name}-unknown"
            self.candidates.append((name, model, result))
            logger.info("ensemble: %s (model=%s) produced a candidate", name, model)
        if not self.candidates:
            if errors:
                # A sub-backend that errored (rate limit, auth, crash) is a
                # transient failure: retry, do not burn a 24h decline
                # cooldown on it the way an all-declined outcome deserves.
                detail = "; ".join(errors)
                raise BackendError(f"ensemble: no candidate from {self._backend_names}: {detail}")
            raise BackendDecline(f"ensemble: no candidate from {self._backend_names}: all declined")
        # First success wins for the catalog; all candidates stay on self.
        return self.candidates[0][2]
