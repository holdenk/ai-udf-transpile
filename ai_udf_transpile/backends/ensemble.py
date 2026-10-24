# SPDX-License-Identifier: Apache-2.0
"""Fallback-chain backend: sub-backends in order, first candidate wins.

When ``backend="ensemble"`` the worker tries each configured sub-backend
(coco, cursor, ...) on the same UDF until one produces a rewrite.  A
decline falls through to the next sub-backend; so does a transient error
(auth, rate limit, crash) -- later sub-backends never run once one
answers.  All declined => BackendDecline (the models saw it and said no);
no candidate and any error => BackendError (retryable, so a dead CLI does
not burn the row's 24h decline cooldown).

The winner's ``model`` is stamped ``<sub-backend>/<its model>`` so the
catalog records which sub-backend actually served the rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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

    def _sub_backends(self) -> list[tuple[str, Any]]:
        # Lazy construction, inside run(), so a missing binary does not crash
        # enable(). A leg that cannot even be built counts as an error, not a
        # decline -- it never got its turn to say no.
        out: list[tuple[str, Any]] = []
        for name in self._backend_names:
            try:
                out.append((name, get_backend(name, self.spark)))
            except Exception:
                logger.debug("ensemble: sub-backend %s unavailable", name, exc_info=True)
                out.append((name, None))
        return out

    def reported_model(self) -> Optional[str]:
        return None

    def run(self, job: TranspileJob, sandbox: Path) -> TranspileResult:
        errors: list[str] = []
        for name, backend in self._sub_backends():
            if backend is None:
                errors.append(f"{name}: unavailable")
                continue
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
            # First candidate wins: this is a fallback chain, not a vote --
            # later sub-backends never run once one answers. Copy-on-stamp:
            # some backends (the fake) return shared objects.
            stamped = replace(result, model=f"{name}/{result.model or 'unknown'}")
            logger.info("ensemble: %s (model=%s) produced a candidate", name, stamped.model)
            return stamped
        if errors:
            # A sub-backend that errored (rate limit, auth, crash) is a
            # transient failure: retry, do not burn a 24h decline
            # cooldown on it the way an all-declined outcome deserves.
            detail = "; ".join(errors)
            raise BackendError(f"ensemble: no candidate from {self._backend_names}: {detail}")
        raise BackendDecline(f"ensemble: no candidate from {self._backend_names}: all declined")
