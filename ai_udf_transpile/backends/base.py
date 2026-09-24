# SPDX-License-Identifier: Apache-2.0
"""Backend protocol and errors."""

from __future__ import annotations

from typing import Any, Protocol

from ai_udf_transpile.targets import TranspileJob, TranspileResult


class BackendDecline(Exception):
    """The backend will not (or cannot) rewrite this UDF."""


class BackendError(Exception):
    """The backend ran but failed unexpectedly."""


class Backend(Protocol):
    name: str

    def run(self, job: TranspileJob, sandbox: Any) -> TranspileResult:
        """kind=catalyst|java_udf plus payloads. Raise BackendDecline or BackendError."""
