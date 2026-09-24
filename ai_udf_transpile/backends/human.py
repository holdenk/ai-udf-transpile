# SPDX-License-Identifier: Apache-2.0
"""Human-provided impls are written via register_impl(), not a CLI backend."""

from __future__ import annotations

# Intentionally empty of a Backend.run: humans skip the worker CLI path.
ORIGIN = "human"
