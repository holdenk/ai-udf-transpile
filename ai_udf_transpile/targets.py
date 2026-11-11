# SPDX-License-Identifier: Apache-2.0
"""Rewrite payloads a cache hit can reconstruct."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

KIND_CATALYST = "catalyst"
KIND_JAVA_UDF = "java_udf"
KIND_RUST = "rust"
KIND_GPU_KERNEL = "gpu_kernel"
KIND_BINARY = "binary"

# Appended to the spark_version a cache key is built from, for the twin row
# a session that cannot run Rust gets when the main row's rewrite is Rust.
# The worker never offers (or accepts) Rust for such a row.
NO_RUST_SUFFIX = "+no-rust"

V1_KINDS = frozenset({KIND_CATALYST, KIND_JAVA_UDF})
# Served only when spark.sql.experimental.aiUdfTranspile.targets opts in: a
# Rust UDF runs only inside Comet's native execution (see rustc.py).
OPT_IN_KINDS = frozenset({KIND_RUST})
SERVABLE_KINDS = V1_KINDS | OPT_IN_KINDS
RESERVED_KINDS = frozenset({KIND_GPU_KERNEL, KIND_BINARY})


@dataclass
class TranspileJob:
    udf_key: str
    source_text: str
    param_names: list[str]
    input_types: list[str]
    input_categories: list[str]
    return_type: str
    captures: dict[str, Any] = field(default_factory=dict)
    spark_version: str = ""
    closure_fingerprint: str = ""
    # Rewrite kinds a backend may answer with. The worker adds `rust` when
    # the session can build, verify and serve one for these types.
    targets: list[str] = field(default_factory=lambda: [KIND_CATALYST, KIND_JAVA_UDF])


@dataclass
class TranspileResult:
    kind: str
    sql: Optional[str] = None
    java_source: Optional[str] = None
    class_name: Optional[str] = None
    binary: Optional[bytes] = None
    entry: Optional[str] = None
    model: Optional[str] = None
    # kind=rust: the row function (`pub fn udf(...)`); rustc.py wraps it in
    # the Arrow glue and builds the cdylib Comet loads.
    rust_source: Optional[str] = None

    @property
    def impl_source(self) -> Optional[str]:
        """The one source payload a catalog row stores (``impl_source``)."""
        return self.rust_source if self.kind == KIND_RUST else self.java_source

    def reconstructable(self) -> bool:
        if self.kind == KIND_CATALYST:
            return bool(self.sql and self.sql.strip())
        if self.kind == KIND_JAVA_UDF:
            return bool(self.class_name or self.java_source or self.binary)
        if self.kind == KIND_RUST:
            return bool(self.rust_source and self.rust_source.strip())
        return False
