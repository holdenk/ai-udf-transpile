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

V1_KINDS = frozenset({KIND_CATALYST, KIND_JAVA_UDF})
RESERVED_KINDS = frozenset({KIND_RUST, KIND_GPU_KERNEL, KIND_BINARY})


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


@dataclass
class TranspileResult:
    kind: str
    sql: Optional[str] = None
    java_source: Optional[str] = None
    class_name: Optional[str] = None
    binary: Optional[bytes] = None
    entry: Optional[str] = None
    model: Optional[str] = None

    def reconstructable(self) -> bool:
        if self.kind == KIND_CATALYST:
            return bool(self.sql and self.sql.strip())
        if self.kind == KIND_JAVA_UDF:
            return bool(self.class_name or self.java_source or self.binary)
        return False
