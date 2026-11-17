# SPDX-License-Identifier: Apache-2.0
"""served_kind reads the rewrite kind off the served expression (no JVM)."""

from __future__ import annotations

import types

from ai_udf_transpile.transpiler import served_kind


def _udf(*parts: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(transpiled=list(parts))


def test_served_kind_from_the_expression():
    assert served_kind(types.SimpleNamespace(transpiled=None)) is None
    assert served_kind(_udf("Column<'CAST(reverse(_udf_param_0) AS STRING)'>")) == "catalyst"
    assert served_kind(_udf("Column<'CAST(ai_udf_0123456789abcdef(_udf_param_0) AS STRING)'>")) == "java_udf"
    assert served_kind(_udf("Column<'ai_udf_gpu_0123456789abcdef(x)'>")) == "gpu_kernel"
    assert served_kind(_udf("Column<'ai_udf_rs_0123456789abcdef(CAST(x AS BIGINT))'>")) == "rust"
    # A SQL rewrite that merely mentions something ai_udf-shaped is still SQL.
    assert served_kind(_udf("Column<'concat(_udf_param_0, \"ai_udf_x\")'>")) == "catalyst"
