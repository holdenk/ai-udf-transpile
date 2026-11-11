# SPDX-License-Identifier: Apache-2.0
"""Two pinned rewrites for each FHI pipeline UDF (backends/fhi_udfs.py).

`naive` is verbatim what an AI backend served for the function on its first
verified attempt: faithful to Python's `strip()`, `splitlines()` and unicode
regex classes, and for most of them slower than the python it replaces.
`tuned` is hand-written, equally faithful, and shaped by what the naive one
turned out to cost:

  fhi_clean_health_history     btrim over Python's whitespace set instead of
                               a regex strip, and no leading NULL guard (the
                               guard is the only always-evaluated WHEN, so
                               Spark shares none of the repeated strips
                               after it; on a non-nullable column the
                               optimizer deletes the guard and the naive
                               rewrite recovers by itself)
  fhi_verdict_has_legal_check  a Java UDF that walks the lines with plain
  fhi_parse_verdict            loops instead of regexes over the whole text
                               (parse_verdict's naive SQL needs transform /
                               filter lambdas, which Spark will not serve)
  fhi_treatment_from_findings  the naive expression twice behind
  fhi_normalize_entity_text    `length(x) = octet_length(x)`: for pure-ASCII
                               text java.util.regex's default classes mean
                               what Python's mean, so that arm drops the
                               spelled-out unicode classes
  fhi_deidentify_text          the same split inside the Java UDF: an ASCII
                               pattern set and the faithful one, chosen per row
  fhi_diagnosis_from_findings  no tuned variant: the naive rewrite already wins

Both are kept on purpose. The naive ones are the exhibit for when a correct
transpilation is a regression; examples/bench_fhi_udfs.py times the pair
side by side, on a nullable and a non-nullable column.
"""

from __future__ import annotations

from importlib import resources
from typing import Optional

from ai_udf_transpile.javac import extract_java_class
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileResult

VARIANTS = ("naive", "tuned")


def fhi_rewrite(name: str, variant: str) -> Optional[TranspileResult]:
    """The pinned rewrite, or None when the function has no such variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    base = resources.files("ai_udf_transpile") / "resources" / "fhi"
    sql = base / f"{name}.{variant}.sql"
    if sql.is_file():
        return TranspileResult(kind=KIND_CATALYST, sql=sql.read_text(encoding="utf-8").strip())
    java = base / f"{name}.{variant}.java"
    if java.is_file():
        source = java.read_text(encoding="utf-8")
        return TranspileResult(kind=KIND_JAVA_UDF, java_source=source, class_name=extract_java_class(source))
    return None
