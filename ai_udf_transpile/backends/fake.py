# SPDX-License-Identifier: Apache-2.0
"""Deterministic backend for CI: exact canonical source → TranspileResult."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.keys import canonical_source_from_func, canonical_source_text
from ai_udf_transpile.targets import KIND_CATALYST, KIND_JAVA_UDF, TranspileJob, TranspileResult


def plus_one(x: int) -> int:
    return x + 1


def is_none_branch(x: int) -> int:
    if x is None:
        return -1
    return x


def both_positive(x: int, y: int) -> bool:
    return x > 0 and y > 0


def greet(name: str) -> str:
    return "hi " + name


def upper_useragent(useragent: str) -> str:
    return useragent.upper()


def contains_ingredient(text: str, needle: str) -> bool:
    # Rewritten from a contributed UDF (recipe/ingredient): None -> False.
    return text is not None and needle in text.lower()


def timestamp_to_epoch(ts: datetime) -> str:
    # Scalar modernization of the pandas_udf t.dt.strftime("%s").apply(str):
    # on NaT the pandas version yields NaN and .apply(str) makes it 'nan'.
    return "nan" if ts is None else ts.strftime("%s")


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def widget_name(doc: str) -> str:
    import json

    if doc is None:
        return None
    return json.loads(doc).get("widget")


def scatter_to_seconds(start_text: str, dur_text: str) -> list[str]:
    import datetime

    out = []
    try:
        start_text = str(start_text)
        secs = int(float(dur_text))
        if len(start_text) < 19:
            return out
        start_text = start_text[:19]
        parsed = datetime.datetime.strptime(start_text, "%Y-%m-%d %H:%M:%S")
        for i in range(0, secs + 1):
            stamp = (parsed + datetime.timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
            out.append(stamp)
        return out
    except Exception:
        return out


# The BSP/EBR fixtures below are rewrites of contributed telecom UDFs. The
# pasted original of BSPIn was `old_lac in lac_lst_bsp & new_lac in
# lac_lst_mid`, which parses as `old_lac in (lac_lst_bsp & new_lac) in
# lac_lst_mid` (& binds tighter than `in`, and comparisons chain) and raises
# TypeError (list & str) on EVERY row; `and` was the intent.
# Nulls: None in list is False (-> 0); x in None raises (sql may return).


def BSPIn(prev_lac: str, cur_lac: str, bsp_lacs: list[str], mid_lacs: list[str]) -> int:
    return 1 if prev_lac in bsp_lacs and cur_lac in mid_lacs else 0


def EBRIn(prev_lac: str, cur_lac: str, ebr_lacs: list[str], mid_lacs: list[str]) -> int:
    return 1 if prev_lac in ebr_lacs and cur_lac in mid_lacs else 0


def BSPOut(prev_lac: str, cur_lac: str, bsp_lacs: list[str], mid_lacs: list[str]) -> int:
    return 1 if prev_lac in mid_lacs and cur_lac in bsp_lacs else 0


def EBROut(prev_lac: str, cur_lac: str, ebr_lacs: list[str], mid_lacs: list[str]) -> int:
    return 1 if prev_lac in mid_lacs and cur_lac in ebr_lacs else 0


def always_decline(x: int) -> int:
    import os

    return len(os.getcwd())


def _result(sql: str) -> TranspileResult:
    return TranspileResult(kind=KIND_CATALYST, sql=sql)


BACKWARDS_JAVA = """package ai_udf;

import org.apache.spark.sql.api.java.UDF1;

public class Backwards implements UDF1<Object, Object> {
    @Override
    public Object call(Object s) {
        return s == null ? null : new StringBuilder((String) s).reverse().toString();
    }
}
"""

FIXTURES: dict[str, TranspileResult] = {
    canonical_source_from_func(plus_one): _result("_udf_param_0 + 1"),
    canonical_source_from_func(is_none_branch): _result(
        "CASE WHEN _udf_param_0 IS NULL THEN -1 ELSE _udf_param_0 END"
    ),
    canonical_source_from_func(both_positive): _result("_udf_param_0 > 0 AND _udf_param_1 > 0"),
    canonical_source_from_func(greet): _result("concat('hi ', _udf_param_0)"),
    canonical_source_from_func(upper_useragent): _result("upper(_udf_param_0)"),
    canonical_source_from_func(contains_ingredient): _result(
        "coalesce(instr(lower(_udf_param_0), _udf_param_1) > 0, false)"
    ),
    canonical_source_from_func(timestamp_to_epoch): _result(
        "coalesce(cast(unix_timestamp(date_trunc('SECOND', _udf_param_0)) as string), 'nan')"
    ),
    canonical_source_from_func(widget_name): _result("get_json_object(_udf_param_0, '$.widget')"),
    # NOTE: the _udf_param_N refs stay OUTSIDE the transform lambda -- Spark's
    # TranspiledPythonUDF placeholder substitution does not descend into
    # higher-order function lambda bodies, so the lambda closes over only its
    # own argument and literals.
    canonical_source_from_func(scatter_to_seconds): _result(
        "CASE WHEN _udf_param_0 IS NULL THEN array() "
        "WHEN length(_udf_param_0) < 19 THEN array() "
        "WHEN try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss') IS NULL THEN array() "
        "WHEN try_cast(_udf_param_1 AS DOUBLE) IS NULL THEN array() "
        "WHEN isnan(try_cast(_udf_param_1 AS DOUBLE)) THEN array() "
        "WHEN abs(try_cast(_udf_param_1 AS DOUBLE)) = cast('inf' AS DOUBLE) THEN array() "
        "WHEN cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT) < 0 THEN array() "
        "ELSE transform(sequence("
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss'), "
        "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT), "
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')), "
        "interval 1 second), "
        "x -> date_format(x, 'yyyy-MM-dd HH:mm:ss')) END"
    ),
    canonical_source_from_func(backwards): TranspileResult(
        kind=KIND_JAVA_UDF,
        java_source=BACKWARDS_JAVA,
        class_name="ai_udf.Backwards",
    ),
    # In-UDFs check old in the first list and new in the second; Out-UDFs swap.
    canonical_source_from_func(BSPIn): _result(
        "CASE WHEN array_contains(_udf_param_2, _udf_param_0) "
        "AND array_contains(_udf_param_3, _udf_param_1) THEN 1 ELSE 0 END"
    ),
    canonical_source_from_func(EBRIn): _result(
        "CASE WHEN array_contains(_udf_param_2, _udf_param_0) "
        "AND array_contains(_udf_param_3, _udf_param_1) THEN 1 ELSE 0 END"
    ),
    canonical_source_from_func(BSPOut): _result(
        "CASE WHEN array_contains(_udf_param_3, _udf_param_0) "
        "AND array_contains(_udf_param_2, _udf_param_1) THEN 1 ELSE 0 END"
    ),
    canonical_source_from_func(EBROut): _result(
        "CASE WHEN array_contains(_udf_param_3, _udf_param_0) "
        "AND array_contains(_udf_param_2, _udf_param_1) THEN 1 ELSE 0 END"
    ),
}


class FakeBackend:
    name = "fake"

    def run(self, job: TranspileJob, sandbox: Path) -> TranspileResult:
        try:
            key = canonical_source_text(job.source_text)
        except Exception:
            key = (job.source_text or "").strip()
        result = FIXTURES.get(key)
        if result is None:
            raise BackendDecline(f"no fake fixture for source {key!r}")
        if result.sql:
            (sandbox / "OUT.sql").write_text(result.sql + "\n", encoding="utf-8")
        if result.java_source:
            (sandbox / "OUT.java").write_text(result.java_source + "\n", encoding="utf-8")
        return result
