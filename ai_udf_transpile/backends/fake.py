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


def flatlist(groups: list[list[str]]) -> list[str]:
    # Rewritten from a contributed UDF (`def flatlist(s): fl = [item for
    # sublist in s for item in sublist]; return fl`, registered as
    # ArrayType(StringType())). flatten() is faithful: it preserves order,
    # duplicates, and null elements; a null outer or inner array raises
    # TypeError in python, where verification allows any sql result.
    return [item for sub in groups for item in sub]


def mod_bucket(value: int, modulus: int) -> int:
    # python `%` floors (the result takes the divisor's sign); Spark `%`
    # truncates toward zero, and pmod only matches for positive divisors.
    return value % modulus


def round_half(x: float) -> int:
    # python round is banker's rounding (round(2.5) == 2, round(3.5) == 4);
    # Spark round is half-up -- bround is the banker's one.
    return round(x)


def weekday_of(t: datetime) -> int:
    # python weekday(): Monday=0; Spark dayofweek: Sunday=1.
    return t.weekday()


def split_dot(s: str) -> list[str]:
    # Spark's split takes a REGEX: an unescaped '.' matches every character.
    return s.split(".")


def zfill3(s: str) -> str:
    # python zfill never truncates and pads after a +/- sign; lpad truncates
    # and pads before the sign.
    return s.zfill(3)


def edit_distance(a: str, b: str) -> int:
    # The algorithmic arm: a quadratic DP per row in interpreted python, one
    # native Catalyst builtin once transpiled. -1 on NULL, like a SQL citizen.
    if a is None or b is None:
        return -1
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def kms_encrypt(text: str, key_id: str) -> str:
    # The canonical untranspilable UDF: an AWS KMS Encrypt call per row.
    # There is no Catalyst equivalent -- the ciphertext comes from the
    # service and is non-deterministic (the same plaintext encrypts
    # differently every call), so any backend SQL "rewrite" is a
    # hallucination. In an environment without boto3/credentials the python
    # side raises on every example, which is the vacuous-pass trap: "python
    # raise + sql value allowed" would accept ANY rewrite, so verification
    # requires at least one successful python evaluation and fails closed.
    # (A module-level client would be marginally faster but is a closure
    # capture the keyer cannot fingerprint; per-call construction also keeps
    # this UDF self-contained for executors.)
    import base64

    import boto3

    if text is None:
        return None
    client = boto3.client("kms", region_name="us-west-2")
    resp = client.encrypt(KeyId=key_id, Plaintext=text.encode("utf-8"))
    return base64.b64encode(resp["CiphertextBlob"]).decode("utf-8")


def always_decline(x: int) -> int:
    import os

    return len(os.getcwd())


# --- AI-only examples: string ops, numpy, math, regex, datetime ---
# Every function below uses method calls (.title, .strip, .replace, ...),
# imports (re, math, numpy), or module-level functions that neither the
# static transpiler (no rules for them) nor the bytecode backend
# (LOAD_ATTR / IMPORT_NAME / CALL are unsupported) can handle. Only the
# AI backend maps them to Catalyst SQL.


# String method calls -> SQL string functions.


def title_case(s: str) -> str:
    return s.title()


def trim_whitespace(s: str) -> str:
    return s.strip()


def snake_case(s: str) -> str:
    return s.replace(" ", "_")


def pad_left5(s: str) -> str:
    # rjust never truncates; lpad does -- guard with a length check.
    return s.rjust(5, "0")


def lower_useragent(ua: str) -> str:
    return ua.lower()


# Regex (re module) -> SQL regex functions.


def clean_phone(s: str) -> str:
    import re

    return re.sub(r"[^0-9]", "", s)


def extract_domain(url: str) -> str:
    import re

    m = re.search(r"://([^/]+)", url)
    return m.group(1) if m else ""


def is_email(s: str) -> bool:
    import re

    return bool(re.match(r"^[^@]+@[^@]+\.[^@]+$", s))


def count_vowels(s: str) -> int:
    import re

    return len(re.findall(r"[aeiouAEIOU]", s))


# numpy -> SQL math functions.


def np_sqrt(x: float) -> float:
    import numpy as np

    return float(np.sqrt(x))


def np_log(x: float) -> float:
    import numpy as np

    return float(np.log(x))


def np_clip(x: float, lo: float, hi: float) -> float:
    import numpy as np

    return float(np.clip(x, lo, hi))


def np_exp(x: float) -> float:
    import numpy as np

    return float(np.exp(x))


# math module -> SQL math functions.


def math_floor(x: float) -> int:
    import math

    return math.floor(x)


def math_ceil(x: float) -> int:
    import math

    return math.ceil(x)


def math_log10(x: float) -> float:
    import math

    return math.log10(x)


# datetime method calls -> SQL date functions.


def to_iso_date(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


# Java-UDF twins for the bench: same body, different def name, so the
# canonical source maps to a DIFFERENT catalog key -- one key, one rewrite.
# The bench measures a catalyst arm and a java arm for the same computation.
def plus_one_java(x: int) -> int:
    return x + 1


def backwards_java(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def zfill3_java(s: str) -> str:
    return s.zfill(3)


def edit_distance_java(a: str, b: str) -> int:
    if a is None or b is None:
        return -1
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


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

PLUS_ONE_JAVA = """package ai_udf;

import org.apache.spark.sql.api.java.UDF1;

public class PlusOneJava implements UDF1<Object, Object> {
    @Override
    public Object call(Object x) {
        // python raises on None: a null here is "python raise + sql value", allowed.
        // longValue()+1 wraps at 2^63-1: out-of-contract (the interpreted UDF
        // errors on output conversion there), so verification never samples it.
        return x == null ? null : ((Number) x).longValue() + 1L;
    }
}
"""

# zfill(3): pad with '0' AFTER a leading sign, never truncate, total width 3.
# Width and sign are decided in CODE POINTS: python len() counts code points
# while String.length() counts UTF-16 units, so an astral char must be width 1.
ZFILL3_JAVA = """package ai_udf;

import org.apache.spark.sql.api.java.UDF1;

public class Zfill3Java implements UDF1<Object, Object> {
    @Override
    public Object call(Object v) {
        // python raises on None: a null here is "python raise + sql value", allowed.
        if (v == null) {
            return null;
        }
        String s = (String) v;
        int n = s.codePointCount(0, s.length());
        if (n >= 3) {
            return s;
        }
        // "" has no sign ("".zfill(3) == "000"), so the sign needs a guard.
        int sign = n > 0 ? s.codePointAt(0) : -1;
        String body = s;
        int width = 3;
        if (sign == '-' || sign == '+') {
            body = s.substring(Character.charCount(sign));
            width = 2;
        }
        StringBuilder b = new StringBuilder();
        if (sign == '-' || sign == '+') {
            b.appendCodePoint(sign);
        }
        int bodyN = body.codePointCount(0, body.length());
        for (int i = bodyN; i < width; i++) {
            b.append('0');
        }
        b.append(body);
        return b.toString();
    }
}
"""

# Levenshtein over CODE POINTS: python len()/slicing counts code points while
# String.length()/charAt count UTF-16 units, and a surrogate-pair compare would
# disagree with python on astral characters.
EDIT_DISTANCE_JAVA = """package ai_udf;

import org.apache.spark.sql.api.java.UDF2;

public class EditDistanceJava implements UDF2<Object, Object, Object> {
    private static int[] toCodePoints(String s) {
        int n = s.codePointCount(0, s.length());
        int[] out = new int[n];
        int idx = 0;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            out[idx++] = cp;
            i += Character.charCount(cp);
        }
        return out;
    }

    @Override
    public Object call(Object a, Object b) {
        String s = (String) a;
        String t = (String) b;
        if (s == null || t == null) {
            return -1L;
        }
        int[] sc = toCodePoints(s);
        int[] tc = toCodePoints(t);
        int[] prev = new int[tc.length + 1];
        for (int j = 0; j <= tc.length; j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= sc.length; i++) {
            int[] cur = new int[tc.length + 1];
            cur[0] = i;
            for (int j = 1; j <= tc.length; j++) {
                int cost = sc[i - 1] == tc[j - 1] ? 0 : 1;
                cur[j] = Math.min(prev[j] + 1, Math.min(cur[j - 1] + 1, prev[j - 1] + cost));
            }
            prev = cur;
        }
        return (long) prev[tc.length];
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
    canonical_source_from_func(backwards): _result("reverse(_udf_param_0)"),
    canonical_source_from_func(backwards_java): TranspileResult(
        kind=KIND_JAVA_UDF,
        java_source=BACKWARDS_JAVA,
        class_name="ai_udf.Backwards",
    ),
    canonical_source_from_func(plus_one_java): TranspileResult(
        kind=KIND_JAVA_UDF,
        java_source=PLUS_ONE_JAVA,
        class_name="ai_udf.PlusOneJava",
    ),
    canonical_source_from_func(zfill3_java): TranspileResult(
        kind=KIND_JAVA_UDF,
        java_source=ZFILL3_JAVA,
        class_name="ai_udf.Zfill3Java",
    ),
    canonical_source_from_func(edit_distance_java): TranspileResult(
        kind=KIND_JAVA_UDF,
        java_source=EDIT_DISTANCE_JAVA,
        class_name="ai_udf.EditDistanceJava",
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
    # flatten preserves order, duplicates, and null elements -- all verified.
    canonical_source_from_func(flatlist): _result("flatten(_udf_param_0)"),
    # Adversarial pack: the faithful forms behind python/Spark lookalikes.
    # Floored modulo: truncated remainder plus a sign correction.
    canonical_source_from_func(mod_bucket): _result(
        "(_udf_param_0 % _udf_param_1) + CASE WHEN (_udf_param_0 % _udf_param_1) <> 0 "
        "AND ((_udf_param_0 < 0) <> (_udf_param_1 < 0)) THEN _udf_param_1 ELSE 0 END"
    ),
    # Banker's rounding; bigint so 1e16 does not overflow (int32 would raise).
    canonical_source_from_func(round_half): _result("cast(bround(_udf_param_0) as bigint)"),
    # dayofweek Sunday=1 -> python weekday() Monday=0.
    canonical_source_from_func(weekday_of): _result("pmod(dayofweek(_udf_param_0) + 5, 7)"),
    # Literal dot, not regex any-char.
    canonical_source_from_func(split_dot): _result("split(_udf_param_0, '\\\\.')"),
    # Sign-aware, never truncating.
    canonical_source_from_func(zfill3): _result(
        "CASE WHEN length(_udf_param_0) >= 3 THEN _udf_param_0 "
        "WHEN substr(_udf_param_0, 1, 1) IN ('-', '+') "
        "THEN concat(substr(_udf_param_0, 1, 1), lpad(substr(_udf_param_0, 2), 2, '0')) "
        "ELSE lpad(_udf_param_0, 3, '0') END"
    ),
    # The quadratic python DP is one native builtin; -1 on NULL, like python.
    canonical_source_from_func(edit_distance): _result(
        "coalesce(levenshtein(_udf_param_0, _udf_param_1), -1)"
    ),
    # --- AI-only examples: string ops, numpy, math, regex, datetime ---
    # String method calls -> SQL string functions.
    canonical_source_from_func(title_case): _result("initcap(_udf_param_0)"),
    canonical_source_from_func(trim_whitespace): _result("trim(BOTH ' \t\n\r\f\v' FROM _udf_param_0)"),
    canonical_source_from_func(snake_case): _result("replace(_udf_param_0, ' ', '_')"),
    canonical_source_from_func(lower_useragent): _result("lower(_udf_param_0)"),
    # rjust never truncates; lpad does -- guard with a length check.
    canonical_source_from_func(pad_left5): _result(
        "CASE WHEN length(_udf_param_0) >= 5 THEN _udf_param_0 ELSE lpad(_udf_param_0, 5, '0') END"
    ),
    # Regex (re module) -> SQL regex functions.
    canonical_source_from_func(clean_phone): _result("regexp_replace(_udf_param_0, '[^0-9]', '')"),
    canonical_source_from_func(extract_domain): _result("regexp_extract(_udf_param_0, '://([^/]+)', 1)"),
    canonical_source_from_func(is_email): _result("_udf_param_0 rlike '^[^@]+@[^@]+\\\\.[^@]+$'"),
    canonical_source_from_func(count_vowels): _result(
        "length(regexp_replace(_udf_param_0, '[^aeiouAEIOU]', ''))"
    ),
    # numpy -> SQL math functions.
    canonical_source_from_func(np_sqrt): _result("sqrt(_udf_param_0)"),
    canonical_source_from_func(np_log): _result("ln(_udf_param_0)"),
    canonical_source_from_func(np_clip): _result("least(greatest(_udf_param_0, _udf_param_1), _udf_param_2)"),
    canonical_source_from_func(np_exp): _result("exp(_udf_param_0)"),
    # math module -> SQL math functions.
    canonical_source_from_func(math_floor): _result("floor(_udf_param_0)"),
    canonical_source_from_func(math_ceil): _result("ceil(_udf_param_0)"),
    canonical_source_from_func(math_log10): _result("log10(_udf_param_0)"),
    # datetime method calls -> SQL date functions.
    canonical_source_from_func(to_iso_date): _result("date_format(_udf_param_0, 'yyyy-MM-dd')"),
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
