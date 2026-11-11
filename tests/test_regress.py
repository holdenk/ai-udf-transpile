# SPDX-License-Identifier: Apache-2.0
"""regress.py: the static guess at whether a rewrite is slower than the
python. No JVM. The FHI table pins the guess next to what was measured."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ai_udf_transpile.backends.fhi_rewrites import fhi_rewrite
from ai_udf_transpile.backends.fhi_udfs import FHI_UDFS
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.regress import (
    AT_RISK,
    LIKELY_REGRESSION,
    NOT_SERVED,
    OK,
    python_regex_calls,
    regression_risk,
)
from ai_udf_transpile.targets import TranspileResult


def _sql(text: str) -> TranspileResult:
    return TranspileResult(kind="catalyst", sql=text)


@pytest.mark.parametrize(
    "name,calls",
    [
        ("fhi_verdict_has_legal_check", 0),
        ("fhi_clean_health_history", 0),
        ("fhi_diagnosis_from_findings", 2),  # a loop over a 2-tuple of patterns
        ("fhi_parse_verdict", 4),  # the loop is over the data's lines: counted once
        ("fhi_deidentify_text", 8),  # 1 + a loop over 6 + 1
        ("fhi_treatment_from_findings", 10),
        ("fhi_normalize_entity_text", 19),  # 2 + a loop over 17
    ],
)
def test_python_regex_calls(name, calls):
    assert python_regex_calls(canonical_source_from_func(FHI_UDFS[name][0])) == calls


def test_python_regex_calls_sees_compiled_patterns_and_aliases():
    src = (
        "def f(s: str) -> str:\n"
        "    import re as r\n"
        "    p = r.compile('a+')\n"
        "    for q in ('x', 'y', 'z'):\n"
        "        s = r.sub(q, '', s)\n"
        "    return p.sub('', s) if s.strip() else s\n"
    )
    assert python_regex_calls(src) == 4
    assert python_regex_calls("def f(s: str) -> str:\n    return s.strip().upper()\n") == 0
    assert python_regex_calls("not python at all (") == 0


# (function, variant) -> (guess, measured speedup vs python on a nullable
# column; None = Spark does not serve it). Measured on one 4-core machine:
# the guess is about the sign, and `at-risk` means "measure before serving".
FHI_GUESSES = {
    ("fhi_clean_health_history", "naive"): (LIKELY_REGRESSION, 0.3),
    ("fhi_clean_health_history", "tuned"): (OK, 1.9),
    ("fhi_verdict_has_legal_check", "naive"): (LIKELY_REGRESSION, 0.5),
    ("fhi_verdict_has_legal_check", "tuned"): (OK, 4.5),
    ("fhi_parse_verdict", "naive"): (NOT_SERVED, None),
    ("fhi_parse_verdict", "tuned"): (AT_RISK, 9.4),  # a miss: one regex, only on the verdict line
    ("fhi_treatment_from_findings", "naive"): (LIKELY_REGRESSION, 0.8),
    ("fhi_treatment_from_findings", "tuned"): (AT_RISK, 1.2),
    ("fhi_diagnosis_from_findings", "naive"): (OK, 3.0),
    ("fhi_normalize_entity_text", "naive"): (LIKELY_REGRESSION, 0.7),
    ("fhi_normalize_entity_text", "tuned"): (AT_RISK, 0.8),
    ("fhi_deidentify_text", "naive"): (LIKELY_REGRESSION, 0.4),
    ("fhi_deidentify_text", "tuned"): (LIKELY_REGRESSION, 0.8),
}


@pytest.mark.parametrize("name,variant", sorted(FHI_GUESSES))
def test_fhi_guess(name, variant):
    guess, measured = FHI_GUESSES[(name, variant)]
    risk = regression_risk(canonical_source_from_func(FHI_UDFS[name][0]), fhi_rewrite(name, variant))
    assert risk.verdict == guess, str(risk)
    # The property worth having: nothing called a likely regression won,
    # and nothing called ok lost.
    if measured is not None:
        assert not (guess == LIKELY_REGRESSION and measured >= 1.0)
        assert not (guess == OK and measured < 1.0)


def test_the_null_guard_that_defeats_sharing_is_named():
    src = canonical_source_from_func(FHI_UDFS["fhi_clean_health_history"][0])
    naive = regression_risk(src, fhi_rewrite("fhi_clean_health_history", "naive"))
    assert {"null-guard-defeats-cse", "repeated-regex", "regex-introduced"} <= set(naive.flags)
    assert naive.rewrite_regex_evals == 4 and naive.python_regex_calls == 0
    tuned = regression_risk(src, fhi_rewrite("fhi_clean_health_history", "tuned"))
    assert tuned.rewrite_regex_evals == 0 and "null-guard-defeats-cse" not in tuned.flags


NO_REGEX = "def f(s: str) -> str:\n    return s.strip()\n"
TWO_REGEX = "def f(s: str) -> str:\n    import re\n    return re.sub('a', '', re.sub('b', '', s))\n"
MANY_REGEX = (
    "def f(s: str) -> str:\n    import re\n"
    "    for p in ('a', 'b', 'c', 'd', 'e', 'f'):\n        s = re.sub(p, '', s)\n    return s\n"
)


def test_verdict_rules():
    # A lambda is not served at all; nothing else about the text matters.
    assert (
        regression_risk(NO_REGEX, _sql("transform(split(_udf_param_0, ','), x -> upper(x))[0]")).verdict
        == NOT_SERVED
    )
    # ...but an arrow inside a string literal is data.
    assert regression_risk(NO_REGEX, _sql("replace(_udf_param_0, '->', '')")).verdict == OK
    # No regex on either side: the round trip is pure saving.
    assert regression_risk(NO_REGEX, _sql("btrim(_udf_param_0, ' ')")).verdict == OK
    # A C string method became one regex: maybe; two or more: no.
    assert regression_risk(NO_REGEX, _sql("regexp_replace(_udf_param_0, '^ +| +$', '')")).verdict == AT_RISK
    two = "regexp_replace(regexp_replace(_udf_param_0, '^ +', ''), ' +$', '')"
    assert regression_risk(NO_REGEX, _sql(two)).verdict == LIKELY_REGRESSION
    assert regression_risk(NO_REGEX, _sql("_udf_param_0 RLIKE 'a' AND _udf_param_0 rlike 'b'")).verdict == (
        LIKELY_REGRESSION
    )
    # split is a regex; split_part is not; a function name inside a literal is data.
    assert regression_risk(NO_REGEX, _sql("split(_udf_param_0, '::')[0]")).rewrite_regex_evals == 1
    assert regression_risk(NO_REGEX, _sql("split_part(_udf_param_0, '::', 1)")).rewrite_regex_evals == 0
    assert (
        regression_risk(NO_REGEX, _sql("replace(_udf_param_0, 'regexp_replace(', '')")).rewrite_regex_evals
        == 0
    )
    # A couple of regexes moved engines: the round trip still dominates.
    moved = "regexp_replace(regexp_replace(_udf_param_0, 'b', ''), 'a', '')"
    assert regression_risk(TWO_REGEX, _sql(moved)).verdict == OK
    # A regex-dominated body can at best break even...
    six = "_udf_param_0"
    for p in "abcdef":
        six = f"regexp_replace({six}, '{p}', '')"
    assert regression_risk(MANY_REGEX, _sql(six)).verdict == AT_RISK
    # ...and loses once python's unicode classes are spelled out for Java,
    emulated = six.replace("'a'", "'[\\\\s\\\\x85\\\\u2028]a'")
    risk = regression_risk(MANY_REGEX, _sql(emulated))
    assert risk.verdict == LIKELY_REGRESSION and "unicode-classes" in risk.flags
    # unless an ASCII guard keeps that spelling off the common path.
    guarded = f"CASE WHEN length(_udf_param_0) = octet_length(_udf_param_0) THEN {six} ELSE {emulated} END"
    risk = regression_risk(MANY_REGEX, _sql(guarded))
    assert risk.verdict == AT_RISK and "ascii-guard" in risk.flags
    # Java: no regex is fine, regex is worth measuring.
    plain = TranspileResult(kind="java_udf", java_source="class T { Object call(Object s) { return s; } }")
    assert regression_risk(NO_REGEX, plain).verdict == OK
    rx = TranspileResult(
        kind="java_udf", java_source='class T { static final Pattern P = Pattern.compile("a"); }'
    )
    assert regression_risk(TWO_REGEX, rx).verdict == AT_RISK


def test_no_documented_win_is_called_a_likely_regression():
    """The documented examples are wins measured by other benches. If one of
    them is ever flagged, the rules have a false alarm to explain."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "EXAMPLES_TRANSPILED.md"
    if not doc.exists():
        pytest.skip("no docs/EXAMPLES_TRANSPILED.md")
    scored = 0
    for section in re.split(r"^## ", doc.read_text(encoding="utf-8"), flags=re.M)[1:]:
        name = section.split("\n", 1)[0].strip()
        source = re.search(r"```python\n(.*?)```", section, re.S)
        served = re.search(
            r"^- \*\*AI \([^)]*\)\*\*: `(.*?)` \(attempt_count.*?\(([0-9.]+)x vs python\)",
            section,
            re.M | re.S,
        )
        if not source or not served:
            continue
        risk = regression_risk(source.group(1), _sql(served.group(1)))
        scored += 1
        if float(served.group(2)) >= 1.5:
            assert risk.verdict != LIKELY_REGRESSION, f"{name}: {risk} but documented at {served.group(2)}x"
    assert scored >= 20  # the doc format moved if this trips
