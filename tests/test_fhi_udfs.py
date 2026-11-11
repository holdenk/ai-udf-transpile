# SPDX-License-Identifier: Apache-2.0
"""The FHI pipeline UDFs (backends/fhi_udfs.py): what each port returns, and
that every one of them gets past the gate -- so the bench asks a backend
about them instead of silently skipping. No JVM."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from ai_udf_transpile import keys
from ai_udf_transpile import types as udf_types
from ai_udf_transpile.backends.fhi_udfs import (
    FHI_UDFS,
    fhi_clean_health_history,
    fhi_deidentify_text,
    fhi_diagnosis_from_findings,
    fhi_normalize_entity_text,
    fhi_parse_verdict,
    fhi_treatment_from_findings,
    fhi_verdict_has_legal_check,
)


@pytest.mark.parametrize("name", sorted(FHI_UDFS))
def test_port_passes_the_gate(name):
    func, return_type = FHI_UDFS[name]
    assert func.__name__ == name
    function_ast = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    params = [a.arg for a in function_ast.args.args]
    # Self-contained on purpose: the originals reach module-level compiled
    # patterns and helpers, which extract_captures cannot key.
    assert udf_types.types_known(function_ast, params, return_type)
    assert keys.extract_captures(func, function_ast, params) == {}


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, False),
        ("", False),
        ("VERDICT: VIABLE\nLEGAL: OK", True),
        ("VERDICT: VIABLE\n  LEGAL: ERISA 503 cited  ", True),
        ("LEGAL: UNCHECKED", False),
        ("LEGAL:", False),
        ("LEGAL:   ", False),
        ("legal: ok", False),  # the marker is case-sensitive
        ("NOTE LEGAL: ok", False),  # and must start the stripped line
        ("LEGAL: UNCHECKED\nLEGAL: OK", False),  # first LEGAL line decides
        ("VERDICT: VIABLE\rLEGAL: OK", True),  # splitlines, not split('\n')
        ("VERDICT: VIABLE LEGAL: OK", True),
        (" LEGAL: OK", True),  # strip() takes NBSP; SQL trim does not
        ("LEGAL: a:b", True),
    ],
)
def test_verdict_has_legal_check(text, expected):
    assert fhi_verdict_has_legal_check(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, None),
        ("no decision here", None),
        ("VERDICT: VIABLE", True),
        ("VERDICT: NOT_VIABLE", False),
        ("- **VERDICT: NOT VIABLE**", False),
        ("1. VERDICT: VIABLE (dates unverified)", True),
        ("Verdict: non-viable", False),
        ("verdict: unviable", False),
        ("VERDICT: VIABLE or NOT_VIABLE", None),  # both words: unclear
        # Two different negative words are still one answer. A rewrite that
        # counts distinct matched words instead of distinct meanings returns
        # NULL here -- and one did, verified, because the gate's generated
        # strings (16 chars at most) cannot hold two verdict words.
        ("VERDICT: UNVIABLE and NON VIABLE", False),
        ("VERDICT: VIABLE, viable", True),
        ("VERDICT", None),  # no colon, no value
        ("VERDICT: maybe", None),
        ("REASON: x\n> `VERDICT: VIABLE`", True),
        ("VERDICT: NOT_VIABLE\nVERDICT: VIABLE", False),  # first VERDICT line decides
        ("VERDICT: INVIABLE", None),  # \b: VIABLE inside a word is not the word
    ],
)
def test_parse_verdict(text, expected):
    assert fhi_parse_verdict(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("UNKNOWN", None),
        (" unknown ", None),
        ("None.", None),
        ("none", None),
        ("..NONE..", None),
        ("None reported", "None reported"),
        ("  Type 2 diabetes.  ", "Type 2 diabetes."),
        (" asthma ", "asthma"),
    ],
)
def test_clean_health_history(text, expected):
    assert fhi_clean_health_history(text) == expected


@pytest.mark.parametrize(
    "findings,tc,tsc,expected",
    [
        (None, None, None, None),
        (
            "The patient requested coverage for 12 sessions of physical therapy. Denied.",
            "Rehab",
            "PT",
            "physical therapy",
        ),
        ("At issue is whether Humira 40mg is medically necessary.", "Pharmacy", "Other", "Humira 40mg"),
        ("The Health Plan determined the lumbar MRI was not needed.", None, None, "lumbar MRI"),
        ("The physicians recommended inpatient rehab.", "A", "B", "inpatient rehab"),
        # No regex hit: the coarse category pair, unless either side is Other/missing.
        ("nothing to extract", "Rehab", "PT", "RehabPT"),
        ("nothing to extract", "Other", "PT", None),
        ("nothing to extract", "Rehab", "Other", None),
        ("nothing to extract", "Rehab", None, None),
        (None, "Rehab", "PT", "RehabPT"),
        ("", "Rehab", "PT", "RehabPT"),
        # The ninth pattern has no `treatment` group: its match raises, and
        # the registered UDF's except turns that into None -- not the fallback.
        ("She is seeking  for asthma.", "Rehab", "PT", None),
        # A captured treatment of two characters or fewer is skipped.
        ("The physicians recommended PT.", "Rehab", "PT", "RehabPT"),
    ],
)
def test_treatment_from_findings(findings, tc, tsc, expected):
    assert fhi_treatment_from_findings(findings, tc, tsc) == expected


@pytest.mark.parametrize(
    "findings,dc,dsc,expected",
    [
        (None, None, None, None),
        ("She was admitted to the hospital for acute pancreatitis. Then", None, None, "acute pancreatitis"),
        ("He is seeking  for chronic migraine. Denied.", "X", "Y", "chronic migraine"),
        ("ADMITTED TO THE HOSPITAL FOR sepsis", "X", "Y", "sepsis"),
        ("nothing to extract", "Digestive", "Pancreatitis", "DigestivePancreatitis"),
        ("nothing to extract", "Other", "Pancreatitis", None),
        ("nothing to extract", "Digestive", "Other", None),
        ("nothing to extract", None, "Pancreatitis", None),
    ],
)
def test_diagnosis_from_findings(findings, dc, dsc, expected):
    assert fhi_diagnosis_from_findings(findings, dc, dsc) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, None),
        ("", ""),
        ("**The diagnosis is** <answer>Type 2 diabetes</answer> ---", "Type 2 diabetes"),
        ("Treatment: physical therapy", "physical therapy"),
        ("the DIAGNOSIS  is\tasthma", "asthma"),  # (?i) and \s+
        ("The procedure is an MRI", "MRI"),
        ("**bold** and **unclosed", "bold and unclosed"),
        # `.` stops at \n only: the lead-in is cut on its own line, and the
        # greedy `.*` keeps everything before the last lead-in on that line.
        ("first line\nDiagnosis: asthma", "first line\nasthma"),
        ("Diagnosis: a Diagnosis: b", "b"),
        (" UNKNOWN ", "UNKNOWN"),
    ],
)
def test_normalize_entity_text(text, expected):
    assert fhi_normalize_entity_text(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, None),
        ("", ""),
        ("Patient SSN 123-45-6789", "Patient SSN {{SSN}}"),
        ("Call (415) 555-1212 now", "Call {{Your Phone Number}} now"),
        ("Call +1 415.555.1212 now", "Call {{Your Phone Number}} now"),
        # A phone-shaped run inside a longer digit run is left alone.
        ("Claim number 9874155551212", "Claim number 9874155551212"),
        ("mail jo.smith+x@example.org today", "mail {{Your Email Address}} today"),
        ("DOB: 01/02/1980", "{{DOB}}"),
        ("born 1980-01-02", "{{DOB}}"),
        ("Member ID: AB1234567", "{{MEMBER_ID}}"),
        ("I live at 12 Main Street.", "I live at {{ADDRESS}}."),
        # Existing placeholders survive, whatever they contain.
        ("{{Your Phone Number}} [NAME 123-45-6789]", "{{Your Phone Number}} [NAME 123-45-6789]"),
        # \d is unicode in Python: Arabic-Indic digits are an SSN here, and
        # a rewrite on an ASCII-only \d would leave them in the clear.
        ("١٢٣-٤٥-٦٧٨٩", "{{SSN}}"),
        # A token-shaped run that was already in the input is not an index
        # this call handed out, so it comes back untouched.
        ("\x00PH7\x00 {{NAME}}", "\x00PH7\x00 {{NAME}}"),
    ],
)
def test_deidentify_text(text, expected):
    assert fhi_deidentify_text(text) == expected


def test_deidentify_text_is_idempotent():
    letter = (
        "Dear {{FIRST_NAME}}, reach me at 415-555-1212 or jo@example.org. "
        "SSN 123-45-6789, DOB: 01/02/1980, Member ID: AB1234567, 12 Main Street."
    )
    once = fhi_deidentify_text(letter)
    assert "415" not in once and "6789" not in once and "example.org" not in once
    assert fhi_deidentify_text(once) == once


@pytest.mark.spark
def test_bench_frames_exercise_every_port(spark):
    """The bench's synthetic frames line up with each port's parameters and
    drive more than one outcome out of it -- a frame that only ever produced
    NULL would bench the early return, not the function."""
    import importlib.util
    from pathlib import Path

    from pyspark.sql.types import _parse_datatype_string
    from pyspark.sql.udf import UserDefinedFunction

    path = Path(__file__).resolve().parents[1] / "examples" / "bench_fhi_udfs.py"
    spec = importlib.util.spec_from_file_location("bench_fhi_udfs", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    assert set(bench.FRAME_EXPRS) == set(FHI_UDFS)
    for name, (func, return_type) in FHI_UDFS.items():
        cols = list(bench.FRAME_EXPRS[name])
        assert cols == list(inspect.signature(func).parameters), name
        udf = UserDefinedFunction(func, _parse_datatype_string(return_type))
        rows = bench.frame(spark, name, 60).select(*cols, udf(*cols).alias("out")).collect()
        assert len(rows) == 60
        for row in rows:  # the UDF path returns what a direct call returns
            assert row["out"] == func(*[row[c] for c in cols]), name
        assert len({row["out"] for row in rows}) >= 2, name
