# SPDX-License-Identifier: Apache-2.0
"""Real pipeline UDFs: the scalar text UDFs from Fight Health Insurance's
training-data pipeline (fighthealthinsurance/healthinsurance-llm-internal,
dataset_tools/, Apache-2.0), ported for the benches.

Unlike the rest of the corpus these were not written to be transpiled. They
post-process LLM output (verdict blocks, extracted entities) and scrub PII,
and every one of them is a regex or a line loop over free text -- which is
where Python `re` and `java.util.regex` stop agreeing (`\\s`/`\\d`/`\\b` are
Unicode-aware in Python and ASCII in Java, `.` stops at different line
terminators, `str.strip()` is not `trim`, `splitlines()` is not
`split('\\n')`).

The originals are un-annotated or `Optional[str]`, and reach module-level
compiled patterns and helpers, so none of them passes this plugin's gate as
written (types.py, keys.extract_captures). Each port below is the same
logic made self-contained: `str` annotations, `import re` in the body,
helpers and pattern tables inlined. Where the original is registered through
a wrapper (the `@udf` try/except around the findings extractors) the port is
wrapper and callee together, so it returns what the registered UDF returns.
Executor-side prints are dropped; nothing else is.
"""

from __future__ import annotations


def fhi_verdict_has_legal_check(verdict_text: str) -> bool:
    # legal_check.verdict_has_legal_check: was the stored verdict computed
    # with a legal check in hand? False on NULL, never NULL.
    for raw in (verdict_text or "").splitlines():
        line = raw.strip()
        if line.startswith("LEGAL:"):
            return line.split(":", 1)[1].strip() not in ("", "UNCHECKED")
    return False


def fhi_parse_verdict(text: str) -> bool:
    # appeal_grounding.parse_verdict with _clean_line and _verdict_value
    # inlined: True/False from the first VERDICT line, None without one or
    # when that line carries both words.
    import re

    markup = r"^[\s\-*#>`]*|[\s*`]*$"
    for raw in (text or "").splitlines():
        line = re.sub(markup, "", raw)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = re.sub(markup, "", line)
        upper = line.upper()
        if upper.startswith("VERDICT"):
            value = upper.split(":", 1)[1] if ":" in upper else ""
            words = re.findall(r"\b((?:NOT|UN|NON)[\s_-]*VIABLE|VIABLE)\b", value.upper())
            found = {m == "VIABLE" for m in words}
            return found.pop() if len(found) == 1 else None
    return None


def fhi_clean_health_history(text: str) -> str:
    # appeal_qa_chain.clean_health_history with _clean inlined: the
    # extracted patient history, or None for an empty / NONE / UNKNOWN one.
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned or cleaned.upper() == "UNKNOWN":
        return None
    if cleaned.strip().strip(".").upper() in ("NONE", "UNKNOWN"):
        return None
    return cleaned


def fhi_treatment_from_findings(findings: str, tc: str, tsc: str) -> str:
    # ca_data_spark.get_treatment_from_findings_and_categories_udf: first
    # regex hit over the reviewer's findings, else the coarse category pair.
    # The ninth pattern has no `treatment` group, so a match on it raises
    # and the wrapper's `except Exception` turns that into None -- kept.
    import re

    patterns = (
        r"\s*(The|An|A)?\s*(parent|father|mother|patient|beneficiary|enrollee|member|provider)\s*[^.]*"
        r"(requested|required|asked|requires|reimbursement|coverage|requesting|has)\s*[^.]*"
        r"(of|for|medication|reimbursement|coverage|services)\s+(?P<treatment>\d*\w+.+?)\.",
        r"At issue\s*(in this case|)\s*(is|)\s*(whether|if)\s+(?P<treatment>\d*\w+.+?) (is|were|was)"
        r" medically (necessary|indicated)",
        r"the requested (medication|treatment|service|procedure)\s+(?P<treatment>\d*\w+.+?) (is|were|was)"
        r" (likely to be|medically necessary|medically indicated)",
        r"patients provider has prescribed the medication\s+(?P<treatment>[^.]+?).",
        r"(wishes|desires|like) to (undergo|take)\s+(?P<treatment>[^.]+?).",
        r"The (Health Plan|Plan|Insurance Company) (determined the|determined|indicates)"
        r" (?P<treatment>.+?) (is|was|were) not",
        r"treatment[^.]*with\s+(?P<treatment>[^.]+?) (is|were|was)",
        r"(requested|required|asked|requires|reimbursement|coverage|request|requesting)\s*[^.]*"
        r"(for|medication|reimbursement|coverage|of)\s+(?P<treatment>\d*\w+.+?)\.",
        r"is seeking (?P<treatement>) for (?P<diagnosis>[^.]+)",
        r"physicians recommended (?P<treatment>[^.]+)",
    )
    try:
        if findings is None and tc is None and tsc is None:
            return None
        if findings:
            for pattern in patterns:
                matches = re.search(pattern, findings, re.IGNORECASE)
                if matches is not None:
                    t = matches.group("treatment")
                    if t is not None and len(t) > 2:
                        return t
        if tc and tc == "Other":
            return None
        if tsc and tsc == "Other":
            tsc = ""
        if tsc and tc:
            return f"{tc}{tsc}"
        return None
    except Exception:
        return None


def fhi_diagnosis_from_findings(findings: str, dc: str, dsc: str) -> str:
    # ca_data_spark.get_diagnosis_from_findings_and_categories_udf.
    import re

    patterns = (
        r"is seeking (?P<treatement>) for (?P<diagnosis>[^.]+)",
        r"admitted to the hospital for (?P<diagnosis>[^.]+)",
    )
    try:
        if findings is None and dc is None and dsc is None:
            return None
        if findings:
            for pattern in patterns:
                matches = re.search(pattern, findings, re.IGNORECASE)
                if matches is not None:
                    d = matches.group("diagnosis")
                    if d is not None and len(d) > 2:
                        return d
        if dc and dc == "Other":
            return None
        if dsc and dsc == "Other":
            dsc = ""
        if dc and dsc:
            return f"{dc}{dsc}"
        return None
    except Exception:
        return None


def fhi_normalize_entity_text(text: str) -> str:
    # spark_utils.normalize_entity_text, the StringType registration: strip
    # the markup and the "The diagnosis is ..." lead-ins a model wraps an
    # extracted entity in. (The original also takes a list; that is the
    # ArrayType registration, a different UDF.)
    import re

    if text is None:
        return None
    cleaned_text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    cleaned_text = re.sub(r"<answer>(.*?)</answer>", r"\1", cleaned_text)
    patterns_to_remove = (
        r"(?i).*The\s+diagnosis\s+is\s+",
        r"(?i).*Diagnosis:\s+",
        r"(?i).*The\s+procedure\s+is\s+an\s+",
        r"(?i).*The\s+procedure\s+is\s+",
        r"(?i).*Procedure:\s+",
        r"(?i).*The\s+treatment\s+is\s+",
        r"(?i).*Treatment:\s+",
        r"(?i).*The\s+insurance\s+company\s+is\s+",
        r"(?i).*Insurance\s+company:\s+",
        r"(?i).*The\s+phone\s+number\s+is\s+",
        r"(?i).*Phone\s+number:\s+",
        r"(?i).*The\s+fax\s+number\s+is\s+",
        r"(?i).*Fax\s+number:\s+",
        r"(?i).*The\s+email\s+address\s+is\s+",
        r"(?i).*Email\s+address:\s+",
        r"\*\*",
        r"---",
    )
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text)
    return cleaned_text.strip()


def fhi_deidentify_text(text: str) -> str:
    # deid.deidentify_text: the regex PII pass. Existing {{PLACEHOLDER}} /
    # [PLACEHOLDER] tokens are parked behind NUL-delimited indexes, six PII
    # patterns run in order (SSN before phone), then the tokens come back.
    # A PII scrubber fails in one direction: a rewrite that redacts less
    # than this does is a leak, not a mismatch.
    import re

    if not text:
        return text
    placeholder_re = (
        r"\{\{(?:FIRST_NAME|LAST_NAME|NAME|PHONE|Your Phone Number|SSN|EMAIL|Your Email Address|"
        r"DOB|ADDRESS|MEMBER_ID|PLAN_ID|CASEID|SCSID|GPID|FAX|DATE|ID|REDACTED)[^}]*\}\}"
        r"|"
        r"\[(?:NAME|PHONE|SSN|EMAIL|DOB|ADDRESS|MEMBER_ID|PLAN_ID|FAX|DATE|ID|REDACTED)[^\]]*\]"
    )
    pii_patterns = (
        ("{{SSN}}", r"\b\d{3}-\d{2}-\d{4}\b"),
        (
            "{{Your Phone Number}}",
            r"(?<![\d-])(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?![\d-])",
        ),
        ("{{Your Email Address}}", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        (
            "{{DOB}}",
            r"(?i)(?:DOB|date\s+of\s+birth|born|birthday|d\.o\.b\.?)\s*:?\s*"
            r"(?:"
            r"(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}"
            r"|(?:19|20)\d{2}[/\-](?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])"
            r")",
        ),
        (
            "{{MEMBER_ID}}",
            r"(?i)(?:member|group|subscriber|policy|plan)\s*"
            r"(?:id|#|number|no\.?)\s*:?\s*"
            r"[A-Z]{0,4}\d{6,12}\b",
        ),
        (
            "{{ADDRESS}}",
            r"(?i)\b\d{1,5}\s+(?:[A-Z][a-z]+\s*){1,3}"
            r"(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Ln|Lane|Rd|Road|Ct|Court|Way|Pl|Place|Cir|Circle)"
            r"\.?\b",
        ),
    )
    placeholders_found = []

    def _save_placeholder(m):
        placeholders_found.append(m.group(0))
        return f"\x00PH{len(placeholders_found) - 1}\x00"

    working = re.sub(placeholder_re, _save_placeholder, text)
    for placeholder, pattern in pii_patterns:
        working = re.sub(pattern, placeholder, working)
    if placeholders_found:

        def _restore_placeholder(m):
            idx = int(m.group(1))
            if idx >= len(placeholders_found):
                return m.group(0)
            return placeholders_found[idx]

        working = re.sub(r"\x00PH(\d+)\x00", _restore_placeholder, working)
    return working


# name -> (function, Spark return type simpleString). The benches and the
# tests both walk this.
FHI_UDFS = {
    "fhi_verdict_has_legal_check": (fhi_verdict_has_legal_check, "boolean"),
    "fhi_parse_verdict": (fhi_parse_verdict, "boolean"),
    "fhi_clean_health_history": (fhi_clean_health_history, "string"),
    "fhi_treatment_from_findings": (fhi_treatment_from_findings, "string"),
    "fhi_diagnosis_from_findings": (fhi_diagnosis_from_findings, "string"),
    "fhi_normalize_entity_text": (fhi_normalize_entity_text, "string"),
    "fhi_deidentify_text": (fhi_deidentify_text, "string"),
}
