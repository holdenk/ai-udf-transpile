# SPDX-License-Identifier: Apache-2.0
"""kms_encrypt: the canonical untranspilable UDF (boto3 KMS call per row).

There is no Catalyst equivalent of AWS KMS Encrypt -- the ciphertext comes
from the service and is non-deterministic -- so any backend SQL "rewrite"
is a hallucination. The safety property this file pins down: verification
must FAIL CLOSED when the python side can never be observed to succeed.

Two sandbox realities, both covered:

- boto3 not installed: every call raises ImportError, which verification
  already treats as a mismatch (not a per-row python error).
- boto3 installed but no credentials: every call raises a plain Exception
  (botocore NoCredentialsError). Under "python raise + sql value allowed"
  that used to accept ANY rewrite vacuously; verification now requires at
  least one successful python evaluation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.types import StringType
from pyspark.sql.udf import UserDefinedFunction

from ai_udf_transpile import enable, register_impl
from ai_udf_transpile.backends.fake import kms_encrypt
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult
from ai_udf_transpile.transpiler import get_catalog
from ai_udf_transpile.verify import hypothesis_check

pytestmark = pytest.mark.spark


def creds_missing(text: str) -> str:
    # Simulates "boto3 installed, no credentials": every call raises a plain
    # Exception (botocore's NoCredentialsError), not an ImportError.
    raise RuntimeError("NoCredentialsError: Unable to locate credentials")


def picky_echo(text: str) -> str:
    # Control: raises on some inputs, succeeds on others -- verification
    # still has signal and must accept a faithful rewrite.
    if text == "boom":
        raise RuntimeError("explode")
    return text


def _check(func, sql, spark, arity=1):
    return hypothesis_check(
        source_text=canonical_source_from_func(func),
        captures={},
        result=TranspileResult(kind=KIND_CATALYST, sql=sql),
        input_types=["string"] * arity,
        return_type="string",
        spark=spark,
        max_examples=20,
    )


def test_all_raised_fails_closed(spark):
    # base64 "looks plausible" for an encryption UDF and python never
    # contradicts it (it raises everywhere) -- reject for lack of signal.
    ok, err = _check(creds_missing, "base64(_udf_param_0)", spark)
    assert not ok
    assert "no signal" in (err or "")


def test_mixed_raise_and_success_still_verifies(spark):
    ok, err = _check(picky_echo, "_udf_param_0", spark)
    assert ok, err


def test_hallucinated_sql_rejected_without_boto3(spark):
    # boto3 is not installed in this environment: ImportError on every call.
    ok, err = _check(kms_encrypt, "base64(_udf_param_0)", spark, arity=2)
    assert not ok
    assert "ModuleNotFoundError" in (err or "")  # ImportError subclass


def test_fake_backend_declines_and_cooldown(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=True)
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    UserDefinedFunction(kms_encrypt, StringType())
    assert get_catalog().count() == 1, "typed UDF should be queued (decline happens later)"

    import time

    deadline = time.time() + 60
    status = None
    while time.time() < deadline:
        row = get_catalog()._conn.execute("SELECT status, origin, error FROM cache").fetchone()
        if row and row[0] == "failed":
            status = row
            break
        time.sleep(0.5)
    assert status is not None, "fake backend never declined kms_encrypt"
    assert status[1] == "fake"
    assert "no fake fixture" in status[2]

    before = get_catalog().count()
    UserDefinedFunction(kms_encrypt, StringType())
    assert get_catalog().count() == before, "cooldown should prevent re-queue"


def test_register_impl_rejects_hallucinated_sql(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError):
        register_impl(
            spark,
            kms_encrypt,
            kind="catalyst",
            catalyst_sql="base64(_udf_param_0)",
            return_type=StringType(),
        )
