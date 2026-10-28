# SPDX-License-Identifier: Apache-2.0
"""kms_encrypt: the canonical untranspilable UDF (boto3 KMS call per row).

There is no Catalyst equivalent of AWS KMS Encrypt -- the ciphertext comes
from the service and is non-deterministic -- so any backend SQL "rewrite"
is a hallucination. What verification does about it depends on the env:

- boto3 absent: every call raises ImportError, verification fails closed.
- boto3 installed, no credentials (conftest scrubs every source): the
  NULL-guard row (text is None -> None) supplies the one successful python
  evaluation the vacuous-pass guard requires, base64(NULL)=NULL matches
  it, and every service-calling row is a "python raise + sql value"
  allowance -- so the differential gate ACCEPTS the rewrite. Black-box
  verification cannot refute a service-calling UDF when the service cannot
  succeed in the verification env. Pinned below as the documented gap, not
  treated as a bug: the gate is only as strong as the env it runs in.
"""

from __future__ import annotations

import importlib.util

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


BOTO3 = importlib.util.find_spec("boto3") is not None


def test_hallucinated_sql_fails_closed_without_boto3(spark):
    if BOTO3:
        pytest.skip("boto3 installed: the ImportError reality is not reachable")
    ok, err = _check(kms_encrypt, "base64(_udf_param_0)", spark, arity=2)
    assert not ok
    assert "ModuleNotFoundError" in (err or "")  # ImportError subclass


def test_hallucinated_sql_with_boto3_is_the_documented_gap(spark):
    # See the module docstring: the NULL-guard row gives verification its
    # one successful python eval, base64(NULL)=NULL matches it, and every
    # service-calling row is a python-raise + sql-value allowance. The gate
    # cannot refute a service it cannot reach -- pin the acceptance so a
    # change in this behavior is seen, not silently shipped.
    if not BOTO3:
        pytest.skip("boto3 absent: the no-credentials reality is not reachable")
    ok, err = _check(kms_encrypt, "base64(_udf_param_0)", spark, arity=2)
    assert ok, err


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


def test_register_impl_rejects_hallucinated_sql_without_boto3(spark, sqlite_path):
    if BOTO3:
        pytest.skip("boto3 installed: see the documented-gap test below")
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    with pytest.raises(ValueError):
        register_impl(
            spark,
            kms_encrypt,
            kind="catalyst",
            catalyst_sql="base64(_udf_param_0)",
            return_type=StringType(),
        )


def test_register_impl_accepts_hallucinated_sql_with_boto3(spark, sqlite_path):
    # The same gap through the register_impl path: with boto3 installed and
    # no credentials, the hallucinated rewrite verifies and would be served.
    # That is the gate's black-box limit, not a register_impl bug.
    if not BOTO3:
        pytest.skip("boto3 absent: the no-credentials reality is not reachable")
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    register_impl(
        spark,
        kms_encrypt,
        kind="catalyst",
        catalyst_sql="base64(_udf_param_0)",
        return_type=StringType(),
    )
