# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from pyspark.sql.functions import udf
from pyspark.sql.types import LongType

from ai_udf_transpile import enable
from ai_udf_transpile.transpiler import get_catalog

pytestmark = pytest.mark.spark


def untyped_plus(x):
    return x + 1


def plus_one(x: int) -> int:
    return x + 1


def test_untyped_udf_never_inserts(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    udf(untyped_plus, LongType())
    catalog = get_catalog()
    assert catalog is not None
    assert catalog.count() == 0


def test_annotated_udf_inserts_pending(spark, sqlite_path):
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)
    udf(plus_one, LongType())
    catalog = get_catalog()
    assert catalog is not None
    assert catalog.count() == 1
    row = catalog.oldest_pending() or catalog.get(next(_iter_keys(catalog)))
    assert row is not None
    assert row.status in {"pending", "running", "success"}
    assert row.input_types == ["bigint"]
    assert row.return_type in {"bigint", "long"}
    assert row.param_names == ["x"]


def _iter_keys(catalog):
    # SqliteCatalog doesn't list keys; use a direct query.
    cur = catalog._conn.execute("SELECT udf_key FROM cache")
    for (key,) in cur.fetchall():
        yield key
