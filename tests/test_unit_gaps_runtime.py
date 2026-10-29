# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sqlite3
import types

import pytest

import ai_udf_transpile.worker as worker_mod
from ai_udf_transpile import (
    _ansi_on,
    _default_sqlite_path,
    _function_ast,
    _public_params,
    conf,
    enable,
    register_impl,
    shutdown,
)
from ai_udf_transpile.backends.fake import FakeBackend, always_decline, plus_one
from ai_udf_transpile.catalog import (
    CacheRow,
    configured_tolerance,
    loads_dict,
    loads_list,
    open_catalog,
    parse_iso,
)
from ai_udf_transpile.catalog.sqlite import SqliteCatalog
from ai_udf_transpile.catalog.writeback import WritebackCatalog
from ai_udf_transpile.javac import JavaCompileError, _stage_jar, compile_java
from ai_udf_transpile.keys import canonical_source_from_func
from ai_udf_transpile.targets import TranspileResult
from ai_udf_transpile.transpiler import get_session, set_catalog
from ai_udf_transpile.verify import (
    VerifyFailed,
    _copy_defining_module_imports,
    _hydrate_captures,
    _quote_closes,
    _smoke_test_reconstruction,
    _spark_type,
    _strategy_for,
    _within_tolerance,
    collect_imported_module_names,
    hypothesis_check,
)
from ai_udf_transpile.worker import (
    _reported_model,
    inline_thread_alive,
    main,
    maybe_start_inline,
    poll_once,
    process_row,
    stop_inline,
)


def _pending(cat: SqliteCatalog, key: str = "k", func=plus_one) -> None:
    cat.insert_pending(
        udf_key=key,
        source_text=canonical_source_from_func(func),
        param_names=["x"],
        input_types=["bigint"],
        input_categories=["numeric"],
        return_type="bigint",
        spark_version="test",
        closure_fingerprint="",
        captures={},
    )


# -- worker.py -----------------------------------------------------------


def test_reported_model_swallows_exception():
    class _Explodes:
        def reported_model(self):
            raise RuntimeError("nope")

    assert _reported_model(_Explodes()) is None


def test_process_row_human_origin_reuses_existing_candidate(tmp_path, monkeypatch):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k", "human")
    cat.mark_success(
        "k", TranspileResult(kind="catalyst", sql="_udf_param_0 + 1"), origin="human", hypothesis_passed=True
    )
    cat._conn.execute("UPDATE cache SET status = 'pending' WHERE udf_key = 'k'")
    assert cat.claim("k", "human")

    class _ExplodingBackend:
        name = "human"

        def run(self, job, sandbox):
            raise AssertionError("backend must not run for a human-origin candidate")

    monkeypatch.setattr(worker_mod, "smoke_test_reconstruction", lambda *a, **k: None)
    process_row(cat, cat.get("k"), _ExplodingBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "success"
    assert row.origin == "human"


def test_process_row_rejects_forbidden_rewrite(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k", "coco")

    class _ForbiddenBackend:
        name = "coco"

        def run(self, job, sandbox):
            return TranspileResult(kind="catalyst", sql="reflect('java.lang.Runtime', 'getRuntime')")

    process_row(cat, cat.get("k"), _ForbiddenBackend(), spark=None, verify_fn=lambda **k: (True, None))
    row = cat.get("k")
    assert row.status == "failed"
    assert "reflect" in row.error


def test_process_row_swallows_samples_for_failure(tmp_path, monkeypatch):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k", "fake")
    monkeypatch.setattr(cat, "samples_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    assert cat.get("k").status == "success"


class _WeakrefableDummySpark:
    pass


def test_process_row_pins_session_when_given_one(tmp_path, monkeypatch):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k", "fake")
    dummy_spark = _WeakrefableDummySpark()
    monkeypatch.setattr(worker_mod, "smoke_test_reconstruction", lambda *a, **k: None)
    process_row(cat, cat.get("k"), FakeBackend(), spark=dummy_spark, verify_fn=lambda **k: (True, None))
    assert get_session() is dummy_spark


def test_process_row_decline_mark_failed_failure_is_logged(tmp_path, monkeypatch, caplog):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat, func=always_decline)
    assert cat.claim("k", "coco")
    monkeypatch.setattr(cat, "mark_failed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
    with caplog.at_level(logging.ERROR):
        process_row(cat, cat.get("k"), FakeBackend(), spark=None, verify_fn=lambda **k: (True, None))
    assert "failed to mark declined row" in caplog.text


def test_process_row_generic_exception_mark_failed_failure_is_logged(tmp_path, monkeypatch, caplog):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    assert cat.claim("k", "coco")

    class _ExplodingBackend:
        name = "coco"

        def run(self, job, sandbox):
            raise RuntimeError("backend blew up")

    monkeypatch.setattr(
        cat, "mark_failed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone too"))
    )
    with caplog.at_level(logging.ERROR):
        process_row(cat, cat.get("k"), _ExplodingBackend(), spark=None, verify_fn=lambda **k: (True, None))
    assert "failed to mark error row" in caplog.text


def test_poll_once_claim_race_returns_false(tmp_path, monkeypatch):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    monkeypatch.setattr(cat, "claim", lambda *a, **k: False)
    assert poll_once(cat, FakeBackend(), spark=None, verify_fn=lambda **k: (True, None)) is False


def test_maybe_start_inline_noop_when_already_running(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    try:
        maybe_start_inline(None, cat)
        assert inline_thread_alive()
        first = worker_mod._thread
        maybe_start_inline(None, cat)
        assert worker_mod._thread is first
    finally:
        stop_inline()


def test_worker_main_requires_sqlite_path():
    with pytest.raises(SystemExit):
        main(["--no-spark"])


def test_worker_main_once_no_spark(tmp_path):
    path = str(tmp_path / "c.sqlite")
    rc = main(["--sqlite-path", path, "--backend", "fake", "--once", "--no-spark"])
    assert rc == 0


# -- javac.py --------------------------------------------------------------


def test_compile_java_requires_spark():
    with pytest.raises(JavaCompileError, match="SparkSession"):
        compile_java(None, "class Foo {}")


def test_compile_java_requires_class_declaration():
    with pytest.raises(JavaCompileError, match="no class declaration"):
        compile_java(object(), "not valid java at all")


def test_compile_java_falls_back_to_janino_when_jdk_unavailable(monkeypatch):
    import ai_udf_transpile.javac as javac_mod

    monkeypatch.setattr(
        javac_mod, "_compile_jdk", lambda *a: (_ for _ in ()).throw(JavaCompileError("no javac"))
    )
    monkeypatch.setattr(javac_mod, "_compile_janino", lambda *a: b"jarbytes")
    compiled = compile_java(object(), "public class Foo {}")
    assert compiled.janino is True
    assert compiled.jar_bytes == b"jarbytes"
    assert compiled.class_name == "Foo"


def test_stage_jar_rewrites_when_content_changes():
    path1 = _stage_jar("ai_udf_transpile_test_stage_jar", b"one")
    assert open(path1, "rb").read() == b"one"
    path2 = _stage_jar("ai_udf_transpile_test_stage_jar", b"two")
    assert path1 == path2
    assert open(path2, "rb").read() == b"two"


@pytest.mark.spark
def test_compile_and_register_java_udf_real_jdk(spark):
    from ai_udf_transpile.javac import register_java_udf, verify_function_name

    java_source = (
        "package ai_udf_test;\n"
        "import org.apache.spark.sql.api.java.UDF2;\n"
        "public class Adder implements UDF2<Integer, Integer, Integer> {\n"
        "    @Override\n"
        "    public Integer call(Integer a, Integer b) { return a + b; }\n"
        "}\n"
    )
    try:
        compiled = compile_java(spark, java_source)
    except JavaCompileError as exc:
        pytest.skip(f"no javac/janino available: {exc}")
    assert compiled.class_name == "ai_udf_test.Adder"
    from pyspark.sql.types import IntegerType

    fname = verify_function_name("ai_udf_test_adder", compiled.jar_bytes)
    register_java_udf(
        spark, fname, compiled.class_name, compiled.jar_bytes, IntegerType(), janino=compiled.janino
    )
    df = spark.createDataFrame([(1, 2)], ["a", "b"])
    result = df.selectExpr(f"{fname}(CAST(a AS INT), CAST(b AS INT))").collect()[0][0]
    assert result == 3


# -- ai_udf_transpile/__init__.py ------------------------------------------


def test_function_ast_rejects_lambda():
    with pytest.raises(ValueError, match="def statement"):
        _function_ast(lambda x: x)


def test_public_params_strips_self_for_bound_method():
    class _C:
        def f(self, x, y):
            return x + y

    c = _C()
    fa = _function_ast(c.f)
    assert _public_params(c.f, fa) == ["x", "y"]


def test_default_sqlite_path_no_spark():
    assert _default_sqlite_path(None) == "/tmp/ai_udf_transpile.sqlite"


def test_default_sqlite_path_splits_comma_list():
    spark = types.SimpleNamespace(conf=types.SimpleNamespace(get=lambda k, d: "/var/tmp,/other"))
    assert _default_sqlite_path(spark) == "/var/tmp/ai_udf_transpile.sqlite"


def test_default_sqlite_path_swallows_conf_exception():
    spark = types.SimpleNamespace(
        conf=types.SimpleNamespace(get=lambda k, d: (_ for _ in ()).throw(RuntimeError("no")))
    )
    assert _default_sqlite_path(spark) == "/tmp/ai_udf_transpile.sqlite"


def test_ansi_on_swallows_exception():
    spark = types.SimpleNamespace(
        conf=types.SimpleNamespace(get=lambda *a: (_ for _ in ()).throw(RuntimeError()))
    )
    assert _ansi_on(spark) is False


def test_shutdown_tolerates_catalog_close_failure():
    class _BadCatalog:
        def close(self):
            raise RuntimeError("nope")

    set_catalog(_BadCatalog())
    shutdown()


def test_register_impl_requires_enable():
    def f(x: int) -> int:
        return x + 1

    with pytest.raises(RuntimeError, match="enable"):
        register_impl(None, f, return_type="bigint")


def test_register_impl_requires_return_type():
    set_catalog(object())

    def f(x: int):
        return x

    with pytest.raises(ValueError, match="return_type"):
        register_impl(None, f, return_type=None)


_UNSUPPORTED_CAPTURE = [1, 2, 3]


def test_register_impl_rejects_unsupported_capture():
    set_catalog(object())

    def f(x: int) -> int:
        return x + len(_UNSUPPORTED_CAPTURE)

    with pytest.raises(ValueError, match="captures"):
        register_impl(None, f, return_type="bigint", catalyst_sql="_udf_param_0 + 3")


def test_register_impl_rejects_unknown_kind():
    set_catalog(object())

    def f(x: int) -> int:
        return x + 1

    with pytest.raises(ValueError, match="catalyst or java_udf"):
        register_impl(None, f, return_type="bigint", kind="rust", catalyst_sql="_udf_param_0 + 1")


def test_register_impl_requires_reconstructable_payload():
    set_catalog(object())

    def f(x: int) -> int:
        return x + 1

    with pytest.raises(ValueError, match="needs catalyst_sql"):
        register_impl(None, f, return_type="bigint")


def test_register_impl_swallows_samples_for_failure(tmp_path, monkeypatch):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    monkeypatch.setattr(cat, "samples_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    set_catalog(cat)

    def f(x: int) -> int:
        return x + 1

    with pytest.raises(ValueError, match="Hypothesis failed"):
        register_impl(None, f, return_type="bigint", catalyst_sql="_udf_param_0 + 1")


@pytest.mark.spark
def test_enable_stamps_table_and_env_max_examples(spark, sqlite_path, monkeypatch):
    monkeypatch.setenv("AI_UDF_TRANSPILE_MAX_EXAMPLES", "7")
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False, table="some_delta_table")
    assert conf.get_value(conf.MAX_EXAMPLES, spark) == "7"
    assert conf.get_value(conf.TABLE, spark) == "some_delta_table"


@pytest.mark.spark
def test_enable_tolerates_conf_set_and_get_failures(spark, sqlite_path, monkeypatch):
    real_set, real_get = spark.conf.set, spark.conf.get

    def flaky_set(key, value):
        if "transpilePyUDFs" in key or "pyTranspilers" in key:
            raise RuntimeError("nope")
        return real_set(key, value)

    def flaky_get(key, default=None):
        if "pyTranspilers" in key:
            raise RuntimeError("nope")
        return real_get(key, default)

    monkeypatch.setattr(spark.conf, "set", flaky_set)
    monkeypatch.setattr(spark.conf, "get", flaky_get)
    enable(spark, sqlite_path=sqlite_path, backend="fake", inline_worker=False)


# -- catalog/__init__.py ---------------------------------------------------


def test_parse_iso_and_json_helpers():
    assert parse_iso(None) is None
    assert parse_iso("") is None
    assert loads_list("") == []
    assert loads_dict("") == {}


def test_cache_row_reconstructable_false_for_unknown_kind():
    row = CacheRow(udf_key="k", status="success", target_kind="rust")
    assert row.reconstructable() is False


def test_configured_tolerance_rejects_negative_and_nan():
    conf.set_value(conf.TOLERANCE, "-1.0")
    assert configured_tolerance() == 0.0
    conf.set_value(conf.TOLERANCE, "nan")
    assert configured_tolerance() == 0.0


def test_open_catalog_requires_sqlite_path():
    with pytest.raises(ValueError, match="sqlitePath"):
        open_catalog(None)


def test_open_catalog_rejects_unknown_kind():
    conf.set_value(conf.CATALOG, "nonsense")
    with pytest.raises(ValueError, match="unknown catalog"):
        open_catalog(None, sqlite_path="/tmp/x.sqlite")


def test_open_catalog_delta_requires_spark():
    conf.set_value(conf.CATALOG, "delta")
    with pytest.raises(ValueError, match="delta catalog requires a SparkSession"):
        open_catalog(None)


# -- catalog/sqlite.py -------------------------------------------------------


def test_lookup_falls_through_on_unknown_status(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    cat._conn.execute("UPDATE cache SET status = 'weird' WHERE udf_key = 'k'")
    from ai_udf_transpile.catalog import WAIT

    kind, row = cat.lookup("k")
    assert kind == WAIT
    assert row is not None and row.status == "weird"


class _FailingOnClaimConnection(sqlite3.Connection):
    def execute(self, sql, *args, **kwargs):
        if "SET status = 'running'" in sql:
            raise sqlite3.OperationalError("boom")
        return super().execute(sql, *args, **kwargs)


def test_claim_rolls_back_on_sql_error(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    cat._conn.close()
    cat._conn = sqlite3.connect(
        cat.path, factory=_FailingOnClaimConnection, check_same_thread=False, isolation_level=None
    )
    cat._conn.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.OperationalError):
        cat.claim("k", "fake")


def test_upsert_success_requires_known_types(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    with pytest.raises(ValueError, match="known types"):
        cat.upsert_success(
            udf_key="k",
            source_text="",
            param_names=[],
            input_types=[],
            input_categories=[],
            return_type="",
            spark_version="t",
            closure_fingerprint="",
            captures={},
            result=TranspileResult(kind="catalyst", sql="1"),
            origin="human",
            hypothesis_passed=True,
        )


def test_mark_written_back_empty_list_is_noop(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    cat.mark_written_back([])
    assert cat.get("k").written_back_at is None


def test_record_samples_round_trips_through_samples_for(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(cat)
    cat.record_samples("k", [[1], [2]])
    assert sorted(cat.samples_for("k")) == [[1], [2]]


# -- catalog/writeback.py ----------------------------------------------------


def test_writeback_catalog_rejects_bad_format(tmp_path):
    cat = SqliteCatalog(tmp_path / "c.sqlite")
    with pytest.raises(ValueError, match="writeback format"):
        WritebackCatalog(cat, None, "tbl", "csv", 1)


def test_writeback_catalog_delegates_when_table_unavailable(tmp_path):
    staging = SqliteCatalog(tmp_path / "c.sqlite")
    _pending(staging)
    bad_spark = types.SimpleNamespace(sql=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no table")))
    wb = WritebackCatalog(staging, bad_spark, "tbl", "parquet", 1)
    assert wb.get("missing") is None  # local miss + _remote_success short-circuits on _available=False
    assert wb.reclaim_stale(60, 3) == staging.reclaim_stale(60, 3)
    assert wb.count() == staging.count()
    wb.record_samples("k", [[9]])
    assert wb.samples_for("k") == [[9]]
    assert wb.path == staging.path
    assert wb.oldest_pending().udf_key == "k"


def test_writeback_catalog_remote_lookup_failure_is_swallowed(tmp_path):
    staging = SqliteCatalog(tmp_path / "c.sqlite")

    def flaky_sql(query):
        if query.startswith("CREATE") or query.startswith("ALTER"):
            return types.SimpleNamespace()
        raise RuntimeError("remote unreachable")

    spark = types.SimpleNamespace(sql=flaky_sql)
    wb = WritebackCatalog(staging, spark, "tbl", "parquet", 1)
    assert wb.get("missing") is None


# -- verify.py ---------------------------------------------------------------


def test_quote_closes_counts_backslash_parity():
    assert _quote_closes("a\\\\'", 3) is True
    assert _quote_closes("a\\'", 2) is False


def test_collect_imported_module_names():
    assert collect_imported_module_names("def f(:\n  bad") == set()
    names = collect_imported_module_names("import sys\nfrom os.path import join\ndef f(x):\n    return x")
    assert names == {"sys", "os"}


def test_copy_defining_module_imports_handles_source_free_callable():
    ns: dict = {}
    _copy_defining_module_imports(len, ns)
    assert ns == {}


def test_hydrate_captures_decodes_bytes_and_passes_through_plain_values():
    out = _hydrate_captures({"b": {"__bytes__": "68656c6c6f"}, "x": 5})
    assert out == {"b": b"hello", "x": 5}


def test_strategy_for_decimal_and_binary_and_rejects_unknown():
    from hypothesis.strategies import SearchStrategy

    assert isinstance(_strategy_for("decimal(10,2)"), SearchStrategy)
    assert isinstance(_strategy_for("binary"), SearchStrategy)
    with pytest.raises(VerifyFailed, match="unsupported input type"):
        _strategy_for("not_a_real_type")


def test_spark_type_timestamp_ntz():
    from pyspark.sql.types import TimestampNTZType

    assert _spark_type("timestamp_ntz") == TimestampNTZType()


def test_within_tolerance_overflow_is_not_within():
    assert _within_tolerance(10**400, 1, 1.0) is False


def test_hypothesis_check_syntax_error_in_source():
    ok, err = hypothesis_check(
        source_text="def f(x) return x",
        captures={},
        result=TranspileResult(kind="catalyst", sql="_udf_param_0"),
        input_types=["bigint"],
        return_type="bigint",
        spark=None,
    )
    assert ok is False
    assert "SyntaxError" in err


def test_hypothesis_check_java_udf_without_binary_or_source():
    result = TranspileResult(kind="java_udf", class_name="Foo")
    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x",
        captures={},
        result=result,
        input_types=["bigint"],
        return_type="bigint",
        spark=object(),
    )
    assert ok is False
    assert "no class/binary" in err


def test_hypothesis_check_java_udf_compile_failure_reported():
    result = TranspileResult(kind="java_udf", java_source="class Foo {}", class_name="Foo")
    ok, err = hypothesis_check(
        source_text="def f(x: int) -> int:\n    return x",
        captures={},
        result=result,
        input_types=["bigint"],
        return_type="bigint",
        spark=object(),
    )
    assert ok is False
    assert "java_udf verify setup failed" in err


def test_smoke_test_reconstruction_confs_failure_reported():
    spark = types.SimpleNamespace(
        conf=types.SimpleNamespace(
            set=lambda *a: (_ for _ in ()).throw(RuntimeError("no")), get=lambda *a, **k: ""
        )
    )
    err = _smoke_test_reconstruction(
        spark,
        source_text="def f(x):\n    return x",
        captures={},
        input_types=["bigint"],
        return_type="bigint",
    )
    assert err == "smoke test could not arm the transpile confs"
