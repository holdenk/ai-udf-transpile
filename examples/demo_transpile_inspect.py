# SPDX-License-Identifier: Apache-2.0
"""Inspect what the transpiler produced, how many attempts it took, and
whether different "agents" would have produced different (wrong) SQL.

Three sections:

1. TRANSPILATION RESULTS -- for each UDF: the transpiled SQL/Java,
   attempt_count, origin, backend, and Hypothesis verdict, pulled from
   the catalog. Shows what the code actually became, not just a boolean.
2. ATTEMPT COUNTS -- how many worker claims each transpile took (a
   backend that fails Hypothesis cools down and retries, bumping the
   count).
3. MULTI-AGENT COMPARISON -- for adversarial UDFs, several candidate
   rewrites (as if from different agents) are run through Hypothesis
   directly. Some pass, some fail -- the point: different agents produce
   different SQL, and verification is what decides which one ships.

    export SPARK_HOME=/path/to/spark-master
    export PYTHONPATH="$SPARK_HOME/python:$(echo $SPARK_HOME/python/lib/py4j-*-src.zip | awk '{print $1}')"
    export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"
    export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
    python examples/demo_transpile_inspect.py
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import BooleanType, LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, shutdown  # noqa: E402
from ai_udf_transpile.backends import ai_backend  # noqa: E402
from ai_udf_transpile.backends.fake import (  # noqa: E402
    backwards_java,
    contains_ingredient,
    greet,
    is_none_branch,
    mod_bucket,
    plus_one,
    round_half,
    upper_useragent,
    weekday_of,
    zfill3,
)
from ai_udf_transpile.catalog import configured_tolerance  # noqa: E402
from ai_udf_transpile.keys import (  # noqa: E402
    canonical_source,
    canonical_source_from_func,
    extract_captures,
)
from ai_udf_transpile.targets import KIND_CATALYST, TranspileResult  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402
from ai_udf_transpile.types import input_spark_types, return_spark_type  # noqa: E402
from ai_udf_transpile.verify import hypothesis_check  # noqa: E402


def gave_up(func) -> bool:
    """The backend already failed func; failed rows cool down instead of re-queueing."""
    catalog = get_catalog()
    with catalog._lock:
        row = catalog._conn.execute(
            "SELECT status FROM cache WHERE source_text = ?", (canonical_source_from_func(func),)
        ).fetchone()
    return row is not None and row[0] == "failed"


def wait_transpiled(func, rtype, timeout=900):
    """Re-create the UDF until the cache hit transpiles it."""
    deadline = time.time() + timeout
    udf = UserDefinedFunction(func, rtype)
    while time.time() < deadline and not udf.transpiled and not gave_up(func):
        time.sleep(1.0)
        udf = UserDefinedFunction(func, rtype)
    return udf


def verify_candidate(spark, func, return_type, candidate_sql):
    """Run Hypothesis on a candidate SQL rewrite without touching the catalog."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    function_ast = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    params = [arg.arg for arg in function_ast.args.args]
    source = canonical_source(function_ast)
    captures = extract_captures(func, function_ast, params) or {}
    in_types = input_spark_types(function_ast, params) or []
    out_type = return_spark_type(return_type)
    result = TranspileResult(kind=KIND_CATALYST, sql=candidate_sql)
    ok, err = hypothesis_check(
        source_text=source,
        captures=captures,
        result=result,
        input_types=in_types,
        return_type=out_type,
        spark=spark,
        max_examples=conf.get_int(conf.MAX_EXAMPLES, spark, int(conf.default_max_examples())),
        func=func,
        samples=[],
        tolerance=configured_tolerance(spark),
        party_on_strings=conf.get_bool(conf.PARTY_ON_STRINGS, spark, False),
    )
    return ok, err


def _faithful_sql(func):
    """This run's verified catalyst SQL for func, or None."""
    catalog = get_catalog()
    with catalog._lock:  # the inline worker writes through the same connection
        row = catalog._conn.execute(
            "SELECT catalyst_sql FROM cache WHERE source_text = ? AND status = 'success'",
            (canonical_source_from_func(func),),
        ).fetchone()
    return row[0] if row else None


def _truncate(text, width=80):
    if text is None:
        return "(none)"
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _func_name(source_text):
    if not source_text:
        return "?"
    return source_text.split("(")[0].replace("def ", "").strip()


def main() -> int:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ai-udf-transpile-inspect")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    sqlite = os.environ.get("AI_UDF_TRANSPILE_SQLITE") or os.path.join(
        tempfile.mkdtemp(prefix="ai-udf-inspect-"), "cache.sqlite"
    )
    enable(spark, sqlite_path=sqlite, backend=ai_backend(spark), inline_worker=True)
    catalog = get_catalog()
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")
    # weekday_of takes a datetime (timestamp category); opt in.
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,timestamp", spark)
    # These fixtures' strings are text (greet, upper, zfill), not dates-in-strings.
    conf.set_value(conf.PARTY_ON_STRINGS, "true", spark)

    print("=== TRANSPILATION RESULTS (what each UDF became) ===")
    udfs = [
        (plus_one, LongType()),
        (is_none_branch, LongType()),
        (greet, StringType()),
        (upper_useragent, StringType()),
        (contains_ingredient, BooleanType()),
        (mod_bucket, LongType()),
        (round_half, LongType()),
        (weekday_of, LongType()),
        (zfill3, StringType()),
        (backwards_java, StringType()),
    ]
    for func, rtype in udfs:
        wait_transpiled(func, rtype)

    rows = catalog._conn.execute(
        "SELECT source_text, target_kind, catalyst_sql, impl_class, "
        "attempt_count, origin, backend, hypothesis_passed, status "
        "FROM cache ORDER BY created_at"
    ).fetchall()
    for source_text, kind, sql, impl_class, attempts, origin, backend, hyp, status in rows:
        name = _func_name(source_text)
        if kind == "java_udf":
            target = f"Java UDF -> {impl_class}"
        elif sql:
            target = f"SQL -> {sql}"
        else:
            target = "(no payload)"
        print(f"  {name:22}")
        print(f"    {_truncate(target, 90)}")
        print(f"    attempts={attempts} origin={origin} backend={backend} hyp={hyp} status={status}")

    print()
    print("=== ATTEMPT COUNTS (worker claims per transpile) ===")
    for source_text, _kind, _sql, _cls, attempts, _o, _b, _h, status in rows:
        print(f"  {_func_name(source_text):22} attempts={attempts} status={status}")
    print("(a backend that fails Hypothesis cools down 24h, then retries --\n bumping attempt_count.)")

    print()
    print("=== MULTI-AGENT COMPARISON (different agents, different SQL) ===")
    print("Each UDF below gets 2-3 candidate rewrites, as if from different")
    print("agents. Hypothesis decides which ship and which get rejected.")
    print()

    candidates = [
        (
            "mod_bucket (value % modulus; python floors, Spark truncates)",
            mod_bucket,
            LongType(),
            [
                ("Agent A (naive %)", "_udf_param_0 % _udf_param_1"),
                ("Agent B (pmod)", "pmod(_udf_param_0, _udf_param_1)"),
                ("Agent C (this run)", _faithful_sql(mod_bucket)),
            ],
        ),
        (
            "round_half (banker's rounding; Spark round is half-up)",
            round_half,
            LongType(),
            [
                ("Agent A (round)", "round(_udf_param_0)"),
                ("Agent B (int cast)", "cast(round(_udf_param_0) as int)"),
                ("Agent C (this run)", _faithful_sql(round_half)),
            ],
        ),
        (
            "weekday_of (Monday=0; Spark dayofweek Sunday=1)",
            weekday_of,
            LongType(),
            [
                ("Agent A (dayofweek)", "dayofweek(_udf_param_0)"),
                ("Agent B (offset -1)", "dayofweek(_udf_param_0) - 1"),
                ("Agent C (this run)", _faithful_sql(weekday_of)),
            ],
        ),
        (
            "contains_ingredient (None -> False, never NULL)",
            contains_ingredient,
            BooleanType(),
            [
                ("Agent A (naive instr)", "instr(lower(_udf_param_0), _udf_param_1) > 0"),
                ("Agent B (this run)", _faithful_sql(contains_ingredient)),
            ],
        ),
    ]
    for label, func, rtype, rewrites in candidates:
        print(f"  {label}")
        for agent, sql in rewrites:
            if sql is None:  # this run's backend did not land one
                print(f"    {agent + ':':22} (none)")
                continue
            ok, err = verify_candidate(spark, func, rtype, sql)
            verdict = "ACCEPTED" if ok else f"REJECTED ({_truncate(err, 70)})"
            print(f"    {agent + ':':22} {_truncate(sql, 56):58} {verdict}")
        print()

    print("=== BYTECODE vs AI BACKEND (deterministic vs this run) ===")
    print("The bytecode backend walks CPython bytecode -- no AI, no network,")
    print("no attempts. It declines anything it cannot prove (CALL, IMPORT, ...).")
    print()
    from ai_udf_transpile.backends.bytecode import BytecodeBackend as _BC
    from ai_udf_transpile.keys import canonical_source_from_func as _csf
    from ai_udf_transpile.sandbox import make_sandbox as _ms
    from ai_udf_transpile.targets import TranspileJob as _TJ

    class _NoopSandbox:
        def __truediv__(self, o):
            return self

        def write_text(self, *a, **k):
            pass

    bc = _BC()
    import inspect as _insp

    for func, rtype in udfs:
        src = _csf(func)
        params = list(func.__code__.co_varnames[: func.__code__.co_argcount])
        sig = _insp.signature(func)
        in_types = ["string" if p.annotation is str else "bigint" for p in sig.parameters.values()]
        rtype_simple = return_spark_type(rtype) or "bigint"
        job = _TJ(
            udf_key="k",
            source_text=src,
            param_names=params,
            input_types=in_types,
            input_categories=[],
            return_type=rtype_simple,
            captures={},
        )
        ai_sql = _faithful_sql(func)
        try:
            with _ms(job) as sandbox:
                bc_result = bc.run(job, sandbox)
            bc_sql = bc_result.sql
            match = "same" if bc_sql == ai_sql else "equiv"
            print(
                f"  {func.__name__:22} bytecode: {_truncate(bc_sql, 50):52} "
                f"AI: {_truncate(ai_sql, 50)}  [{match}]"
            )
        except Exception as exc:
            reason = str(exc).replace("unsupported bytecode: ", "").replace("declined: ", "")
            print(
                f"  {func.__name__:22} bytecode: DECLINED ({_truncate(reason, 40):42}) "
                f"AI: {_truncate(ai_sql, 50)}"
            )
    print()

    shutdown()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
