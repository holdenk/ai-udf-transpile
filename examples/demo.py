# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: working rewrites and deliberate declines.

Run with a packaged Spark master:

    export SPARK_HOME=/path/to/spark-master
    export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip"
    export PYSPARK_PYTHON="$(pwd)/.venv/bin/python"   # workers must match the driver
    export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"
    python examples/demo.py
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import ArrayType, BooleanType, LongType, StringType  # noqa: E402
from pyspark.sql.udf import UserDefinedFunction  # noqa: E402

from ai_udf_transpile import conf, enable, register_impl, shutdown  # noqa: E402
from ai_udf_transpile.transpiler import get_catalog  # noqa: E402


def plus_one(x: int) -> int:
    return x + 1


def greet(name: str) -> str:
    return "hi " + name


def upper_useragent(useragent: str) -> str:
    # SPARK-21935: this exact UDF OOM'd executors via Python worker memory
    # overhead; as Catalyst SQL there is no Python worker at all.
    return useragent.upper()


def contains_ingredient(recipe: str, ingredient: str) -> bool:
    # The boolean NULL trap: python returns False for a NULL recipe, so the
    # rewrite must coalesce -- instr(...) > 0 alone returns NULL, which only
    # "looks gucci" inside WHERE clauses.
    if recipe is not None:
        return ingredient in recipe.lower()
    return False


def timestamp_to_epoch(t: datetime) -> str:
    # Scalar modernization of the pandas_udf t.dt.strftime("%s").apply(str)
    # over NYC taxi tpep_pickup_datetime: on NaT the pandas version yields
    # NaN and .apply(str) makes it 'nan' -- a faithful rewrite must too.
    # Also: python drops microseconds *before* epoch conversion, so the SQL
    # needs date_trunc('SECOND', ...) or pre-1970 fractional times are 1s off.
    if t is None:
        return "nan"
    return t.strftime("%s")


def scatter_to_seconds(start: str, duration: str) -> list[str]:
    # Every failure path (null/short/unparseable start, non-numeric/NaN/inf/
    # negative duration) returns [] via the bare except, and the loop is
    # range(duration + 1) -- off-by-one bait. First array<string> return: the
    # SQL keeps _udf_param_N refs outside the transform lambda (placeholder
    # substitution does not descend into higher-order function lambdas).
    import datetime

    ret = []
    try:
        start = str(start)
        duration = int(float(duration))
        if len(start) < 19:
            return ret
        start = start[:19]
        start_struct = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        for i in range(duration + 1):
            cur = (start_struct + datetime.timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
            ret.append(cur)
        return ret
    except Exception:
        return ret


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def widget_name(payload: str) -> str:
    import json

    if payload is None:
        return None
    return json.loads(payload).get("widget")


def both_positive(x: int, y: int) -> bool:
    return x > 0 and y > 0


def is_none_branch(x: int) -> int:
    if x is None:
        return -1
    return x


def pymap_to_json(inp: dict[str, str]) -> str:
    import ast
    import json

    if inp is None:
        return None
    new = {}
    for k, v in inp.items():
        try:
            new[k] = json.loads(v)
        except Exception:
            try:
                new[k] = ast.literal_eval(v)
            except Exception:
                new[k] = v
    return json.dumps(new)


def untyped(x):
    return x + 1


def uses_helper(x: int) -> int:
    return plus_one(x)


def always_decline(x: int) -> int:
    import os

    return len(os.getcwd())


def wait_for_status(catalog, key, want, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = catalog.get(key)
        if row is not None and row.status == want:
            return row
        time.sleep(0.5)
    return catalog.get(key)


def main() -> int:
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ai-udf-transpile-demo")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    sqlite = os.path.join(tempfile.mkdtemp(prefix="ai-udf-demo-"), "cache.sqlite")
    enable(spark, sqlite_path=sqlite, backend="fake", inline_worker=True)
    catalog = get_catalog()
    spark.conf.set("spark.sql.experimental.optimizer.pyTranspilers", "ai")

    print("=== WORKING CASES (fake backend fixtures, Hypothesis-verified) ===")
    df = spark.createDataFrame([(1,), (41,), (None,)], ["x"])

    for func, rtype in [
        (plus_one, LongType()),
        (is_none_branch, LongType()),
    ]:
        first = UserDefinedFunction(func, rtype)
        print(f"{func.__name__}: first call transpiled={bool(first.transpiled)} (cache miss -> Python)")
        # wait for the inline worker
        deadline = time.time() + 120
        while time.time() < deadline:
            second = UserDefinedFunction(func, rtype)
            if second.transpiled:
                break
            time.sleep(1.0)
        print(f"{func.__name__}: second call transpiled={bool(second.transpiled)} (cache hit)")
        rows = df.filter("x is not null").select(second("x").alias("r")).collect()
        print(f"{func.__name__}: results={[r[0] for r in rows]}")

    g = UserDefinedFunction(greet, StringType())
    deadline = time.time() + 120
    while time.time() < deadline:
        g = UserDefinedFunction(greet, StringType())
        if g.transpiled:
            break
        time.sleep(1.0)
    names = spark.createDataFrame([("bo",), ("holden",)], ["name"])
    greet_rows = [r[0] for r in names.select(g("name")).collect()]
    print(f"greet: transpiled={bool(g.transpiled)} results={greet_rows}")

    ua = UserDefinedFunction(upper_useragent, StringType())
    deadline = time.time() + 120
    while time.time() < deadline:
        ua = UserDefinedFunction(upper_useragent, StringType())
        if ua.transpiled:
            break
        time.sleep(1.0)
    uas = spark.createDataFrame([("Mozilla/5.0 (Windows NT 10.0; Win64; x64)",)], ["ua"])
    print(
        f"upper_useragent (SPARK-21935): transpiled={bool(ua.transpiled)} "
        f"results={[r[0] for r in uas.select(ua('ua')).collect()]}"
    )

    ci = UserDefinedFunction(contains_ingredient, BooleanType())
    deadline = time.time() + 120
    while time.time() < deadline:
        ci = UserDefinedFunction(contains_ingredient, BooleanType())
        if ci.transpiled:
            break
        time.sleep(1.0)
    recipes = spark.createDataFrame(
        [("2 cups flour, 1 tsp salt", "salt"), ("100% whole wheat", "%"), (None, "salt")],
        ["recipe", "ingredient"],
    )
    print(
        f"contains_ingredient: transpiled={bool(ci.transpiled)} "
        f"results={[r[0] for r in recipes.select(ci('recipe', 'ingredient')).collect()]} "
        "(NULL recipe -> False, never NULL)"
    )

    # Timestamp inputs are gated out by default; opt in for this one.
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,map,timestamp", spark)
    te = UserDefinedFunction(timestamp_to_epoch, StringType())
    deadline = time.time() + 120
    while time.time() < deadline:
        te = UserDefinedFunction(timestamp_to_epoch, StringType())
        if te.transpiled:
            break
        time.sleep(1.0)
    pickups = spark.createDataFrame(
        [
            (datetime.datetime(2015, 1, 1, 0, 12, 0),),
            (datetime.datetime(1969, 12, 31, 23, 59, 59),),
            (None,),
        ],
        ["tpep_pickup_datetime"],
    )
    stamped = pickups.select(te("tpep_pickup_datetime").alias("timestamp_copy"))
    print(
        f"timestamp_to_epoch: transpiled={bool(te.transpiled)} "
        f"results={[r[0] for r in stamped.collect()]} "
        f"distinct_count={stamped.distinct().count()} (NULL -> 'nan', like the pandas original)"
    )

    sc = UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))
    deadline = time.time() + 120
    while time.time() < deadline:
        sc = UserDefinedFunction(scatter_to_seconds, ArrayType(StringType()))
        if sc.transpiled:
            break
        time.sleep(1.0)
    starts = spark.createDataFrame(
        [
            ("2015-01-01 00:00:00", "2"),
            ("2015-12-31 23:59:59", "2"),  # year rollover
            ("2015-01-01", "5"),  # short start -> []
            (None, "2"),  # null start -> []
        ],
        ["start", "duration"],
    )
    print(
        f"scatter_to_seconds (array<string> return): transpiled={bool(sc.transpiled)} "
        f"results={[r[0] for r in starts.select(sc('start', 'duration')).collect()]} "
        "(failure paths -> [], never NULL)"
    )

    print()
    print("=== REAL-ROW SAMPLING (string inputs feed Hypothesis) ===")
    # First call is a cache miss: it runs as a Python UDF, and the real string
    # arguments are sampled into the catalog so the verifier sees realistic
    # JSON instead of only random strings (which would let a wrong JSON path
    # pass vacuously).
    payloads = spark.createDataFrame(
        [('{"widget": "gizmo", "n": 1}',), ('{"widget": "sprocket", "n": 2}',)],
        ["payload"],
    )
    w = UserDefinedFunction(widget_name, StringType())
    first_rows = [r[0] for r in payloads.select(w("payload")).collect()]
    print(f"widget_name: first call transpiled={bool(w.transpiled)} results={first_rows} (miss -> Python)")
    deadline = time.time() + 120
    while time.time() < deadline:
        w = UserDefinedFunction(widget_name, StringType())
        if w.transpiled:
            break
        time.sleep(1.0)
    sampled = catalog._conn.execute("SELECT args_json FROM samples").fetchall()
    print(f"widget_name: sampled real rows={len(sampled)} e.g. {sampled[0][0] if sampled else None}")
    print(
        f"widget_name: transpiled={bool(w.transpiled)} "
        f"results={[r[0] for r in payloads.select(w('payload')).collect()]} (verified against samples)"
    )

    print()
    print("=== JAVA UDF TARGET (compiled on the driver, Hypothesis-verified) ===")
    bw = UserDefinedFunction(backwards, StringType())
    deadline = time.time() + 120
    while time.time() < deadline:
        bw = UserDefinedFunction(backwards, StringType())
        if bw.transpiled:
            break
        time.sleep(1.0)
    bw_rows = [r[0] for r in names.select(bw("name")).collect()]
    print(f"backwards: transpiled={bool(bw.transpiled)} results={bw_rows} (Java UDF from cache)")

    b = UserDefinedFunction(both_positive, BooleanType())
    deadline = time.time() + 120
    while time.time() < deadline:
        b = UserDefinedFunction(both_positive, BooleanType())
        if b.transpiled:
            break
        time.sleep(1.0)
    pairs = spark.createDataFrame([(1, 2), (-1, 2)], ["a", "b"])
    print(
        f"both_positive: transpiled={bool(b.transpiled)} "
        f"results={[r[0] for r in pairs.select(b('a', 'b')).collect()]}"
    )

    print()
    print("=== HUMAN-PROVIDED IMPL (register_impl) ===")

    def times_ten(x: int) -> int:
        return x * 10

    register_impl(
        spark,
        times_ten,
        kind="catalyst",
        catalyst_sql="_udf_param_0 * 10",
        return_type=LongType(),
    )
    t = UserDefinedFunction(times_ten, LongType())
    print(f"times_ten: transpiled={bool(t.transpiled)} (human impl, Hypothesis-verified)")
    print(f"times_ten: results={[r[0] for r in df.filter('x is not null').select(t('x')).collect()]}")

    print()
    print("=== NOT WORKING / BY-DESIGN DECLINES ===")

    u = UserDefinedFunction(untyped, LongType())
    print(
        f"untyped UDF: transpiled={bool(u.transpiled)}, catalog rows={catalog.count()} "
        "(no type annotations -> never queued)"
    )

    h = UserDefinedFunction(uses_helper, LongType())
    print(
        f"closure over a Python function: transpiled={bool(h.transpiled)} (cannot key captures -> declined)"
    )

    UserDefinedFunction(always_decline, LongType())
    cur = catalog._conn.execute("SELECT udf_key FROM cache WHERE source_text LIKE '%always_decline%'")
    found = cur.fetchone()
    if found:
        row = wait_for_status(catalog, found[0], "failed", timeout=60)
        print(
            f"always_decline: status={row.status} origin={row.origin} backend={row.backend} "
            f"error={row.error!r} (no fixture -> failed, cooldown applies)"
        )
    before = catalog.count()
    UserDefinedFunction(always_decline, LongType())
    print(f"always_decline second call: catalog rows {before} -> {catalog.count()} (cooldown, no re-queue)")

    print()
    print("=== WRONG REWRITE REJECTED BY HYPOTHESIS ===")

    def plus_two(x: int) -> int:
        return x + 2

    try:
        register_impl(
            spark,
            plus_two,
            kind="catalyst",
            catalyst_sql="_udf_param_0 + 99",  # wrong on purpose
            return_type=LongType(),
        )
        print("BUG: wrong impl was accepted")
    except ValueError as exc:
        print(f"register_impl rejected wrong SQL: {exc}")

    # The classic "looks gucci but is not" rewrite: to_json type-checks and
    # eyeballs fine, but keeps values as strings, uses compact JSON spacing,
    # and Spark maps do not preserve Python's key insertion order.
    try:
        register_impl(
            spark,
            pymap_to_json,
            kind="catalyst",
            catalyst_sql="to_json(_udf_param_0)",
            return_type=StringType(),
        )
        print("BUG: naive to_json was accepted")
    except ValueError as exc:
        print(f"register_impl rejected naive to_json for pymap_to_json (map input): {exc}")

    # The boolean NULL trap: fine inside WHERE (NULL filters like false),
    # wrong as a selected column.
    try:
        register_impl(
            spark,
            contains_ingredient,
            kind="catalyst",
            catalyst_sql="instr(lower(_udf_param_0), _udf_param_1) > 0",
            return_type=BooleanType(),
        )
        print("BUG: naive instr was accepted")
    except ValueError as exc:
        print(f"register_impl rejected naive instr for contains_ingredient: {exc}")

    # Guard-complete but wrong rewrites of scatter_to_seconds, so only the
    # specific trap remains (a guard-less variant would be caught by the
    # built-in examples before the interesting part is even reached).
    scatter_faithful = (
        "CASE WHEN _udf_param_0 IS NULL THEN array() "
        "WHEN length(_udf_param_0) < 19 THEN array() "
        "WHEN try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss') IS NULL "
        "THEN array() "
        "WHEN try_cast(_udf_param_1 AS DOUBLE) IS NULL THEN array() "
        "WHEN isnan(try_cast(_udf_param_1 AS DOUBLE)) THEN array() "
        "WHEN abs(try_cast(_udf_param_1 AS DOUBLE)) = cast('inf' AS DOUBLE) THEN array() "
        "WHEN cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT) < 0 THEN array() "
        "ELSE transform(sequence("
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss'), "
        "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT), "
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')), "
        "interval 1 second), x -> date_format(x, 'yyyy-MM-dd HH:mm:ss')) END"
    )
    # The range(duration + 1) off-by-one: this rewrite stops one second short.
    # Caught by the cross-combined built-ins (a valid timestamp start paired
    # with a parseable duration -- the one-param-at-a-time built-ins never
    # produce that pair), with the real rows recorded earlier as a second
    # layer of defense.
    scatter_off_by_one = scatter_faithful.replace(
        "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT),",
        "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT) - 1,",
    )
    try:
        register_impl(
            spark,
            scatter_to_seconds,
            kind="catalyst",
            catalyst_sql=scatter_off_by_one,
            return_type=ArrayType(StringType()),
        )
        print("BUG: off-by-one scatter_to_seconds was accepted")
    except ValueError as exc:
        print(f"register_impl rejected off-by-one scatter_to_seconds: {exc}")

    # Value-correct but undeliverable: this transform lambda closes over
    # _udf_param_0, and TranspiledPythonUDF placeholder substitution does not
    # descend into lambda bodies -- the query would fail at analysis with
    # UNRESOLVED_COLUMN. Hypothesis (which binds params as plain columns)
    # passes it; the reconstruction smoke test does not.
    scatter_lambda_closing = scatter_faithful.replace(
        "ELSE transform(sequence("
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss'), "
        "timestampadd(SECOND, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT), "
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')), "
        "interval 1 second), x -> date_format(x, 'yyyy-MM-dd HH:mm:ss'))",
        "ELSE transform(sequence(0, cast(int(try_cast(_udf_param_1 AS DOUBLE)) AS INT)), "
        "i -> date_format(timestampadd(SECOND, i, "
        "try_to_timestamp(substr(_udf_param_0, 1, 19), 'yyyy-MM-dd HH:mm:ss')), "
        "'yyyy-MM-dd HH:mm:ss'))",
    )
    try:
        register_impl(
            spark,
            scatter_to_seconds,
            kind="catalyst",
            catalyst_sql=scatter_lambda_closing,
            return_type=ArrayType(StringType()),
        )
        print("BUG: lambda-closing scatter_to_seconds was accepted")
    except ValueError as exc:
        print(f"register_impl rejected lambda-closing scatter_to_seconds: {str(exc)[:140]}...")

    print()
    print("=== INPUT-CATEGORY GATE (int/float only vs int/float/string) ===")

    def shout(name: str) -> str:
        return name.upper()

    conf.set_value(conf.INPUT_CATEGORIES, "numeric,bool,binary", spark)
    before = catalog.count()
    s = UserDefinedFunction(shout, StringType())
    print(
        f"shout (string input, gate=numeric only): transpiled={bool(s.transpiled)}, "
        f"catalog rows {before} -> {catalog.count()} (gated out, never queued)"
    )
    conf.set_value(conf.INPUT_CATEGORIES, conf.DEFAULTS[conf.INPUT_CATEGORIES], spark)
    s = UserDefinedFunction(shout, StringType())
    print(
        f"shout (gate=default): transpiled={bool(s.transpiled)}, "
        f"catalog rows {before} -> {catalog.count()} (queued for the backend)"
    )

    shutdown()
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
