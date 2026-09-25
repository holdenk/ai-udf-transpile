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
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, LongType, StringType  # noqa: E402
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


def contains_ingredient(text: str, needle: str) -> bool:
    # The boolean NULL trap: python returns False for a NULL input, so the
    # rewrite must coalesce -- instr(...) > 0 alone returns NULL, which only
    # "looks gucci" inside WHERE clauses.
    return text is not None and needle in text.lower()


def timestamp_to_epoch(ts: datetime) -> str:
    # Scalar modernization of the pandas_udf t.dt.strftime("%s").apply(str)
    # over NYC taxi tpep_pickup_datetime: on NaT the pandas version yields
    # NaN and .apply(str) makes it 'nan' -- a faithful rewrite must too.
    # Also: python drops microseconds *before* epoch conversion, so the SQL
    # needs date_trunc('SECOND', ...) or pre-1970 fractional times are 1s off.
    return "nan" if ts is None else ts.strftime("%s")


def scatter_to_seconds(start_text: str, dur_text: str) -> list[str]:
    # Every failure path (null/short/unparseable start, non-numeric/NaN/inf/
    # negative duration) returns [] via the bare except, and the loop is
    # range(0, secs + 1) -- off-by-one bait. First array<string> return: the
    # SQL keeps _udf_param_N refs outside the transform lambda (placeholder
    # substitution does not descend into higher-order function lambdas).
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


LAC_BSP = ["14503", "13403", "11518", "1452", "1343", "1518"]
LAC_MID = ["14506", "1462"]


def BSPIn(prev_lac: str, cur_lac: str, bsp_lacs: list[str], mid_lacs: list[str]) -> int:
    # Rewritten from a pasted telecom UDF whose condition was written as
    #   old_lac in lac_lst_bsp & new_lac in lac_lst_mid
    # which raises TypeError on EVERY row: `&` binds tighter than `in` and
    # comparisons chain, so it parses as
    #   old_lac in (lac_lst_bsp & new_lac) in lac_lst_mid   # list & str -> TypeError
    # `and` is the intent. First array<string> *input* params: the lists
    # arrive as F.array(F.lit(...)) columns at the call site.
    return 1 if prev_lac in bsp_lacs and cur_lac in mid_lacs else 0


def flatlist(groups: list[list[str]]) -> list[str]:
    # Rewritten from a contributed UDF (`def flatlist(s): fl = [item for
    # sublist in s for item in sublist]; return fl`, registered via
    # F.udf(flatlist, ArrayType(StringType()))). flatten() is faithful: it
    # preserves order, duplicates, and null elements; a null outer or inner
    # array raises TypeError in python (verify allows any sql result there).
    return [item for sub in groups for item in sub]


def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]


def widget_name(doc: str) -> str:
    import json

    if doc is None:
        return None
    return json.loads(doc).get("widget")


def both_positive(x: int, y: int) -> bool:
    return x > 0 and y > 0


def is_none_branch(x: int) -> int:
    if x is None:
        return -1
    return x


def pymap_to_json(mapping: dict[str, str]) -> str:
    import ast
    import json

    if mapping is None:
        return None
    out = {}
    for key, val in mapping.items():
        try:
            out[key] = json.loads(val)
        except Exception:
            try:
                out[key] = ast.literal_eval(val)
            except Exception:
                out[key] = val
    return json.dumps(out)


def untyped(x):
    return x + 1


def uses_helper(x: int) -> int:
    return plus_one(x)


def kms_encrypt(text: str, key_id: str) -> str:
    # The canonical untranspilable UDF: an AWS KMS Encrypt call per row.
    # There is no Catalyst equivalent -- the ciphertext comes from the
    # service and is non-deterministic (the same plaintext encrypts
    # differently every call), so any backend SQL "rewrite" is a
    # hallucination. In an environment without boto3/credentials the python
    # side raises on every example, which is the vacuous-pass trap: "python
    # raise + sql value allowed" would accept ANY rewrite, so verification
    # requires at least one successful python evaluation and fails closed.
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

    # Array inputs are gated out by default; opt in for the LAC-membership UDF.
    conf.set_value(conf.INPUT_CATEGORIES, "numeric,string,bool,binary,map,timestamp,array", spark)
    lac = spark.createDataFrame(
        [
            ("14503", "14506"),  # in bsp, in mid -> 1
            ("14503", "99999"),  # in bsp, not mid -> 0
            ("99999", "14506"),  # not bsp, in mid -> 0
            (None, "14506"),  # null old_lac -> 0
        ],
        ["OLD_LAC", "NEW_LAC"],
    )
    # Same call-site shape as the original: the lists arrive as array columns.
    lac = lac.withColumn("bsp", F.array(*[F.lit(x) for x in LAC_BSP]))
    lac = lac.withColumn("mid", F.array(*[F.lit(x) for x in LAC_MID]))
    bi = UserDefinedFunction(BSPIn, IntegerType())
    first_rows = [r[0] for r in lac.select(bi("OLD_LAC", "NEW_LAC", "bsp", "mid")).collect()]
    print(
        f"BSPIn: first call transpiled={bool(bi.transpiled)} results={first_rows} "
        "(miss -> Python, rows sampled)"
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        bi = UserDefinedFunction(BSPIn, IntegerType())
        if bi.transpiled:
            break
        time.sleep(1.0)
    print(
        f"BSPIn (array<string> inputs): transpiled={bool(bi.transpiled)} "
        f"results={[r[0] for r in lac.select(bi('OLD_LAC', 'NEW_LAC', 'bsp', 'mid')).collect()]}"
    )

    # First nested array input: array<array<string>> -> array<string>.
    nested = spark.createDataFrame(
        [
            ([["a", "b"], ["c"]],),
            ([["a", None], ["b"]],),  # null element survives
            ([["x", "y"], ["x"]],),  # duplicate survives
        ],
        "s array<array<string>>",
    )
    fl = UserDefinedFunction(flatlist, ArrayType(StringType()))
    first_rows = [r[0] for r in nested.select(fl("s")).collect()]
    print(
        f"flatlist: first call transpiled={bool(fl.transpiled)} results={first_rows} "
        "(miss -> Python, rows sampled)"
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        fl = UserDefinedFunction(flatlist, ArrayType(StringType()))
        if fl.transpiled:
            break
        time.sleep(1.0)
    print(
        f"flatlist (array<array<string>> input): transpiled={bool(fl.transpiled)} "
        f"results={[r[0] for r in nested.select(fl('s')).collect()]}"
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

    # boto3 KMS encryption: untranspilable by nature (service call per row,
    # non-deterministic ciphertext). The type gate queues it (it is fully
    # annotated), the fake backend declines, and even a hallucinated rewrite
    # fails closed: python raises on every example in an environment without
    # boto3/credentials, and verification requires at least one successful
    # python evaluation -- otherwise "python raise + sql value allowed"
    # would accept ANY sql vacuously.
    UserDefinedFunction(kms_encrypt, StringType())
    cur = catalog._conn.execute("SELECT udf_key FROM cache WHERE source_text LIKE '%kms_encrypt%'")
    found = cur.fetchone()
    if found:
        row = wait_for_status(catalog, found[0], "failed", timeout=60)
        print(
            f"kms_encrypt (boto3 KMS per row): status={row.status} origin={row.origin} "
            f"backend={row.backend} error={row.error!r} (no fixture -> failed, cooldown applies)"
        )
    try:
        register_impl(
            spark,
            kms_encrypt,
            kind="catalyst",
            catalyst_sql="base64(_udf_param_0)",  # hallucinated "encryption"
            return_type=StringType(),
        )
        print("BUG: hallucinated KMS rewrite was accepted")
    except ValueError as exc:
        print(f"register_impl rejected hallucinated KMS rewrite (fail-closed): {str(exc)[:110]}...")

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

    # BSPIn traps, in increasing subtlety. (a) Bare boolean without the CASE:
    # array_contains(arr, NULL) is NULL, but the python returns 0 on null
    # input -- caught by the built-in null examples.
    try:
        register_impl(
            spark,
            BSPIn,
            kind="catalyst",
            catalyst_sql="array_contains(_udf_param_2, _udf_param_0) "
            "AND array_contains(_udf_param_3, _udf_param_1)",
            return_type=IntegerType(),
        )
        print("BUG: bare-boolean BSPIn was accepted")
    except ValueError as exc:
        print(f"register_impl rejected bare-boolean BSPIn (NULL vs 0): {exc}")

    # (b) OR instead of AND -- caught by the cross-combined built-ins (an
    # old_lac that is in the list paired with a new_lac that is not).
    try:
        register_impl(
            spark,
            BSPIn,
            kind="catalyst",
            catalyst_sql="CASE WHEN array_contains(_udf_param_2, _udf_param_0) "
            "OR array_contains(_udf_param_3, _udf_param_1) THEN 1 ELSE 0 END",
            return_type=IntegerType(),
        )
        print("BUG: OR-instead-of-AND BSPIn was accepted")
    except ValueError as exc:
        print(f"register_impl rejected OR-instead-of-AND BSPIn: {exc}")

    # (c) Swapped list params -- value-plausible (same shape, same functions),
    # caught only because the first BSPIn call above recorded the real rows
    # as samples: on ("14503", "14506") the swap flips 1 -> 0.
    try:
        register_impl(
            spark,
            BSPIn,
            kind="catalyst",
            catalyst_sql="CASE WHEN array_contains(_udf_param_3, _udf_param_0) "
            "AND array_contains(_udf_param_2, _udf_param_1) THEN 1 ELSE 0 END",
            return_type=IntegerType(),
        )
        print("BUG: swapped-lists BSPIn was accepted")
    except ValueError as exc:
        print(f"register_impl rejected swapped-lists BSPIn (caught by recorded samples): {exc}")

    # flatlist traps. (a) array_distinct drops duplicates -- caught by the
    # built-in nested example [['x', 'y'], ['x']].
    try:
        register_impl(
            spark,
            flatlist,
            kind="catalyst",
            catalyst_sql="array_distinct(flatten(_udf_param_0))",
            return_type=ArrayType(StringType()),
        )
        print("BUG: array_distinct flatlist was accepted")
    except ValueError as exc:
        print(f"register_impl rejected array_distinct flatlist (duplicates): {exc}")

    # (b) Filtering out nulls drops elements python passes through (flatten
    # preserves them) -- caught by the built-in [['a', None], []].
    try:
        register_impl(
            spark,
            flatlist,
            kind="catalyst",
            catalyst_sql="filter(flatten(_udf_param_0), x -> x IS NOT NULL)",
            return_type=ArrayType(StringType()),
        )
        print("BUG: null-filtering flatlist was accepted")
    except ValueError as exc:
        print(f"register_impl rejected null-filtering flatlist: {exc}")

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
