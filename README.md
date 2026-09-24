# AI UDF transpile

Partial, non-blocking Python UDF transpilation for Apache Spark. Spark's
built-in Catalyst transpiler covers simple functions; this plugin asks an
agent (CoCo / Cursor / Claude, or a deterministic fake backend in CI) to
rewrite the rest into Catalyst SQL or a Java UDF, then Hypothesis-checks the
rewrite against the original Python before it is served.

Queries never wait on the agent. A miss inserts a `pending` catalog row and
Spark keeps the interpreted Python UDF. A background worker (inline driver
thread by default, or `python -m ai_udf_transpile.worker`) fills the cache.

Targets **classic Spark master** (the `AbstractTranspiler` hook landed for
4.3 / current master). Connect is not supported. This is experimental.

## Install

```bash
pip install -e ".[dev]"
export SPARK_HOME=/path/to/spark-master   # package-only build, tests skipped
export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-*-src.zip"
```

PyPI PySpark is not used for integration tests until 4.3+ is the pinned
target; CI builds Spark from `apache/spark@master` with `-DskipTests`.

## Usage

```python
from pyspark.sql import SparkSession
from ai_udf_transpile import enable, register_impl

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.ansi.enabled", "true")
enable(spark, backend="fake")  # or coco / cursor / claude / auto

from pyspark.sql.functions import udf
from pyspark.sql.types import LongType


def plus_one(x: int) -> int:
    return x + 1


f = udf(plus_one, LongType())  # first call: Python path; worker may fill cache
```

Types must be known: every public parameter needs a recognized annotation
(`int` / `float` / `str` / `bool` / `bytes`) and `udf(..., returnType)` must
be an atomic Spark type. Untyped UDFs are ignored by this plugin (Spark's
built-in `catalyst` transpiler may still try them).

Human-provided rewrite (skips the agent):

```python
register_impl(
    spark,
    plus_one,
    kind="catalyst",
    catalyst_sql="_udf_param_0 + 1",
    return_type=LongType(),
)
```

Standalone worker when `inlineWorker` is false:

```bash
python -m ai_udf_transpile.worker --sqlite-path /tmp/ai_udf_transpile.sqlite --backend fake
```

## Catalog

Default catalog is **SQLite** (real `UPDATE` CAS). Optional `catalog=delta`
requires Delta Lake on the classpath. Vanilla Hive/Parquet tables cannot
`UPDATE` a row and are not used.

Failed rewrites cool down for 24h (`failCooldownSeconds`) and stop retrying
after `maxRetries` (default 3).

## Tests

```bash
pytest -m "not spark" tests/   # no JVM
SPARK_HOME=/path/to/spark-master pytest tests/   # plus classic SparkSession tests
```

The Spark tests need a package-only build of `apache/spark@master`
(`./build/mvn -DskipTests -Phadoop-3 package -pl sql/core,assembly -am`).
`tests/conftest.py` puts `$SPARK_HOME/python` and py4j on `sys.path` and pins
`PYSPARK_PYTHON` to the running interpreter so workers match the driver.

CI uses `backend=fake` with a handful of annotated fixtures (`plus_one`,
`is_none_branch`, `both_positive`, `greet`). Live CoCo/Cursor/Claude CLIs
are not invoked in default GHA.
