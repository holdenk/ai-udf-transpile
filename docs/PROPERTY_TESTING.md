# Property testing: how a rewrite earns "verified"

Every AI rewrite (Catalyst SQL, Java UDF, Rust UDF) goes through a
differential property test before it is served: the original Python UDF and
the rewrite run on the same inputs, and every result must agree. The
property is

> for every input the declared column types can hold, the rewrite returns
> what the Python UDF returns (or the Python UDF raises).

[Hypothesis](https://hypothesis.readthedocs.io/) generates the inputs. The
code is `hypothesis_check` in
[`ai_udf_transpile/verify.py`](../ai_udf_transpile/verify.py); the built-in
example tables live in
[`ai_udf_transpile/sampling.py`](../ai_udf_transpile/sampling.py).

This is evidence, not proof. Read
[the README's caveats](../README.md#dont-use-this-code-yet) before you
lean on it.

## The pipeline

1. **Static rejection** (`rejected_rewrite`). Before anything runs: SQL
   that calls `reflect` / `java_method`, Java that names a denylisted API
   (`Runtime`, `ProcessBuilder`, reflection, class loaders, `java.io`,
   `java.net`, ...), and Rust that reaches `std::fs` / `process` / `net` /
   `env`, `unsafe`, FFI, or compile-time includes are refused outright.
   These are denylists, not a sandbox.
2. **Load the Python side** in a sandbox namespace (stdlib allowlist plus
   the defining module's imports and captured values). A `NameError` /
   `ImportError` here fails verification: the gate cannot test what it
   cannot run.
3. **Build the inputs** (next section) and run the Python UDF on each.
4. **Run the rewrite** on the same rows in a real `SparkSession`, about 100
   rows per Spark job (`SQL_BATCH_ROWS`). A batch that throws (one ANSI
   divide-by-zero fails the whole job) is retried row by row so the
   failing row is identified. Each query is cancelled after
   `verifyQueryTimeoutSeconds` (a Java/Rust loop that never ends).
5. **Judge each row** (below).
6. **Reconstruction checks.** Verification binds the params as plain
   columns, which hides one difference from production: Spark's
   placeholder substitution does not reach inside lambda bodies. So a SQL
   rewrite whose `lambda` captures a `_udf_param_N` is refused
   (`sql_reconstruction_error`), and the rewrite is then pushed through
   the real `TranspiledPythonUDF` path on an empty DataFrame
   (`smoke_test_reconstruction`). Until both pass, the row stays
   provisional and no other query is served it.
7. Optionally, **manual approval** for the input types the generator is
   weakest on (see [Manual approval](../README.md#manual-approval)).

## Where the inputs come from

Three sources, in this order:

| source | what | when |
|---|---|---|
| real samples | up to `maxSamples` argument tuples per executor process, captured from the interpreted Python UDF on a cache miss (64 stored per key) | always, if any were captured |
| built-in examples | hand-picked values per type, one param at a time (others `NULL`), then a diagonal cross product across params | always (strings only with `partyOnStrings`) |
| random draws | `maxExamples` Hypothesis draws per type (default 20; 5 on GitHub Actions; `AI_UDF_TRANSPILE_MAX_EXAMPLES` overrides) | always |

Samples and built-ins run as Hypothesis `@example`s, so they run on every
verification regardless of `maxExamples`. Explicit examples are capped at
160 per check.

### Built-in examples per type

| type | built-ins | what they catch |
|---|---|---|
| `bigint` | 0, ±1, ±3, ±7, int32 edges and ±1 past them, 2^40, 2^53−1 / 2^53 / 2^53+1 | truncating vs flooring division/modulo; int32 casts; int/float precision past 2^53 |
| `int` / `smallint` / `tinyint` | the `bigint` list filtered to the type's range, plus the type's own min and max | ANSI overflow at the input type's edge (`int + 1`, `tinyint * tinyint`, `-smallint_min`) |
| `decimal(p,s)` | 0, ±1, ±0.5, ±2.5, ±0.01 where exactly representable at scale `s`, plus ±max | HALF_UP vs HALF_EVEN rounding, floor vs round, scale loss, precision overflow |
| `double` / `float` | 0, ±1, 0.5, ±2.5, 3.5, 1e16, 1e37 | banker's rounding; out-of-range results (1e37 → `bigint`) |
| `boolean` | `True`, `False` | three-valued logic: `NULL AND FALSE` is `FALSE` in SQL, `None` in Python |
| `string` | `""`, `" "`, numbers, `nan`, `inf`, JSON fragments, a timestamp, `é`, `ß`, `a.b`, `1_000`, `-5`, `2147483648` (only with `partyOnStrings`) | unescaped regex in `split`, ASCII-only case mapping, `int()` vs `CAST`, `zfill` vs `lpad`, missing `isnan` guards |
| `timestamp` | `NULL`, epoch, 1960 (negative epoch), 2038-01-19 03:14:07, a leap day | sign of pre-1970 epochs, 32-bit rollover |
| `array<string>` | `[]`, `[""]`, `["0"]`, `["a","b"]`, LAC-style codes | empty and singleton membership |
| `array<array<string>>` | empties at both levels, duplicates across inners, a `NULL` element | `array_distinct` / `explode`-style rewrites of `flatten` |
| `map<string,string>` | values that are JSON, Python literals, both, or neither | `to_json` vs `str()`, `json.loads` vs `ast.literal_eval` |

### Random draws per type

Integers draw from the full range of the declared column type; decimals
draw an unscaled integer and shift it by the scale, so every value is
exactly representable at `(p, s)`; doubles draw finite float32 values;
strings draw from assigned, non-surrogate code points (and only with
`partyOnStrings`); timestamps are naive, 1900–2100. Every type also draws
`None`.

**Every input must fit its column.** A value the column cannot hold is not
an input the UDF will ever see, and `createDataFrame` refuses it. The gate
used to hand `int`/`smallint`/`tinyint` columns int64 values and build
`decimal` columns as `bigint`, so a *correct* rewrite of any of those UDFs
was rejected as "sql raised". `tests/test_property_inputs.py` now pins
this as a property of the generators.

### Strings are off by default

A `string` parameter may hold a date, JSON, or a number. Random text says
nothing about a JSON parser: Python raises, the SQL returns `NULL`, and
under the policy below that counts as agreement. So the gate invents string
values only with `spark.sql.experimental.aiUdfTranspile.partyOnStrings`;
real samples always run. A string UDF verified with neither samples nor
`partyOnStrings` logs a warning: it was checked only on `NULL`.

## Judging a row

| Python | rewrite | verdict |
|---|---|---|
| value | equal value (within `tolerance` for numbers) | agree |
| value | different value | **mismatch** |
| value | raises | **reject** ("sql raised") |
| raises | anything | allowed (Spark's own transpiler policy) |
| `NameError` / `ImportError` | — | **reject** (sandbox could not run it) |
| value that does not fit the declared return type | anything | skipped (a real Python UDF fails there too) |

- `tolerance` (`spark.sql.experimental.aiUdfTranspile.tolerance`,
  default 0) is an absolute numeric distance. It never applies to strings,
  booleans, lists, or timestamps. A cached rewrite is served only to a
  reader whose tolerance is at least the one it was verified at.
- The rewrite's result must fit the declared return type too: verification
  rejects a value the production cast would overflow on.
- **No vacuous passes.** If Python raised on *every* example (say a
  `boto3` UDF with no credentials), "Python raises → anything goes" would
  pass any rewrite, so verification fails closed.

## How we test the tester

The gate is itself tested, at three levels:

- **Generators** (`tests/test_property_inputs.py`, no JVM). Hypothesis
  property tests of the input strategies: every draw and built-in fits
  its column (integer ranges, decimal precision and scale up to
  `decimal(38,18)`), the edges are present, and `_representable` matches
  the type ranges. These found that `st.decimals` and unary minus round in
  Python's default 28-digit context and produce `-1E20` for
  `decimal(38,18)`, which is why decimals are drawn as scaled integers.
- **Mutation testing** (`tests/test_property_mutants.py`, Spark). For each
  UDF, the correct rewrite must verify and every mutant — a plausible wrong
  rewrite (truncating division, swapped `coalesce` args, SQL three-valued
  `AND`, narrow-int overflow, HALF_UP vs HALF_EVEN, `floor` for `round`,
  dropped decimal scale) — must be rejected with the expected reason.
  Mutants run with `max_examples=1`, so the kill must come from a
  deterministic built-in example, never a lucky random draw.
- **Adversarial and differential cases** (`tests/test_hypothesis_gate.py`,
  `tests/test_adversarial.py`, `tests/test_sampling.py`, plus the
  per-feature files such as `test_map_inputs.py`, `test_flatlist.py`,
  `test_timestamp_to_epoch.py`, `test_kms_encrypt.py`). Rewrites that look
  right and are not (Python's floored `%` vs `pmod`, `round` vs `bround`,
  `lower` vs `casefold`, `split` on an unescaped `.`, `int("1_000")`), the
  null/raise policy, tolerance, Java UDFs, and a wrong key that passes on
  random strings but is caught by one real sampled row.

When you add a built-in example or a strategy, add the mutant it is meant
to kill to `CASES` in `tests/test_property_mutants.py`. If no mutant needs
it, it is probably not pulling its weight.

## Known blind spots

- **Sampled, not exhaustive.** A rewrite wrong only on inputs no sample,
  built-in, or draw reaches will pass.
- **Strings** are only as good as the real samples, unless you turn on
  `partyOnStrings`, and even then random text rarely hits a parser's valid
  inputs.
- **Python raises → anything goes.** A UDF that raises on most inputs gives
  the gate little to compare. Service-calling UDFs in an environment
  that cannot reach the service are the extreme case; gate them with
  `requireApproval=network`.
- **Doubles** draw float32 values; float64-only behavior is reached only
  through the built-ins and real samples.
- **Explicit examples are capped at 160**, so for UDFs with many parameters
  the cross product is truncated (it is enumerated diagonally, so every
  parameter's values still appear early).
- **Nested types** beyond `array<string>`, `array<array<string>>` and
  `map<string,string>` are not generated; structs are not supported.
