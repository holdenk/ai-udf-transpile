# Note: riding Iceberg's SQL UDF catalog

Status: design note, not a plan of record. Nothing here is implemented.
Written 2026-09-29 against the write-back code, and corrected 2026-10-02
after checking the spec that had already shipped.

Today `writebackFormat=iceberg` uses Iceberg as a *table*: a place to append
rows whose columns happen to describe UDFs. Iceberg neither knows nor cares
that `catalyst_sql` is executable. The interesting version is a rewrite we
verified becoming a *first-class function* other engines can resolve.

The metadata format for that exists. [Iceberg's SQL UDF spec](https://iceberg.apache.org/udf-spec/)
(format-version 1, merged February 2026 in apache/iceberg#14117) is a real
document, not a hope. What is still missing, and why this stays a note:

- Representations are **SQL only** (`type` must be `"sql"`). There is no
  Java, Rust, or GPU payload. `java_udf` and the reserved kinds in
  `targets.py` do not fit.
- A definition version may carry **one SQL body per dialect**. There is no
  engine-version field on the representation. Spark 4.1 and Spark 4.3 cannot
  both be current `dialect: "spark"` bodies.
- REST catalog support that has landed is **read-only** (list and load).
  Create/replace is not something to code against yet.

Check the spec again before writing a client. The shape below is format-version
1 as published, not a guess.

## What we would actually be binding to

A function's *name is not in the metadata file*. The catalog maps a name to a
metadata location. The file itself is immutable; a change is a new file and an
atomic swap of that pointer. Inside one file:

```
function metadata (format-version 1)
  function-uuid        generated once
  definitions[]        one per signature, keyed by definition-id
    definition-id      parameter types only, e.g. "long" or "long,string"
                       (Iceberg spells Spark bigint as long, array as list)
    parameters[]       { name, type }   -- SQL must use these names
    return-type
    versions[]
      version-id
      deterministic    default false
      on-null-input    "call" (default) or "return-null"
      representations[]
        { type: "sql", dialect: "spark", sql: "x + 1" }
        { type: "sql", dialect: "trino", sql: "x + 1" }
    current-version-id
  properties           string map, hints only
```

`definition-id` is the type tuple, not our content hash. Two different Python
bodies with the same signature are two *versions* of one definition, and only
one of them is current.

## Why our current row only half-maps

- **`udf_key` is content-addressed, and the spec has no such field.**
  `keys.udf_key()` hashes canonical source, param names, input types, return
  type, Spark version, and the closure fingerprint. That hash belongs in
  `properties`, and a reader still has to check it. It is not `definition-id`
  and it is not `function-uuid`.
- **`target_kind=catalyst` can become one `sql` representation with
  `dialect: "spark"`.** `java_udf`, `rust`, `gpu_kernel`, and `binary` cannot.
  The spec will not grow a blob representation just because `targets.py`
  reserved the names.
- **`spark_version` is in our key and is not in the spec.** Publishing a
  second Spark version means a new definition version that *replaces* the
  spark representation, with the old body kept only for rollback. A reader
  cannot ask for "spark 4.3" specifically.
- **The SQL body must name parameters.** The spec requires `sql` to reference
  the names in `parameters`, not `_udf_param_N`. A published body is a rename
  of the cached expression, and that rename is itself something to verify.
- **Verification metadata is a property, and properties are hints.**
  `hypothesis_passed`, `origin`, `model`, `error`, and `tolerance` can be
  copied into the string map. Nothing in the spec makes a reader honor them.
- **`on-null-input` defaults to `call`, which is the one we want.** Several
  of our rewrites return a value on NULL (`contains_ingredient` returns
  false). Publishing `return-null` would change them. `deterministic`
  defaults to false; a pure rewrite should set true, and we should not set
  it for anything we declined as side-effecting.

## What would have to change

**1. The catalog name is not the key.**
Resolution is the catalog's name-to-metadata mapping, then a content check
against `udf_key` in properties. Two jobs both defining `normalize` must not
share a metadata file unless the hash matches. A one-character body change
must not keep serving the old version. Name lookup alone is wrong.

**2. Reads must re-verify, not trust.**
This is the important one and it is a hole in the *current* write-back path
too. `WritebackCatalog` lookup will serve a remote `catalyst_sql` through
`F.expr()` and a remote `impl_source` through `compile_java`. `hypothesis_passed`
and `tolerance` are numbers in a table. Anyone who can append to that table
can run SQL and Java in every reader's driver. A tighter local tolerance
refuses a *looser* recorded tolerance; it does not re-run the check.

A real catalog makes that worse, because the object came from somewhere else.
The rule:

> A representation fetched from a shared catalog is a **candidate**, exactly
> like a backend's output. It goes through `hypothesis_check` (at the local
> tolerance) and `smoke_test_reconstruction` before it is served, and the
> local success row records that *we* verified it.

Verification is the cheap half compared to an agent call, and it is the step
that makes a shared cache safe to read. Cache the verdict keyed by
(`udf_key`, representation body) so each process pays it once. Skipping that
requires a signature from a trusted signer, not a boolean in a file anyone
can publish.

**3. Atomic update replaces append-and-dedupe.**
Write-back is append-only with "latest `updated_at` wins" because parquet has
no row-level UPDATE. A catalog metadata swap removes that workaround:
publishing becomes a commit, dedupe-on-read goes away, and `written_back_at`
plus `writebackThreshold` lose their reason to exist. Keep SQLite anyway. It
is the hot path for claims, cooldowns, and samples, none of which belong in
a shared catalog.

```
SQLite (local)      claims, attempt counts, cooldowns, samples, verified verdicts
Iceberg functions   one current SQL body per dialect, named by the catalog, atomically swapped
```

**4. Claims stay local. Do not put the work queue in the catalog.**
`pending` / `running` / `claimed_at` / `attempt_count` are our worker's
coordination state, not facts about the function. A shared catalog is the
wrong substrate for a lock. `DeltaCatalog.claim()` already shows why: two
workers can both read `pending`, both run the `MERGE`, and the loser still
observes `attempt_count == before + 1` and believes it owns the row. Publish
only verified successes. Failures stay local too. A rewrite that failed here
may succeed on a different engine, and publishing failures invites a wrong
cooldown.

**5. Types have to be Iceberg types, not Spark's spelling.**
`return_spark_type()` returns a Spark simpleString (`bigint`, `array<string>`)
and, in the fallback where `simpleString()` fails, a class name
(`DecimalType`). The spec wants Iceberg types: `long`, `list` of `string`,
`decimal(9,2)` with no spaces. That translation has to exist before any
publish path, and the class-name fallback is already the wrong string for our
own cache key.

## The part that actually gets better

- **Cross-engine, for the SQL subset.** A body that is valid Spark SQL and
  valid Trino SQL can be stored as two representations on one version. The
  spec will not pretend one string is universal, and neither should we.
  Java UDFs stay ours until a representation type exists for them.
- **Cold start, for named SQL.** Write-back already lets a fresh process skip
  the backend. A catalog extends that across engines, and the name mapping
  makes the rewrite visible to SQL written by hand, not just to our hook.

## Not doing this yet, and why

The metadata format is real, and it still does not cover the targets we
actually ship besides Catalyst SQL. REST cannot create a function yet. The
trust rule above is the hard part, and the current write-back path needs it
whether or not Iceberg is involved: a shared row is a candidate, not a hit.
