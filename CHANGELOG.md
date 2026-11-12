# Changelog

Alpha software: no releases yet, so everything sits under Unreleased, newest
first. Perf numbers are from one local machine (4 cores, PyPI PySpark 4.3.0,
JDK 21, local mode, min of 3 rounds) and the `claude` CLI backend; they are
evidence for a direction, not a promise about your cluster.

## Unreleased

### Rust target: Comet native UDFs, opt-in (2026-11-12)

A new rewrite kind, `rust`, for Apache DataFusion Comet 1.2+'s experimental
native UDFs (`comet-udf-sdk` / `CometNativeUDF`). Turn it on with
`spark.sql.experimental.aiUdfTranspile.targets=catalyst,java_udf,rust` in a
session where Comet is live. SQL is still the first choice; Rust is offered
before Java for UDFs with no SQL form.

**The model writes one row function**, `pub fn udf(Option<..>, ..) ->
Result<Option<..>, String>`, with the signature fixed by the UDF's declared
types. `ai_udf_transpile/rustc.py` generates the Arrow glue, builds a cdylib
with cargo against the Comet SDK of the session's jar (or `$COMET_HOME`),
caches it by content, and registers it. Verification runs the built library
through Comet on the same Hypothesis inputs as the other targets. A scan
refuses `unsafe`, FFI, filesystem/process/network/env access, include
macros, `#[path]`, `use` aliases that hide those, and recursion (a Rust stack
overflow aborts the process).

**Numbers.** From `examples/demo_rust_udf.py` (2M rows, native range, Comet
1.2.0): `collatz_steps` 20.5x and `is_prime` 16.8x over the interpreted UDF,
both served as Rust. `longest_digit_run` was answered as SQL. The catalog
bench (smaller cached samples) measures the same two Rust rewrites at 4.1x
and 3.6x. In the micro bench every `*_rust` twin was answered as SQL
(`reverse`, `levenshtein`, ...), so those rows time SQL, not Rust.

**Not fail-open.** Spark cannot evaluate a Comet native UDF: a query whose
operator Comet does not take fails instead of falling back to Python. That
includes a Rust UDF beside a still-interpreted Python UDF in the same
`select`, scans over `createDataFrame`/`spark.range`/cached frames without
the matching Comet conversion confs, and a global sort keyed on the call.
A session that cannot run Rust gets a separate `+no-rust` cache row with its
own SQL/Java rewrite instead of a Rust one it would fail on. Atomic types
only; `local` masters only unless `rust.sharedBuildDir` says the build
directory is cluster-visible. See the README's Rust target section.

### Manual approval for hard-to-verify rewrites (2026-11-12)

`spark.sql.experimental.aiUdfTranspile.requireApproval` (default `none`)
parks AI rewrites in a new `awaiting_approval` status after they pass
Hypothesis and the reconstruction smoke test, so a person signs off before
they are served or written back. Gate everything (`all`) or only the cases
Hypothesis is weakest on: `string` / `struct` / `map` / `array` / `binary` /
`timestamp` types, `network` (the UDF imports or calls a network client),
or `hard` (`string,struct,map,array,network`). Human (`register_impl`) and
`bytecode` rows are exempt.

`python -m ai_udf_transpile.approve` walks the queue, showing the Python,
the rewrite, the flags it tripped and sampled inputs, and prompts
`[y/n/skip]`; `--browse` is the same view read-only, and `--list` /
`--approve` / `--reject` script it. `reviewed_by` / `reviewed_at` are
recorded on the row (SQLite, Delta, and the write-back table) and cleared
whenever the payload changes. Lookups enforce the policy too: unreviewed
successes it covers park on their next lookup, and an unreviewed shared
write-back row is copied locally for review instead of served.

### Real pipeline UDFs, and when a faithful rewrite is a regression (2026-11-11)

Seven scalar text UDFs from Fight Health Insurance's training-data pipeline
(verdict parsing, entity cleanup, the findings regex extractors, the regex PII
scrubber) joined the benches: ported self-contained into
`ai_udf_transpile/backends/fhi_udfs.py`, timed by
`examples/bench_fhi_udfs.py` (`examples/run_demos.sh --bench-fhi`).

**The finding.** Every one got a rewrite that passed the Hypothesis gate on
the first attempt, and most of those rewrites were slower than the Python
they replaced. The slowdown is per-row, not fixed overhead: it held from
100k to 50M rows (0.4x for `fhi_clean_health_history` at 10M and at 50M).
A rewrite removes one thing for certain, the trip to the Python worker
(about 0.2 us/row here); a faithful rewrite of `str.strip()`,
`splitlines()` or Python's unicode `\s`/`\d`/`\b` that spells them out as
`java.util.regex` classes and lookarounds costs more than that per regex
evaluation.

**Both rewrites are kept.** Each function has two pinned, faithful rewrites
(`ai_udf_transpile/backends/fhi_rewrites.py`, `resources/fhi/`): `naive` is
verbatim what the backend served on its first verified attempt, `tuned` is
hand-written from what the naive one cost. The bench times python, naive,
tuned and (with `--ai`) whatever the backend serves today, each over a
column Spark must treat as nullable and one it can prove non-null.

Speedup vs the interpreted Python UDF, nullable column (`naive` on a
non-null column in brackets where it differs):

| Function | naive (AI, old prompt) | AI, new prompt | tuned (hand) |
|---|---|---|---|
| `fhi_verdict_has_legal_check` | 0.5x | 2.2x (Java loop) | 7.2x (Java loop) |
| `fhi_parse_verdict` | not served (lambda) | 20.0x (Java loop) | 8.5x (Java loop) |
| `fhi_clean_health_history` | 0.3x (0.7x) | 1.7x (`btrim`) | 1.8x (`btrim`) |
| `fhi_treatment_from_findings` | 0.9x | 1.5x (Java, ASCII path) | 1.4x (SQL, ASCII guard) |
| `fhi_diagnosis_from_findings` | 2.5x | 2.3x (same shape) | -- |
| `fhi_normalize_entity_text` | 0.8x | 6.3x (Java, keyword prefilter) | 0.9x (SQL, ASCII guard) |
| `fhi_deidentify_text` | 0.4x | 0.8x (Java, ASCII path) | 0.8x (Java, ASCII path) |

2M rows (500k for treatment/normalize, 200k for the scrubber), cached
inputs, noop sink. Between runs the AI column moved by up to 2x for the
fastest rewrites (the legal check measured 3.9x-4.1x in two earlier runs),
so read the signs and the orders of magnitude, not the decimals. Five of the
seven naive rewrites are slower than Python; with the new prompt, six of
seven AI rewrites are faster.

What the tuned rewrites change, each measured on its own first:

- `clean_health_history`: `btrim(x, <Python's 29 whitespace code points>)`
  instead of a regex strip, and no leading `CASE WHEN x IS NULL THEN NULL`.
  That guard is the only WHEN condition Spark always evaluates, so
  subexpression elimination shares nothing after it and all three repeated
  strips run. On a non-nullable column the optimizer deletes the guard and
  the naive rewrite improves from 0.3x to 0.7x by itself. The same guard in
  front of the `btrim` shape takes it from 1.8x to 1.0x on a nullable column
  and costs nothing on a non-null one. **That is what strong typing
  (provably non-null inputs) buys: it undoes the guard. It does nothing for
  a rewrite written without one, and nothing about the regex**, which is
  where the rest of the cost was (the naive rewrite is still 0.7x
  non-null).
- `verdict_has_legal_check`, `parse_verdict`: a Java UDF walking the lines
  with plain loops instead of regexes over the whole text.
  `parse_verdict`'s naive SQL needs `transform`/`filter` lambdas, which Spark
  4.3 never serves (see Fixed).
- `treatment_from_findings`, `normalize_entity_text`: the naive expression
  twice, behind `length(x) = octet_length(x)`. On pure-ASCII text Java's
  cheap classes mean what Python's mean, so that arm drops the spelled-out
  unicode classes.
- `deidentify_text`: the same split inside the Java UDF.
- `diagnosis_from_findings`: no tuned variant; two simple patterns already
  win.

### Prompt: "Faster than the Python, or it is not a rewrite" (2026-11-11)

`ai_udf_transpile/resources/prompt.md` gained a section on speed, written
from the measurements above:

- never turn a `str` method into a regex (`strip()` is `btrim` with
  Python's whitespace set, spelled out in the prompt);
- no leading `IS NULL` guard; put a reused subexpression in the first WHEN;
- line- and character-level logic belongs in a Java UDF with plain loops;
- avoid `->` lambdas (Spark will not serve them);
- an ASCII fast path, but only when the Python makes five or more `re`
  calls per row. The first wording had no threshold, and the backend built
  an `instr`/`lower` fast path around `diagnosis_from_findings`'s two simple
  patterns, taking it from 2.2x to 1.5x; with the threshold it writes the
  plain `regexp_extract` shape again (2.3x-2.7x across runs).

A Java trap was added too: a `java.util.regex` lookbehind over a character
class steps back one UTF-16 unit (see Fixed).

Before and after, same backend: the `naive` and `AI, new prompt` columns
above. The backend now picks a Java line loop for the two verdict
functions, `btrim` for the strip, and for `normalize_entity_text` a Java UDF
that runs each lead-in regex only when its keyword occurs in the
lowercased text (0.8x to 6.3x).

The new rewrites agree with python on all 311 hand-built adversarial inputs
(`tests/fhi_adversarial.py`), including the supplementary-plane cases. The scrubber is still slower than CPython's
`re`; nothing yet stops a verified-but-slower rewrite from being served.

### Added (2026-11-11)

- `ai_udf_transpile/regress.py`: a static, report-only guess at whether a
  rewrite will be slower than the Python, from the two texts alone
  (`ok` / `at-risk` / `likely-regression` / `not-served`). It counts regex
  calls in the Python (loops over literal pattern tuples multiply) and regex
  evaluations in the rewrite (WHEN/THEN copies counted separately), and
  flags regex introduced where the Python used C string methods, a
  regex-dominated body with spelled-out unicode classes or lookarounds, a
  leading null guard in front of repeated work, and lambdas.
  `bench_fhi_udfs.py` prints the guess beside every measured ratio. On the
  13 pinned FHI rewrites every `likely-regression` measured slower and every
  `ok` measured faster; none of the 48 documented examples in
  `docs/EXAMPLES_TRANSPILED.md` is flagged as a likely regression. One machine, one corpus: treat
  `at-risk` as "measure before serving".
- `tests/fhi_adversarial.py`: hand-built inputs aimed at where a rewrite and
  Python part ways (unicode whitespace and line breaks, non-ASCII digits and
  case partners, supplementary-plane characters, two verdict words on a
  line). The gate generates strings of at most 16 characters and cannot hold
  most of them.

### Fixed (2026-11-11)

- **Java rewrites with the same class name ran each other's code.** Every
  jar a session registers shares one classloader, so two different rewrites
  declaring `ai_udf.FhiDeidentifyText` (retry attempts, a re-verify after
  the UDF changed, two functions with the same name) all ran whichever was
  loaded first, in verification as well as serving. `compile_java` now
  compiles under the declared name plus a hash of the source
  (`javac.unique_class_name`) and returns that name; the catalog stores it,
  and the serve path registers what it compiled. Rows stored before this
  keep their jar and name. Found because the bench's pinned naive scrubber
  measured exactly as fast as the AI's.
- **A verified rewrite Spark will not serve was reported as transpiled.**
  Spark 4.3 drops a transpiled expression that contains a
  higher-order-function lambda and plans the Python UDF instead, silently;
  `udf.transpiled` stays truthy. `bench_fhi_udfs.py` and `bench_catalog.py`
  check the physical plan before timing and report such a row as Python.
- **Supplementary-plane characters broke `\b` emulation.** A
  `java.util.regex` lookbehind over a class without supplementary characters
  reads one UTF-16 unit. `(?<![\p{L}\p{N}_])` after U+1D400 sees a lone
  surrogate, so the tuned `parse_verdict` flipped verdicts, and the
  scrubber's faithful pattern set left `SSN 123-45-678𝟗` (U+1D7D7)
  unredacted. Both tuned rewrites now use `[...]{1,2}`. The naive exhibits
  keep the bug verbatim, pinned in `tests/test_fhi_udfs.py`. Known, not
  fixed: 622 CJK Extension I letters (U+2EBF0..U+2EE5D) are word characters
  to Python 3.13 (Unicode 15.1) and unassigned to JDK 21 (Unicode 15.0), so
  boundaries next to them follow the JVM.
- **Forbidden-API scan gaps.** SQL `try_reflect(...)` (reflect with the
  exception swallowed, present in Spark 4.3) passed the `reflect` ban; Java
  `System.getenv`/`getProperty`, `java.nio.file`, `java.nio.channels`,
  `javax.net`, `ClassLoader` and the `getDeclared*` reflection spellings
  passed the `java.io`/`java.net` ban. It is still a denylist.
- `bench_fhi_udfs.py`: a 4g driver heap (the cached letter frame OOMed the
  1g default), a catalog wait scoped to this session's Spark version, a
  `string_gate` of `cached` when a rewrite was reused rather than verified;
  `bench_graph_aggregate.py` routes kind-labeled rows to the AI arms;
  `dump_results.py` keeps the catalog bench's Comet number when the FHI CSV
  takes over a function's line.
