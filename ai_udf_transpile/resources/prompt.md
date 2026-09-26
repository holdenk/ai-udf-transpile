# Transpile this Python UDF into a Spark rewrite. Prefer Catalyst SQL.

Your output is differentially tested against the original Python on hundreds
of generated inputs (NULLs, empty strings/arrays, duplicates, unicode, real
sampled rows) before anyone trusts it. A rewrite that mismatches on any input
where Python returns a value is rejected; a rewrite that crashes at analysis
is rejected. DECLINE is cheap and final -- a wrong guess is wasted work.

## Inputs (authoritative)

- `udf.py` — the function to rewrite.
- `types.json` — Spark SQL types. `param_names[i]` is placeholder `_udf_param_i`
  with Spark type `input_types[i]`. `return_type` is the declared UDF return type.
  Do not guess types; these fields are required.
- `captures.json` — lexical captures. **Inline these values** in the rewrite;
  do not emit the capture names (for example, if `OFFSET` is `10`, write `10`).

## Output (exactly one)

1. **Preferred:** write a Spark SQL expression to `OUT.sql` (rules below).
2. If the UDF cannot be expressed as Spark SQL, write a Java UDF class to
   `OUT.java` (rules below).
3. If neither is possible, print `DECLINE` and write nothing else.

## Catalyst SQL rules (`OUT.sql`)

- A raw Spark SQL expression using only `_udf_param_N` placeholders. No table
  scan, no Python, no markdown fences, no prose. Target ANSI Spark SQL
  semantics (overflow raises, divide-by-zero raises).
- **Never reference `_udf_param_N` inside a higher-order function lambda**
  (`transform(arr, x -> ...)`, `filter(arr, x -> ...)`, `exists(...)`):
  placeholder substitution does not descend into lambda bodies, so the query
  fails at analysis with UNRESOLVED_COLUMN. Keep parameter references outside
  lambdas: `transform(sequence(_udf_param_0, _udf_param_1), x -> date_format(x, 'yyyy-MM-dd'))`.
- **NULL vs False/0:** SQL NULL propagation is not Python `None` handling. If
  Python returns `False`/`0` on NULL input, a bare boolean expression is wrong
  (`instr(...) > 0` returns NULL): wrap it — `CASE WHEN ... THEN 1 ELSE 0 END`
  or `coalesce(..., false)`.
- **Reproduce Python's failure-path defaults, not NULL.** With
  `except: return []` (or `return "nan"`, `return 0`), unparseable input must
  yield that default. Use `try_cast` / `try_to_timestamp` (NULL on bad input)
  plus guards that map NULL to the Python default; a plain `cast` raises under
  ANSI where Python's try/except returns a default.
- **Off-by-one, order, duplicates, null elements:** `range(n + 1)` includes
  both endpoints (`sequence(start, end)`); `flatten` preserves order,
  duplicates, and NULL elements while `explode` + `collect_list` drops NULLs;
  `array_distinct` drops duplicates; `sort_array` reorders.
- **String formatting:** Python `str()` differs from SQL casts — `str(float)`
  yields `'inf'`/`'nan'`, timestamps stringify as `yyyy-MM-dd HH:mm:ss`, and
  Spark date patterns use `java.time` letters (`yyyy-MM-dd HH:mm:ss`), not
  Python strftime (`%Y-%m-%d`).
- The expression must type-check to `return_type` (e.g. for an `int` return,
  emit `THEN 1 ELSE 0`, not a bare boolean).
- Use only functions that exist in Spark SQL. If you are unsure a function
  exists, do not invent it — use Java or DECLINE.

### Python/Spark lookalikes that bite (all verified against real Spark)

- **Modulo/division:** Python `%` and `//` floor (sign of the divisor); Spark
  `%` and `div` truncate toward zero, and `pmod` matches Python only for
  POSITIVE divisors (`pmod(7, -3)` is `1`, Python `7 % -3` is `-2`). Floored
  modulo: `(x % y) + CASE WHEN (x % y) <> 0 AND ((x < 0) <> (y < 0)) THEN y ELSE 0 END`;
  floored division: `floor(x / y)`.
- **Rounding:** Python `round` is banker's rounding; Spark `round` is
  half-up. Use `bround`, and cast to `bigint` — `cast(bround(1e16) as int)`
  overflows where Python returns `10000000000000000`.
- **`split` takes a regex:** `split(s, '.')` turns `'a.b'` into four empty
  strings. Escape literal dots: `split(s, '\\.')`.
- **`substr` is 1-based** and `lpad`/`rpad` TRUNCATE over-long input and pad
  before a sign: Python `'-5'.zfill(3)` is `'-05'` but `lpad('-5', 3, '0')`
  is `'0-5'`, and `'abcd'.zfill(3)` stays `'abcd'` while `lpad` gives `'abc'`.
- **Weekdays:** Python `date.weekday()` is Monday=0; `dayofweek` is
  Sunday=1. Faithful: `pmod(dayofweek(t) + 5, 7)`.
- **Case:** `lower` is not `casefold` — `'ß'.casefold()` is `'ss'`. No
  faithful SQL exists; use Java or DECLINE.
- **Float stringification:** `cast(x as string)` yields `'1.0E16'` where
  Python `str(x)` yields `'1e+16'`, and `str(None)` is the string `'None'`,
  not NULL. Match Python's spelling or DECLINE.
- **`int()` accepts underscores** (`int('1_000')` is `1000`): strip them
  before casting — `try_cast(regexp_replace(s, '_', '') as int)`.
- **`trim` strips spaces only**; Python `strip()` removes all whitespace
  (tabs, newlines, NBSP). If the input can contain non-space whitespace, no
  faithful `trim` rewrite exists.

### Example (Catalyst)

`udf.py`:
```python
def contains_ingredient(text: str, needle: str) -> bool:
    return text is not None and needle in text.lower()
```
`OUT.sql`:
```
coalesce(instr(lower(_udf_param_0), _udf_param_1) > 0, false)
```
(The bare comparison returns NULL on NULL input; Python returns False.)

## Java UDF rules (`OUT.java`)

Implement `org.apache.spark.sql.api.java.UDFN` (N = arity):

- Declare `package ai_udf;` and exactly one public class with a public
  no-arg constructor.
- Use erased signatures only: `implements UDF1<Object, Object>` and
  `public Object call(Object arg0)`. Cast arguments inside the body, e.g.
  `((String) arg0).toUpperCase()`. Never use typed generics
  (`UDF1<String, String>`) or primitive parameters: the class may be
  compiled by Janino, which emits no generic signatures and no bridge
  methods, so typed signatures fail at registration.
- Arguments arrive boxed per `types.json`: `Long`/`Integer`/`Double`/`Float`/
  `Boolean`/`String`/`byte[]`/`java.sql.Timestamp`;
  `array<string>` arrives as `scala.collection.Seq<Object>` (NULL elements
  arrive as `null`); `map<string,string>` arrives as
  `scala.collection.Map<Object, Object>`.
- Return values: boxed atomics for atomic returns; for `array<string>` return
  a `java.util.List<String>` (NULL elements allowed).
- Match Python `None` semantics: return `null` when an argument is `null`
  unless the Python function handles `None` differently.
- Java traps: compare strings with `.equals` (never `==`); Python ints are
  arbitrary precision, so a `long` can overflow where Python cannot;
  `Double.toString` spells non-finite values `Infinity`/`NaN` where Python
  spells `inf`/`nan` — match Python's spelling if the UDF stringifies.

### Example (Java)

`udf.py`:
```python
def backwards(name: str) -> str:
    if name is None:
        return None
    return name[::-1]
```
`OUT.java`:
```java
package ai_udf;

import org.apache.spark.sql.api.java.UDF1;

public class Backwards implements UDF1<Object, Object> {
    @Override
    public Object call(Object s) {
        return s == null ? null : new StringBuilder((String) s).reverse().toString();
    }
}
```

## When to DECLINE

- The UDF calls a network/service/SDK (boto3, requests), reads the clock or
  randomness, or does I/O: no faithful rewrite exists.
- You cannot match Python semantics for the declared types with confidence.

Match Python semantics for the given types, including NULL (`None`).
