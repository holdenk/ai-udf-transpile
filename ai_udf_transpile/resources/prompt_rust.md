## Rust target (`OUT.rs`) -- accepted for this job

This session runs Apache DataFusion Comet, which runs a Rust function natively
on Arrow data with no JVM or Python hop per row. That changes the output
choice above to (still exactly one):

1. **Preferred:** Spark SQL in `OUT.sql`, whenever the UDF is expressible as SQL.
2. Otherwise a Rust row function in `OUT.rs` (rules below).
3. Otherwise a Java UDF in `OUT.java`.
4. Otherwise print `DECLINE`.

### Rust rules (`OUT.rs`)

- `OUT.rs` is one Rust module defining exactly this function -- same name,
  same parameter types in the same order, same return type (parameter names
  are yours to choose):

  ```rust
  @@SIGNATURE@@
  ```

  It is called once per row. `None` is SQL NULL / Python `None`, in both
  directions. Return `Err(message)` where Python raises: it fails the query.
- The types are fixed by `types.json`: bigint `i64`, int `i32`, smallint
  `i16`, tinyint `i8`, double `f64`, float `f32`, boolean `bool`, string
  `&str` in and `String` out, binary `&[u8]` in and `Vec<u8>` out. Do not
  write any Arrow code; the array handling is generated around your function.
- Standard library only. Helper functions and `const`s in the same file are
  fine. No external crates, no `arrow`, no `mod` declarations, no attributes,
  no `macro_rules!`.
- It must be a pure function of its arguments: no clock (`SystemTime`,
  `Instant`), no randomness, no global or static mutable state. Comet may
  fold a call over constants or evaluate a repeated call once. `HashMap` and
  `HashSet` iterate in a random order: where order reaches the result, use
  `BTreeMap`/`BTreeSet`, a `Vec`, or sort.
- Rejected before it is ever compiled: `unsafe`, `extern`, anything under
  `std::fs`, `std::process`, `std::net`, `std::env`, `std::io`, `std::os`,
  `std::thread`, `std::path`, `std::ffi`, `std::ptr`; `include!`,
  `include_str!`, `include_bytes!`, `env!`, `macro_rules!`; `#[path]`,
  `#[link]`, `#[no_mangle]`. A row function needs none of them.
- No recursion (a function that calls itself is rejected): a Rust stack
  overflow aborts the whole process instead of failing the query. Write the
  loop. Every loop must terminate on every input, NULL and empty included.
  The check is by name: any mention of a function's name inside a function
  that it can reach from itself counts as a call, so never name a local or a
  parameter after a function. Also rejected, because they can call a
  function without naming it: trait impls (`impl Trait for Type`) and
  `trait` items, `dyn`, `Any`, a struct/enum/`type` holding a `fn(..)` or
  `Fn`, and a function used as a value in a `static` or `const`. Use plain
  functions, inherent methods (`impl Type { .. }`), and closures.
- A panic fails the query, and integer overflow panics (overflow checks are
  on). Python ints never overflow: where an intermediate can leave `i64`
  while the result still fits, widen to `i128`; where the result itself
  cannot fit, return `Err`.

### Python/Rust lookalikes that bite

- **Division and modulo:** Rust `/` truncates toward zero and `%` takes the
  sign of the dividend; Python `//` floors and `%` takes the sign of the
  divisor. `div_euclid`/`rem_euclid` match Python only for a positive divisor.
  Floored: `let (q, r) = (a / b, a % b); if r != 0 && ((r < 0) != (b < 0)) { q - 1 } else { q }`
  and `if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r }`. Python raises
  `ZeroDivisionError` for integer AND float division by zero: return `Err`
  (Rust float division gives `inf`/`NaN` silently).
- **Strings are UTF-8 bytes in Rust, code points in Python.** `len(s)` is
  `s.chars().count()`, not `s.len()`. Index, slice and reverse through
  `chars()`; a byte-offset slice inside a multi-byte character panics.
  `s[::-1]` is `s.chars().rev().collect::<String>()`.
- **Whitespace:** `trim()`/`split_whitespace()` use Unicode `White_Space`;
  Python `strip()`/`split()` also treat U+001C..U+001F as whitespace. Handle
  those four or DECLINE if the input can contain them.
- **Case:** `to_uppercase`/`to_lowercase` are the full Unicode mappings, like
  Python `upper`/`lower`. There is no `casefold` in std.
- **Rounding:** Python `round(x)` is banker's rounding: `x.round_ties_even()`.
  `f64::round` rounds half away from zero.
- **Float to int:** Python `int(x)` truncates and raises on `inf`/`nan`;
  `x as i64` saturates silently. Check `is_finite()` and the range, then cast.
- **Formatting:** `format!("{}", 1e16_f64)` is `10000000000000000`, Python
  `str(1e16)` is `1e+16`; Rust prints `inf`/`NaN`, Python `inf`/`nan`;
  `str(True)` is `True`, `format!("{}", true)` is `true`; `str(None)` is the
  string `None`. Match Python's spelling exactly or DECLINE.
- **Parsing:** Python `int(s)` accepts surrounding whitespace, a sign, and
  single underscores between digits; `str::parse::<i64>` accepts none of the
  whitespace or underscores. Python `float(s)` accepts `inf`, `nan`,
  `infinity` in any case, and underscores.
- **Truthiness:** `0`, `0.0`, `""`, `b""` and `None` are falsy in Python.

### Example (Rust)

`udf.py`:
```python
def collatz_steps(n: int) -> int:
    if n is None or n < 1:
        return None
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps
```
`OUT.rs`:
```rust
pub fn udf(n: Option<i64>) -> Result<Option<i64>, String> {
    // i128: 3n + 1 can leave i64 on the way to a result that fits.
    let mut n = match n {
        Some(v) if v >= 1 => v as i128,
        _ => return Ok(None),
    };
    let mut steps: i64 = 0;
    while n != 1 {
        n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        steps += 1;
    }
    Ok(Some(steps))
}
```
(A data-dependent loop has no Spark SQL form; the `None`/`n < 1` guard comes
first because Python checks it first.)
