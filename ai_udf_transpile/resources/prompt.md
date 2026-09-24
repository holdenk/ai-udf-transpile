# Transpile this Python UDF into a Spark rewrite. Prefer Catalyst SQL.

## Inputs (authoritative)

- `udf.py` — the function to rewrite.
- `types.json` — Spark SQL types. `param_names[i]` is placeholder `_udf_param_i`
  with Spark type `input_types[i]`. `return_type` is the declared UDF return type.
  Do not guess types; these fields are required.
- `captures.json` — lexical captures. **Inline these values** in the rewrite;
  do not emit the capture names (for example, if `OFFSET` is `10`, write `10`).

## Output (exactly one)

1. **Preferred:** write a Spark SQL expression to `OUT.sql` using only `_udf_param_N`
   placeholders. Target ANSI Spark SQL semantics (overflow raises, divide-by-zero
   raises). No table scan, no Python.
2. If the UDF cannot be expressed as Spark SQL, write a Java UDF class to `OUT.java`
   implementing `org.apache.spark.sql.api.java.UDFN` (N = arity) with `call(...)`
   arguments in parameter order.
3. If neither is possible, print `DECLINE` and write nothing else.

Match Python semantics for the given types, including NULL (`None`).
