# References

Work this experiment leans on, borrows from, or is measured against.

## Upstream Spark

- [SPARK-54783](https://issues.apache.org/jira/browse/SPARK-54783) — the
  experimental Python UDF transpiler surface (`AbstractTranspiler`,
  `pyTranspilers`, `transpilePyUDFs`) this plugin hooks, on Spark
  master (4.3+).
- [SPARK-55206](https://issues.apache.org/jira/browse/SPARK-55206) —
  array-return UDFs skip the transpiler hook entirely; the reason sampling
  never records rows for `scatter_to_seconds` and several tests xfail.
- [SPARK-21935](https://issues.apache.org/jira/browse/SPARK-21935) — the
  trivial `upper()` UDF that OOM'd executors through Python worker memory;
  the motivating bug this repo demos against.
- [SPARK-58626](https://issues.apache.org/jira/browse/SPARK-58626) /
  [SPARK-58650](https://issues.apache.org/jira/browse/SPARK-58650) —
  lowered UDFs evaluate each used argument once per row, and lambda source
  identity must be pinned: constraints every faithful rewrite here has to
  respect.

## Talks

- Oliphant, T., Lam, S. K., & Florisson, M. (2013). *Numba: A Dynamic
  Python compiler for Science*. PyCon US 2013.
  <https://us.pycon.org/2013/schedule/presentation/130/> — lays out the
  bytecode-translation vs AST-translation paths and their trade-offs; this
  repo ships both.

## Key references

- Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python
  JIT compiler. *Proceedings of the Second Workshop on the LLVM Compiler
  Infrastructure in HPC*, 1–6. <https://doi.org/10.1145/2833157.2833162> —
  bytecode compiled to native code. The Java bytecode evaluator here is the
  same idea held at JVM fidelity: interpret the opcodes instead of lowering
  them.
- Bolz, C. F., Cuni, A., Fijałkowski, M., & Rigo, A. (2009). Tracing the
  meta-level: PyPy's tracing JIT compiler. *Proceedings of the 4th Workshop
  on the Implementation, Compilation, Optimization of Object-Oriented
  Languages and Programming Systems*, 18–25.
  <https://doi.org/10.1145/1565824.1565827> — the interpreter *is* the
  program. Our fixed Java interpreter pasted into each UDF jar is a
  non-optimizing cousin of that shape.
- Behnel, S., Bradshaw, R., Citro, C., Dalcin, L., Seljebotn, D. S., &
  Smith, K. (2011). Cython: The best of both worlds. *Computing in Science
  & Engineering, 13*(2), 31–39. <https://doi.org/10.1109/mcse.2010.118> —
  ahead-of-time typed lowering of Python. The static bytecode→SQL backend
  does the same, lowering to Catalyst instead of C.
- Wimmer, C., & Würthinger, T. (2012). Truffle: A self-optimizing runtime
  system. *Proceedings of the 3rd Annual Conference on Systems,
  Programming, and Applications: Software for Humanity*, 13–14.
  <https://doi.org/10.1145/2384716.2384723> — guest languages as
  interpreters the runtime optimizes. Our Java interpreter is a fixed
  guest: no partial evaluation, just honest speedups over CPython.
- Larose, O., Kaleba, S., Burchell, H., & Marr, S. (2023). AST vs.
  bytecode: Interpreters in the age of meta-compilation. *Proceedings of
  the ACM on Programming Languages, 7*(OOPSLA2), 318–346.
  <https://doi.org/10.1145/3622808> — the empirical answer to "which IR
  should you consume," which is why this repo walks both roads: Spark's
  built-in transpiler consumes the AST, ours consumes the bytecode.

## Related work

- *Predicate Pushdown for Data Science Pipelines* (SIGMOD 2023 Best Paper)
  introduced MagicPush, which symbolically analyzes Python UDFs to push
  their work into the relational engine — the same "understand the UDF,
  lower it into the database" goal, attacked symbolically where this repo
  attacks it with static bytecode walking and AI rewrites under a
  differential-testing gate. Cong Yan co-authored.
