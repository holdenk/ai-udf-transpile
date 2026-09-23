Current state: hopes and dreams
Next step: proof of concept

Idea: Python UDFS can be expensive to evaluate, especially when the core engine is in another language and a data copy is required (like Java for Spark, or Rust for LakeSail) or *parts* of evaluation are happening elsewhere (like nv rapids)

One possible solution is transpilation, taking the Python code and turning it into the other language to avoid the data copy. Even when a data copy is not required, moving Python evaluation into a more performant language can be beneficial.

However, creating a complete alternative compiler for Python is *hard* (see Jython). We can do *partial* transpilation, that is transpile *when it makes sense* instead of aiming for 100% success. This is the idea behind https://issues.apache.org/jira/browse/SPARK-54783

This repo explore taking the idea a step further, using AI to non-deterministically transpile Python which is too complicated for a simple transpiler.

Since, as every AI agent should remind you, AI can make mistakes -- we also need mechanisms to detect when transpilation is incorrect. We can do this (to an extend) with automated hypothesis powered tests since we have a reference implementation we _know_ is correct.
Similarily, if you've ever asked Claude (or Codex or Cursor or ...) to write some code only to come back an hour later and see it still not finished; niavely transpiling and waiting for it to finish (and then testing it) could easily take a one minute job and turn it into hours. Instead we'll kick this off non-blocking.
