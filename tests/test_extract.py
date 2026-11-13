# SPDX-License-Identifier: Apache-2.0
"""Static UDF extraction: shapes, skips, and key parity with the runtime hook.

No JVM. The parity test imports a real script against a stub ``pyspark``
and keys each live function exactly the way ``AITranspiler`` does (Spark's
getsource/dedent/parse, then the decorator strip, captures, udf_key), so a
divergence between the static and runtime views fails here, not in prod.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
import types

import pytest

from ai_udf_transpile.extract import extract_source
from ai_udf_transpile.keys import (
    canonical_source,
    closure_fingerprint,
    extract_captures,
    strip_udf_decorators,
    udf_function_ast,
    udf_key,
)
from ai_udf_transpile.types import input_spark_types

SPARK_VERSION = "4.3.0-test"


def _extract(src: str):
    return extract_source(textwrap.dedent(src), "job.py")


def _names(result):
    return sorted(u.name for u in result.udfs)


def _skips(result):
    return {s.name: s.reason for s in result.skipped}


# -- shapes --------------------------------------------------------------------


def test_decorator_shapes_and_return_types():
    result = _extract(
        """
        import pyspark.sql.functions as F
        from pyspark.sql import functions as G
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DecimalType, LongType

        RT = LongType()

        @udf(returnType=LongType())
        def a(x: int) -> int:
            return x + 1

        @F.udf("double")
        def b(x: float) -> float:
            return x * 2

        @udf
        def c(s: str) -> str:
            return s.upper()

        @G.udf(RT)
        def d(x: int) -> int:
            return x - 1

        @udf(DecimalType(12, 2))
        def e(x: float) -> float:
            return x

        @udf("long")
        def f(x: int) -> int:
            return x
        """
    )
    assert result.error is None
    rt = {u.name: u.return_type for u in result.udfs}
    assert rt == {
        "a": "bigint",
        "b": "double",
        "c": "string",  # udf()'s default returnType
        "d": "bigint",
        "e": "decimal(12,2)",
        "f": "bigint",  # DDL "long" parses to LongType
    }
    for u in result.udfs:
        assert "@" not in u.source_text, "the pyspark udf decorator must not be keyed"
    assert result.skipped == []


def test_call_and_register_shapes():
    result = _extract(
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import BooleanType

        def both(x: int, y: int) -> bool:
            return x > 0 and y > 0

        def main(spark):
            spark.udf.register("later", later, "bigint")
            spark.udf.register("both_again", both_udf)

        def later(x: int) -> int:
            return x - 1

        both_udf = udf(f=both, returnType=BooleanType())
        """
    )
    assert _names(result) == ["both", "later"]
    assert result.skipped == []
    assert {u.name: u.input_types for u in result.udfs}["both"] == ["bigint", "bigint"]


def test_unkeyable_shapes_are_skipped_with_reasons():
    result = _extract(
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, LongType, StringType
        from mylib import imported_fn
        import numpy as np

        CFG = {"a": 1}

        def typed(x: int) -> int:
            return x

        def uses_dict(x: int) -> int:
            return CFG["a"] + x

        def uses_numpy(x: float) -> float:
            return np.sqrt(x)

        def untyped(x):
            return x

        def defaulted(x: int, y: int = 1) -> int:
            return x + y

        def helper(x: int) -> int:
            return x

        def calls_helper(x: int) -> int:
            return helper(x)

        a = udf(uses_dict, LongType())
        b = udf(uses_numpy, "double")
        c = udf(untyped, LongType())
        d = udf(lambda x: x + 1, LongType())
        e = udf(typed, ArrayType(StringType()))
        f = udf(typed, LongType()).asNondeterministic()
        g = udf(imported_fn, LongType())
        h = udf(defaulted, LongType())
        i = udf(calls_helper, LongType())
        j = udf(typed, LongType(), useArrow=True)

        def outer():
            k = 5

            def inner(x: int) -> int:
                return x + k

            return udf(inner, LongType())
        """
    )
    assert result.udfs == []
    reasons = [s.reason for s in result.skipped]
    for needle in [
        "'CFG', a dict",
        "non-stdlib module 'numpy'",
        "every parameter needs a supported type annotation",
        "lambda",
        "not a static atomic type",
        "nondeterministic",
        "defined in another module (mylib.imported_fn)",
        "default, variadic, or keyword-only",
        "module-level function 'helper'",
        "Arrow",
        "enclosing function",
    ]:
        assert any(needle in r for r in reasons), (needle, reasons)


def test_rebinding_and_global_are_not_static():
    result = _extract(
        """
        from pyspark.sql.functions import udf

        SCALE = 2
        SCALE = 3
        LIMIT = 10

        def bump():
            global LIMIT
            LIMIT += 1

        @udf("bigint")
        def scaled(x: int) -> int:
            return x * SCALE

        @udf("bigint")
        def limited(x: int) -> int:
            return min(x, LIMIT)
        """
    )
    skips = _skips(result)
    assert "bound 2 times" in skips["scaled"]
    assert "global" in skips["limited"]


def test_star_import_shadowed_builtins():
    result = _extract(
        """
        from pyspark.sql.functions import *

        @udf("double")
        def rounded(x: float) -> float:
            return round(x, 2)

        @udf("bigint")
        def length(s: str) -> int:
            return len(s)
        """
    )
    assert _names(result) == ["length"]
    assert "pyspark.sql.functions.round" in _skips(result)["rounded"]


def test_other_decorators_and_methods_are_skipped():
    result = _extract(
        """
        import functools
        from pyspark.sql.functions import udf

        @udf("bigint")
        @functools.lru_cache(maxsize=None)
        def cached(x: int) -> int:
            return x

        class Job:
            @udf("bigint")
            def method(x: int) -> int:
                return x
        """
    )
    skips = _skips(result)
    assert "something other than pyspark's udf" in skips["cached"]
    assert "method" in skips["method"]


def test_not_pyspark_udf_is_not_a_udf():
    result = _extract(
        """
        from mylib import udf

        @udf("bigint")
        def mine(x: int) -> int:
            return x
        """
    )
    assert result.udfs == [] and result.skipped == []


def test_syntax_error_reports_not_raises():
    result = _extract("def broken(:\n")
    assert result.error and "SyntaxError" in result.error
    assert len(result.sha256) == 64


def test_stdlib_captures_match_runtime_shape():
    result = _extract(
        """
        import math
        import os.path as osp
        import re as regex
        from math import pi
        from os import path
        from pyspark.sql.functions import udf

        PREFIX = "x-"

        @udf("double")
        def area(r: float) -> float:
            return pi * r * math.e

        @udf("string")
        def tagged(s: str) -> str:
            return PREFIX + regex.sub("a", "b", osp.basename(path.join(s, s)))
        """
    )
    caps = {u.name: u.captures for u in result.udfs}
    assert caps["area"] == {"math": {"__module__": "math"}, "pi": pytest.approx(3.141592653589793)}
    posix = sys.modules["os"].path.__name__  # posixpath / ntpath
    assert caps["tagged"] == {
        "PREFIX": "x-",
        "osp": {"__module__": posix},
        "path": {"__module__": posix},
        "regex": {"__module__": "re"},
    }


# -- decorator stripping (runtime side) ---------------------------------------


def test_strip_returns_same_node_when_nothing_to_drop():
    fn = ast.parse("def f(x: int) -> int:\n    return x").body[0]
    assert strip_udf_decorators(fn, lambda parts: True) is fn


def test_strip_never_mutates_input():
    fn = ast.parse("@udf('bigint')\ndef f(x: int) -> int:\n    return x").body[0]
    stripped = strip_udf_decorators(fn, lambda parts: parts == ["udf"])
    assert stripped is not fn
    assert len(fn.decorator_list) == 1 and stripped.decorator_list == []


# -- runtime parity -------------------------------------------------------------

_DDL = {"bigint": "bigint", "long": "bigint", "double": "double", "boolean": "boolean", "string": "string"}


def _stub_pyspark(monkeypatch):
    """Just enough pyspark for a job script to import and build its UDFs."""

    class _Type:
        _simple = ""

        def simpleString(self):
            return self._simple

    def _type(name, simple):
        return type(name, (_Type,), {"_simple": simple})

    types_mod = types.ModuleType("pyspark.sql.types")
    for name, simple in [
        ("LongType", "bigint"),
        ("DoubleType", "double"),
        ("StringType", "string"),
        ("BooleanType", "boolean"),
    ]:
        setattr(types_mod, name, _type(name, simple))

    created = []

    class _UDF:
        def __init__(self, func, return_type):
            if isinstance(return_type, str):
                return_type = _type("Parsed", _DDL[return_type])()
            self.func, self.returnType = func, return_type
            created.append(self)

    def udf(f=None, returnType=None, useArrow=None):
        rt = returnType if returnType is not None else types_mod.StringType()
        if f is None or isinstance(f, (str, _Type)):
            rt = f if f is not None else rt
            return lambda fn: _UDF(fn, rt)
        return _UDF(f, rt)

    udf.__module__ = "pyspark.sql.functions.builtin"
    functions_mod = types.ModuleType("pyspark.sql.functions")
    functions_mod.udf = udf
    sql_mod = types.ModuleType("pyspark.sql")
    sql_mod.functions = functions_mod
    sql_mod.types = types_mod
    root = types.ModuleType("pyspark")
    root.sql = sql_mod
    for name, mod in [
        ("pyspark", root),
        ("pyspark.sql", sql_mod),
        ("pyspark.sql.functions", functions_mod),
        ("pyspark.sql.types", types_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return created


def _runtime_key(func, return_type) -> str:
    """What AITranspiler._transpile_from_ast keys, given Spark's parse of ``func``."""
    src = textwrap.dedent(inspect.getsource(func)).strip()
    function_ast = udf_function_ast(func, ast.parse(src).body[0])
    params = [a.arg for a in function_ast.args.args]
    captures = extract_captures(func, function_ast, params)
    assert captures is not None, f"runtime would decline {func.__name__}"
    in_types = input_spark_types(function_ast, params)
    return udf_key(
        canonical_source(function_ast),
        params,
        in_types,
        return_type.simpleString(),
        SPARK_VERSION,
        closure_fingerprint(captures),
    )


PARITY_JOB = '''
import math
import re as regex
from math import pi

import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, DoubleType, LongType

SCALE = 3
RT = LongType()


@udf(returnType=LongType())
def plus_one(x: int) -> int:
    return x + 1


@F.udf("double")
def area(r: float) -> float:
    # comments and blank lines do not survive ast.unparse on either side

    return pi * r * r * SCALE


@udf
def shout(s: str) -> str:
    """Docstrings do."""
    return s.upper()


@udf(RT)
def doubled(x: int) -> int:
    return x * 2


def both(x: int, y: int) -> bool:
    return x > 0 and y > 0


def floor_count(x: float) -> float:
    return math.floor(x) + len(regex.findall("a", "aaa"))


both_udf = udf(both, BooleanType())
floor_udf = F.udf(f=floor_count, returnType=DoubleType())


def build():
    def nested(x: int) -> int:
        return x + SCALE

    return udf(nested, "bigint")


nested_udf = build()
'''


def test_static_keys_match_runtime_keys(tmp_path, monkeypatch):
    created = _stub_pyspark(monkeypatch)
    path = tmp_path / "parity_job.py"
    path.write_text(PARITY_JOB)
    spec = importlib.util.spec_from_file_location("parity_job", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "parity_job", module)
    spec.loader.exec_module(module)

    runtime = {u.func.__name__: _runtime_key(u.func, u.returnType) for u in created}
    static = {u.name: u.key(SPARK_VERSION) for u in extract_source(path.read_bytes(), str(path)).udfs}
    assert set(runtime) == {"plus_one", "area", "shout", "doubled", "both", "floor_count", "nested"}
    assert static == runtime


def test_runtime_keeps_a_non_pyspark_udf_decorator(monkeypatch):
    """A user's own ``udf`` decorator may change behavior: never strip it."""
    ns: dict = {}
    exec(
        compile("def udf(f):\n    return f\n", "<mine>", "exec"),
        ns,
    )
    src = "@udf\ndef g(x: int) -> int:\n    return x\n"
    exec(compile(src, "<mine>", "exec"), ns)
    fn_ast = ast.parse(src).body[0]
    assert udf_function_ast(ns["g"], fn_ast) is fn_ast


def test_return_type_resolves_in_the_call_scope_not_the_def_scope():
    result = _extract(
        """
        from pyspark.sql.functions import udf

        RT = "string"

        def f(x: int) -> int:
            return x

        def g(x: int) -> int:
            return x - 1

        def build():
            RT = "bigint"
            return udf(f, RT)

        def build_local_only(spark):
            LOCAL_RT = "bigint"
            spark.udf.register("g", g, LOCAL_RT)
        """
    )
    assert result.skipped == []
    assert {u.name: u.return_type for u in result.udfs} == {"f": "bigint", "g": "bigint"}


def test_only_vetted_modules_are_ever_imported(monkeypatch):
    """`import unittest.__main__ as runner` would start a test runner on import."""
    imported = []
    real = __import__("importlib").import_module
    monkeypatch.setattr(
        "ai_udf_transpile.extract.importlib.import_module", lambda n: imported.append(n) or real(n)
    )
    result = _extract(
        """
        import unittest.__main__ as runner
        from pyspark.sql.functions import udf

        @udf("bigint")
        def f(x: int) -> int:
            return runner and x
        """
    )
    assert imported == []
    assert "not on the vetted stdlib import list" in _skips(result)["f"]


def test_type_constructors_resolve_through_imports():
    result = _extract(
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        def LongType():
            return StringType()

        @udf(LongType())
        def shadowed(x: int) -> int:
            return x

        @udf(DecimalType(10, 2))
        def unimported(x: float) -> float:
            return x
        """
    )
    assert result.udfs == []
    assert set(_skips(result)) == {"shadowed", "unimported"}


def test_assigned_return_type_resolves_in_the_assignments_scope():
    result = _extract(
        """
        from pyspark.sql.functions import udf

        BASE = "bigint"
        RT = BASE

        def f(x: int) -> int:
            return x

        def build():
            BASE = "string"
            return udf(f, RT)
        """
    )
    assert [(u.name, u.return_type) for u in result.udfs] == [("f", "bigint")]


def test_unpacked_arguments_are_not_static():
    result = _extract(
        """
        from pyspark.sql.functions import udf

        OPTIONS = {"returnType": "bigint"}

        def f(x: int) -> int:
            return x

        def g(x: int) -> int:
            return x

        a = udf(f, **OPTIONS)
        b = udf(*(g, "bigint"))

        @udf(**OPTIONS)
        def h(x: int) -> int:
            return x

        def main(spark):
            spark.udf.register("f2", f, **OPTIONS)
        """
    )
    assert result.udfs == []
    assert len(result.skipped) == 4
    assert all("unpacked" in s.reason for s in result.skipped)


def test_non_literal_use_arrow_is_skipped():
    result = _extract(
        """
        from pyspark.sql.functions import udf

        ARROW = True

        def f(x: int) -> int:
            return x

        a = udf(f, "bigint", useArrow=ARROW)
        b = udf(f, "double", useArrow=False)
        """
    )
    assert [(u.name, u.return_type) for u in result.udfs] == [("f", "double")]
    assert "useArrow is not a literal" in result.skipped[0].reason


def test_udf_imported_inside_a_factory_is_skipped():
    """Decorators run in the enclosing scope; the hook only sees globals."""
    result = _extract(
        """
        def build():
            from pyspark.sql.functions import udf

            @udf("bigint")
            def inner(x: int) -> int:
                return x + 1

            return inner
        """
    )
    assert result.udfs == []
    assert "imported inside the enclosing function" in _skips(result)["inner"]


def test_a_later_star_import_may_shadow_an_explicit_binding():
    result = _extract(
        """
        from pyspark.sql.types import LongType

        LIMIT = 10

        from my_types import *  # may export LongType (or LIMIT) of its own
        from pyspark.sql.functions import udf

        @udf(LongType())
        def typed(x: int) -> int:
            return x

        @udf("bigint")
        def limited(x: int) -> int:
            return x + LIMIT
        """
    )
    skips = _skips(result)
    assert result.udfs == []
    assert "not a static atomic type" in skips["typed"]
    assert "later `import *`" in skips["limited"]


def test_a_later_pyspark_star_import_keeps_pyspark_bindings():
    """`from pyspark.sql.functions import *` after explicit imports is routine."""
    result = _extract(
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import LongType
        from pyspark.sql.functions import *
        from pyspark.sql.types import *

        @udf(LongType())
        def f(x: int) -> int:
            return x
        """
    )
    assert [(u.name, u.return_type) for u in result.udfs] == [("f", "bigint")]
