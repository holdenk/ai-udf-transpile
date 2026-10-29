# SPDX-License-Identifier: Apache-2.0
"""Coverage-gap tests for bytecode.py, bytecode_eval.py, sampling.py,
transpiler.py, sandbox.py. Fills decline/fallback paths the existing
backend tests don't reach -- see test_bytecode_backend.py /
test_bytecode_eval.py for the happy-path tests these sit next to."""

from __future__ import annotations

import ast
import dis
import json
import sys
import types
from types import SimpleNamespace

import pytest

import ai_udf_transpile.backends.bytecode as bytecode_mod
import ai_udf_transpile.backends.bytecode_eval as bytecode_eval_mod
import ai_udf_transpile.sampling as sampling
import ai_udf_transpile.transpiler as transpiler_mod
from ai_udf_transpile.backends.base import BackendDecline
from ai_udf_transpile.backends.bytecode import BytecodeBackend, _Evaluator
from ai_udf_transpile.backends.bytecode_eval import emit_java_eval, emit_java_eval_source, opcode_numbers
from ai_udf_transpile.backends.fake import plus_one
from ai_udf_transpile.catalog import HIT, CacheRow
from ai_udf_transpile.sandbox import disassembly_text, job_appendix, make_sandbox
from ai_udf_transpile.targets import TranspileJob

_DUMMY_CODE = compile("def _f(): pass", "<t>", "exec")


def _ins(opname, argval=None, argrepr="", arg=None, offset=0):
    return SimpleNamespace(opname=opname, argval=argval, argrepr=argrepr, arg=arg, offset=offset)


def _job_for(func, **kwargs) -> TranspileJob:
    src = kwargs.get("source_text")
    if src is None:
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(func))
    params = kwargs.get("param_names", list(func.__code__.co_varnames[: func.__code__.co_argcount]))
    return TranspileJob(
        udf_key="k",
        source_text=src,
        param_names=params,
        input_types=kwargs.get("input_types", ["bigint"] * len(params)),
        input_categories=kwargs.get("input_categories", ["numeric"] * len(params)),
        return_type=kwargs.get("return_type", "bigint"),
        captures={},
    )


def _run(func, **kwargs) -> str:
    backend = BytecodeBackend()
    job = _job_for(func, **kwargs)
    with make_sandbox(job) as sandbox:
        return backend.run(job, sandbox).sql


# --- bytecode.py ---


def test_varchar_type_is_string_truthiness():
    def f(s) -> int:
        return 1 if s else 0

    sql = _run(f, input_types=["varchar(20)"], return_type="bigint")
    assert "length(_udf_param_0) > 0" in sql


def test_unparseable_source_declines():
    job = _job_for(plus_one, source_text="x = 1\n", param_names=["x"])
    with pytest.raises(BackendDecline, match="could not parse source text"):
        BytecodeBackend().run(job, make_sandbox(job).__enter__())


def test_no_function_code_object_declines(monkeypatch):
    monkeypatch.setattr(bytecode_mod, "canonical_source_text", lambda src: "x = 1")
    job = _job_for(plus_one)
    with pytest.raises(BackendDecline, match="no function code object"):
        BytecodeBackend().run(job, make_sandbox(job).__enter__())


def test_no_return_path_declines(monkeypatch):
    monkeypatch.setattr(bytecode_mod._Evaluator, "evaluate", lambda self, instrs: None)
    job = _job_for(plus_one)
    with pytest.raises(BackendDecline, match="no return path"):
        BytecodeBackend().run(job, make_sandbox(job).__enter__())


def test_fallthrough_with_no_return_yields_top_of_stack():
    # Mirrors the shape a branch's shared tail produces: a walk that ends
    # without hitting RETURN returns whatever is on top of the stack.
    code = compile("def f(x):\n    return x + 1\n", "<t>", "exec")
    func_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    instrs = [i for i in dis.get_instructions(func_code) if i.opname != "RETURN_VALUE"]
    ev = _Evaluator(["x"], func_code, ["numeric"])
    assert ev.evaluate(instrs) == ("(_udf_param_0 + 1)", "numeric")


def test_load_fast_load_fast_without_name_declines():
    ev = _Evaluator(["x", "y"], _DUMMY_CODE, ["numeric", "numeric"])
    ev._idx = {}
    ev._budget = 10
    with pytest.raises(BackendDecline, match="without name"):
        ev._eval_range([_ins("LOAD_FAST_LOAD_FAST", argval="x")], 0, 1, [])


def test_load_fast_unknown_var_declines():
    code = compile("def f(x):\n    return x\n", "<t>", "exec")
    func_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    ev = _Evaluator(["y"], func_code, ["numeric"])
    with pytest.raises(BackendDecline, match="unknown var"):
        ev.evaluate(list(dis.get_instructions(func_code)))


def test_unsupported_compare_op_declines():
    ev = _Evaluator(["x"], _DUMMY_CODE, ["numeric"])
    ev._idx = {0: 0}
    ev._budget = 10
    stack = [("_udf_param_0", "numeric"), ("5", "numeric")]
    with pytest.raises(BackendDecline, match="unsupported COMPARE_OP"):
        ev._eval_range([_ins("COMPARE_OP", argval="in")], 0, 1, stack)


def test_is_op_none_check_lowers_to_is_null():
    def f(x: int) -> bool:
        return x is None

    assert _run(f, return_type="boolean") == "(_udf_param_0 IS NULL)"


def test_is_op_none_first_operand_lowers_to_is_null():
    def f(x: int) -> bool:
        return None is x

    assert _run(f, return_type="boolean") == "(_udf_param_0 IS NULL)"


def test_is_op_on_non_none_declines():
    def f(x: int, y: int) -> bool:
        return x is y

    with pytest.raises(BackendDecline, match="IS_OP on non-None"):
        _run(f, input_types=["bigint", "bigint"], return_type="boolean")


def test_jump_target_out_of_range_declines():
    ev = _Evaluator(["x"], _DUMMY_CODE, ["bool"])
    ev._idx = {0: 0}
    ev._budget = 10
    with pytest.raises(BackendDecline, match="out of range"):
        ev._eval_range([_ins("POP_JUMP_IF_FALSE", argval=99, offset=0)], 0, 1, [("_udf_param_0", "bool")])


def test_branch_with_no_return_declines(monkeypatch):
    ev = _Evaluator(["x"], _DUMMY_CODE, ["bool"])
    ev._idx = {0: 0, 1: 1}
    results = iter([None, ("true", "bool")])
    monkeypatch.setattr(ev, "_eval_range", lambda *a, **k: next(results))
    with pytest.raises(BackendDecline, match="branch with no return"):
        ev._branch(
            [_ins("POP_JUMP_IF_FALSE", argval=1, offset=0), _ins("NOP", offset=1)],
            0,
            2,
            [],
            "_udf_param_0",
            False,
        )


def test_truthiness_on_other_kind_declines():
    def f(x) -> int:
        return 1 if x else 2

    with pytest.raises(BackendDecline, match="no truthiness SQL"):
        _run(f, input_types=["timestamp"], return_type="bigint")


def test_true_false_consts():
    def f(x: int) -> bool:
        return True if x else False

    sql = _run(f, return_type="boolean")
    assert sql == "CASE WHEN (_udf_param_0 IS NOT NULL AND _udf_param_0 <> 0) THEN true ELSE false END"


def test_negative_infinite_float_const_emits_cast():
    def f(x: int) -> float:
        return x * -1e500

    assert _run(f, return_type="double") == "(_udf_param_0 * cast('-Infinity' as double))"


def test_tuple_const_declines():
    def f(x: int) -> bool:
        return x in (1, 2, 3)

    with pytest.raises(BackendDecline, match="unsupported const type"):
        _run(f, return_type="boolean")


# --- bytecode_eval.py ---


def test_opcode_numbers_too_few_constants_declines():
    with pytest.raises(BackendDecline, match="missing opcode constants"):
        opcode_numbers(java_source="public static final int OP_FOO = 1;\n")


def test_eval_unparseable_source_declines():
    with pytest.raises(BackendDecline, match="could not parse source text"):
        emit_java_eval_source("x = 1\n")


def test_eval_no_function_code_object_declines(monkeypatch):
    monkeypatch.setattr(bytecode_eval_mod, "canonical_source_text", lambda src: "x = 1")
    with pytest.raises(BackendDecline, match="no function code object"):
        emit_java_eval_source("def f(x: int) -> int:\n    return x\n")


def test_eval_try_except_declines():
    src = "def f(x: int) -> int:\n    try:\n        return x + 1\n    except Exception:\n        return 0\n"
    with pytest.raises(BackendDecline, match="try/except"):
        emit_java_eval_source(src)


def test_eval_zero_arity_declines():
    def f() -> int:
        return 1

    with pytest.raises(BackendDecline, match="arity"):
        emit_java_eval(f)


def test_eval_unary_negative():
    def f(x: int) -> int:
        return -x

    ops = opcode_numbers()
    code = emit_java_eval(f).code
    assert ops["OP_UNARY_NEG"] in code


def test_lower_one_load_fast_load_fast_without_name_declines():
    with pytest.raises(BackendDecline, match="without name"):
        bytecode_eval_mod._lower_one(_ins("LOAD_FAST_LOAD_FAST", argval="x"), ["x", "y"], [])


def test_lower_one_unsupported_compare_op_declines():
    with pytest.raises(BackendDecline, match="unsupported COMPARE_OP"):
        bytecode_eval_mod._lower_one(_ins("COMPARE_OP", argval="in", argrepr="in"), ["x"], [])


def test_jump_origin_out_of_range_declines():
    with pytest.raises(BackendDecline, match="out of range"):
        bytecode_eval_mod._jump_origin(999, [])


def test_local_unknown_var_declines():
    with pytest.raises(BackendDecline, match="unknown var"):
        bytecode_eval_mod._local("z", ["x", "y"])


def test_java_const_bool_and_long_edges():
    assert bytecode_eval_mod._java_const(True) == "Boolean.TRUE"
    assert bytecode_eval_mod._java_const(False) == "Boolean.FALSE"
    assert bytecode_eval_mod._java_const(-(2**63)) == "Long.MIN_VALUE"
    with pytest.raises(BackendDecline, match="does not fit long"):
        bytecode_eval_mod._java_const(2**70)


def test_java_const_float_specials():
    assert bytecode_eval_mod._java_const(float("nan")) == "Double.NaN"
    assert bytecode_eval_mod._java_const(float("inf")) == "Double.POSITIVE_INFINITY"
    assert bytecode_eval_mod._java_const(float("-inf")) == "Double.NEGATIVE_INFINITY"
    assert bytecode_eval_mod._java_const(3.5) == "Double.valueOf(3.5d)"


def test_java_const_unsupported_type_declines():
    with pytest.raises(BackendDecline, match="unsupported const type"):
        bytecode_eval_mod._java_const([1, 2])


def test_java_string_escapes_all_special_chars():
    def f(s: str) -> str:
        return s + '\\"\n\r\t\x01'

    java = emit_java_eval(f).java_source
    assert "\\\\" in java
    assert '\\"' in java
    assert "\\n" in java
    assert "\\r" in java
    assert "\\t" in java
    assert "\\u0001" in java


def test_interpreter_body_missing_class_declines(tmp_path, monkeypatch):
    bogus = tmp_path / "Bogus.java"
    bogus.write_text("// no class here\n", encoding="utf-8")
    monkeypatch.setattr(bytecode_eval_mod, "_JAVA_PATH", bogus)
    with pytest.raises(BackendDecline, match="no class"):
        bytecode_eval_mod._interpreter_body()


def test_render_three_args_uses_evaln():
    def f(a: int, b: int, c: int) -> int:
        return a + b + c

    assert "Eval.evalN" in emit_java_eval(f).java_source


# --- sampling.py ---


def test_encode_bytes():
    assert sampling._encode(b"\x01\x02") == {"__bytes__": "0102"}


def test_encode_unsupported_type_falls_back_to_repr():
    class Weird:
        def __repr__(self):
            return "<Weird>"

    assert sampling._encode(Weird()) == {"__repr__": "<Weird>"}


@pytest.mark.parametrize(
    "item",
    [
        {"__bytes__": "zz"},
        {"__map__": [["k", {"__bytes__": "zz"}]]},
        {"__datetime__": "not-a-date"},
        {"__array__": [{"__bytes__": "zz"}]},
        {"__tuple__": [{"__bytes__": "zz"}]},
        {"__repr__": "<object at 0x1>"},
    ],
)
def test_decode_args_undecodable_item_returns_none(item):
    assert sampling.decode_args(json.dumps([item])) is None


def test_decode_args_invalid_json_returns_none():
    assert sampling.decode_args("not json") is None


def test_decode_args_non_list_returns_none():
    assert sampling.decode_args("{}") is None


def test_record_samples_without_sqlite_path_is_noop():
    assert sampling.record_samples("", "k", [["a"]]) is None


def test_record_samples_bad_path_never_raises(tmp_path):
    bad = tmp_path / "missing_dir" / "db.sqlite"
    sampling.record_samples(str(bad), "k", [["a"]])
    assert not bad.exists()


def test_maybe_record_never_raises_on_noniterable_args():
    assert sampling.maybe_record("db", "k", 123, 5) is None


def test_udf_instance_from_spark_stack_no_frame_returns_none():
    assert sampling.udf_instance_from_spark_stack() is None


def test_udf_instance_import_failure_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyspark.sql.udf", None)
    assert sampling.udf_instance_from_spark_stack() is None


# --- transpiler.py ---


def test_pin_session_none_is_noop():
    assert transpiler_mod._pin_session(None) is None


def test_pin_session_swallows_exception(monkeypatch):
    # _pin_session sets SparkSession._activeSession/_instantiatedSession
    # directly (not via a fixture) before the call that raises, so a bare
    # call here would leak a fake session into every later spark-marked
    # test in this process. Snapshot/restore those two class attrs.
    from pyspark.sql import SparkSession

    monkeypatch.setattr(SparkSession, "_activeSession", None, raising=False)
    monkeypatch.setattr(SparkSession, "_instantiatedSession", None, raising=False)

    class FakeSpark:
        _jvm = object()
        _jsparkSession = object()

    transpiler_mod._pin_session(FakeSpark())  # must not raise


def test_spark_version_none():
    assert transpiler_mod._spark_version(None) == "unknown"


def test_spark_version_raises_falls_back():
    class Bad:
        @property
        def version(self):
            raise RuntimeError("boom")

    assert transpiler_mod._spark_version(Bad()) == "unknown"


def test_reconstruct_column_unknown_target_kind_returns_none():
    row = SimpleNamespace(target_kind="unknown", catalyst_sql=None, param_names=[])
    assert transpiler_mod._reconstruct_column(row, "bigint", None) is None


def test_reconstruct_column_java_udf_without_class_returns_none():
    row = SimpleNamespace(
        target_kind="java_udf",
        catalyst_sql=None,
        param_names=["x"],
        impl_class=None,
        impl_binary=None,
        impl_source="public class Foo {}",
    )
    assert transpiler_mod._reconstruct_column(row, "bigint", None) is None


class _FakeConf:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeSpark:
    def __init__(self, values):
        self.conf = _FakeConf(values)


def test_maybe_wrap_sampling_disabled_returns_early(monkeypatch):
    called = []
    monkeypatch.setattr(sampling, "udf_instance_from_spark_stack", lambda: called.append(True))
    spark = _FakeSpark({transpiler_mod.conf.SAMPLING: "false"})
    transpiler_mod._maybe_wrap_sampling(spark, "k", ["string"])
    assert called == []


def test_maybe_wrap_sampling_non_sqlite_catalog_returns_early(monkeypatch):
    called = []
    monkeypatch.setattr(sampling, "udf_instance_from_spark_stack", lambda: called.append(True))
    spark = _FakeSpark({transpiler_mod.conf.SAMPLING: "true", transpiler_mod.conf.CATALOG: "other"})
    transpiler_mod._maybe_wrap_sampling(spark, "k", ["string"])
    assert called == []


def test_maybe_wrap_sampling_no_instance_returns_early(monkeypatch):
    monkeypatch.setattr(sampling, "udf_instance_from_spark_stack", lambda: None)

    def boom(*a, **k):
        raise AssertionError("should not wrap when there is no live UDF instance")

    monkeypatch.setattr(sampling, "wrap_for_sampling", boom)
    spark = _FakeSpark({transpiler_mod.conf.SAMPLING: "true", transpiler_mod.conf.CATALOG: "sqlite"})
    transpiler_mod._maybe_wrap_sampling(spark, "k", ["string"])  # must not raise


pytest.importorskip("pyspark")


def test_transpile_from_ast_type_mismatch_returns_none(monkeypatch):
    monkeypatch.setattr(transpiler_mod, "_SESSION_REF", None)
    monkeypatch.setattr(transpiler_mod, "return_spark_type", lambda rt: None)
    cls = transpiler_mod.get_transpiler_class()
    tree = ast.parse("def f(x: int) -> int:\n    return x + 1\n")
    fn = tree.body[0]
    assert cls()._transpile_from_ast(None, tree, fn, ["x"], "bigint") is None


def test_transpile_from_ast_unresolvable_capture_returns_none(monkeypatch):
    monkeypatch.setattr(transpiler_mod, "_SESSION_REF", None)
    cls = transpiler_mod.get_transpiler_class()
    tree = ast.parse("def f(x: int) -> int:\n    return SOME_UNDEFINED_GLOBAL + x\n")
    fn = tree.body[0]
    assert cls()._transpile_from_ast(None, tree, fn, ["x"], "bigint") is None


def test_transpile_from_ast_no_catalog_returns_none(monkeypatch):
    monkeypatch.setattr(transpiler_mod, "_SESSION_REF", None)
    monkeypatch.setattr(transpiler_mod, "_CATALOG", None)
    cls = transpiler_mod.get_transpiler_class()
    tree = ast.parse("def f(x: int) -> int:\n    return x + 1\n")
    fn = tree.body[0]
    assert cls()._transpile_from_ast(None, tree, fn, ["x"], "bigint") is None


def test_transpile_from_ast_swallows_unexpected_exception(monkeypatch):
    monkeypatch.setattr(transpiler_mod, "_SESSION_REF", None)
    monkeypatch.setattr(transpiler_mod, "_CATALOG", None)
    cls = transpiler_mod.get_transpiler_class()
    assert cls()._transpile_from_ast(None, None, None, ["x"], "bigint") is None


class _FakeHitCatalog:
    def __init__(self, row):
        self._row = row

    def lookup(self, udf_key, *, spark=None):
        return HIT, self._row


def test_transpile_from_ast_reconstruction_failure_returns_none(monkeypatch):
    monkeypatch.setattr(transpiler_mod, "_SESSION_REF", None)
    row = CacheRow(
        udf_key="k",
        param_names=["x"],
        target_kind="catalyst",
        catalyst_sql="_udf_param_0 + 1",
        status="success",
    )
    monkeypatch.setattr(transpiler_mod, "_CATALOG", _FakeHitCatalog(row))
    cls = transpiler_mod.get_transpiler_class()
    tree = ast.parse("def f(x: int) -> int:\n    return x + 1\n")
    fn = tree.body[0]
    # F.expr(...) raises (no live SparkContext when run alone; forced here
    # so it's deterministic when a prior spark-marked test leaves one up),
    # and the inner try/except must swallow it rather than propagate.
    import pyspark.sql.functions as F

    monkeypatch.setattr(F, "expr", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no column")))
    assert cls()._transpile_from_ast(None, tree, fn, ["x"], "bigint") is None


# --- sandbox.py ---


def test_disassembly_text_syntax_error_returns_empty():
    assert disassembly_text("def f(:\n") == ""


def test_disassembly_text_no_function_returns_empty():
    assert disassembly_text("x = 1\n") == ""


def test_job_appendix_bytecode_mode_with_no_function_omits_listing():
    job = TranspileJob(
        udf_key="k",
        source_text="x = 1\n",
        param_names=[],
        input_types=[],
        input_categories=[],
        return_type="bigint",
        captures={},
    )
    body = job_appendix(job, inputs="bytecode")
    # The file *sections* are correctly gated; only the intro sentence's
    # "on disk as ..." listing is not (see BUG note below).
    assert "### disassembly.txt" not in body
    assert "### udf.py" not in body
