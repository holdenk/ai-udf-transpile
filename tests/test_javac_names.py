# SPDX-License-Identifier: Apache-2.0
"""javac.unique_class_name: the name a Java rewrite is compiled under. No JVM;
the collision it prevents is pinned in test_java_udf.py."""

from __future__ import annotations

from ai_udf_transpile.javac import _rename_class, unique_class_name


def test_unique_class_name_is_a_function_of_the_source():
    src = (
        "package ai_udf;\n// Same is named in a comment\n"
        "public class Same implements UDF1<Object, Object> {\n"
        '    static final String N = "Same";\n    public Same() {}\n'
        "    static Same make() { return new Same(); }\n"
        "    public Object call(Object s) { return SameOther.X + s; }\n}\n"
    )
    name = unique_class_name(src)
    assert name.startswith("ai_udf.Same_") and name != "ai_udf.Same"
    # Either name handed back in for the same source gives the same answer...
    assert unique_class_name(src, "ai_udf.Same") == name == unique_class_name(src, name)
    # ...and different source gives a different one.
    assert unique_class_name(src.replace("return SameOther", "return  SameOther")) != name
    renamed = _rename_class(src, "Same", name.rpartition(".")[2])
    simple = name.rpartition(".")[2]
    assert f"public class {simple} " in renamed and f"public {simple}()" in renamed
    assert f"new {simple}()" in renamed and f"static {simple} make" in renamed
    # Comments, string literals and longer identifiers are left alone.
    assert "// Same is named" in renamed and '"Same"' in renamed and "SameOther.X" in renamed
