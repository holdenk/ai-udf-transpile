# SPDX-License-Identifier: Apache-2.0
"""The default backend prompt must keep the hard-won rules.

Each assertion maps to a failure observed with a real backend: lambda-closing
placeholders crash at analysis (coco), bare boolean NULL traps pass vacuously
in WHERE but not as columns, typed Java generics fail under Janino, and
side-effecting UDFs must be declined rather than hallucinated about.
"""

from ai_udf_transpile.sandbox import default_prompt


def test_default_prompt_covers_hard_won_rules():
    prompt = default_prompt()
    # Placeholders and their lambda restriction (UNRESOLVED_COLUMN crash).
    assert "_udf_param_0" in prompt
    assert "lambda" in prompt
    # NULL-vs-False guidance with a worked coalesce example.
    assert "coalesce" in prompt
    # Output discipline and the decline escape hatch.
    assert "DECLINE" in prompt
    assert "OUT.sql" in prompt and "OUT.java" in prompt
    # Java target: erased signatures for Janino, boxed arrival types.
    assert "Janino" in prompt
    assert "UDF1<Object, Object>" in prompt
    assert "scala.collection.Seq" in prompt  # array<string> arrival
    # Verification notice and side-effect decline guidance.
    assert "differentially tested" in prompt
    assert "boto3" in prompt
