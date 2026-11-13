# SPDX-License-Identifier: Apache-2.0
"""Manual approval gate for AI rewrites.

Hypothesis is value-equality over sampled inputs, and some UDFs are much
harder to sample than others: string columns that hold dates/JSON/numbers,
nested struct/map/array values, and UDFs that call a network service
(verification cannot refute a service it cannot reach). Set
``spark.sql.experimental.aiUdfTranspile.requireApproval`` and a rewrite that
passes every automated check parks in ``awaiting_approval`` instead of being
served; a person approves or rejects it with ``python -m
ai_udf_transpile.approve``.

Conf values (comma-separated, case-insensitive):

- ``none`` / ``false`` / empty (default): no manual approval.
- ``all`` / ``true``: every AI rewrite.
- any of ``string``, ``struct``, ``map``, ``array``, ``binary``,
  ``timestamp``: rewrites whose input or return type contains that type.
- ``network``: rewrites of UDFs whose source imports or references a
  network client (boto3, requests, socket, urllib, ...).
- ``hard``: shorthand for ``string,struct,map,array,network``.

Human rewrites (``register_impl``) and the deterministic bytecode backend
are never gated: a person or no AI wrote them.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable, Optional

from ai_udf_transpile import conf

# Origins that are not AI output.
EXEMPT_ORIGINS = frozenset({"human", "bytecode"})

TYPE_TRIGGERS = ("string", "struct", "map", "array", "binary", "timestamp")
NETWORK = "network"
ALL = "all"
HARD = ("string", "struct", "map", "array", NETWORK)
KNOWN_TRIGGERS = frozenset(TYPE_TRIGGERS) | {NETWORK, ALL}

_OFF = {"", "none", "false", "off", "0", "no"}
_ON = {"all", "true", "on", "1", "yes"}

_TYPE_PATTERNS = {
    "string": re.compile(r"\bstring\b"),
    "struct": re.compile(r"\bstruct\s*<"),
    "map": re.compile(r"\bmap\s*<"),
    "array": re.compile(r"\barray\s*<"),
    "binary": re.compile(r"\bbinary\b"),
    "timestamp": re.compile(r"\btimestamp(?:_ntz|_ltz)?\b"),
}

# Top-level module names whose use means the UDF talks to something off-box.
# Denylist, so a client not named here is not caught: extend it, or use
# requireApproval=all when unsure.
NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "anthropic",
        "azure",
        "boto3",
        "botocore",
        "cassandra",
        "confluent_kafka",
        "elasticsearch",
        "ftplib",
        "google",  # google.cloud.*, google.auth
        "grpc",
        "http",
        "httplib2",
        "httpx",
        "imaplib",
        "kafka",
        "ldap3",
        "openai",
        "paramiko",
        "poplib",
        "psycopg",
        "psycopg2",
        "pymongo",
        "pymysql",
        "redis",
        "requests",
        "smtplib",
        "snowflake",
        "socket",
        "sqlalchemy",
        "ssl",
        "telnetlib",
        "urllib",
        "urllib3",
        "websocket",
        "websockets",
        "xmlrpc",
    }
)

# Calls that reach the network without an obvious module import in the
# function body (e.g. a module-level ``from urllib.request import urlopen``).
_NETWORK_CALLS = frozenset({"urlopen", "create_connection", "getaddrinfo", "gethostbyname"})


def parse_triggers(raw: Optional[str]) -> frozenset[str]:
    """Normalized trigger set for a ``requireApproval`` value. Unknown tokens are ignored."""
    out: set[str] = set()
    for part in (raw or "").split(","):
        token = part.strip().lower()
        if token in _OFF:
            continue
        if token in _ON:
            out.add(ALL)
        elif token == "hard":
            out.update(HARD)
        elif token in KNOWN_TRIGGERS:
            out.add(token)
    return frozenset(out)


def configured_triggers(spark: Any = None) -> frozenset[str]:
    return parse_triggers(conf.get_value(conf.REQUIRE_APPROVAL, spark, conf.DEFAULTS[conf.REQUIRE_APPROVAL]))


def _root(name: str) -> str:
    return name.split(".", 1)[0]


def network_modules_used(source_text: str, captures: Optional[dict[str, Any]] = None) -> list[str]:
    """Network-client modules/calls a UDF's source or captures reference (sorted, deduped)."""
    found: set[str] = set()
    for value in (captures or {}).values():
        if isinstance(value, dict) and isinstance(value.get("__module__"), str):
            if _root(value["__module__"]) in NETWORK_MODULES:
                found.add(value["__module__"])
    try:
        tree = ast.parse(source_text or "")
    except SyntaxError:
        # Unparseable source: fall back to a token scan rather than say "no".
        for word in re.findall(r"[A-Za-z_][\w.]*", source_text or ""):
            if _root(word) in NETWORK_MODULES or word in _NETWORK_CALLS:
                found.add(word)
        return sorted(found)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) in NETWORK_MODULES:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and _root(node.module) in NETWORK_MODULES:
                found.add(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in _NETWORK_CALLS:
                found.add(name)
            if name == "__import__" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if _root(arg.value) in NETWORK_MODULES:
                        found.add(arg.value)
        elif isinstance(node, ast.Name) and node.id in NETWORK_MODULES:
            # Module-level import used as a free name (requests.get(...)).
            found.add(node.id)
    return sorted(found)


def approval_reasons(
    *,
    input_types: Iterable[str],
    return_type: str,
    source_text: str = "",
    captures: Optional[dict[str, Any]] = None,
    origin: Optional[str] = None,
    spark: Any = None,
    triggers: Optional[frozenset[str]] = None,
) -> list[str]:
    """Why this rewrite needs a person to sign off. Empty list: it does not."""
    if origin in EXEMPT_ORIGINS:
        return []
    active = configured_triggers(spark) if triggers is None else triggers
    if not active:
        return []
    reasons: list[str] = []
    if ALL in active:
        reasons.append("requireApproval=all")
    types = [str(t).strip().lower() for t in input_types] + [str(return_type or "").strip().lower()]
    for trigger in TYPE_TRIGGERS:
        if trigger in active and any(_TYPE_PATTERNS[trigger].search(t) for t in types):
            reasons.append(f"{trigger} type")
    if NETWORK in active:
        used = network_modules_used(source_text, captures)
        if used:
            reasons.append("network call (" + ", ".join(used) + ")")
    return reasons


def row_approval_reasons(row: Any, spark: Any = None, *, origin: Optional[str] = None) -> list[str]:
    """``approval_reasons`` for a CacheRow. ``origin`` overrides the row's (worker path)."""
    return approval_reasons(
        input_types=row.input_types,
        return_type=row.return_type,
        source_text=row.source_text,
        captures=row.captures,
        origin=origin if origin is not None else row.origin,
        spark=spark,
    )


def blocks_serving(row: Any, spark: Any = None) -> bool:
    """A success row that approval policy says must not be served yet."""
    return not getattr(row, "reviewed_at", None) and bool(row_approval_reasons(row, spark))
