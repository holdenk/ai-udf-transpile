# SPDX-License-Identifier: Apache-2.0
"""Walk the rewrites parked in ``awaiting_approval`` and ask a person about each.

  python -m ai_udf_transpile.approve --sqlite-path /tmp/ai_udf_transpile.sqlite
  python -m ai_udf_transpile.approve --sqlite-path ... --list
  python -m ai_udf_transpile.approve --sqlite-path ... --approve 3fa2c1 --reviewer alice
  python -m ai_udf_transpile.approve --sqlite-path ... --reject 3fa2c1 --reason "drops the tz"

Rows land here when ``spark.sql.experimental.aiUdfTranspile.requireApproval``
covers them (see ``ai_udf_transpile.approval``). Approving serves the rewrite
(and, with a write-back table, re-publishes it carrying the sign-off on the
next flush); rejecting fails it with the reviewer's reason, keeping the
candidate for the record. A rejected UDF goes back through the backend after
the usual failCooldownSeconds, bounded by maxRetries.

Keys may be given as any unique prefix of the full udf_key.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import textwrap
from typing import Any, Callable, Optional, TextIO

from ai_udf_transpile.approval import approval_reasons, parse_triggers
from ai_udf_transpile.catalog import CacheRow

_SAMPLES_SHOWN = 3


def default_reviewer() -> str:
    for var in ("AI_UDF_TRANSPILE_REVIEWER", "USER", "USERNAME"):
        value = os.environ.get(var)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _indent(text: str) -> str:
    return textwrap.indent((text or "").rstrip() or "(empty)", "    ")


def _reasons(row: CacheRow, triggers: Any) -> list[str]:
    # Reasons under the triggers the reviewer passed (or "hard" by default),
    # so a row parked under requireApproval=all still says what is risky about it.
    return approval_reasons(
        input_types=row.input_types,
        return_type=row.return_type,
        source_text=row.source_text,
        captures=row.captures,
        origin=row.origin,
        triggers=triggers,
    )


def describe(row: CacheRow, catalog: Any = None, *, triggers: Any = None) -> str:
    """Human-readable summary of one parked rewrite."""
    triggers = parse_triggers("hard") if triggers is None else triggers
    params = ", ".join(f"{p}: {t}" for p, t in zip(row.param_names, row.input_types))
    lines = [
        f"udf_key   {row.udf_key}",
        f"signature ({params}) -> {row.return_type}",
        f"origin    {row.origin or '?'}  backend {row.backend or '?'}  model {row.model or '?'}",
        f"attempts  {row.attempt_count}  tolerance {row.tolerance if row.tolerance is not None else 0}"
        f"  verified-at {row.updated_at or '?'}",
    ]
    reasons = [r for r in _reasons(row, triggers) if r != "requireApproval=all"]
    lines.append("flags     " + ("; ".join(reasons) if reasons else "none (requireApproval=all)"))
    if row.captures:
        lines.append(f"captures  {row.captures}")
    lines.append("python:")
    lines.append(_indent(row.source_text))
    if row.target_kind == "java_udf":
        lines.append(f"rewrite (java_udf, class {row.impl_class or '?'}):")
        lines.append(_indent(row.impl_source or "(binary only, no source stored)"))
    else:
        lines.append("rewrite (catalyst sql; _udf_param_N is the Nth argument):")
        lines.append(_indent(row.catalyst_sql or ""))
    if catalog is not None:
        try:
            samples = catalog.samples_for(row.udf_key, limit=_SAMPLES_SHOWN)
        except Exception:
            samples = []
        if samples:
            lines.append(f"real inputs it was checked on (latest {len(samples)}):")
            lines.extend(f"    {s!r}" for s in samples)
        elif any(t == "string" for t in row.input_types):
            lines.append("no sampled real inputs: string columns were checked on NULL/fixed examples only")
    return "\n".join(lines)


def resolve(rows: list[CacheRow], prefix: str) -> CacheRow:
    matches = [row for row in rows if row.udf_key.startswith(prefix)]
    if not matches:
        raise KeyError(f"no row awaiting approval matches {prefix!r}")
    if len(matches) > 1:
        raise KeyError(f"{prefix!r} is ambiguous ({len(matches)} rows); give more of the key")
    return matches[0]


def review(
    catalog: Any,
    *,
    reviewer: str,
    input_fn: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
    triggers: Any = None,
) -> dict[str, int]:
    """Interactive loop over every parked rewrite. Returns counts per decision."""
    counts = {"approved": 0, "rejected": 0, "skipped": 0}
    rows = catalog.awaiting_approval()
    if not rows:
        print("nothing awaiting approval", file=out)
        return counts
    print(f"{len(rows)} rewrite(s) awaiting approval; reviewing as {reviewer}", file=out)
    for index, row in enumerate(rows, 1):
        print(f"\n=== [{index}/{len(rows)}] ===", file=out)
        print(describe(row, catalog, triggers=triggers), file=out)
        while True:
            try:
                answer = input_fn("[a]pprove / [r]eject / [s]kip / [q]uit? ").strip().lower()
            except EOFError:
                answer = "q"
            if answer in {"a", "approve", "y", "yes"}:
                if catalog.approve(row.udf_key, reviewer):
                    counts["approved"] += 1
                    print("approved", file=out)
                else:
                    print("not approved: row is no longer awaiting approval", file=out)
                break
            if answer in {"r", "reject", "n", "no"}:
                try:
                    reason = input_fn("reason (optional): ").strip()
                except EOFError:
                    reason = ""
                if catalog.reject(row.udf_key, reviewer, reason):
                    counts["rejected"] += 1
                    print("rejected", file=out)
                else:
                    print("not rejected: row is no longer awaiting approval", file=out)
                break
            if answer in {"s", "skip", ""}:
                counts["skipped"] += 1
                break
            if answer in {"q", "quit", "exit"}:
                counts["skipped"] += len(rows) - index + 1
                _summary(counts, out)
                return counts
            print("please answer a, r, s, or q", file=out)
    _summary(counts, out)
    return counts


def _summary(counts: dict[str, int], out: TextIO) -> None:
    print(
        f"\napproved {counts['approved']}, rejected {counts['rejected']}, skipped {counts['skipped']}",
        file=out,
    )


def main(argv: Optional[list[str]] = None, *, out: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(
        description="Approve or reject AI UDF rewrites parked by requireApproval",
    )
    parser.add_argument("--sqlite-path", default=os.environ.get("AI_UDF_TRANSPILE_SQLITE"))
    parser.add_argument("--reviewer", default=None, help="name recorded on the row (default: $USER)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="print parked rewrites and exit")
    action.add_argument("--approve", nargs="+", metavar="KEY", help="approve these keys (prefixes ok)")
    action.add_argument("--reject", nargs="+", metavar="KEY", help="reject these keys (prefixes ok)")
    parser.add_argument("--reason", default="", help="rejection reason (with --reject)")
    parser.add_argument(
        "--flags",
        default="hard",
        help="which risk flags to show per row (requireApproval syntax; default: hard)",
    )
    args = parser.parse_args(argv)
    if not args.sqlite_path:
        parser.error("--sqlite-path (or $AI_UDF_TRANSPILE_SQLITE) is required")
    if args.reason and not args.reject:
        parser.error("--reason only applies with --reject")

    from ai_udf_transpile.catalog.sqlite import SqliteCatalog

    catalog = SqliteCatalog(args.sqlite_path)
    reviewer = args.reviewer or default_reviewer()
    triggers = parse_triggers(args.flags)
    try:
        if args.list:
            rows = catalog.awaiting_approval()
            if not rows:
                print("nothing awaiting approval", file=out)
            for row in rows:
                print(describe(row, catalog, triggers=triggers), file=out)
                print(file=out)
            return 0
        if args.approve or args.reject:
            rows = catalog.awaiting_approval()
            status = 0
            for prefix in args.approve or args.reject:
                try:
                    row = resolve(rows, prefix)
                except KeyError as exc:
                    print(f"error: {exc.args[0]}", file=out)
                    status = 1
                    continue
                if args.approve:
                    ok = catalog.approve(row.udf_key, reviewer)
                else:
                    ok = catalog.reject(row.udf_key, reviewer, args.reason)
                verb = "approved" if args.approve else "rejected"
                print(f"{verb if ok else 'unchanged'} {row.udf_key}", file=out)
                status = status if ok else 1
            return status
        review(catalog, reviewer=reviewer, out=out, triggers=triggers)
        return 0
    finally:
        catalog.close()


if __name__ == "__main__":
    raise SystemExit(main())
