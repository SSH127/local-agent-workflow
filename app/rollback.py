"""Operator CLI for write snapshots: inspect, scan and roll back write transactions.

Run it outside the agent (the agent itself has the approval-gated
``rollback_file`` tool, limited to its own run)::

    python -m app.rollback --scan
    python -m app.rollback --list --run <run_id>
    python -m app.rollback --apply <tx_id> --run <run_id>
    python -m app.rollback --apply <tx_id> --run <run_id> --force

``--scan`` reports transactions left ``prepared`` by a crashed process. It never
auto-restores: a transaction whose target no longer matches the pre-write
snapshot needs human judgement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import write_guard


def _manifests(root: Path, run_id: str | None) -> list[dict]:
    pattern = (
        f"{run_id}/*/{write_guard.MANIFEST_NAME}"
        if run_id
        else f"*/{write_guard.MANIFEST_NAME}"
    )
    items: list[dict] = []
    for path in sorted(root.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["_manifest"] = str(path)
        items.append(data)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rollback", description="写入快照与回滚工具"
    )
    parser.add_argument("--workspace", default=None, help="工作区路径（用于定位快照目录）")
    parser.add_argument("--run", default=None, help="只处理指定 run_id")
    parser.add_argument("--list", action="store_true", help="列出写入事务")
    parser.add_argument("--scan", action="store_true", help="扫描未完成的写入事务")
    parser.add_argument("--apply", default=None, metavar="TX_ID", help="回滚指定事务")
    parser.add_argument("--force", action="store_true", help="即使文件已被外部修改也强制回滚")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else None
    root = write_guard.backup_root(workspace)

    if args.scan:
        findings = write_guard.scan_incomplete(workspace)
        print(
            json.dumps(findings, ensure_ascii=False, indent=2)
            if args.json
            else _format_findings(findings)
        )
        return 0

    if args.list:
        items = _manifests(root, args.run)
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2))
        else:
            for item in items:
                print(
                    f"{item.get('tx_id')}  run={item.get('run_id')}  "
                    f"status={item.get('status')}  path={item.get('path')}  "
                    f"before={str(item.get('before_sha256'))[:12]}  "
                    f"after={str(item.get('after_sha256'))[:12]}"
                )
            if not items:
                print(f"没有写入事务（{root}）。")
        return 0

    if args.apply:
        if not args.run:
            print(
                "--apply 需要同时提供 --run <run_id>（审批/回滚均绑定任务）。",
                file=sys.stderr,
            )
            return 2
        try:
            result = write_guard.rollback(
                run_id=args.run,
                tx_id=args.apply,
                workspace=workspace or Path.cwd(),
                force=args.force,
            )
        except (write_guard.WriteGuardError, write_guard.WriteConflict, OSError) as exc:
            print(f"回滚失败：{exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(result, ensure_ascii=False, indent=2)
            if args.json
            else f"回滚完成：{result}"
        )
        return 0

    parser.print_help()
    return 2


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "没有未完成的写入事务。"
    lines = [f"发现 {len(findings)} 个未完成的写入事务："]
    for item in findings:
        lines.append(
            f"- {item['tx_id']}  {item['verdict']}  {item['path']}\n"
            f"  {item['detail']}\n  manifest: {item['manifest']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
