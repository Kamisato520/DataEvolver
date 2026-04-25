#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    string_rows = [[format_value(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(cells: list[str]) -> str:
        padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    header_line = render_row(headers)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in string_rows]
    return "\n".join([header_line, separator, *body])


def collect_obj_ids(summary: dict[str, Any], evolution_dir: Path) -> list[str]:
    obj_ids: set[str] = set()
    results = summary.get("results", {})
    if isinstance(results, dict):
        obj_ids.update(str(obj_id) for obj_id in results.keys())
    final_scores = summary.get("final_scores", {})
    if isinstance(final_scores, dict):
        obj_ids.update(str(obj_id) for obj_id in final_scores.keys())
    for child in evolution_dir.glob("obj_*"):
        if child.is_dir():
            obj_ids.add(child.name)
    return sorted(obj_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze evolution outputs and print three summary tables."
    )
    parser.add_argument(
        "--evolution-dir",
        default=".",
        help="Directory containing evolution_summary.json and obj_*/evolution_result.json",
    )
    args = parser.parse_args()

    evolution_dir = Path(args.evolution_dir).expanduser().resolve()
    summary_path = evolution_dir / "evolution_summary.json"
    summary = load_json(summary_path)
    summary_results = summary.get("results", {})
    if not isinstance(summary_results, dict):
        summary_results = {}
    final_scores = summary.get("final_scores", {})
    if not isinstance(final_scores, dict):
        final_scores = {}

    per_object_rows: list[list[Any]] = []
    issue_counter: Counter[str] = Counter()
    exit_counter: Counter[str] = Counter()

    for obj_id in collect_obj_ids(summary, evolution_dir):
        result_path = evolution_dir / obj_id / "evolution_result.json"
        if result_path.exists():
            result = load_json(result_path)
        else:
            result = {}

        summary_result = summary_results.get(obj_id, {})
        if not isinstance(summary_result, dict):
            summary_result = {}

        state_log = result.get("state_log", summary_result.get("state_log", []))
        if not isinstance(state_log, list):
            state_log = []

        final_entry = state_log[-1] if state_log else {}
        if not isinstance(final_entry, dict):
            final_entry = {}
        exit_reason = str(final_entry.get("exit_reason", "max_rounds"))
        exit_counter[exit_reason] += 1

        for entry in state_log:
            if not isinstance(entry, dict):
                continue
            for tag in normalize_tags(entry.get("issue_tags")):
                issue_counter[tag] += 1

        obj_name = result.get("obj_id", summary_result.get("obj_id", obj_id))
        final_hybrid = result.get(
            "final_hybrid",
            summary_result.get("final_hybrid", final_scores.get(obj_id, "")),
        )
        probes = result.get("probes_run", summary_result.get("probes_run", ""))
        updates = result.get("updates_run", summary_result.get("updates_run", ""))
        accepted = result.get("accepted", summary_result.get("accepted", ""))

        per_object_rows.append(
            [obj_name, final_hybrid, probes, updates, accepted, exit_reason]
        )

    per_object_rows.sort(key=lambda row: str(row[0]))

    total_tags = sum(issue_counter.values())
    issue_rows = [
        [tag, count, f"{(count / total_tags * 100) if total_tags else 0.0:.1f}%"]
        for tag, count in sorted(issue_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    total_objects = len(per_object_rows)
    exit_rows = [
        [reason, count, f"{(count / total_objects * 100) if total_objects else 0.0:.1f}%"]
        for reason, count in sorted(exit_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    print("Table 1: Per-object metrics")
    print(
        render_markdown_table(
            ["obj_id", "final_hybrid", "probes", "updates", "accepted", "exit_reason"],
            per_object_rows,
        )
    )
    print()
    print("Table 2: Issue tag distribution")
    print(render_markdown_table(["tag_name", "count", "percentage"], issue_rows))
    print()
    print("Table 3: Exit reason distribution")
    print(render_markdown_table(["exit_reason", "count", "percentage"], exit_rows))


if __name__ == "__main__":
    main()
