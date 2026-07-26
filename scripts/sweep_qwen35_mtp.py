#!/usr/bin/env python3
"""Sweep Qwen3.5 MTP draft parameters for kraken-infer."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


MTP_RE = re.compile(
    r"mtp: (?P<enabled>enabled|disabled), layers=(?P<layers>\d+), "
    r"draft_tokens=(?P<draft_tokens>\d+), p_min=(?P<p_min>[^,]+), "
    r"drafted=(?P<drafted>\d+), accepted=(?P<accepted>\d+), "
    r"verify_steps=(?P<verify_steps>\d+), confidence_stops=(?P<confidence_stops>\d+)"
    r"(?P<rest>.*)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./build/debug/kraken-infer")
    parser.add_argument(
        "--model",
        default="models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about Metal acceleration in local inference.",
    )
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--draft-tokens",
        default="1,2,3",
        help="comma-separated draft token budgets",
    )
    parser.add_argument(
        "--p-min",
        default="0,0.1,0.2,0.3",
        help="comma-separated MTP confidence thresholds",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", default="build/qwen35-mtp-sweep.json")
    parser.add_argument("--csv-out", default="")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    parsed: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        number = int(stripped)
        if number < 0:
            raise ValueError("draft token values must be non-negative")
        parsed.append(number)
    if not parsed:
        raise ValueError("at least one draft token value is required")
    return parsed


def parse_float_list(value: str) -> list[float]:
    parsed: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        number = float(stripped)
        if number < 0.0 or number > 1.0:
            raise ValueError("p-min values must be in [0, 1]")
        parsed.append(number)
    if not parsed:
        raise ValueError("at least one p-min value is required")
    return parsed


def require_path(label: str, path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def kraken_command(
    args: argparse.Namespace,
    draft_tokens: int | None,
    p_min: float | None,
) -> list[str]:
    command = [
        args.binary,
        "infer",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_tokens),
    ]
    if draft_tokens is None:
        command.append("--no-mtp")
    else:
        command += [
            "--mtp",
            "--mtp-draft-tokens",
            str(draft_tokens),
            "--mtp-p-min",
            format_float(p_min if p_min is not None else 0.0),
        ]
    return command


def format_float(value: float) -> str:
    return f"{value:.6g}"


def run_command(name: str, command: list[str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "name": name,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "wall_seconds": time.perf_counter() - started,
            "command": command,
            "output": completed.stdout,
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "wall_seconds": time.perf_counter() - started,
            "command": command,
            "output": output,
            "error": f"timed out after {timeout:.1f}s",
        }


def parse_mtp(output: str) -> dict[str, Any]:
    match = MTP_RE.search(output)
    if not match:
        return {}
    stats: dict[str, Any] = {
        "enabled": match.group("enabled") == "enabled",
        "layers": int(match.group("layers")),
        "draft_tokens": int(match.group("draft_tokens")),
        "p_min": float(match.group("p_min")),
        "drafted": int(match.group("drafted")),
        "accepted": int(match.group("accepted")),
        "verify_steps": int(match.group("verify_steps")),
        "confidence_stops": int(match.group("confidence_stops")),
    }
    rest = match.group("rest")
    for key in ("adaptive_budget", "adaptive_changes"):
        extra = re.search(rf"{key}=([^,\s]+)", rest)
        if extra:
            stats[key] = int(extra.group(1))
    position = re.search(r"accepted_by_position=(\[[^\]]*\])", rest)
    if position:
        stats["accepted_by_position"] = position.group(1)
    return stats


def output_tail(output: str, limit: int = 280) -> str:
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
        and not line.startswith("request_id:")
        and not line.startswith("mtp:")
    ]
    return "\n".join(lines)[-limit:]


def summarize_run(
    result: dict[str, Any],
    mode: str,
    repeat: int,
    requested_draft_tokens: int | None,
    requested_p_min: float | None,
) -> dict[str, Any]:
    mtp = parse_mtp(result["output"])
    accepted = int(mtp.get("accepted", 0))
    drafted = int(mtp.get("drafted", 0))
    verify_steps = int(mtp.get("verify_steps", 0))
    adaptive_budget = mtp.get("adaptive_budget")
    if adaptive_budget is None and requested_draft_tokens is not None and mtp:
        adaptive_budget = requested_draft_tokens
    return {
        "mode": mode,
        "repeat": repeat,
        "ok": result["ok"],
        "returncode": result["returncode"],
        "wall_seconds": result["wall_seconds"],
        "requested_draft_tokens": requested_draft_tokens,
        "requested_p_min": requested_p_min,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": (accepted / drafted) if drafted else 0.0,
        "verify_steps": verify_steps,
        "confidence_stops": int(mtp.get("confidence_stops", 0)),
        "adaptive_budget": adaptive_budget,
        "adaptive_changes": mtp.get("adaptive_changes", 0),
        "accepted_by_position": mtp.get("accepted_by_position", ""),
        "command": result["command"],
        "error": result.get("error", ""),
        "output_tail": output_tail(result["output"]),
    }


def group_key(run: dict[str, Any]) -> tuple[str, int | None, float | None]:
    return (
        run["mode"],
        run["requested_draft_tokens"],
        run["requested_p_min"],
    )


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int | None, float | None], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(group_key(run), []).append(run)
    rows: list[dict[str, Any]] = []
    for key, values in groups.items():
        mode, draft_tokens, p_min = key
        ok_values = [value for value in values if value["ok"]]
        source = ok_values if ok_values else values
        rows.append(
            {
                "mode": mode,
                "draft_tokens": draft_tokens,
                "p_min": p_min,
                "runs": len(values),
                "ok_runs": len(ok_values),
                "wall_mean": mean(value["wall_seconds"] for value in source),
                "drafted_mean": mean(value["drafted"] for value in source),
                "accepted_mean": mean(value["accepted"] for value in source),
                "acceptance_mean": mean(value["acceptance"] for value in source),
                "verify_steps_mean": mean(value["verify_steps"] for value in source),
                "confidence_stops_mean": mean(
                    value["confidence_stops"] for value in source
                ),
                "adaptive_budget_last": source[-1]["adaptive_budget"],
                "adaptive_changes_mean": mean(
                    value["adaptive_changes"] for value in source
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["mode"] == "baseline" else 1,
            row["draft_tokens"] if row["draft_tokens"] is not None else -1,
            row["p_min"] if row["p_min"] is not None else -1.0,
        )
    )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    print("mode      draft  p_min  ok  wall_s  drafted  accepted  acc%   verify  budget")
    print("--------  -----  -----  --  ------  -------  --------  -----  ------  ------")
    for row in rows:
        draft = "-" if row["draft_tokens"] is None else str(row["draft_tokens"])
        p_min = "-" if row["p_min"] is None else format_float(row["p_min"])
        budget = (
            "-"
            if row["adaptive_budget_last"] is None
            else str(row["adaptive_budget_last"])
        )
        print(
            f"{row['mode'][:8]:8}  {draft:>5}  {p_min:>5}  "
            f"{row['ok_runs']}/{row['runs']:<2} "
            f"{row['wall_mean']:7.2f}  {row['drafted_mean']:7.1f}  "
            f"{row['accepted_mean']:8.1f}  {row['acceptance_mean'] * 100:5.1f}  "
            f"{row['verify_steps_mean']:6.1f}  {budget:>6}"
        )


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    require_path("binary", args.binary)
    require_path("model", args.model)
    draft_values = parse_int_list(args.draft_tokens)
    p_min_values = parse_float_list(args.p_min)
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    runs: list[dict[str, Any]] = []
    for repeat in range(args.repeat):
        if not args.skip_baseline:
            print(
                f"running baseline repeat {repeat + 1}/{args.repeat}...",
                file=sys.stderr,
            )
            result = run_command(
                "baseline",
                kraken_command(args, None, None),
                args.timeout,
            )
            runs.append(summarize_run(result, "baseline", repeat, None, None))
        for draft_tokens in draft_values:
            for p_min in p_min_values:
                name = f"mtp_d{draft_tokens}_p{format_float(p_min)}"
                print(
                    f"running {name} repeat {repeat + 1}/{args.repeat}...",
                    file=sys.stderr,
                )
                result = run_command(
                    name,
                    kraken_command(args, draft_tokens, p_min),
                    args.timeout,
                )
                runs.append(
                    summarize_run(result, "mtp", repeat, draft_tokens, p_min)
                )

    rows = aggregate(runs)
    print_table(rows)

    output = {"aggregate": rows, "runs": runs}
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\njson: {json_path}")
    if args.csv_out:
        write_csv(args.csv_out, rows)
        print(f"csv: {args.csv_out}")

    if args.strict and any(not run["ok"] for run in runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
