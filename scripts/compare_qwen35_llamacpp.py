#!/usr/bin/env python3
"""Compare kraken-infer and llama.cpp on the local Qwen3.5 0.8B GGUF."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


KRAKEN_MTP_RE = re.compile(
    r"mtp: (?P<enabled>enabled|disabled), layers=(?P<layers>\d+), "
    r"draft_tokens=(?P<draft_tokens>\d+), p_min=(?P<p_min>[^,]+), "
    r"drafted=(?P<drafted>\d+), accepted=(?P<accepted>\d+), "
    r"verify_steps=(?P<verify_steps>\d+), confidence_stops=(?P<confidence_stops>\d+)"
    r"(?P<rest>.*)"
)
LLAMA_TIMING_RE = re.compile(
    r"\[ Prompt: (?P<prompt_tps>[0-9.]+) t/s \| "
    r"Generation: (?P<generation_tps>[0-9.]+) t/s \]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kraken-binary", default="./build/debug/kraken-infer")
    parser.add_argument(
        "--llama-cli",
        default="/Users/bill/code/llama.cpp/build/bin/llama-cli",
    )
    parser.add_argument(
        "--model",
        default="models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short paragraph about Metal acceleration in local inference.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--p-min", type=float, default=0.30)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument(
        "--include-llama-mtp",
        action="store_true",
        help="also try llama.cpp --spec-type draft-mtp",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero if any requested run fails",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="optional path to write the full JSON summary",
    )
    return parser.parse_args()


def require_path(label: str, path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


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
        wall = time.perf_counter() - started
        return {
            "name": name,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "wall_seconds": wall,
            "command": command,
            "output": completed.stdout,
        }
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "wall_seconds": wall,
            "command": command,
            "output": output,
            "error": f"timed out after {timeout:.1f}s",
        }


def parse_kv_rest(rest: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key in ("adaptive_budget", "adaptive_changes"):
        match = re.search(rf"{key}=([^,\s]+)", rest)
        if match:
            try:
                parsed[key] = int(match.group(1))
            except ValueError:
                parsed[key] = match.group(1)
    for key in ("accepted_by_position",):
        match = re.search(rf"{key}=(\[[^\]]*\])", rest)
        if match:
            parsed[key] = match.group(1)
    reason = re.search(r"reason=([^,\n]+)", rest)
    if reason:
        parsed["reason"] = reason.group(1)
    return parsed


def summarize_kraken(result: dict[str, Any]) -> dict[str, Any]:
    output = result["output"]
    summary: dict[str, Any] = {
        "name": result["name"],
        "ok": result["ok"],
        "returncode": result["returncode"],
        "wall_seconds": result["wall_seconds"],
    }
    match = KRAKEN_MTP_RE.search(output)
    if match:
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
        stats.update(parse_kv_rest(match.group("rest")))
        summary["mtp"] = stats
    summary["request_id"] = parse_request_id(output)
    summary["text_tail"] = text_tail(output)
    return summary


def summarize_llama(result: dict[str, Any]) -> dict[str, Any]:
    output = result["output"]
    summary: dict[str, Any] = {
        "name": result["name"],
        "ok": result["ok"],
        "returncode": result["returncode"],
        "wall_seconds": result["wall_seconds"],
    }
    match = LLAMA_TIMING_RE.search(output)
    if match:
        summary["timing"] = {
            "prompt_tps": float(match.group("prompt_tps")),
            "generation_tps": float(match.group("generation_tps")),
        }
    if "error" in result:
        summary["error"] = result["error"]
    summary["text_tail"] = text_tail(output)
    return summary


def parse_request_id(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("request_id:"):
            return line.split(":", 1)[1].strip()
    return ""


def text_tail(output: str, limit: int = 360) -> str:
    lines: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("mtp:", "request_id:", "build", "model", "modalities")):
            continue
        if stripped.startswith(("[ Prompt:", "Loading model", "available commands:")):
            continue
        if stripped in {"Exiting...", "▄▄ ▄▄"}:
            continue
        if stripped.startswith(("/exit", "/regen", "/clear", "/read", "/glob")):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    return text[-limit:]


def kraken_command(args: argparse.Namespace, enable_mtp: bool) -> list[str]:
    command = [
        args.kraken_binary,
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
    if enable_mtp:
        command += [
            "--mtp",
            "--mtp-draft-tokens",
            str(args.draft_tokens),
            "--mtp-p-min",
            str(args.p_min),
        ]
    else:
        command.append("--no-mtp")
    return command


def llama_command(args: argparse.Namespace, enable_mtp: bool) -> list[str]:
    command = [
        args.llama_cli,
        "-m",
        args.model,
        "-p",
        args.prompt,
        "-n",
        str(args.max_tokens),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--no-warmup",
        "--no-display-prompt",
        "--single-turn",
        "--simple-io",
        "--log-disable",
    ]
    if enable_mtp:
        command += [
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            str(args.draft_tokens),
            "--spec-draft-p-min",
            str(args.p_min),
        ]
    return command


def print_table(summaries: list[dict[str, Any]]) -> None:
    print("name                 ok   wall_s  mtp")
    print("-------------------  ---  ------  ------------------------------")
    for item in summaries:
        mtp = item.get("mtp")
        mtp_text = ""
        if mtp:
            mtp_text = (
                f"enabled={int(mtp['enabled'])} drafted={mtp['drafted']} "
                f"accepted={mtp['accepted']} verify={mtp['verify_steps']}"
            )
            if "adaptive_budget" in mtp:
                mtp_text += f" budget={mtp['adaptive_budget']}"
        elif "timing" in item:
            timing = item["timing"]
            mtp_text = (
                f"prompt={timing['prompt_tps']:.1f}t/s "
                f"gen={timing['generation_tps']:.1f}t/s"
            )
        print(
            f"{item['name'][:19]:19}  {str(item['ok']):3}  "
            f"{item['wall_seconds']:6.2f}  {mtp_text}"
        )


def main() -> int:
    args = parse_args()
    require_path("kraken binary", args.kraken_binary)
    require_path("model", args.model)
    if not args.skip_llama:
        require_path("llama-cli", args.llama_cli)

    runs: list[tuple[str, list[str], str]] = [
        ("kraken_no_mtp", kraken_command(args, False), "kraken"),
        ("kraken_mtp", kraken_command(args, True), "kraken"),
    ]
    if not args.skip_llama:
        runs.append(("llama_no_mtp", llama_command(args, False), "llama"))
        if args.include_llama_mtp:
            runs.append(("llama_mtp", llama_command(args, True), "llama"))

    summaries: list[dict[str, Any]] = []
    full_runs: list[dict[str, Any]] = []
    for name, command, kind in runs:
        print(f"running {name}...", file=sys.stderr)
        result = run_command(name, command, args.timeout)
        full_runs.append(result)
        if kind == "kraken":
            summaries.append(summarize_kraken(result))
        else:
            summaries.append(summarize_llama(result))

    print_table(summaries)
    output = {"runs": summaries, "raw": full_runs}
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\njson: {path}")
    else:
        print("\nsummary_json:")
        print(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2))

    if args.strict and any(not item["ok"] for item in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
