#!/usr/bin/env python3
"""Basic maintainability measurement pipeline.

A small, self-contained pipeline that wraps Radon and Pylint to compute the
four maintainability metrics used in this thesis and checks them against the
thresholds defined in Section 4.1 of the methodology:

    - Cyclomatic Complexity (CC)   Radon cc    mean <= 5, max <= 10   (FAIL)
    - Halstead Volume              Radon hal   <= 8000 / module       (WARN)
    - Logical Lines of Code (LLOC) Radon raw   <= 500 / module        (WARN)
    - Pylint score                 Pylint      >= 7.0 / 10            (FAIL)

It prints a human-readable text report and can optionally write a JSON report.
The process exits with code 0 if all FAIL thresholds are met and 1 otherwise,
so it can be used directly in the prompting feedback loop.

Usage
-----
    python metric_pipeline_basic.py <target>
    python metric_pipeline_basic.py <target> --output report.json

`<target>` may be a single .py file or a directory (scanned recursively).

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# --- Thresholds (see Section 4.1). "fail" entries cause a non-zero exit. ---
THRESHOLDS = {
    "mean_cc": {"limit": 5.0, "kind": "fail"},
    "max_cc": {"limit": 10.0, "kind": "fail"},
    "halstead_volume": {"limit": 8000.0, "kind": "warn"},
    "lloc": {"limit": 500.0, "kind": "warn"},
    "pylint_score": {"limit": 7.0, "kind": "fail"},
}

# Pylint checks disabled at file level: module/class/function docstring missing,
# since docstring presence is enforced separately by the prompting requirements.
PYLINT_DISABLE = "C0114,C0115,C0116"
PYLINT_MAX_LINE_LENGTH = "100"


def run_tool(args: list[str]) -> str:
    """Run a Python module tool and return its stdout as text.

    Parameters
    ----------
    args : list[str]
        Arguments passed after the Python executable, e.g. ["-m", "radon", ...].

    Returns
    -------
    str
        The captured standard output (may be empty on error).
    """
    result = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def collect_files(target: Path) -> list[Path]:
    """Return the list of Python files to analyse for a file or directory."""
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    return sorted(p for p in target.rglob("*.py"))


def analyse_cc(target: Path) -> dict[str, dict]:
    """Compute per-module mean and maximum cyclomatic complexity via Radon."""
    raw = run_tool(["-m", "radon", "cc", "-j", str(target)])
    data = json.loads(raw) if raw.strip() else {}
    out: dict[str, dict] = {}
    for module, blocks in data.items():
        complexities: list[int] = []
        for block in blocks:
            complexities.append(block["complexity"])
            for method in block.get("methods", []):
                complexities.append(method["complexity"])
        if complexities:
            out[module] = {
                "mean_cc": round(sum(complexities) / len(complexities), 2),
                "max_cc": max(complexities),
                "num_blocks": len(complexities),
            }
        else:
            out[module] = {"mean_cc": 0.0, "max_cc": 0, "num_blocks": 0}
    return out


def analyse_halstead(target: Path) -> dict[str, dict]:
    """Compute per-module total Halstead Volume via Radon."""
    raw = run_tool(["-m", "radon", "hal", "-j", str(target)])
    data = json.loads(raw) if raw.strip() else {}
    out: dict[str, dict] = {}
    for module, report in data.items():
        total = report.get("total")
        # Radon may return the totals as a list [..., volume, ...] or a dict.
        if isinstance(total, dict):
            volume = total.get("volume", 0.0)
        elif isinstance(total, list) and len(total) >= 6:
            volume = total[5]
        else:
            volume = 0.0
        out[module] = {"halstead_volume": round(float(volume), 2)}
    return out


def analyse_raw(target: Path) -> dict[str, dict]:
    """Compute per-module raw size metrics (LOC, SLOC, LLOC, comments) via Radon."""
    raw = run_tool(["-m", "radon", "raw", "-j", str(target)])
    data = json.loads(raw) if raw.strip() else {}
    out: dict[str, dict] = {}
    for module, report in data.items():
        sloc = report.get("sloc", 0)
        comments = report.get("comments", 0)
        out[module] = {
            "loc": report.get("loc", 0),
            "sloc": sloc,
            "lloc": report.get("lloc", 0),
            "comments": comments,
            "comment_ratio": round(comments / sloc, 3) if sloc else 0.0,
        }
    return out


def analyse_pylint(files: list[Path]) -> dict:
    """Run Pylint once over all files and return score plus violation summary."""
    if not files:
        return {"score": 0.0, "violations": {}}
    str_files = [str(f) for f in files]
    common = [
        "-m",
        "pylint",
        f"--disable={PYLINT_DISABLE}",
        f"--max-line-length={PYLINT_MAX_LINE_LENGTH}",
    ]

    # Pass 1: JSON output for the full message list.
    json_out = run_tool([*common, "--output-format=json", *str_files])
    violations: dict[str, int] = {}
    try:
        messages = json.loads(json_out) if json_out.strip() else []
        for msg in messages:
            symbol = msg.get("symbol", msg.get("message-id", "unknown"))
            violations[symbol] = violations.get(symbol, 0) + 1
    except json.JSONDecodeError:
        pass

    # Pass 2: text output to read the numeric score from the score line.
    text_out = run_tool([*common, *str_files])
    score = 0.0
    for line in text_out.splitlines():
        if "rated at" in line:
            try:
                score = float(line.split("rated at")[1].split("/")[0].strip())
            except (IndexError, ValueError):
                score = 0.0
            break
    return {"score": score, "violations": violations}


def evaluate(report: dict) -> list[dict]:
    """Compare measured values against thresholds and return a list of checks."""
    checks: list[dict] = []

    def add(name: str, value: float, ok: bool, scope: str) -> None:
        checks.append(
            {
                "metric": name,
                "scope": scope,
                "value": value,
                "limit": THRESHOLDS[name]["limit"],
                "kind": THRESHOLDS[name]["kind"],
                "passed": ok,
            }
        )

    for module, m in report["modules"].items():
        add("mean_cc", m["mean_cc"], m["mean_cc"] <= THRESHOLDS["mean_cc"]["limit"], module)
        add("max_cc", m["max_cc"], m["max_cc"] <= THRESHOLDS["max_cc"]["limit"], module)
        add(
            "halstead_volume",
            m["halstead_volume"],
            m["halstead_volume"] <= THRESHOLDS["halstead_volume"]["limit"],
            module,
        )
        add("lloc", m["lloc"], m["lloc"] <= THRESHOLDS["lloc"]["limit"], module)

    score = report["pylint"]["score"]
    add("pylint_score", score, score >= THRESHOLDS["pylint_score"]["limit"], "project")
    return checks


def build_report(target: Path) -> dict:
    """Run all four analysis stages and assemble the structured report."""
    files = collect_files(target)
    cc = analyse_cc(target)
    hal = analyse_halstead(target)
    raw = analyse_raw(target)

    modules: dict[str, dict] = {}
    for module in sorted(set(cc) | set(hal) | set(raw)):
        modules[module] = {
            "mean_cc": cc.get(module, {}).get("mean_cc", 0.0),
            "max_cc": cc.get(module, {}).get("max_cc", 0),
            "num_blocks": cc.get(module, {}).get("num_blocks", 0),
            "halstead_volume": hal.get(module, {}).get("halstead_volume", 0.0),
            "lloc": raw.get(module, {}).get("lloc", 0),
            "sloc": raw.get(module, {}).get("sloc", 0),
            "comment_ratio": raw.get(module, {}).get("comment_ratio", 0.0),
        }

    report = {
        "target": str(target),
        "num_files": len(files),
        "modules": modules,
        "pylint": analyse_pylint(files),
    }
    report["checks"] = evaluate(report)
    report["passed"] = all(
        c["passed"] for c in report["checks"] if c["kind"] == "fail"
    )
    return report


def print_text_report(report: dict) -> None:
    """Print a readable summary of the report to the terminal."""
    line = "=" * 72
    print(line)
    print(f"  MAINTAINABILITY REPORT  -  target: {report['target']}")
    print(f"  files analysed: {report['num_files']}")
    print(line)

    header = f"{'Module':<42}{'meanCC':>7}{'maxCC':>6}{'Halstead':>10}{'LLOC':>6}"
    print(header)
    print("-" * 72)
    for module, m in report["modules"].items():
        name = module if len(module) <= 41 else "..." + module[-38:]
        print(
            f"{name:<42}{m['mean_cc']:>7}{m['max_cc']:>6}"
            f"{m['halstead_volume']:>10}{m['lloc']:>6}"
        )

    print("-" * 72)
    pl = report["pylint"]
    print(f"Pylint score: {pl['score']:.2f} / 10")
    if pl["violations"]:
        top = sorted(pl["violations"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        smells = ", ".join(f"{sym} ({n})" for sym, n in top)
        print(f"Top Pylint findings: {smells}")

    print(line)
    print("  THRESHOLD CHECKS")
    print(line)
    for c in report["checks"]:
        status = "PASS" if c["passed"] else ("FAIL" if c["kind"] == "fail" else "WARN")
        comp = "<=" if c["metric"] != "pylint_score" else ">="
        print(
            f"  [{status:>4}] {c['metric']:<16} {c['value']:>8} "
            f"{comp} {c['limit']:<6} ({c['scope']})"
        )
    print(line)
    verdict = "PASSED" if report["passed"] else "FAILED"
    print(f"  RESULT: {verdict} (FAIL thresholds {'all met' if report['passed'] else 'violated'})")
    print(line)


def main() -> int:
    """Parse arguments, run the pipeline, print/write reports, and set exit code."""
    parser = argparse.ArgumentParser(description="Basic maintainability metric pipeline.")
    parser.add_argument("target", help="Python file or directory to analyse.")
    parser.add_argument("--output", help="Optional path to write the JSON report.")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"error: target not found: {target}", file=sys.stderr)
        return 2

    report = build_report(target)
    print_text_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {out_path}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
