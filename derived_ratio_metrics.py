#!/usr/bin/env python3
"""Derived (ratio) maintainability metrics.

Small, standalone script that computes four additional *descriptive* ratio
metrics on top of the primary metrics used in metric_pipeline_basic.py. These
ratios are not part of the FAIL/WARN pass verdict; they only give a more
fine-grained view of *where* a module's complexity or size is concentrated.

    - LLOC / number of functions           <= 15   avg. function length
    - CC total / number of functions        <= 3    avg. complexity per function
    - Halstead Volume / LLOC                <= 20   operator/operand density per line
    - Public methods / number of classes    <= 5    avg. public interface size

"Functions" = standalone functions + methods defined inside classes.
"Public methods" = methods whose name does not start with "_" (this excludes
dunder methods such as __init__ or __call__, as well as private/protected
helpers), following the usual convention for a class's public interface.

Usage
-----
    python derived_ratio_metrics.py <target.py>

`<target>` may be a single .py file or a directory (scanned recursively by
Radon). Requires the `radon` package (already used by metric_pipeline_basic.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LIMITS = {
    "lloc_per_function": 15.0,
    "cc_per_function": 3.0,
    "halstead_per_lloc": 20.0,
    "public_methods_per_class": 5.0,
}


def run_radon(command: str, target: Path) -> dict:
    """Run `radon <command> -j <target>` and return the parsed JSON output."""
    result = subprocess.run(
        [sys.executable, "-m", "radon", command, "-j", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def count_functions(blocks: list[dict]) -> tuple[int, int]:
    """Count functions/methods and their total complexity for one module.

    Only standalone ``function`` blocks and methods nested inside ``class``
    blocks are counted. Radon additionally lists some class methods a second
    time as flat top-level ``method`` blocks; those duplicates are skipped
    here to avoid double-counting.
    """
    num_functions = 0
    cc_total = 0
    for block in blocks:
        if block["type"] == "function":
            num_functions += 1
            cc_total += block["complexity"]
        elif block["type"] == "class":
            for method in block.get("methods", []):
                num_functions += 1
                cc_total += method["complexity"]
    return num_functions, cc_total


def count_classes(blocks: list[dict]) -> tuple[int, int]:
    """Count classes and their public methods (names not starting with '_')."""
    num_classes = 0
    public_methods = 0
    for block in blocks:
        if block["type"] == "class":
            num_classes += 1
            for method in block.get("methods", []):
                if not method["name"].startswith("_"):
                    public_methods += 1
    return num_classes, public_methods


def halstead_volume(report: dict) -> float:
    """Extract the total Halstead Volume from one module's Radon `hal` report."""
    total = report.get("total")
    if isinstance(total, dict):
        return float(total.get("volume", 0.0))
    if isinstance(total, list) and len(total) >= 6:
        return float(total[5])
    return 0.0


def compute_ratios(module: str, cc_data: dict, hal_data: dict, raw_data: dict) -> dict:
    """Compute the four derived ratios for a single module."""
    num_functions, cc_total = count_functions(cc_data.get(module, []))
    num_classes, public_methods = count_classes(cc_data.get(module, []))
    lloc = raw_data.get(module, {}).get("lloc", 0)
    volume = halstead_volume(hal_data.get(module, {}))

    return {
        "lloc_per_function": round(lloc / num_functions, 2) if num_functions else 0.0,
        "cc_per_function": round(cc_total / num_functions, 2) if num_functions else 0.0,
        "halstead_per_lloc": round(volume / lloc, 2) if lloc else 0.0,
        "public_methods_per_class": round(public_methods / num_classes, 2) if num_classes else 0.0,
    }


def main() -> int:
    """Parse arguments, compute the ratios for every module, and print a report."""
    parser = argparse.ArgumentParser(description="Compute derived maintainability ratios.")
    parser.add_argument("target", help="Python file or directory to analyse.")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"error: target not found: {target}", file=sys.stderr)
        return 2

    cc_data = run_radon("cc", target)
    hal_data = run_radon("hal", target)
    raw_data = run_radon("raw", target)

    modules = sorted(set(cc_data) | set(hal_data) | set(raw_data))
    if not modules:
        print(f"no Python modules found under: {target}")
        return 0

    header = (
        f"{'Module':<42}{'LLOC/Fn':>9}{'CC/Fn':>8}{'Hal/LLOC':>10}{'PubM/Cls':>10}"
    )
    print(header)
    print("-" * len(header))
    for module in modules:
        ratios = compute_ratios(module, cc_data, hal_data, raw_data)
        name = module if len(module) <= 41 else "..." + module[-38:]
        print(
            f"{name:<42}{ratios['lloc_per_function']:>9}{ratios['cc_per_function']:>8}"
            f"{ratios['halstead_per_lloc']:>10}{ratios['public_methods_per_class']:>10}"
        )
    print()
    print(
        "limits: LLOC/Fn <= 15, CC/Fn <= 3, Hal/LLOC <= 20, PubM/Cls <= 5 "
        "(descriptive only, not part of the pass/fail verdict)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
