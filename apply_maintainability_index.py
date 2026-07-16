#!/usr/bin/env python3
"""Retroactively add Maintainability Index (MI) to existing result files.

This is a one-off utility, not part of the original experiment protocol. It
re-runs the (now MI-aware) metric_pipeline_basic.py against every already
generated module in src/generated/{strategy}/{module}.py, and merges the
resulting "maintainability_index" / "mi_rank" fields into the *final accepted*
version JSON report for that strategy/module pair in src/results/.

MI is added purely as a descriptive, secondary metric (see the module
docstring of metric_pipeline_basic.py). It does not change any FAIL/WARN
threshold, any "passed" verdict, or any other previously recorded value in
the result files - it only adds two new keys to the relevant module entry.

Usage
-----
    python apply_maintainability_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

import metric_pipeline_basic as mp

REPO_ROOT = Path(__file__).resolve().parent

FINAL_VERSIONS = [
    ("zero_shot", "data_types", 1),
    ("zero_shot", "fpoints_controller", 3),
    ("zero_shot", "orchestrator", 2),
    ("few_shot", "data_types", 2),
    ("few_shot", "fpoints_controller", 2),
    ("few_shot", "orchestrator", 1),
    ("constraint_based", "data_types", 2),
    ("constraint_based", "fpoints_controller", 2),
    ("constraint_based", "orchestrator", 1),
    ("chain_of_thoughts", "data_types", 2),
    ("chain_of_thoughts", "orchestrator", 1),
]


def main() -> None:
    summary_rows = []

    for strategy, module, version in FINAL_VERSIONS:
        target = Path("src") / "generated" / strategy / f"{module}.py"
        result_path = REPO_ROOT / "src" / "results" / strategy / f"{module}_v{version}.json"

        if not (REPO_ROOT / target).exists():
            print(f"skip (no generated code): {target}")
            continue
        if not result_path.exists():
            print(f"skip (no result file): {result_path}")
            continue

        import os

        old_cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            report = mp.build_report(target)
        finally:
            os.chdir(old_cwd)

        module_key = str(target)
        mi_data = report["modules"].get(module_key, {})
        mi_value = mi_data.get("maintainability_index", 0.0)
        mi_rank = mi_data.get("mi_rank", "?")

        with open(result_path, encoding="utf-8") as f:
            existing = json.load(f)

        if module_key not in existing["modules"]:
            print(f"warning: module key mismatch for {result_path}, skipping merge")
            continue

        existing["modules"][module_key]["maintainability_index"] = mi_value
        existing["modules"][module_key]["mi_rank"] = mi_rank
        existing.setdefault(
            "notes",
            "maintainability_index / mi_rank added retroactively (descriptive "
            "only, not part of the FAIL/WARN threshold checks or the 'passed' verdict) "
            "via apply_maintainability_index.py",
        )

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
            f.write("\n")

        print(f"updated {result_path.relative_to(REPO_ROOT)}: MI={mi_value} ({mi_rank})")
        summary_rows.append((strategy, module, version, mi_value, mi_rank))

    print("\nSummary (strategy, module, final version, MI, rank):")
    for row in summary_rows:
        print(f"  {row[0]:<18}{row[1]:<20}v{row[2]:<3}{row[3]:>7}   {row[4]}")


if __name__ == "__main__":
    main()
