"""
PF3 Experiment Analyzer - CLI Tool for Reading and Plotting Experiment Data.

Reads TDMS files from PF3 experiment results and creates analysis plots.

Usage:
    # Interactive wizard (no arguments):
    python pf3_post_processing.py

    # Direct file path:
    python pf3_post_processing.py experiment-series/260325_day1/results/platform-computer/764-15-1651.tdms

    # With simulation comparison:
    python pf3_post_processing.py data.tdms --sim simulation.pkl

    # With PXI pressure sensor data:
    python pf3_post_processing.py data.tdms --meas-tdms pxi_data.tdms
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from src.experiment import (
    align_signals_multi_rms,
    align_signals_rms,
    load_simulation,
    plot_hillchart,
    plot_pcb_sensors_aligned,
    plot_results_aligned,
    plot_runner_sensors_aligned,
    read_platform_tdms,
    read_pxi_tdms,
)

# =============================================================================
# Overview.md Parser and File Wizard
# =============================================================================


@dataclass
class ExperimentEntry:
    """Single experiment entry from overview.md."""

    date: str  # e.g., "2026-03-25"
    testing_day: str  # e.g., "day1"
    series_number: str  # e.g., "1651"
    series_name: str  # e.g., "S60"
    video: str
    comment: str

    @property
    def folder_name(self) -> str:
        """Return folder name like '260325_day1'."""
        # Convert 2026-03-25 -> 260325
        parts = self.date.split("-")
        yy = parts[0][2:]  # "26" from "2026"
        mm = parts[1]
        dd = parts[2]
        return f"{yy}{mm}{dd}_{self.testing_day}"

    @property
    def display_name(self) -> str:
        """Return display name for selection."""
        return f"{self.series_name} (#{self.series_number})"


def parse_overview(overview_path: Path) -> list[ExperimentEntry]:
    """Parse overview.md markdown table into list of ExperimentEntry."""
    entries = []
    with open(overview_path) as f:
        lines = f.readlines()

    # Skip header and separator lines
    for line in lines:
        line = line.strip()
        if not line or line.startswith("|--") or line.startswith("| Date"):
            continue
        if not line.startswith("|"):
            continue

        # Parse: | Date | Testing Day | Series Number | Series Name | Video | Comment |
        parts = [p.strip() for p in line.split("|")]
        # parts[0] is empty (before first |), parts[-1] is empty (after last |)
        if len(parts) < 7:
            continue

        date = parts[1]
        testing_day = parts[2]
        series_number = parts[3]
        series_name = parts[4]
        video = parts[5]
        comment = parts[6] if len(parts) > 6 else ""

        # Skip if no valid series number
        if not series_number.isdigit():
            continue

        entries.append(
            ExperimentEntry(
                date=date,
                testing_day=testing_day,
                series_number=series_number,
                series_name=series_name,
                video=video,
                comment=comment,
            )
        )

    return entries


@dataclass
class WizardResult:
    """Result from file search wizard."""

    platform_tdms: Path
    meas_tdms: Path | None
    sim_pkl: Path | None
    entry: ExperimentEntry | None = None
    t_start: float | None = None
    t_end: float | None = None


def build_title(tdms_path: Path, series_dir: Path | None = None) -> str:
    """Build descriptive title from path: DATE DAY SERIES #NUMBER.

    Parses path structure: .../260325_day1/results/platform-computer/764-15-1651.tdms
    Optionally looks up series name from overview.md.
    """
    # Extract series number from filename (e.g., "764-15-1651" -> "1651")
    filename = tdms_path.stem
    parts = filename.split("-")
    series_number = parts[-1] if len(parts) >= 3 else filename

    # Try to extract date and day from path (e.g., "260325_day1")
    date_str = ""
    day_str = ""
    for parent in tdms_path.parents:
        name = parent.name
        if "_day" in name:
            # Parse "260325_day1" -> date="2026-03-25", day="day1"
            folder_parts = name.split("_")
            if len(folder_parts) >= 2:
                raw_date = folder_parts[0]  # "260325"
                day_str = folder_parts[1]  # "day1"
                if len(raw_date) == 6:
                    date_str = f"20{raw_date[:2]}-{raw_date[2:4]}-{raw_date[4:6]}"
            break

    # Try to look up series name from overview.md
    series_name = ""
    if series_dir is not None:
        overview_path = series_dir / "overview.md"
        if overview_path.exists():
            entries = parse_overview(overview_path)
            for entry in entries:
                if entry.series_number == series_number:
                    series_name = entry.series_name
                    break

    # Build title
    title_parts = []
    if date_str:
        title_parts.append(date_str)
    if day_str:
        title_parts.append(day_str)
    if series_name:
        title_parts.append(series_name)
    title_parts.append(f"#{series_number}")

    return " ".join(title_parts) if title_parts else filename


def find_experiment_files(entry: ExperimentEntry, base_dir: Path) -> WizardResult:
    """Find all related files for an experiment entry."""
    day_dir = base_dir / entry.folder_name

    # Platform TDMS: results/platform-computer/764-15-{series_number}.tdms
    platform_dir = day_dir / "results" / "platform-computer"
    platform_tdms = platform_dir / f"764-15-{entry.series_number}.tdms"

    # Measurement TDMS: results/measurement-computer/*{series_number}*_processed.tdms
    meas_dir = day_dir / "results" / "measurement-computer"
    meas_tdms = None
    if meas_dir.exists():
        meas_files = list(meas_dir.glob(f"*{entry.series_number}*_processed.tdms"))
        if meas_files:
            meas_tdms = meas_files[0]

    # Simulation: setup/{series_name}/simulation_{series_name}.pkl
    sim_dir = day_dir / "setup" / entry.series_name
    sim_pkl = sim_dir / f"simulation_{entry.series_name}.pkl"
    if not sim_pkl.exists():
        sim_pkl = None

    return WizardResult(
        platform_tdms=platform_tdms,
        meas_tdms=meas_tdms,
        sim_pkl=sim_pkl,
        entry=entry,
    )


def interactive_select(prompt: str, options: list[str]) -> int:
    """Simple interactive selection using numbered list."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")

    while True:
        try:
            choice = input("\nEnter number: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            raise SystemExit(0)


def run_file_wizard(base_dir: Path) -> WizardResult | None:
    """Interactive wizard to select experiment files.

    Flow:
    1. Select testing day (date + day number)
    2. Select series from that day
    3. Find and confirm files
    """
    overview_path = base_dir / "overview.md"
    if not overview_path.exists():
        print(f"Error: overview.md not found at {overview_path}")
        return None

    entries = parse_overview(overview_path)
    if not entries:
        print("Error: No entries found in overview.md")
        return None

    # Group by testing day
    days: dict[str, list[ExperimentEntry]] = {}
    for entry in entries:
        key = entry.folder_name
        if key not in days:
            days[key] = []
        days[key].append(entry)

    # Step 1: Select testing day
    day_options = sorted(days.keys())
    day_labels = []
    for d in day_options:
        # Show date and day number more clearly
        first_entry = days[d][0]
        day_labels.append(f"{d} ({first_entry.date})")

    print("\n" + "=" * 50)
    print("PF3 Experiment File Wizard")
    print("=" * 50)

    day_idx = interactive_select("Select testing day:", day_labels)
    selected_day = day_options[day_idx]
    day_entries = days[selected_day]

    # Step 2: Select series
    series_labels = []
    for e in day_entries:
        label = f"{e.series_name} (#{e.series_number})"
        if e.comment:
            label += f" - {e.comment[:40]}"
        series_labels.append(label)

    series_idx = interactive_select("Select series:", series_labels)
    selected_entry = day_entries[series_idx]

    # Step 3: Find files
    result = find_experiment_files(selected_entry, base_dir)

    # Display results
    print("\n" + "-" * 50)
    print("Found files:")
    print(f"  Platform:    {result.platform_tdms}")
    if result.platform_tdms.exists():
        print("               [OK]")
    else:
        print("               [NOT FOUND]")

    if result.meas_tdms:
        print(f"  Measurement: {result.meas_tdms.name}")
        print("               [OK]")
    else:
        print("  Measurement: [not found]")

    if result.sim_pkl:
        print(f"  Simulation:  {result.sim_pkl.name}")
        print("               [OK]")
    else:
        print("  Simulation:  [not found]")

    print("-" * 50)

    if not result.platform_tdms.exists():
        print("Error: Platform TDMS file not found")
        return None

    # Step 4: Ask for time window
    print("\nTime window (leave empty for full range):")
    try:
        t_start_str = input("  Start time [s]: ").strip()
        t_end_str = input("  End time [s]: ").strip()
        result.t_start = float(t_start_str) if t_start_str else None
        result.t_end = float(t_end_str) if t_end_str else None
    except (ValueError, KeyboardInterrupt, EOFError):
        pass  # Keep None values

    # Print equivalent command for direct execution
    cmd_parts = ["python", "pf3_post_processing.py", str(result.platform_tdms)]
    if result.sim_pkl:
        cmd_parts.extend(["--sim", str(result.sim_pkl)])
    if result.meas_tdms:
        cmd_parts.extend(["--meas-tdms", str(result.meas_tdms)])
    if result.t_start is not None:
        cmd_parts.extend(["--t-start", str(result.t_start)])
    if result.t_end is not None:
        cmd_parts.extend(["--t-end", str(result.t_end)])
    print("\nTo run directly without wizard:")
    print(f"  {' '.join(cmd_parts)}")
    print()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read PF3 TDMS measurement results and create plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="If no TDMS file is provided, an interactive wizard helps select files.",
    )
    parser.add_argument(
        "tdms_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to TDMS file (optional - wizard runs if not provided)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path for plot (default: same dir as TDMS with .png extension)",
    )
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=Path("experiment-series"),
        help="Path to experiment-series directory (default: experiment-series)",
    )
    parser.add_argument(
        "--filter-size",
        type=int,
        default=51,
        help="Filter size (default: 51, must be odd for median)",
    )
    parser.add_argument(
        "--filter-type",
        type=str,
        choices=["median", "moving_avg", "none"],
        default="median",
        help="Filter type: median (default), moving_avg, or none",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't show plot (just save)",
    )
    parser.add_argument(
        "--fturb-dat",
        type=Path,
        default=Path("data/pf3/passive_elements/FTURB1.DAT"),
        help="Path to FTURB1.DAT for hillchart (default: data/pf3/passive_elements/FTURB1.DAT)",
    )
    parser.add_argument(
        "--sim",
        type=Path,
        default=None,
        help="Path to simulation .pkl file. Aligns measurement to simulation using y_T "
        "and crops to sim time range.",
    )
    parser.add_argument(
        "--meas-tdms",
        type=Path,
        default=None,
        help="Path to PXI measurement TDMS file (for PCB pressure sensors). "
        "Will be automatically aligned to platform data using turbine speed.",
    )
    parser.add_argument(
        "--pcb-downsample",
        type=int,
        default=100,
        help="Downsample factor for PCB sensor plots (default: 100)",
    )
    parser.add_argument(
        "--pxi-offset",
        type=float,
        default=None,
        help="Manual override for PXI time offset in seconds. "
        "If not specified, offset is computed automatically using step-down matching. "
        "Use: t_platform = t_pxi + offset",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Start time for plot x-axis (simulation time if --sim provided).",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="End time for plot x-axis (simulation time if --sim provided).",
    )

    args = parser.parse_args()

    # If no TDMS file provided, run interactive wizard
    if args.tdms_file is None:
        if not args.series_dir.exists():
            print(f"Error: Series directory not found: {args.series_dir}")
            return 1

        wizard_result = run_file_wizard(args.series_dir)
        if wizard_result is None:
            return 1

        args.tdms_file = wizard_result.platform_tdms
        # Use wizard results for sim and meas if not explicitly provided
        if args.sim is None and wizard_result.sim_pkl is not None:
            args.sim = wizard_result.sim_pkl
        if args.meas_tdms is None and wizard_result.meas_tdms is not None:
            args.meas_tdms = wizard_result.meas_tdms
        if args.t_start is None and wizard_result.t_start is not None:
            args.t_start = wizard_result.t_start
        if args.t_end is None and wizard_result.t_end is not None:
            args.t_end = wizard_result.t_end

    if not args.tdms_file.exists():
        print(f"Error: File not found: {args.tdms_file}")
        return 1

    # =========================================================================
    # Load all data
    # =========================================================================
    print("\n=== Load all data ===")
    print(f"Reading platform: {args.tdms_file}")
    platform_data = read_platform_tdms(args.tdms_file)

    sim = None
    if args.sim is not None and args.sim.exists():
        print(f"Loading simulation: {args.sim}")
        sim = load_simulation(args.sim)

    pxi_data = None
    if args.meas_tdms is not None and args.meas_tdms.exists():
        print(f"Reading PXI measurement: {args.meas_tdms}")
        pxi_data = read_pxi_tdms(args.meas_tdms)

    # =========================================================================
    # Align and crop signals
    # =========================================================================
    print("\n=== Align and crop signals ===")

    # Compute alignment offsets
    # offset means: platform_t = sim_t + offset (or pxi_t = platform_t + offset)
    plat_to_sim_offset = 0.0
    pxi_to_plat_offset = 0.0

    if sim is not None:
        # Use all available signals for robust alignment
        # (handles cases where one or more signals are constant)
        signal_pairs = [
            (sim.y_T.time, sim.y_T.val, platform_data.y_T.time, platform_data.y_T.val, "y_T"),
            (sim.N_T.time, sim.N_T.val, platform_data.N_T.time, platform_data.N_T.val, "N_T"),
            (sim.N_P1.time, sim.N_P1.val, platform_data.N_P1.time, platform_data.N_P1.val, "N_P1"),
            (sim.N_P2.time, sim.N_P2.val, platform_data.N_P2.time, platform_data.N_P2.val, "N_P2"),
        ]
        plat_to_sim_offset, rms_dict, used_signals = align_signals_multi_rms(
            signal_pairs, sample_rate=1.0
        )
        rms_summary = ", ".join(f"{k}={v:.4f}" for k, v in rms_dict.items())
        print(f"  Platform -> Sim: offset = {plat_to_sim_offset:.1f}s")
        print(f"    Used signals: {used_signals}")
        print(f"    RMS errors: {rms_summary}")

    if pxi_data is not None:
        if args.pxi_offset is not None:
            pxi_to_plat_offset = args.pxi_offset
            print(f"  PXI -> Platform: manual offset = {pxi_to_plat_offset:.1f}s")
        else:
            pxi_to_plat_offset, rms_NT = align_signals_rms(
                t_ref=platform_data.N_T.time,
                signal_ref=platform_data.N_T.val,
                t_target=pxi_data.rpm.time,
                signal_target=pxi_data.rpm.val,
                sample_rate=1.0,
            )
            print(
                f"  PXI -> Platform: offset = {pxi_to_plat_offset:.1f}s (RMS N_T = {rms_NT:.1f} RPM)"
            )

    # Crop to simulation time range (time resets to match sim time axis)
    if sim is not None:
        t_start_sim = sim.y_T.time[0]
        t_end_sim = sim.y_T.time[-1]
        print(f"  Sim time range: [{t_start_sim:.1f}, {t_end_sim:.1f}]s")

        # align_signals_rms returns offset where: sim_t = platform_t + offset
        # So: platform_t = sim_t - offset
        platform_data.crop(
            t_start_sim - plat_to_sim_offset,
            t_end_sim - plat_to_sim_offset,
            new_origin=t_start_sim,
        )
        print(f"  Platform: {len(platform_data.N_T)} DAQ samples after cropping")

        # Crop PXI: pxi_t = platform_t + pxi_to_plat_offset
        # Combined: pxi_t = (sim_t - plat_to_sim_offset) + pxi_to_plat_offset
        #         = sim_t - (plat_to_sim_offset - pxi_to_plat_offset)
        # So: pxi_t = sim_t - combined_offset where combined = plat - pxi
        if pxi_data is not None:
            # t_pxi = t_sim - plat_to_sim_offset - pxi_to_plat_offset
            combined_offset = plat_to_sim_offset + pxi_to_plat_offset
            pxi_data.crop(
                t_start_sim - combined_offset,
                t_end_sim - combined_offset,
                new_origin=t_start_sim,
            )
            print(
                f"  PXI: {len(pxi_data.ps1)} raw samples, "
                f"{len(pxi_data.rpm)} RPM points after cropping"
            )
    else:
        print("  No simulation - using full platform data")

    # Build xlim for plots (if user specified time window)
    xlim = None
    if args.t_start is not None or args.t_end is not None:
        xlim = (args.t_start, args.t_end)
        print(f"  Plot window: [{args.t_start}, {args.t_end}]s")

    # =========================================================================
    # Make plots
    # =========================================================================
    print("\n=== Make plots ===")

    # Determine output path
    if args.output is None:
        output_path = args.tdms_file.with_suffix(".png")
    else:
        output_path = args.output

    title = build_title(args.tdms_file, args.series_dir)

    # Create time series plot
    plot_results_aligned(
        platform_data,
        sim=sim,
        pxi=pxi_data,
        output_path=output_path,
        title=title,
        filter_size=args.filter_size,
        filter_type=args.filter_type,
        xlim=xlim,
    )

    # Create hillchart
    # Note: Uses cropped data (same as other plots). For full trajectory,
    # reload the data or call clear_masks() before this.
    hillchart_path = output_path.with_stem(output_path.stem + "_hillchart")
    plot_hillchart(
        platform_data,
        output_path=hillchart_path,
        title=title,
        fturb_dat_path=args.fturb_dat,
    )

    # Create PCB sensor plot if PXI data was loaded
    if pxi_data is not None:
        pcb_path = output_path.with_stem(output_path.stem + "_pcb")
        plot_pcb_sensors_aligned(
            pxi_data,
            platform_data,
            output_path=pcb_path,
            title=title,
            downsample=args.pcb_downsample,
            filter_size=args.filter_size,
            xlim=xlim,
        )

        # Create runner sensors plot (PS1/2/3/5)
        runner_path = output_path.with_stem(output_path.stem + "_runner")
        plot_runner_sensors_aligned(
            pxi_data,
            platform_data,
            output_path=runner_path,
            title=title,
            downsample=args.pcb_downsample,
            xlim=xlim,
        )

    if not args.no_plot:
        plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
