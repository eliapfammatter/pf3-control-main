"""
PF3 Experiment Data Analysis Package.

This package provides tools for reading, processing, and visualizing
measurement data from PF3 pump-turbine experiments.

Main components:
- MeasurementSeries: Core time series data structure with lazy calibration
- Data containers: PlatformData, PXIMeasurementData, SimulationData
- Readers: read_platform_tdms, read_pxi_tdms, load_simulation
- Signal processing: align_signals_rms, compute_spectrogram_normalized
- Plotting: plot_results_aligned, plot_hillchart, plot_pcb_sensors_aligned

Example usage:
    from src.experiment import read_platform_tdms, plot_results_aligned

    # Load platform data
    data = read_platform_tdms(Path("experiment.tdms"))

    # Crop to time range
    data.crop(10, 60)

    # Create plots
    plot_results_aligned(data, output_path=Path("results.png"))
"""

from .calibration import (
    CAL,
    G,
    GVO_TO_Y_FACTOR,
    PCB_INFO,
    PXI_SAMPLE_RATE,
    RHO,
    S1,
    S2,
    CalCoeffs,
)
from .data_types import PlatformData, PXIMeasurementData, SimulationData
from .measurement import MaskableDataMixin, MeasurementSeries
from .plotting import (
    plot_hillchart,
    plot_pcb_sensors_aligned,
    plot_results_aligned,
    plot_runner_sensors_aligned,
    print_summary,
)
from .readers import (
    load_simulation,
    load_simulation_raw,
    read_platform_tdms,
    read_pxi_tdms,
)
from .signal_processing import (
    add_gvo_overlay,
    align_signals_multi_rms,
    align_signals_rms,
    compute_spectrogram_normalized,
    extract_rpm_from_tachometer,
    plot_spectrogram_on_axis,
)

__all__ = [
    # Measurement
    "MeasurementSeries",
    "MaskableDataMixin",
    # Data types
    "PlatformData",
    "PXIMeasurementData",
    "SimulationData",
    # Calibration
    "CalCoeffs",
    "CAL",
    "PCB_INFO",
    "G",
    "RHO",
    "S1",
    "S2",
    "GVO_TO_Y_FACTOR",
    "PXI_SAMPLE_RATE",
    # Readers
    "read_platform_tdms",
    "read_pxi_tdms",
    "load_simulation",
    "load_simulation_raw",
    # Signal processing
    "align_signals_multi_rms",
    "align_signals_rms",
    "extract_rpm_from_tachometer",
    "compute_spectrogram_normalized",
    "plot_spectrogram_on_axis",
    "add_gvo_overlay",
    # Plotting
    "plot_hillchart",
    "plot_results_aligned",
    "plot_pcb_sensors_aligned",
    "plot_runner_sensors_aligned",
    "print_summary",
]
