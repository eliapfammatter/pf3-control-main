"""
PF3 Hydraulic Network - Lumped model using incidence matrix formulation.

Factory functions that build a HydraulicNetwork with combined series pipes,
giving a 5-state model instead of ~140 states in the distributed model.

Topology:
=========
- STANK: Surge tank (provides system capacitance)
- NODE1: Low-pressure junction (penstock outlet, pump inlets)
- NODE2: High-pressure junction (pump outlets, turbine inlet)

Paths:
- Penstock: STANK → NODE1 (combined L1-L4 + elbows)
- Turbine path: NODE2 → STANK via FTURB1 (combined L12-L19_2 + elbows + cone)
- Pump 1 path: NODE1 → NODE2 via PUMP1 (combined L5, LP1, L11 + elbows)
- Pump 2 path: NODE1 → NODE2 via PUMP2 (combined L6, L7, LP2, L10 + elbows)

States (alphabetical by branch name):
    x[0] = Q_penstock    (STANK → NODE1)
    x[1] = Q_pump1_path  (NODE1 → NODE2)
    x[2] = Q_pump2_path  (NODE1 → NODE2)
    x[3] = Q_turbine_path (NODE2 → STANK)
    x[4] = H_STANK       (tank head)

Usage:
    from src.pf3_lumped_network import build_pf3_lumped, build_pf3_lumped_mpc

    # Just the HydraulicNetwork (for simulation):
    network = build_pf3_lumped(interp_method="structured")
    result = network.simulate(t_span=(0, 10), external_inputs_func=...)

    # With MPC adapter:
    adapter = build_pf3_lumped_mpc(interp_method="structured")
    adapter.linearize(state, inputs)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .hydraulic_elements import DiscreteLoss, Pipe, Tank
from .hydraulic_elements.pump_turbine import PumpTurbine
from .incidence_network import HydraulicNetwork


@dataclass
class CombinedPipeParams:
    """Combined hydraulic parameters for series pipe elements."""

    L_total: float  # Total hydraulic inductance [s²/m²]
    R_total: float  # Total resistance coefficient [s²/m⁵]
    Q0: float  # Initial flow [m³/s]

    @classmethod
    def from_elements(
        cls,
        pipes: List[Pipe],
        losses: List[DiscreteLoss],
        Q0: float = None,
    ) -> "CombinedPipeParams":
        """
        Combine series pipes and losses into equivalent parameters.

        For series elements:
        - L_total = Σ L_i (inductances add)
        - R_total = Σ R_i (resistances add)
        """
        L_total = sum(p.hydraulic_L() for p in pipes)
        R_total = sum(p.hydraulic_R() for p in pipes)
        R_total += sum(loss.hydraulic_R() for loss in losses)

        if Q0 is None:
            Q0 = pipes[0].Q0 if pipes else 0.0

        return cls(L_total=L_total, R_total=R_total, Q0=Q0)


def build_pf3_lumped_mpc(
    interp_method: str = "linear",
    data_dir: Path = Path("data/pf3/passive_elements"),
):
    """
    Build lumped PF3 network with MPC adapter.

    Convenience function that creates the network and wraps it with
    MPCNetworkAdapter, pre-configured for the PF3 topology.

    Parameters
    ----------
    interp_method : str
        Interpolation method for characteristic curves ("linear", "structured", etc.)
    data_dir : Path
        Directory containing SIMSEN DAT files.

    Returns
    -------
    MPCNetworkAdapter
        Adapter implementing the MPCCompatibleNetwork interface.
    """
    from src.mpc_network_adapter import MPCNetworkAdapter

    data_dir = Path(data_dir)
    network = build_pf3_lumped(interp_method, data_dir)

    # Load machines for their characteristics (needed for MPC)
    fturb1 = PumpTurbine.from_dat(data_dir / "FTURB1.DAT", interp_method=interp_method)
    pump1 = PumpTurbine.from_dat(data_dir / "PUMP1.DAT", interp_method=interp_method)

    return MPCNetworkAdapter(
        network=network,
        turbine_char=fturb1.characteristic,
        pump_char=pump1.characteristic,
        turbine_Q_name="turbine_path",
        turbine_L=fturb1.hydraulic_L(),
        input_defaults={"y_P1": 1.0, "y_P2": 1.0},
    )
