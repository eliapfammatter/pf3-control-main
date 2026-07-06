import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.hydraulic_elements.element import Element


@dataclass
class Pipe(Element):
    """
    PIPEZ element - a pipe segment with optional distributed discretization.

    Physics:
      - Inertia (L): L_h = L * Ksi / (g * A)
      - Friction (R): R = L * Lambda / (2 * g * D * A²)
      - Compressibility (C): C = g * A / a² (for distributed model)

    Distributed model (Nb > 0):
      Creates Nb+1 flow states and Nb internal head states.
    """

    name: str
    L: float  # Pipe length [m]
    D: float  # Diameter [m]
    A: float  # Cross-sectional area [m²]
    a: float  # Wave speed [m/s]
    Lambda: float  # Darcy friction factor [-]
    Ksi: float = 1.0  # Inertia correction factor [-]
    Nb: int = 0  # Number of discretization elements (0 = lumped)
    Q0: float = 0.0  # Initial flow [m³/s] (first segment, for backward compat)
    Q0_list: List[float] = field(default_factory=list)  # Initial Q for each segment
    Hc0_list: List[float] = field(default_factory=list)  # Initial internal heads
    g: float = 9.806  # Gravity [m/s²]

    def __post_init__(self):
        # Don't call Element.__init__ since we're using dataclass
        pass

    @classmethod
    def from_dat(cls, file_path: Path) -> "Pipe":
        """Load pipe parameters from SIMSEN PIPEZ DAT file."""
        file_path = Path(file_path)
        params: Dict[str, Any] = {"name": file_path.stem}

        with open(file_path, "r") as f:
            content = f.read()

        # Parse scalar parameters (including Ah/Dh for hydraulic area/diameter)
        for key in ["L", "D", "A", "a", "Lambda", "Ksi", "Nb", "g", "Ah", "Dh"]:
            pattern = rf"^{re.escape(key)}\s+\["
            for line in content.split("\n"):
                stripped = line.strip()
                if re.match(pattern, stripped) and ":" in stripped:
                    try:
                        val = line.split(":")[1].strip()
                        if key == "Nb":
                            params[key] = int(float(val))
                        else:
                            params[key] = float(val)
                    except (IndexError, ValueError):
                        pass
                    break

        # Defaults
        params.setdefault("Ksi", 1.0)
        params.setdefault("Nb", 0)
        params.setdefault("g", 9.806)

        # Parse initial conditions
        # SIMSEN uses Q1, Q2, ... (1-indexed) and Hc1, Hc2, ... (1-indexed)
        Q0_list = []
        Hc0_list = []
        in_initial = False

        for line in content.split("\n"):
            stripped = line.strip()
            if "INITIAL CONDITIONS" in stripped:
                in_initial = True
                continue
            if in_initial:
                if stripped.startswith("-") and ":" not in stripped:
                    break
                # Q1, Q2, ... [m3/s] (1-indexed)
                q_match = re.match(
                    r"^Q(\d+)\s+\[m3/s\]\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                    stripped,
                )
                if q_match:
                    idx = int(q_match.group(1)) - 1  # Convert to 0-indexed
                    val = float(q_match.group(2))
                    while len(Q0_list) <= idx:
                        Q0_list.append(0.0)
                    Q0_list[idx] = val
                # Hc1, Hc2, ... [m] (1-indexed)
                hc_match = re.match(
                    r"^Hc(\d+)\s+\[m\]\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                    stripped,
                )
                if hc_match:
                    idx = int(hc_match.group(1)) - 1  # Convert to 0-indexed
                    val = float(hc_match.group(2))
                    while len(Hc0_list) <= idx:
                        Hc0_list.append(0.0)
                    Hc0_list[idx] = val

        # Q0 is the first flow value (inlet flow), Q0_list has all segment Q values
        params["Q0"] = Q0_list[0] if Q0_list else 0.0
        params["Q0_list"] = Q0_list
        params["Hc0_list"] = Hc0_list

        # Use hydraulic area/diameter when D=0 (SIMSEN convention for non-circular sections)
        # Priority: A > Ah > computed from D
        if "A" not in params or params.get("A", 0) == 0:
            if "Ah" in params and params["Ah"] > 0:
                params["A"] = params["Ah"]
            elif "D" in params and params["D"] > 0:
                params["A"] = np.pi * params["D"] ** 2 / 4

        # Use hydraulic diameter when D=0
        if "D" not in params or params.get("D", 0) == 0:
            if "Dh" in params and params["Dh"] > 0:
                params["D"] = params["Dh"]

        # Remove Ah/Dh from params (not part of Pipe dataclass)
        params.pop("Ah", None)
        params.pop("Dh", None)

        required = ["L", "D", "A", "a", "Lambda"]
        missing = [k for k in required if k not in params or params.get(k) is None]
        if missing:
            raise ValueError(f"Missing parameters in {file_path}: {missing}")

        return cls(**params)

    def hydraulic_L(self) -> float:
        """Total hydraulic inductance [s²/m²]."""
        if self.A <= 0:
            return 0.0
        return self.L * self.Ksi / (self.g * self.A)

    def hydraulic_R(self) -> float:
        """Total hydraulic resistance coefficient [s²/m⁵]."""
        if self.A <= 0 or self.D <= 0:
            return 0.0
        return self.L * self.Lambda / (2 * self.g * self.D * self.A**2)

    def hydraulic_C(self) -> float:
        """Hydraulic capacitance per unit length [m²]."""
        if self.a <= 0:
            return 0.0
        return self.g * self.A / (self.a**2)

    @property
    def n_Q(self) -> int:
        """Number of flow states."""
        return self.Nb + 1 if self.Nb > 0 else 1

    @property
    def n_Hc(self) -> int:
        """Number of internal head states."""
        return self.Nb if self.Nb > 0 else 0
