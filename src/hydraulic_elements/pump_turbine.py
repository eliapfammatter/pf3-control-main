import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from ..characteristics import Characteristics
from .element import Element


@dataclass
class PumpTurbine(Element):
    """
    FTURB element - Francis turbine or pump-turbine.

    Based on SIMSEN FTURB model: an LRH circuit where:
      - L (inductance) = Lequ / (g × Amean) - water inertia
      - R (resistance) = guide vane closure effect
      - H (head source) = characteristic curves (energy transfer)

    The turbine is a SEPARATE BRANCH element connecting inlet and outlet nodes,
    not just an H_source attached to a pipe. This properly models the head
    discontinuity across the turbine.

    Momentum equation:
      L_turb × dQ/dt = H_inlet - H_outlet - R×|Q|×Q + H_characteristic

    Sign convention (SIMSEN):
      - Turbine mode: N > 0, Q > 0, T > 0, H > 0
      - Pump mode: N < 0, Q < 0, T > 0, H > 0
    """

    name: str
    Lequ: float  # Equivalent length [m]
    Amean: float  # Mean cross-section area [m²]
    Dref: float  # Reference diameter [m]
    characteristic: Characteristics
    Q0: float = 0.0  # Initial flow [m³/s]
    N0: float = 0.0  # Initial rotational speed [rpm]
    g: float = 9.806  # Gravity [m/s²]

    # Characteristic curve object (set after creation)

    # External input keys
    _y_key: str = field(default="y_T", repr=False)
    _N_key: str = field(default="N_T", repr=False)

    def __post_init__(self):
        pass

    @classmethod
    def from_dat(cls, file_path: Path, interp_method: str = "linear") -> "PumpTurbine":
        """
        Load turbine parameters from SIMSEN FTURB DAT file.

        Parses both FTURB (turbine) and PUMP (also FTURB format in SIMSEN).

        Parameters
        ----------
        file_path : Path
            Path to the DAT file
        interp_method : str
            Interpolation method for characteristic curves ("linear", "structured", etc.)
        """
        file_path = Path(file_path)
        params: dict[str, Any] = {"name": file_path.stem}

        with open(file_path, "r") as f:
            content = f.read()

        # Parse PARAMETERS section
        param_map = {
            "Lequ": "Lequ",
            "Amean": "Amean",
            "Dref": "Dref",
            "g": "g",
        }

        in_params = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "- PARAMETERS :" in stripped:
                in_params = True
                continue
            if in_params:
                if stripped.startswith("-") and ":" not in stripped:
                    break
                for dat_key, param_key in param_map.items():
                    pattern = rf"^{re.escape(dat_key)}\s+\["
                    if re.match(pattern, stripped) and ":" in stripped:
                        try:
                            params[param_key] = float(stripped.split(":")[1].strip())
                        except (IndexError, ValueError):
                            pass

        # Parse RATED VALUES section (for reference)
        in_rated = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "- RATED VALUES :" in stripped:
                in_rated = True
                continue
            if in_rated:
                if stripped.startswith("-") and ":" not in stripped:
                    break
                # Skip lines starting with $ (comments)
                if stripped.startswith("$"):
                    continue
                # Hn, Qn, Nn, Tn
                for key in ["Hn", "Qn", "Nn", "Tn"]:
                    pattern = rf"^{key}\s+\["
                    if re.match(pattern, stripped) and ":" in stripped:
                        try:
                            params[f"_{key}"] = float(stripped.split(":")[1].strip())
                        except (IndexError, ValueError):
                            pass

        # Parse INITIAL CONDITIONS section
        in_initial = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "- INITIAL CONDITIONS :" in stripped:
                in_initial = True
                continue
            if in_initial:
                if stripped.startswith("-") and ":" not in stripped:
                    break
                # Q [m3/s]
                q_match = re.match(
                    r"^Q\s+\[m3/s\]\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                    stripped,
                )
                if q_match:
                    params["Q0"] = float(q_match.group(1))
                # N [rpm]
                n_match = re.match(
                    r"^N\s+\[rpm\]\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                    stripped,
                )
                if n_match:
                    params["N0"] = float(n_match.group(1))

        # Parse <yN11Q11T11 File> section for characteristic file
        char_file_name = None
        in_char_file = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "<yN11Q11T11 File>" in stripped:
                in_char_file = True
                continue
            if in_char_file:
                if "</yN11Q11T11 File>" in stripped:
                    break
                if stripped and not stripped.startswith(";"):
                    char_file_name = stripped
                    break

        # Extract rated values before removing internal keys
        rated_values = {
            "Hn": params.get("_Hn"),
            "Qn": params.get("_Qn"),
            "Nn": params.get("_Nn"),
            "Tn": params.get("_Tn"),
        }

        if char_file_name is None:
            raise ValueError(
                f"No characteristic file (<yN11Q11T11 File> section) found in {file_path}"
            )

        missing_rated = [k for k, v in rated_values.items() if v is None]
        if missing_rated:
            raise ValueError(f"Missing rated values in {file_path}: {missing_rated}")

        # Search for characteristic file in multiple locations
        search_paths = [
            file_path.parent / char_file_name,  # Same directory
            file_path.parent / "missing_files" / char_file_name,  # missing_files subdir
            file_path.parent.parent
            / "missing_files"
            / char_file_name,  # Parent's missing_files
        ]

        char_file_path = None
        for path in search_paths:
            if path.exists():
                char_file_path = path
                break

        if char_file_path is None:
            raise ValueError(
                f"Characteristic file '{char_file_name}' not found. "
                f"Searched in: {[str(p) for p in search_paths]}"
            )

        characteristic = Characteristics(
            d_ref=params["Dref"],
            h_n=rated_values["Hn"],
            q_n=rated_values["Qn"],
            t_n=rated_values["Tn"],
            n_n=rated_values["Nn"],
            char_file=char_file_path,
            interp_method=interp_method,
        )

        params.setdefault("characteristic", characteristic)

        # Remove internal keys (rated values) before creating object
        internal_keys = [k for k in params if k.startswith("_")]
        for k in internal_keys:
            del params[k]

        # Defaults
        params.setdefault("Q0", 0.0)
        params.setdefault("N0", 0.0)
        params.setdefault("g", 9.806)

        required = ["Lequ", "Amean", "Dref"]
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing parameters in {file_path}: {missing}")

        instance = cls(**params)

        return instance

    def hydraulic_L(self) -> float:
        """
        Hydraulic inductance [s²/m²].

        L = Lequ / (g × Amean)

        This represents the water inertia through the turbine passage.

        Note: Elements with Lequ=0 (like SIMSEN pumps) should be added with
        on_pipe parameter, which attaches them as H_sources to the pipe
        instead of creating separate branches. This matches SIMSEN's treatment.
        """
        if self.Amean <= 0 or self.Lequ <= 0:
            return 0.0  # No inductance - should be attached as H_source
        return self.Lequ / (self.g * self.Amean)

    def compute_H_characteristic(self, Q: float, t: float, state_dict: Dict) -> float:
        """
        Compute turbine/pump head from characteristic curves (quasi-steady).

        This returns only the characteristic H from the Suter transform,
        representing the energy transfer between fluid and runner.

        Used internally as H_source in the momentum equation, where the
        L×dQ/dt term is handled separately by the equation structure.

        Parameters
        ----------
        Q : float
            Flow rate [m³/s]
        t : float
            Time [s]
        state_dict : dict
            External inputs (must contain y_key and N_key)

        Returns
        -------
        H : float
            Head [m] from characteristic curves only
        """
        if self.characteristic is None:
            raise RuntimeError(
                f"FrancisTurbine '{self.name}' has no characteristic set. "
                "Ensure the DAT file contains a valid characteristic file path."
            )

        y = state_dict.get(self._y_key, 0.5)
        N = state_dict.get(self._N_key, self.N0)

        return self.characteristic.compute_H(y, N, Q)

    def compute_H(
        self, Q: float, t: float, state_dict: Dict, dQ_dt: float = 0.0
    ) -> float:
        """
        Compute total head across turbine including dynamic effects.

        The FTURB element is modeled as an LRH circuit (see SIMSEN FTurb.pdf):

            H = H_characteristic + L × dQ/dt

        where:
            - H_characteristic = f(y, N, Q) from the Suter transform
            - L = Lequ / (g × Amean) is the hydraulic inductance [s²/m²]
            - dQ/dt is the flow acceleration [m³/s²]

        This matches SIMSEN's FTURB-H output, which represents the total
        head across the machine element including water inertia effects.

        Parameters
        ----------
        Q : float
            Flow rate [m³/s]
        t : float
            Time [s]
        state_dict : dict
            External inputs (must contain y_key and N_key)
        dQ_dt : float
            Flow rate derivative [m³/s²]. Default 0 for steady-state.

        Returns
        -------
        H : float
            Total head [m] = H_characteristic + L × dQ/dt
        """
        H_char = self.compute_H_characteristic(Q, t, state_dict)
        L = self.hydraulic_L()
        return H_char + L * dQ_dt
