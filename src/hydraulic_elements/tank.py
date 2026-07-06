import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.hydraulic_elements.element import Element


@dataclass
class Tank(Element):
    """
    STANK element - surge tank with free surface.

    Physics:
      A * dH/dt = Q_in - Q_out  (mass conservation)

    The tank is a storage node (has H state variable).
    """

    name: str
    A: float  # Surface area [m²]
    H0: float = 0.0  # Initial water level [m]
    g: float = 9.806  # Gravity [m/s²]

    def __post_init__(self):
        pass

    @classmethod
    def from_dat(cls, file_path: Path) -> "Tank":
        """Load from SIMSEN STANK DAT file."""
        file_path = Path(file_path)
        params: Dict[str, Any] = {"name": file_path.stem}

        with open(file_path, "r") as f:
            content = f.read()

        for line in content.split("\n"):
            stripped = line.strip()
            for key in ["A", "g"]:
                pattern = rf"^{re.escape(key)}\s+\["
                if re.match(pattern, stripped) and ":" in stripped:
                    try:
                        params[key] = float(stripped.split(":")[1].strip())
                    except (IndexError, ValueError):
                        pass

        # Parse initial H (look for H or Hc)
        in_initial = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "INITIAL CONDITIONS" in stripped:
                in_initial = True
                continue
            if in_initial:
                if stripped.startswith("-") and ":" not in stripped:
                    break
                h_match = re.match(
                    r"^Hc?\s+\[m\]\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                    stripped,
                )
                if h_match:
                    params["H0"] = float(h_match.group(1))
                    break

        if "A" not in params:
            raise ValueError(f"Missing area A in {file_path}")

        params.setdefault("H0", 0.0)
        params.setdefault("g", 9.806)

        return cls(**params)
