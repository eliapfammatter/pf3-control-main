import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.hydraulic_elements.element import Element


@dataclass
class DiscreteLoss(Element):
    """
    DLOSS element - discrete head loss (elbow, valve, etc.).

    Physics:
      ΔH = K / (2gA²) * |Q| * Q

    This is modeled as a branch with very small inductance and
    resistance coefficient R = K / (2gA²).
    """

    name: str
    K: float  # Loss coefficient [-]
    Aref: float  # Reference area [m²]
    Q0: float = 0.0  # Initial flow [m³/s]
    g: float = 9.806  # Gravity [m/s²]

    def __post_init__(self):
        pass

    @classmethod
    def from_dat(cls, file_path: Path) -> "DiscreteLoss":
        """Load from SIMSEN DLOSS DAT file."""
        file_path = Path(file_path)
        params: Dict[str, Any] = {"name": file_path.stem}

        with open(file_path, "r") as f:
            content = f.read()

        for line in content.split("\n"):
            for key in ["K", "Aref", "g"]:
                pattern = rf"^{re.escape(key)}\s+\["
                stripped = line.strip()
                if re.match(pattern, stripped) and ":" in stripped:
                    try:
                        params[key] = float(line.split(":")[1].strip())
                    except (IndexError, ValueError):
                        pass

        if "K" not in params or "Aref" not in params:
            raise ValueError(f"Missing K or Aref in {file_path}")

        # Parse initial Q (SIMSEN uses Q [m3/s] for DLOSS, not Q1)
        in_initial = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "INITIAL CONDITIONS" in stripped:
                in_initial = True
                continue
            if in_initial:
                if stripped.startswith("-") and ":" not in stripped:
                    break
                # Try both Q [m3/s] and Q1 [m3/s] formats
                q_match = re.match(
                    r"^Q1?\s+\[m3/s\]\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)",
                    stripped,
                )
                if q_match:
                    params["Q0"] = float(q_match.group(1))
                    break

        params.setdefault("Q0", 0.0)
        params.setdefault("g", 9.806)

        return cls(**params)

    def hydraulic_R(self) -> float:
        """Hydraulic resistance coefficient [s²/m⁵]."""
        if self.Aref <= 0:
            return 0.0
        return self.K / (2 * self.g * self.Aref**2)
