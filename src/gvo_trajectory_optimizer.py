"""
Guide Vane Trajectory Optimizer.

Pre-filters y_T (guide vane) trajectories to respect servomotor speed limits.
Uses motor_turns space for linear rate constraints.

This is a standalone optimizer that runs BEFORE the NMPC simulation.
"""

from pathlib import Path
from typing import Optional

import casadi as ca
import numpy as np

from src.gvo_kinematics import GVOKinematics, compute_motor_speed
from src.helpers import FPoints


def optimize_y_T_trajectory(
    fpoints: FPoints,
    dt: float,
    t_end: float,
    motor_speed_max: float = 60.0,
    gvo: Optional[GVOKinematics] = None,
    verbose: bool = False,
) -> FPoints:
    """
    Optimize y_T trajectory to respect servomotor speed limit.

    Resamples input to regular grid, then optimizes.

    Args:
        fpoints: Input FPoints trajectory to optimize
        dt: Time step for optimization grid [s]
        t_end: End time [s]
        motor_speed_max: Max servomotor speed [rpm]
        gvo: GVOKinematics instance (created if None)
        verbose: Print solver output

    Returns:
        FPoints: Optimized trajectory on regular grid
    """
    if gvo is None:
        gvo = GVOKinematics()

    # Resample to regular grid
    fpoints = fpoints.resample(dt=dt, t_end=t_end)
    t_ref = fpoints.time
    y_T_ref = fpoints.values
    N = len(t_ref)

    # Decision variables: motor_turns (linear rate constraint)
    motor_turns = ca.MX.sym("motor_turns", N)

    # Convert reference to motor_turns
    motor_turns_ref = np.array([gvo.y_to_motor_turns(y) for y in y_T_ref])

    # Cost: minimize squared deviation from reference
    cost = ca.sum1((motor_turns - motor_turns_ref) ** 2)

    # Constraints: servomotor rate
    g = []
    lbg = []
    ubg = []

    max_speed_turns_per_sec = motor_speed_max / 60.0  # [turns/s]

    for k in range(1, N):
        dt = t_ref[k] - t_ref[k - 1]
        max_delta = max_speed_turns_per_sec * dt

        # |motor_turns[k] - motor_turns[k-1]| <= max_delta
        delta = motor_turns[k] - motor_turns[k - 1]
        g.append(delta)
        lbg.append(-max_delta)
        ubg.append(max_delta)

    # Bounds on motor_turns
    lbx = [gvo.motor_turns_min] * N
    ubx = [gvo.motor_turns_max] * N

    # Build and solve NLP
    nlp = {"x": motor_turns, "f": cost, "g": ca.vertcat(*g) if g else ca.MX(0)}

    opts = {
        "ipopt.print_level": 5 if verbose else 0,
        "ipopt.sb": "yes",
        "print_time": verbose,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    sol = solver(
        x0=motor_turns_ref,
        lbx=lbx,
        ubx=ubx,
        lbg=lbg if lbg else [],
        ubg=ubg if ubg else [],
    )

    motor_turns_opt = np.array(sol["x"]).flatten()

    # Convert back to y_T
    y_T_opt = np.array([gvo.motor_turns_to_y(m) for m in motor_turns_opt])

    return FPoints.from_arrays(
        time=t_ref,
        values=y_T_opt,
        x_unit=fpoints.x_unit,
        y_unit=fpoints.y_unit,
    )


