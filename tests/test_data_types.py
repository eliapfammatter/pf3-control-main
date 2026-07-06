import numpy as np
from data_types import ModelInputs, ModelOutputs, ModelState, Trajectory, TrajectorySet, Artefact


def test_model_inputs():
    mi = ModelInputs(y_T=0.5, N_T=300.0, N_P=100.0)
    assert mi.y_T == 0.5
    assert mi.N_T == 300.0
    assert mi.N_P == 100.0


def test_model_outputs():
    mo = ModelOutputs(H_T=5.0, Q_T=0.2, H_P1=10.0, H_P2=12.0)
    assert mo.H_T == 5.0
    assert mo.Q_T == 0.2
    assert mo.H_P1 == 10.0
    assert mo.H_P2 == 12.0


def test_model_state():
    mi = ModelInputs(y_T=0.5, N_T=300.0, N_P=100.0)
    mo = ModelOutputs(H_T=5.0, Q_T=0.2, H_P1=10.0, H_P2=12.0)
    ms = ModelState(t=1.0, inputs=mi, outputs=mo)
    assert ms.t == 1.0
    assert ms.inputs is mi
    assert ms.outputs is mo


def test_trajectory_interpolates_linearly():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 10.0, 20.0])
    traj = Trajectory(times=times, values=values)
    assert traj(0.5) == np.interp(0.5, times, values)
    assert traj(1.5) == np.interp(1.5, times, values)


def test_trajectory_set_n_p_optional():
    traj = Trajectory(times=np.array([0.0, 1.0]), values=np.array([0.0, 1.0]))
    ts = TrajectorySet(y_T=traj, N_T=traj, H_ref=traj, N_P=None)
    assert ts.N_P is None


def test_artefact_fields():
    art = Artefact(
        t=np.array([0.0, 1.0]),
        inputs={"y_T": np.array([0.5, 0.6])},
        outputs={"H_T": np.array([5.0, 5.1])},
        metadata={"git_hash": "abc123", "timestamp": "2026-07-03T00:00:00"},
    )
    assert art.predictions is None
    assert art.metadata["git_hash"] == "abc123"
