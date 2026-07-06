"""
Hydraulic Network using Incidence Matrix Formulation

This module provides a principled approach to hydraulic network simulation
based on electrical circuit theory. The incidence matrix encodes network
topology, and Kirchhoff's laws become simple matrix operations.

Architecture:
  1. ELEMENTS (physics) - Load from SIMSEN DAT files
     - Pipe: PIPEZ elements (distributed or lumped)
     - DiscreteLoss: DLOSS elements (elbows, valves)
     - Tank: STANK elements (free surface storage)
     - FrancisTurbine: Turbines and pumps with characteristic curves

  2. NETWORK (topology) - Assemble by connecting elements
     Via ElementNetworkBuilder (for element-level construction):
       builder.add(element, from_node=..., to_node=...)
       network = builder.build()
     Or directly on HydraulicNetwork (for branch/node-level construction):
       network.add_node(...); network.add_branch(...); network.build()

Analogy to electrical circuits:
  - Head (H) ↔ Voltage (V)
  - Flow (Q) ↔ Current (I)
  - Hydraulic resistance ↔ Resistance R
  - Hydraulic inductance (inertia) ↔ Inductance L
  - Hydraulic capacitance (compliance) ↔ Capacitance C

Standard form:
  L @ dQ/dt = A.T @ H - R(Q) + H_sources     (KVL for each branch)
  C @ dH/dt = -A @ Q + q_sources             (KCL for capacitor nodes)
  0 = A_j @ Q                                (KCL at junctions)

where A is the incidence matrix:
  A[i,j] = +1 if branch j leaves node i
  A[i,j] = -1 if branch j enters node i
  A[i,j] = 0  otherwise
"""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from .hydraulic_elements import DiscreteLoss, Pipe, PumpTurbine, Tank
from .hydraulic_elements.element import Element

# =============================================================================
# Node and Branch Types
# =============================================================================


class NodeType(Enum):
    """Node types in the hydraulic network."""

    JUNCTION = "junction"  # No storage (algebraic KCL constraint)
    CAPACITOR = "capacitor"  # Has storage/compressibility (internal Hc)
    TANK = "tank"  # Free surface with area (H state)
    FIXED = "fixed"  # Fixed head boundary (reference/reservoir)


@dataclass
class HydraulicNode:
    """A node in the hydraulic network."""

    name: str
    node_type: NodeType
    C: float = 0.0  # Capacitance [m²] (area for tanks, gA/a² for pipes)
    H0: float = 0.0  # Initial or fixed head [m]

    @property
    def is_state(self) -> bool:
        """Whether this node contributes a state variable."""
        return self.node_type in (NodeType.CAPACITOR, NodeType.TANK)

    @property
    def is_algebraic(self) -> bool:
        """Whether this node requires algebraic constraint (KCL)."""
        return self.node_type == NodeType.JUNCTION

    @property
    def is_fixed(self) -> bool:
        """Whether this node has fixed head."""
        return self.node_type == NodeType.FIXED


@dataclass
class HydraulicBranch:
    """A branch (pipe segment) connecting two nodes."""

    name: str
    node_from: str  # Source node (flow leaves this node)
    node_to: str  # Sink node (flow enters this node)
    L: float  # Inductance [s²/m²] = ρL/(gA) simplified
    R_coef: float  # Resistance coefficient [s²/m⁵]
    Q0: float = 0.0  # Initial flow [m³/s]

    # Optional: machine (pump/turbine) on this branch
    # H_source(Q, t, state_dict) -> head gain [m] (positive = pump)
    H_source: Optional[Callable[[float, float, Dict], float]] = None
    H_source_name: Optional[str] = None  # Name for debugging


# =============================================================================
# Hydraulic Network (Incidence Matrix Formulation)
# =============================================================================


class HydraulicNetwork:
    """
    Hydraulic network using incidence matrix formulation.

    The incidence matrix A encodes the network topology:
      A[i,j] = +1 if branch j leaves node i
      A[i,j] = -1 if branch j enters node i

    This gives Kirchhoff's laws as matrix operations:
      KCL: A @ Q = 0 (at junctions, sum of flows = 0)
      KVL: ΔH_branch = -A.T @ H_node (branch head drop)

    State vector: x = [Q_1, ..., Q_m, H_c1, ..., H_cn]
      - Q_i: flow through branch i
      - H_cj: head at capacitor/tank node j

    Dynamics:
      L_i * dQ_i/dt = (H_from - H_to) - R_i*|Q_i|*Q_i + H_source_i
      C_j * dH_cj/dt = net_inflow_j = -[A @ Q]_j

    Junction nodes (no storage) are handled as algebraic constraints.
    Their heads are solved from KVL equations at each timestep.
    """

    def __init__(self):
        # User-defined components
        self.nodes: Dict[str, HydraulicNode] = {}
        self.branches: Dict[str, HydraulicBranch] = {}

        # Pipe group metadata (set by ElementNetworkBuilder for Q_{pipe}_in/out)
        self._pipe_groups: Dict[str, Dict] = {}

        # Build flag
        self._built = False

        # Matrices (set by build())
        self._A: np.ndarray = None  # Full incidence matrix [n_nodes x n_branches]
        self._A_c: np.ndarray = None  # Rows for capacitor/tank nodes
        self._A_j: np.ndarray = None  # Rows for junction nodes
        self._A_f: np.ndarray = None  # Rows for fixed nodes
        self._L_diag: np.ndarray = None  # Branch inductances
        self._C_diag: np.ndarray = None  # Capacitor node capacitances
        self._R_coef: np.ndarray = None  # Branch resistance coefficients

        # Node/branch ordering (set by build())
        self._branch_order: List[str] = []
        self._node_order: List[str] = []
        self._cap_nodes: List[str] = []  # Capacitor + tank nodes (have H state)
        self._junc_nodes: List[str] = []  # Junction nodes (algebraic)
        self._fixed_nodes: List[str] = []  # Fixed head nodes (boundary)

        # Index mappings
        self._branch_idx: Dict[str, int] = {}
        self._node_idx: Dict[str, int] = {}
        self._Q_indices: Dict[str, int] = {}  # State index for Q
        self._H_indices: Dict[str, int] = {}  # State index for H

        # Dimensions
        self._n_branches: int = 0
        self._n_nodes: int = 0
        self._n_Q: int = 0
        self._n_H: int = 0
        self._n_states: int = 0
        self._n_junctions: int = 0

    # -------------------------------------------------------------------------
    # Building the network
    # -------------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        node_type: Union[NodeType, str],
        C: float = 0.0,
        H0: float = 0.0,
    ):
        """
        Add a node to the network.

        Parameters
        ----------
        name : str
            Unique node identifier
        node_type : NodeType or str
            One of: "junction", "capacitor", "tank", "fixed"
        C : float
            Capacitance [m²]. For tanks, this is the surface area.
            For internal pipe nodes, this is g*A/a².
        H0 : float
            Initial head (for capacitor/tank) or fixed head (for fixed nodes).
        """
        if isinstance(node_type, str):
            node_type = NodeType(node_type.lower())

        self.nodes[name] = HydraulicNode(name, node_type, C, H0)
        self._built = False

    def add_branch(
        self,
        name: str,
        node_from: str,
        node_to: str,
        L: float,
        R_coef: float,
        Q0: float = 0.0,
        H_source: Callable = None,
        H_source_name: str = None,
    ):
        """
        Add a branch connecting two nodes.

        Parameters
        ----------
        name : str
            Unique branch identifier
        node_from : str
            Source node (flow leaves this node when Q > 0)
        node_to : str
            Sink node (flow enters this node when Q > 0)
        L : float
            Hydraulic inductance [s²/m²]
        R_coef : float
            Resistance coefficient [s²/m⁵] for friction term R*|Q|*Q
        Q0 : float
            Initial flow [m³/s]
        H_source : callable, optional
            Head source function H_source(Q, t, state_dict) -> head [m]
            Positive = adds head (pump), negative = removes head (turbine loss)
        H_source_name : str, optional
            Name of the machine for debugging
        """
        self.branches[name] = HydraulicBranch(
            name=name,
            node_from=node_from,
            node_to=node_to,
            L=L,
            R_coef=R_coef,
            Q0=Q0,
            H_source=H_source,
            H_source_name=H_source_name,
        )
        self._built = False

    def build(self):
        """
        Build the incidence matrix and prepare for simulation.

        This method:
        1. Orders nodes and branches
        2. Constructs the incidence matrix A
        3. Partitions A by node type (capacitor, junction, fixed)
        4. Builds state index mappings
        """
        self._n_branches = len(self.branches)
        self._n_nodes = len(self.nodes)

        if self._n_branches == 0:
            raise ValueError("Network has no branches")
        if self._n_nodes == 0:
            raise ValueError("Network has no nodes")

        # Order branches and nodes (deterministic)
        self._branch_order = sorted(self.branches.keys())
        self._node_order = sorted(self.nodes.keys())

        self._branch_idx = {name: i for i, name in enumerate(self._branch_order)}
        self._node_idx = {name: i for i, name in enumerate(self._node_order)}

        # Validate: all branch endpoints exist
        for branch in self.branches.values():
            if branch.node_from not in self.nodes:
                raise ValueError(
                    f"Branch '{branch.name}' references unknown node '{branch.node_from}'"
                )
            if branch.node_to not in self.nodes:
                raise ValueError(
                    f"Branch '{branch.name}' references unknown node '{branch.node_to}'"
                )

        # Build incidence matrix A
        # Convention: A[i,j] = +1 if branch j LEAVES node i (Q flows out)
        #             A[i,j] = -1 if branch j ENTERS node i (Q flows in)
        A = np.zeros((self._n_nodes, self._n_branches))

        for branch_name in self._branch_order:
            branch = self.branches[branch_name]
            j = self._branch_idx[branch_name]
            i_from = self._node_idx[branch.node_from]
            i_to = self._node_idx[branch.node_to]

            A[i_from, j] = +1.0  # Flow leaves from-node
            A[i_to, j] = -1.0  # Flow enters to-node

        self._A = A

        # Partition nodes by type
        self._cap_nodes = [
            n
            for n in self._node_order
            if self.nodes[n].node_type in (NodeType.CAPACITOR, NodeType.TANK)
        ]
        self._junc_nodes = [
            n for n in self._node_order if self.nodes[n].node_type == NodeType.JUNCTION
        ]
        self._fixed_nodes = [
            n for n in self._node_order if self.nodes[n].node_type == NodeType.FIXED
        ]

        # Extract submatrices for each node type
        cap_rows = [self._node_idx[n] for n in self._cap_nodes]
        junc_rows = [self._node_idx[n] for n in self._junc_nodes]
        fixed_rows = [self._node_idx[n] for n in self._fixed_nodes]

        self._A_c = A[cap_rows, :] if cap_rows else np.zeros((0, self._n_branches))
        self._A_j = A[junc_rows, :] if junc_rows else np.zeros((0, self._n_branches))
        self._A_f = A[fixed_rows, :] if fixed_rows else np.zeros((0, self._n_branches))

        # Build diagonal matrices for L and C
        self._L_diag = np.array([self.branches[n].L for n in self._branch_order])
        self._R_coef = np.array([self.branches[n].R_coef for n in self._branch_order])
        self._C_diag = np.array([self.nodes[n].C for n in self._cap_nodes])

        # Build state index mappings
        # State vector: [Q_branch1, Q_branch2, ..., H_cap1, H_cap2, ...]
        self._Q_indices.clear()
        self._H_indices.clear()

        idx = 0
        for name in self._branch_order:
            self._Q_indices[name] = idx
            idx += 1

        self._n_Q = idx

        for name in self._cap_nodes:
            self._H_indices[name] = idx
            idx += 1

        self._n_H = len(self._cap_nodes)
        self._n_states = idx
        self._n_junctions = len(self._junc_nodes)

        self._built = True

    def print_summary(self):
        """Print network summary."""
        if not self._built:
            self.build()

        print(f"HydraulicNetwork Summary:")
        print(f"  Branches: {self._n_branches} (Q states)")
        print(f"  Nodes: {self._n_nodes} total")
        print(f"    - Capacitor/Tank: {len(self._cap_nodes)} (H states)")
        print(f"    - Junction: {len(self._junc_nodes)} (algebraic)")
        print(f"    - Fixed: {len(self._fixed_nodes)} (boundary)")
        print(f"  Total states: {self._n_states}")
        print(f"\nIncidence matrix A shape: {self._A.shape}")

    def print_incidence_matrix(self, max_cols: int = 10):
        """Print the incidence matrix (for debugging)."""
        if not self._built:
            self.build()

        print("\nIncidence Matrix A:")
        print("  Rows = nodes, Cols = branches")
        print("  +1 = flow leaves node, -1 = flow enters node\n")

        # Column headers
        cols_to_show = min(max_cols, self._n_branches)
        header = "            " + "".join(
            f"{name[:8]:>9}" for name in self._branch_order[:cols_to_show]
        )
        if cols_to_show < self._n_branches:
            header += "  ..."
        print(header)

        # Rows
        for i, node_name in enumerate(self._node_order):
            node = self.nodes[node_name]
            type_char = node.node_type.value[0].upper()  # J, C, T, F
            row_str = f"{node_name[:10]:>10}({type_char})"
            for j in range(cols_to_show):
                val = self._A[i, j]
                if val == 0:
                    row_str += "        ."
                elif val > 0:
                    row_str += "       +1"
                else:
                    row_str += "       -1"
            if cols_to_show < self._n_branches:
                row_str += "  ..."
            print(row_str)

    # -------------------------------------------------------------------------
    # State vector operations
    # -------------------------------------------------------------------------

    def get_initial_state(self) -> np.ndarray:
        """Get initial state vector from node/branch initial values."""
        if not self._built:
            self.build()

        state = np.zeros(self._n_states)

        # Q states
        for name in self._branch_order:
            idx = self._Q_indices[name]
            state[idx] = self.branches[name].Q0

        # H states
        for name in self._cap_nodes:
            idx = self._H_indices[name]
            state[idx] = self.nodes[name].H0

        return state

    def unpack_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack state vector into Q and H_c arrays."""
        Q = state[: self._n_Q]
        H_c = state[self._n_Q :]
        return Q, H_c

    def pack_state(self, Q: np.ndarray, H_c: np.ndarray) -> np.ndarray:
        """Pack Q and H_c arrays into state vector."""
        return np.concatenate([Q, H_c])

    # -------------------------------------------------------------------------
    # Physics: resistance and head sources
    # -------------------------------------------------------------------------

    def _compute_friction(self, Q: np.ndarray) -> np.ndarray:
        """
        Compute friction head loss for each branch.

        Returns R_coef * |Q| * Q for each branch.
        """
        return self._R_coef * np.abs(Q) * Q

    def _compute_H_sources(
        self, Q: np.ndarray, t: float, state_dict: Dict
    ) -> np.ndarray:
        """
        Compute head sources (machines) for each branch.

        Returns array of head gains [m] for each branch.
        """
        H_src = np.zeros(self._n_branches)

        for i, name in enumerate(self._branch_order):
            branch = self.branches[name]
            if branch.H_source is not None:
                H_src[i] = branch.H_source(Q[i], t, state_dict)

        return H_src

    # -------------------------------------------------------------------------
    # Junction head solver (algebraic constraint from KCL)
    # -------------------------------------------------------------------------

    def _solve_junction_heads(
        self,
        Q: np.ndarray,
        H_c: np.ndarray,
        t: float,
        state_dict: Dict,
        friction: np.ndarray,
        H_src: np.ndarray,
    ) -> np.ndarray:
        """
        Solve for junction heads using KCL + KVL.

        At junctions with no storage (C=0), we have the constraint that
        the sum of flows must balance. Taking the time derivative of KCL:

            Σ dQ_in/dt = Σ dQ_out/dt

        Substituting KVL for each branch i connected to junction j:

            L_i * dQ_i/dt = (H_from - H_to) - R_i*|Q_i|*Q_i + H_src_i

        For branches entering junction j (flow goes into j):
            L_i * dQ_i/dt = H_neighbor - H_j - friction_i + H_src_i

        For branches leaving junction j (flow goes out of j):
            L_i * dQ_i/dt = H_j - H_neighbor - friction_i + H_src_i

        Summing all branches at junction j and using KCL (Σ dQ_in = Σ dQ_out):

            H_j × Σ(1/L_i) = Σ_entering[(H_neighbor - friction + H_src)/L_i]
                          + Σ_leaving[(H_neighbor + friction - H_src)/L_i]

        This gives a direct algebraic equation for H_j. For multiple junctions,
        it becomes a linear system M @ H_j = b where M is the "junction
        admittance matrix" (analogous to nodal admittance in circuit theory).
        """
        if self._n_junctions == 0:
            return np.array([])

        n_junctions = self._n_junctions
        n_branches = self._n_branches

        # friction and H_src are precomputed by caller to avoid redundant work

        # Fixed node heads
        H_f_dict = {n: self.nodes[n].H0 for n in self._fixed_nodes}

        # Capacitor node heads (from state)
        H_c_dict = {n: H_c[i] for i, n in enumerate(self._cap_nodes)}

        # Build junction admittance matrix M and RHS vector b
        # M[i,j] @ H_j[j] = b[i]  for junction i
        M = np.zeros((n_junctions, n_junctions))
        b = np.zeros(n_junctions)

        # Map junction names to indices
        junc_idx = {name: i for i, name in enumerate(self._junc_nodes)}

        # Process each branch
        for k, branch_name in enumerate(self._branch_order):
            branch = self.branches[branch_name]
            L_k = self._L_diag[k]

            if L_k <= 0:
                continue  # Skip zero-inductance branches

            node_from = branch.node_from
            node_to = branch.node_to
            fric_k = friction[k]
            src_k = H_src[k]

            # Contribution to KCL at each node
            # dQ_k/dt = (H_from - H_to - fric_k + src_k) / L_k
            #
            # At node_from: this branch removes flow (dQ leaves)
            # At node_to: this branch adds flow (dQ enters)

            # Get head at each endpoint (or mark as junction unknown)
            from_is_junc = node_from in junc_idx
            to_is_junc = node_to in junc_idx

            H_from_known = None
            H_to_known = None

            if not from_is_junc:
                if node_from in H_c_dict:
                    H_from_known = H_c_dict[node_from]
                elif node_from in H_f_dict:
                    H_from_known = H_f_dict[node_from]

            if not to_is_junc:
                if node_to in H_c_dict:
                    H_to_known = H_c_dict[node_to]
                elif node_to in H_f_dict:
                    H_to_known = H_f_dict[node_to]

            inv_L = 1.0 / L_k

            # Process contribution to FROM node if it's a junction
            if from_is_junc:
                i = junc_idx[node_from]
                # At from-node, dQ leaves, so we ADD dQ/dt to LHS of KCL
                # dQ_k/dt = (H_from - H_to - fric + src) / L
                # Rearranging: contribution to junction i (node_from):
                #   H_i / L  (self admittance)
                M[i, i] += inv_L

                # Contribution from H_to
                if to_is_junc:
                    j = junc_idx[node_to]
                    M[i, j] -= inv_L  # -H_j / L
                elif H_to_known is not None:
                    b[i] += H_to_known * inv_L

                # Friction and source terms
                b[i] += (fric_k - src_k) * inv_L

            # Process contribution to TO node if it's a junction
            if to_is_junc:
                i = junc_idx[node_to]
                # At to-node, dQ enters, so we SUBTRACT dQ/dt from LHS of KCL
                # -dQ_k/dt = -(H_from - H_to - fric + src) / L
                #          = (-H_from + H_to + fric - src) / L
                # Contribution to junction i (node_to):
                #   H_i / L  (self admittance)
                M[i, i] += inv_L

                # Contribution from H_from
                if from_is_junc:
                    j = junc_idx[node_from]
                    M[i, j] -= inv_L  # -H_j / L
                elif H_from_known is not None:
                    b[i] += H_from_known * inv_L

                # Friction and source terms (note: opposite sign from 'from' case)
                b[i] += (-fric_k + src_k) * inv_L

        # Solve the linear system M @ H_j = b
        if n_junctions == 1:
            # Single junction: direct division
            if abs(M[0, 0]) > 1e-12:
                H_j = np.array([b[0] / M[0, 0]])
            else:
                H_j = np.array([0.0])
        else:
            # Multiple junctions: solve linear system
            try:
                H_j = np.linalg.solve(M, b)
            except np.linalg.LinAlgError:
                # Singular matrix - use pseudo-inverse as fallback
                H_j = np.linalg.lstsq(M, b, rcond=None)[0]

        return H_j

    def _assemble_full_H(self, H_c: np.ndarray, H_j: np.ndarray) -> np.ndarray:
        """Assemble full head vector in node order."""
        H_all = np.zeros(self._n_nodes)

        # Capacitor/tank nodes
        for i, name in enumerate(self._cap_nodes):
            H_all[self._node_idx[name]] = H_c[i]

        # Junction nodes
        for i, name in enumerate(self._junc_nodes):
            H_all[self._node_idx[name]] = H_j[i]

        # Fixed nodes
        for name in self._fixed_nodes:
            H_all[self._node_idx[name]] = self.nodes[name].H0

        return H_all

    # -------------------------------------------------------------------------
    # ODE right-hand side
    # -------------------------------------------------------------------------

    def ode_rhs(
        self, t: float, state: np.ndarray, external_inputs: Dict = None
    ) -> np.ndarray:
        """
        Compute state derivatives using incidence matrix formulation.

        State: [Q_1, ..., Q_m, H_c1, ..., H_cn]

        Equations:
          L_i * dQ_i/dt = -[A.T @ H]_i - R_i*|Q_i|*Q_i + H_src_i
                       = (H_from - H_to) - friction + machine_head

          C_j * dH_cj/dt = -[A_c @ Q]_j = net_inflow_j
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        external_inputs = external_inputs or {}
        state_dict = {"t": t, **external_inputs}

        # Unpack state
        Q, H_c = self.unpack_state(state)

        # Compute friction and H_sources once, reuse in junction solver and dQ/dt
        friction = self._compute_friction(Q)
        H_src = self._compute_H_sources(Q, t, state_dict)

        # Solve for junction heads (algebraic constraint)
        H_j = self._solve_junction_heads(
            Q, H_c, t, state_dict, friction=friction, H_src=H_src
        )

        # Assemble full H vector
        H_all = self._assemble_full_H(H_c, H_j)

        # Compute dQ/dt from KVL
        # L_i * dQ_i/dt = (H_from - H_to) - R_i*|Q_i|*Q_i + H_src_i
        #
        # Incidence matrix convention:
        #   A[from_node, branch] = +1 (flow leaves from_node)
        #   A[to_node, branch] = -1 (flow enters to_node)
        #
        # Therefore: (A.T @ H)[branch] = H_from - H_to
        Delta_H = self._A.T @ H_all  # = H_from - H_to for each branch

        dQ = np.zeros(self._n_Q)
        for i in range(self._n_Q):
            if self._L_diag[i] > 0:
                dQ[i] = (Delta_H[i] - friction[i] + H_src[i]) / self._L_diag[i]

        # Compute dH/dt from KCL for capacitor nodes
        # C_j * dH_cj/dt = net_inflow = -[A_c @ Q]_j
        # (negative because A has +1 for outflow, -1 for inflow)
        dH_c = np.zeros(self._n_H)
        if self._n_H > 0:
            Q_net = -self._A_c @ Q  # Net inflow to each capacitor node
            for i in range(self._n_H):
                if self._C_diag[i] > 0:
                    dH_c[i] = Q_net[i] / self._C_diag[i]

        return self.pack_state(dQ, dH_c)

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray = None,
        t_eval: np.ndarray = None,
        external_inputs_func: Callable[[float], Dict] = None,
        **solve_ivp_kwargs,
    ):
        """
        Simulate the network dynamics.

        Parameters
        ----------
        t_span : tuple
            (t_start, t_end)
        y0 : array, optional
            Initial state. If None, uses get_initial_state().
        t_eval : array, optional
            Times at which to store solution.
        external_inputs_func : callable, optional
            Function f(t) -> dict providing external inputs at time t.
        **solve_ivp_kwargs
            Additional arguments passed to scipy.integrate.solve_ivp.

        Returns
        -------
        result : OdeResult
            Solution object from solve_ivp.
        """
        if not self._built:
            self.build()

        if y0 is None:
            y0 = self.get_initial_state()

        def rhs(t, state):
            ext = external_inputs_func(t) if external_inputs_func else {}
            return self.ode_rhs(t, state, ext)

        solve_ivp_kwargs.setdefault("method", "RK45")

        return solve_ivp(rhs, t_span, y0, t_eval=t_eval, **solve_ivp_kwargs)

    # -------------------------------------------------------------------------
    # Pipe groups (for extract_results convenience keys)
    # -------------------------------------------------------------------------

    def register_pipe_group(self, group_name: str, branch_names: List[str]):
        """Register a group of branches that form a pipe (for extract_results)."""
        self._pipe_groups[group_name] = {"branch_names": branch_names}

    # -------------------------------------------------------------------------
    # Result extraction
    # -------------------------------------------------------------------------

    def extract_results(self, result) -> Dict[str, np.ndarray]:
        """
        Extract results with meaningful names.

        Returns dict with keys:
          - "t": time array
          - "Q_{branch_name}": flow through each branch
          - "H_{node_name}": head at each capacitor/tank node
          - "Q_{pipe}_in", "Q_{pipe}_out": inlet/outlet flows for pipe groups
        """
        out = {"t": result.t}

        # Q states
        for name in self._branch_order:
            idx = self._Q_indices[name]
            out[f"Q_{name}"] = result.y[idx, :]

        # H states
        for name in self._cap_nodes:
            idx = self._H_indices[name]
            out[f"H_{name}"] = result.y[idx, :]

        # Pipe group convenience keys (Q_{pipe}_in / Q_{pipe}_out)
        for group_name, info in self._pipe_groups.items():
            branch_names = info.get("branch_names", [])
            if branch_names:
                inlet = branch_names[0]
                if f"Q_{inlet}" in out:
                    out[f"Q_{group_name}_in"] = out[f"Q_{inlet}"]
                outlet = branch_names[-1]
                if f"Q_{outlet}" in out:
                    out[f"Q_{group_name}_out"] = out[f"Q_{outlet}"]

        return out

    def get_junction_heads(
        self, state: np.ndarray, t: float = 0.0, external_inputs: Dict = None
    ) -> Dict[str, float]:
        """
        Compute junction heads for a given state.

        Returns dict mapping junction name to head value.
        """
        external_inputs = external_inputs or {}
        state_dict = {"t": t, **external_inputs}

        Q, H_c = self.unpack_state(state)
        friction = self._compute_friction(Q)
        H_src = self._compute_H_sources(Q, t, state_dict)
        H_j = self._solve_junction_heads(Q, H_c, t, state_dict, friction, H_src)

        return {name: H_j[i] for i, name in enumerate(self._junc_nodes)}


# =============================================================================
# Element Network Builder
# =============================================================================


class ElementNetworkBuilder:
    """
    Builder for HydraulicNetwork from element-level objects (Pipe, Tank, etc.).

    Use this builder when constructing networks from SIMSEN element objects.
    It handles pipe expansion (Nb > 0 → multiple branches + internal nodes),
    machine attachment, and auto-junction creation.

    For direct branch/node construction (e.g., lumped models with pre-combined
    parameters), use HydraulicNetwork directly.

    Usage::

       >>> builder = ElementNetworkBuilder()
       >>> builder.add(Tank.from_dat("STANK.DAT"))
       >>> builder.add(Pipe.from_dat("L1.DAT"), "STANK", "j1")
       >>> builder.add(DiscreteLoss.from_dat("ELBOW1.DAT"), "j1", "j2")
       >>> builder.add(turbine, on_element="L1")
       >>> network = builder.build()  # Returns HydraulicNetwork

    The returned HydraulicNetwork is self-contained; the builder can be discarded.
    """

    def __init__(self, g: float = 9.806):
        """
        Initialize ElementNetworkBuilder.

        Parameters
        ----------
        g : float
            Gravitational acceleration [m/s²]
        """
        self.g = g

        # Internal network being built
        self._network = HydraulicNetwork()

        # Track elements
        self._elements: Dict[str, Element] = {}
        # {element_name: Element}

        # Track pipe expansions for result extraction
        self._pipe_expansions: Dict[str, Dict] = {}
        # {pipe_name: {"n_branches": int, "n_internal_nodes": int, ...}}

        # Track machines and their attachments
        self._machines: Dict[str, Tuple[str, Callable]] = {}
        # {machine_name: (branch_name, H_func)}

        # Track accumulated losses for pipes
        self._pipe_losses: Dict[str, float] = {}
        # {pipe_name: total_additional_R}

        self._built = False

    @property
    def network(self) -> HydraulicNetwork:
        """Access the underlying HydraulicNetwork (prefer using build() instead)."""
        return self._network

    # -------------------------------------------------------------------------
    # Element-based API (recommended)
    # -------------------------------------------------------------------------

    def add(
        self,
        element: Element,
        from_node: str | None = None,
        to_node: str | None = None,
        on_element: str | None = None,
        at_node: str | None = None,
        negate_Q0: bool | None = False,
    ) -> "ElementNetworkBuilder":
        """
        Add an element to the network.

        Parameters
        ----------
        element : Element
            The element to add (Pipe, DiscreteLoss, Tank, FrancisTurbine)

        For Pipe and DiscreteLoss:
            from_node : str
                Source node name
            to_node : str
                Destination node name
            negate_Q0 : bool
                If True, negate all Q0 values (use when flipping topology direction)

        For Tank:
            at_node : str, optional
                Node name (defaults to element.name)

        For FrancisTurbine:
            If Lequ > 0: from_node, to_node (separate branch)
            If Lequ = 0: on_element (attached as H_source to pipe's last branch)

        Returns
        -------
        self : ElementNetworkBuilder
            For method chaining

        Examples
        --------
        >>> net.add(Pipe.from_dat("L1.DAT"), "tank", "junction")
        >>> net.add(Tank.from_dat("STANK.DAT"))
        >>> pump = FrancisTurbine.from_dat("PUMP1.DAT", interp_method="structured")
        >>> pump._y_key = "y_P1"
        >>> pump._N_key = "N_P"
        >>> net.add(pump, on_element="LP1")  # Lequ=0, attaches to LP1's last branch
        """
        self._elements[element.name] = element
        self._built = False

        if isinstance(element, Pipe):
            if from_node is None or to_node is None:
                raise ValueError(
                    f"Pipe '{element.name}' requires from_node and to_node"
                )
            self._add_pipe_element(element, from_node, to_node, negate_Q0)

        elif isinstance(element, DiscreteLoss):
            if from_node is None or to_node is None:
                raise ValueError(
                    f"DiscreteLoss '{element.name}' requires from_node and to_node"
                )
            self._add_loss_element(element, from_node, to_node)

        elif isinstance(element, Tank):
            node_name = at_node if at_node else element.name
            self._network.add_node(node_name, NodeType.TANK, C=element.A, H0=element.H0)

        elif isinstance(element, PumpTurbine):
            # Check if this is an L=0 element (pump with no water inertia)
            if element.Lequ == 0:
                # L=0: attach as H_source to adjacent pipe (like SIMSEN)
                if on_element is None:
                    raise ValueError(
                        f"FrancisTurbine '{element.name}' has Lequ=0, "
                        f"requires on_pipe parameter to attach H_source"
                    )
                self._add_francis_turbine_as_hsource(element, on_element, negate_Q0)
            else:
                # L>0: create as separate branch
                if from_node is None or to_node is None:
                    raise ValueError(
                        f"FrancisTurbine '{element.name}' requires from_node and to_node"
                    )
                self._add_francis_turbine_element(
                    element, from_node, to_node, negate_Q0
                )

        else:
            raise TypeError(f"Unknown element type: {type(element)}")

        return self

    def _ensure_node_exists(self, node_name: str):
        """Ensure a node exists; create as junction if not."""
        if node_name not in self._network.nodes:
            self._network.add_node(node_name, NodeType.JUNCTION)

    def _add_pipe_element(
        self, pipe: Pipe, from_node: str, to_node: str, negate_Q0: bool = False
    ):
        """Add a Pipe element to the network."""
        # Auto-create junction nodes if they don't exist
        self._ensure_node_exists(from_node)
        self._ensure_node_exists(to_node)

        # Get any accumulated losses for this pipe
        additional_R = self._pipe_losses.get(pipe.name, 0.0)

        # Expand the pipe into branches
        self._expand_pipe(
            node_from=from_node,
            node_to=to_node,
            pipe=pipe,
            additional_R=additional_R,
            negate_Q0=negate_Q0,
        )

    def _add_loss_element(self, loss: DiscreteLoss, from_node: str, to_node: str):
        """Add a DiscreteLoss element as a branch."""
        # Auto-create junction nodes if they don't exist
        self._ensure_node_exists(from_node)
        self._ensure_node_exists(to_node)

        R_coef = loss.hydraulic_R()

        # Very small inductance for numerical stability
        L_small = 1e-6

        self._network.add_branch(
            name=loss.name,
            node_from=from_node,
            node_to=to_node,
            L=L_small,
            R_coef=R_coef,
            Q0=loss.Q0,
        )

        self._pipe_expansions[loss.name] = {
            "type": "discrete_loss",
            "n_branches": 1,
            "n_internal_nodes": 0,
            "branch_names": [loss.name],
        }

    def _add_francis_turbine_element(
        self,
        turbine: PumpTurbine,
        from_node: str,
        to_node: str,
        negate_H: bool = False,
    ):
        """
        Add a FrancisTurbine element as a proper branch.

        The turbine creates its own branch with:
        - L = Lequ / (g × Amean) - turbine inductance
        - R = 0 (guide vane effects are in the characteristic)
        - H_source = characteristic H(y, N, Q)

        Parameters
        ----------
        negate_H : bool
            If True, negate H_source. Use for turbines operating in turbine mode
            where the machine CONSUMES head (extracts energy) rather than
            producing it like a pump.
        """
        # Auto-create junction nodes if they don't exist
        self._ensure_node_exists(from_node)
        self._ensure_node_exists(to_node)

        # Compute hydraulic inductance
        L_turb = turbine.hydraulic_L()

        # Create H_source function from turbine's characteristic
        # Use compute_H_characteristic because L×dQ/dt is handled by the momentum equation
        # negate_H: for turbines, H_source should be negative (consumes head)
        def H_source(Q, t, state_dict):
            H = turbine.compute_H_characteristic(Q, t, state_dict)
            return -H if negate_H else H

        # Add the turbine as a branch
        # R = 0 because guide vane resistance is modeled in characteristic
        self._network.add_branch(
            name=turbine.name,
            node_from=from_node,
            node_to=to_node,
            L=L_turb,
            R_coef=0.0,  # No additional friction - it's in the characteristic
            Q0=turbine.Q0,
            H_source=H_source,
            H_source_name=turbine.name,
        )

        # Track expansion info
        self._pipe_expansions[turbine.name] = {
            "type": "francis_turbine",
            "n_branches": 1,
            "n_internal_nodes": 0,
            "branch_names": [turbine.name],
        }

        # Store in machines dict for reference
        self._machines[turbine.name] = (turbine.name, H_source)

    def _add_francis_turbine_as_hsource(
        self, turbine: PumpTurbine, on_pipe: str, negate_Q: bool = False
    ):
        """
        Attach a FrancisTurbine with Lequ=0 as an H_source on a pipe.

        When Lequ=0 (like SIMSEN pumps), the machine has no water inertia
        and acts as a pure head source. Instead of creating a separate branch,
        we attach the H_source to an adjacent pipe. The pipe provides the
        inductance (inertia) for the combined element.

        This matches SIMSEN's treatment of FTURB elements with Lequ=0.

        Parameters
        ----------
        turbine : FrancisTurbine
            The machine element with Lequ=0
        on_pipe : str
            Name of the pipe to attach the H_source to
        negate_Q : bool
            If True, negate Q before passing to characteristic. Use for pumps
            where the network Q direction is opposite to the pump's reference.
        """

        # Create H_source function from turbine's compute_H
        # Capture negate_Q in closure
        # Use compute_H_characteristic because L×dQ/dt is handled by the pipe's momentum equation
        def H_source(Q, t, state_dict):
            Q_char = -Q if negate_Q else Q
            return turbine.compute_H_characteristic(Q_char, t, state_dict)

        # Find the LAST branch of the target pipe (machines connect at pipe outlet)
        # In SIMSEN, pumps/turbines with Lequ=0 share Q with the adjacent pipe's
        # last segment (Qn), not the first (Q1).
        if on_pipe in self._pipe_expansions:
            # Pipe already expanded - get last branch name
            branch_names = self._pipe_expansions[on_pipe].get("branch_names", [])
            if branch_names:
                branch_name = branch_names[-1]  # Last branch
            else:
                branch_name = on_pipe  # Fallback for discrete loss
        else:
            # Pipe not expanded yet - store for later resolution in build()
            self._machines[turbine.name] = (on_pipe, H_source)
            return

        if branch_name not in self._network.branches:
            # Pipe might not be added yet - store for later resolution in build()
            self._machines[turbine.name] = (on_pipe, H_source)
            return

        # Update the branch with the machine's H_source
        branch = self._network.branches[branch_name]
        branch.H_source = H_source
        branch.H_source_name = turbine.name

        # Track as machine (not as separate expansion)
        self._machines[turbine.name] = (branch_name, H_source)

        # Track in expansions for summary output
        self._pipe_expansions[turbine.name] = {
            "type": "francis_turbine_hsource",
            "on_pipe": on_pipe,
            "n_branches": 0,  # No separate branches
            "n_internal_nodes": 0,
            "branch_names": [],
        }

    # -------------------------------------------------------------------------
    # Pipe expansion (distributed model)
    # -------------------------------------------------------------------------

    def _expand_pipe(
        self,
        node_from: str,
        node_to: str,
        pipe: Pipe,
        additional_R: float,
        negate_Q0: bool = False,
    ):
        """
        Expand a pipe into branches and internal nodes.

        For Nb = 0 (lumped): single branch from node_from to node_to
        For Nb > 0 (distributed): Nb+1 branches with Nb internal Hc nodes

        Each branch uses its corresponding Q0 from Q0_list (from SIMSEN DAT file).
        The last segment Q0 is negated due to SIMSEN's capacitor-centric convention.

        If negate_Q0=True, all Q0 values are negated (use when topology direction
        is flipped from SIMSEN's original).
        """
        g = self.g

        # Compute total parameters
        L_h_total, R_total, C_total = (
            pipe.hydraulic_L(),
            pipe.hydraulic_R(),
            pipe.hydraulic_C(),
        )
        R_total += additional_R

        if pipe.Nb == 0:
            # Lumped model: single branch
            Q0_final = -pipe.Q0 if negate_Q0 else pipe.Q0
            self._network.add_branch(
                name=f"{pipe.name}_Q0",
                node_from=node_from,
                node_to=node_to,
                L=L_h_total,
                R_coef=R_total,
                Q0=Q0_final,
            )

            self._pipe_expansions[pipe.name] = {
                "type": "lumped",
                "n_branches": 1,
                "n_internal_nodes": 0,
                "branch_names": [f"{pipe.name}_Q0"],
            }

        else:
            # Distributed model: Nb+1 branches, Nb internal nodes
            pipe_piece: Pipe = copy.copy(pipe)
            pipe_piece.L = pipe.L / pipe.Nb

            L_h_seg, R_seg, C_seg = (
                pipe_piece.hydraulic_L(),
                pipe_piece.hydraulic_R(),
                pipe_piece.hydraulic_C(),
            )
            R_add_per_branch = additional_R / (pipe.Nb + 1)

            branch_names = []
            internal_node_names = []

            # Create internal Hc nodes
            for i in range(pipe.Nb):
                hc_name = f"{pipe.name}_Hc{i}"
                H0 = pipe.Hc0_list[i] if i < len(pipe.Hc0_list) else 0.0
                self._network.add_node(hc_name, NodeType.CAPACITOR, C=C_seg, H0=H0)
                internal_node_names.append(hc_name)

            # Create branches
            for i in range(pipe.Nb + 1):
                branch_name = f"{pipe.name}_Q{i}"
                branch_names.append(branch_name)

                if i == 0:
                    # First half-branch
                    nf = node_from
                    nt = f"{pipe.name}_Hc0"
                    L_b = L_h_seg / 2
                    R_b = R_seg / 2 + R_add_per_branch
                elif i == pipe.Nb:
                    # Last half-branch
                    nf = f"{pipe.name}_Hc{pipe.Nb - 1}"
                    nt = node_to
                    L_b = L_h_seg / 2
                    R_b = R_seg / 2 + R_add_per_branch
                else:
                    # Full internal branch
                    nf = f"{pipe.name}_Hc{i - 1}"
                    nt = f"{pipe.name}_Hc{i}"
                    L_b = L_h_seg
                    R_b = R_seg + R_add_per_branch

                # Use Q0 from Q0_list for each branch (SIMSEN provides per-segment Q)
                # IMPORTANT: SIMSEN's "capacitor-centric convention" stores the LAST
                # segment's Q with opposite sign (negative for normal inlet→outlet flow).
                # We must negate it to get the correct network Q0.
                Q0_i = pipe.Q0_list[i] if i < len(pipe.Q0_list) else pipe.Q0
                if i == pipe.Nb:  # Last segment: capacitor-centric fix
                    Q0_i = -Q0_i
                if negate_Q0:  # Topology flip: negate all segments
                    Q0_i = -Q0_i
                self._network.add_branch(
                    name=branch_name,
                    node_from=nf,
                    node_to=nt,
                    L=L_b,
                    R_coef=R_b,
                    Q0=Q0_i,
                )

            self._pipe_expansions[pipe.name] = {
                "type": "distributed",
                "n_branches": pipe.Nb + 1,
                "n_internal_nodes": pipe.Nb,
                "branch_names": branch_names,
                "internal_node_names": internal_node_names,
            }

    # -------------------------------------------------------------------------
    # Build and simulate
    # -------------------------------------------------------------------------

    def build(self) -> HydraulicNetwork:
        """
        Build and return the HydraulicNetwork.

        Resolves pending machine attachments, transfers pipe expansion
        metadata as pipe groups, and builds the underlying network.

        Returns
        -------
        HydraulicNetwork
            The built network, ready for simulation.
        """
        # Resolve pending machines (added before their target pipes)
        # Machines with Lequ=0 connect at the pipe's OUTLET (last branch)
        unresolved = []
        for machine_name, (on_element, H_func) in self._machines.items():
            # Find the LAST branch of the target element
            if on_element in self._pipe_expansions:
                branch_names = self._pipe_expansions[on_element].get("branch_names", [])
                if branch_names:
                    branch_name = branch_names[-1]  # Last branch (outlet)
                else:
                    branch_name = on_element  # Discrete loss
            else:
                # Try direct element name (discrete loss)
                branch_name = on_element

            if branch_name not in self._network.branches:
                unresolved.append((machine_name, on_element))
                continue

            # Attach machine to the branch
            branch = self._network.branches[branch_name]
            branch.H_source = H_func
            branch.H_source_name = machine_name

        if unresolved:
            msg = ", ".join(f"'{m}' -> '{e}'" for m, e in unresolved)
            raise ValueError(
                f"Cannot resolve machines (target elements not found): {msg}"
            )

        # Transfer pipe expansion metadata as pipe groups
        for pipe_name, info in self._pipe_expansions.items():
            branch_names = info.get("branch_names", [])
            if branch_names:
                self._network.register_pipe_group(pipe_name, branch_names)

        self._network.build()
        self._built = True
        return self._network

    def print_summary(self):
        """Print network summary."""
        if not self._built:
            self.build()

        print("=" * 60)
        print("ElementNetworkBuilder Summary")
        print("=" * 60)

        # Pipe expansion summary
        n_lumped = sum(
            1 for p in self._pipe_expansions.values() if p.get("type") == "lumped"
        )
        n_distributed = sum(
            1 for p in self._pipe_expansions.values() if p.get("type") == "distributed"
        )
        n_dloss = sum(
            1
            for p in self._pipe_expansions.values()
            if p.get("type") == "discrete_loss"
        )

        print(f"Components:")
        print(f"  Pipes (lumped): {n_lumped}")
        print(f"  Pipes (distributed): {n_distributed}")
        print(f"  Discrete losses: {n_dloss}")
        print(f"  Machines: {len(self._machines)}")
        print()

        self._network.print_summary()

    def get_initial_state(self) -> np.ndarray:
        """Get initial state from component values."""
        if not self._built:
            self.build()
        return self._network.get_initial_state()

    def simulate(
        self,
        t_span: Tuple[float, float],
        y0: np.ndarray = None,
        t_eval: np.ndarray = None,
        external_inputs_func: Callable[[float], Dict] | None = None,
        **solve_ivp_kwargs,
    ):
        """
        Simulate the network.

        See HydraulicNetwork.simulate() for parameters.
        """
        if not self._built:
            self.build()

        # Estimate CFL condition for distributed pipes
        dt_min_list = []
        for name, info in self._pipe_expansions.items():
            if info.get("type") == "distributed" and info["n_branches"] > 1:
                # Get wave speed from a branch (stored during expansion)
                # For now, just warn about potential stiffness
                pass

        return self._network.simulate(
            t_span, y0, t_eval, external_inputs_func, **solve_ivp_kwargs
        )

    def ode_rhs(
        self, t: float, state: np.ndarray, external_inputs: Dict | None = None
    ) -> np.ndarray:
        """Compute state derivatives (for external integrators)."""
        if not self._built:
            self.build()
        return self._network.ode_rhs(t, state, external_inputs)

    # -------------------------------------------------------------------------
    # Result extraction
    # -------------------------------------------------------------------------

    def extract_results(self, result) -> Dict[str, np.ndarray]:
        """Delegate to the underlying network's extract_results."""
        return self._network.extract_results(result)

    # -------------------------------------------------------------------------
    # Access to internal state
    # -------------------------------------------------------------------------

    @property
    def _Q_indices(self) -> Dict[str, int]:
        """Access Q state indices (for backward compatibility)."""
        return self._network._Q_indices

    @property
    def _H_indices(self) -> Dict[str, int]:
        """Access H state indices (for backward compatibility)."""
        return self._network._H_indices

    @property
    def _branch_order(self) -> List[str]:
        """Access branch ordering (for backward compatibility)."""
        return self._network._branch_order

    @property
    def _node_order(self) -> List[str]:
        """Access node ordering (for backward compatibility)."""
        return self._network._cap_nodes  # Only storage nodes are in state
