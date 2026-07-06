import sympy as sp

# States
Q_T, Q_d = sp.symbols("Q_T Q_d")
x = sp.Matrix([Q_T, Q_d])

# Input
N_P = sp.symbols("N_P")
u = sp.Matrix([N_P])

# Time-varying parameters (treated as independent symbols)
y_T, N_T = sp.symbols("y_T N_T")
p = sp.Matrix([y_T, N_T])

# Other symbols (constants)
L1, L2 = sp.symbols("L1 L2")
H1, H2 = sp.symbols("H1 H2")
Sigma_f, Sigma_H = sp.symbols("Sigma_f Sigma_H")
Delta_f, Delta_H = sp.symbols("Delta_f Delta_H")

# Define dynamics
f1 = (1 / L1 + 1 / L2) * (H1 - H2) - Sigma_f + Sigma_H
f2 = (1 / L1 - 1 / L2) * (H1 - H2) - Delta_f + Delta_H
f = sp.Matrix([f1, f2])

# Define output
H_T = sp.Function("H_T")
h = H_T(Q_T, y_T, N_T)

# Jacobians
A = f.jacobian(x)
B = f.jacobian(u)
E = f.jacobian(p)

C = sp.Matrix([h]).jacobian(x)
D = sp.Matrix([h]).jacobian(u)
F = sp.Matrix([h]).jacobian(p)

print(A, B, E, C, D, F)
