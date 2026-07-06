# ============================================================
# Appendix E – Symbolic Derivation of First-Order Model
# Verification that K_P matches Appendix B static gain
# ============================================================
#
# This script derives τ and K_P from:
#   1. Appendix E: Dynamic momentum equations with junction elimination
#   2. Appendix B: Static KVL loop equation (under Q₁=Q₂ simplification)
# and verifies they give the same K_P.

import sympy as sp

sp.init_printing()

# ------------------------------------------------------------
# 1. Symbols
# ------------------------------------------------------------

# Hydraulic inertias
L_P, L_3, L_4 = sp.symbols("L_P L_3 L_4", positive=True)

# Linearized friction slopes: r_i = 2 R_i |Q_i,0|
r_P, r_3, r_4 = sp.symbols("r_P r_3 r_4", positive=True)

# Characteristic derivatives (using Appendix notation)
# Turbine: Q_T = Γ^{Q,T}(y_T, N_T, H_T)
Gamma_TH = sp.symbols("Gamma_TH")  # ∂Q_T/∂H_T (typically > 0)

# Pump: Q_P = Γ^{Q,P}(N_P, H_P)
Gamma_PH = sp.symbols("Gamma_PH")  # ∂Q_P/∂H_P (typically < 0)
Gamma_PN = sp.symbols("Gamma_PN")  # ∂Q_P/∂N_P (typically > 0)

# Convenience: g_P = 1/Γ^{Q,P}_H (as used in Appendix E)
g_P = 1 / Gamma_PH

# Perturbation variables (for clarity, not directly used in expressions)
dH_T, dN_P = sp.symbols("delta_H_T delta_N_P")

# Combined turbine branch resistance
r_34 = r_3 + r_4

# ------------------------------------------------------------
# 2. Abbreviations (Appendix E)
# ------------------------------------------------------------

alpha = 2 / L_P
beta = 1 / L_3
gamma = 1 / L_4

D_prime = alpha * (beta + gamma) + beta * gamma  # Denominator D'

print("=" * 60)
print("PART 1: APPENDIX E - DYNAMIC DERIVATION")
print("=" * 60)

# ------------------------------------------------------------
# 3. Express all quantities in terms of δH_T and δN_P
# ------------------------------------------------------------

# From coordinate transform (setting δy_T = δN_T = 0):
#   δQ_T = Γ^{Q,T}_H δH_T
#   δQ_P = δQ_T / 2 = (Γ^{Q,T}_H / 2) δH_T

# From pump characteristic:
#   δH_P = g_P (δQ_P - Γ^{Q,P}_N δN_P)
#        = g_P (Γ^{Q,T}_H / 2) δH_T - g_P Γ^{Q,P}_N δN_P

# Define coefficients for δH_T and δN_P:
# δQ_P = c_QP_HT * δH_T
c_QP_HT = Gamma_TH / 2

# δQ_T = c_QT_HT * δH_T
c_QT_HT = Gamma_TH

# δH_P = c_HP_HT * δH_T + c_HP_NP * δN_P
c_HP_HT = g_P * Gamma_TH / 2
c_HP_NP = -g_P * Gamma_PN

# ------------------------------------------------------------
# 4. Linearized b₁ and b₂
# ------------------------------------------------------------

# From Appendix E:
#   δb₁ = α(r_P δQ_P - δH_P) - γ r₄ δQ_T
#   δb₂ = -α(r_P δQ_P - δH_P) + β(r₃ δQ_T + δH_T)

# Expand in terms of δH_T, δN_P:
# Φ := r_P δQ_P - δH_P = r_P c_QP_HT δH_T - (c_HP_HT δH_T + c_HP_NP δN_P)
#                       = (r_P c_QP_HT - c_HP_HT) δH_T - c_HP_NP δN_P

Phi_HT = r_P * c_QP_HT - c_HP_HT  # coefficient of δH_T in Φ
Phi_NP = -c_HP_NP  # coefficient of δN_P in Φ

# δb₁ = α Φ - γ r₄ c_QT_HT δH_T
#     = (α Φ_HT - γ r₄ c_QT_HT) δH_T + α Φ_NP δN_P
db1_HT = alpha * Phi_HT - gamma * r_4 * c_QT_HT
db1_NP = alpha * Phi_NP

# δb₂ = -α Φ + β (r₃ c_QT_HT + 1) δH_T
#     = (-α Φ_HT + β (r₃ c_QT_HT + 1)) δH_T - α Φ_NP δN_P
db2_HT = -alpha * Phi_HT + beta * (r_3 * c_QT_HT + 1)
db2_NP = -alpha * Phi_NP

# ------------------------------------------------------------
# 5. Junction head difference: δ(H₁ - H₂)
# ------------------------------------------------------------

# δ(H₁ - H₂) = (β δb₁ - γ δb₂) / D'
dH12_HT = (beta * db1_HT - gamma * db2_HT) / D_prime
dH12_NP = (beta * db1_NP - gamma * db2_NP) / D_prime

# ------------------------------------------------------------
# 6. Momentum equation RHS
# ------------------------------------------------------------

# RHS = δ(H₁ - H₂) - r_P δQ_P + δH_P
#     = dH12_HT δH_T + dH12_NP δN_P - r_P c_QP_HT δH_T + c_HP_HT δH_T + c_HP_NP δN_P
#     = (dH12_HT - r_P c_QP_HT + c_HP_HT) δH_T + (dH12_NP + c_HP_NP) δN_P

a_E = dH12_HT - r_P * c_QP_HT + c_HP_HT  # coefficient of δH_T
b_E = dH12_NP + c_HP_NP  # coefficient of δN_P

# LHS coefficient: L_P Γ^{Q,T}_H / 2
lhs_coeff = L_P * Gamma_TH / 2

# ------------------------------------------------------------
# 7. Extract τ and K_P (Appendix E)
# ------------------------------------------------------------

# Equation: lhs_coeff * d/dt δH_T = a_E δH_T + b_E δN_P
#
# Standard form: τ d/dt δH_T = -δH_T + K_P δN_P
#
# Dividing by a_E:
#   (lhs_coeff / a_E) d/dt δH_T = δH_T + (b_E / a_E) δN_P
#
# For standard form (with -δH_T on RHS), we need a_E < 0:
#   τ = -lhs_coeff / a_E
#   K_P = -b_E / a_E

tau_E = -lhs_coeff / a_E
K_P_E = -b_E / a_E

tau_E_simplified = sp.simplify(tau_E)
K_P_E_simplified = sp.simplify(K_P_E)

print("\nτ (Appendix E, simplified):")
sp.pprint(tau_E_simplified)

print("\nK_P (Appendix E, simplified):")
sp.pprint(K_P_E_simplified)

# ============================================================
print("\n" + "=" * 60)
print("PART 2: APPENDIX B - MATRIX FORMULATION")
print("=" * 60)
# ============================================================

# Appendix B defines the linearized system as:
#   0 = A x + B u   (with d = 0)
#   y = C x
#
# where x = [δH_T, δH_P1, δH_P2]^T, u = δN_P, y = δH_T
#
# The static gain is: K_P = -C A^{-1} B

# For verification under Q₁ = Q₂, we use r_1 = r_2 = r_P
r_1, r_2 = sp.symbols("r_1 r_2", positive=True)

# Full matrices from Appendix B (using r_1, r_2 for generality)
A_mat = sp.Matrix(
    [
        [Gamma_TH, -Gamma_PH, -Gamma_PH],
        [1 + r_34 * Gamma_TH, 0, -1 + r_2 * Gamma_PH],
        [1 + r_34 * Gamma_TH, -1 + r_1 * Gamma_PH, 0],
    ]
)

B_mat = sp.Matrix([[-2 * Gamma_PN], [r_2 * Gamma_PN], [r_1 * Gamma_PN]])

C_mat = sp.Matrix([[1, 0, 0]])

print("\nA matrix:")
sp.pprint(A_mat)

print("\nB vector:")
sp.pprint(B_mat)

print("\nC vector:")
sp.pprint(C_mat)

# Compute K_P = -C A^{-1} B (general case)
A_inv = A_mat.inv()
K_P_B_general = -C_mat * A_inv * B_mat

print("\nK_P = -C A^{-1} B (general, with r_1, r_2):")
sp.pprint(sp.simplify(K_P_B_general[0, 0]))

# ============================================================
print("\n" + "=" * 60)
print("PART 3: VERIFICATION")
print("=" * 60)
# ============================================================

# Appendix E uses r_P = (r_1 + r_2)/2 (from the Q_1 = Q_2 assumption)
# Substitute this into Appendix E's formula and compare to Appendix B's general formula

K_P_E_with_r12 = K_P_E_simplified.subs(r_P, (r_1 + r_2) / 2)
K_P_E_with_r12_simplified = sp.simplify(K_P_E_with_r12)

print("\nK_P (Appendix E, with r_P = (r_1 + r_2)/2):")
sp.pprint(K_P_E_with_r12_simplified)

print("\nK_P (Appendix B, general with r_1, r_2):")
K_P_B_general_simplified = sp.simplify(K_P_B_general[0, 0])
sp.pprint(K_P_B_general_simplified)

# The two formulas are equal when r_1 = r_2 (which is the Q_1 = Q_2 assumption)
# Let's verify this by computing the difference and showing it vanishes when r_1 = r_2
difference_general = sp.simplify(K_P_E_with_r12_simplified - K_P_B_general_simplified)

print("\nK_P(E) - K_P(B) (general):")
sp.pprint(difference_general)

# Factor out (r_1 - r_2) to show the difference is proportional to the asymmetry
difference_factored = sp.factor(difference_general)
print("\nK_P(E) - K_P(B) (factored):")
sp.pprint(difference_factored)

# Verify it's zero when r_1 = r_2
difference_symmetric = difference_general.subs(r_1, r_2)
print("\nK_P(E) - K_P(B) when r_1 = r_2:")
sp.pprint(sp.simplify(difference_symmetric))

if sp.simplify(difference_symmetric) == 0:
    print("\n✓ VERIFIED: K_P from Appendix E (with r_P = (r_1+r_2)/2) equals")
    print("  -C A^{-1} B from Appendix B when r_1 = r_2 (i.e., Q_1 = Q_2)")

# ============================================================
print("\n" + "=" * 60)
print("PART 4: FINAL EXPRESSIONS")
print("=" * 60)
# ============================================================

print("\n" + "-" * 40)
print("Time constant τ (from Appendix E):")
print("-" * 40)
sp.pprint(tau_E_simplified)

print("\n" + "-" * 40)
print("Static gain K_P (equivalent form):")
print("-" * 40)
# Express in the form from Appendix E verification section
K_P_final = -g_P * Gamma_PN / (r_34 * Gamma_TH + (Gamma_TH / 2) * (r_P - g_P) + 1)
print("\nK_P = -g_P Γ^{Q,P}_N / [r_{34} Γ^{Q,T}_H + (Γ^{Q,T}_H/2)(r_P - g_P) + 1]")
print("\nwhere g_P = 1/Γ^{Q,P}_H")
sp.pprint(sp.simplify(K_P_final))
