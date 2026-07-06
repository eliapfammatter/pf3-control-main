import sympy as sp

# --- 1. Define symbols ---
aT, a1, a2, r1, r2, r34, b1, b2 = sp.symbols("aT a1 a2 r1 r2 r34 b1 b2")
k1, k2 = sp.symbols("k1 k2")  # for automatic substitution

# --- 2. Define matrices ---
A = sp.Matrix(
    [[aT, -a1, -a2], [1 - r34 * aT, 0, -1 - r2 * a2], [1 - r34 * aT, -1 - a1 * r1, 0]]
)

B = sp.Matrix([[-b1 - b2], [-r2 * b2], [-r1 * b1]])
C = sp.Matrix([[1, 0, 0]])

# --- 3. Solve K_P ---
x = A.LUsolve(B)
KP = -(C * x)[0]
KP = sp.together(sp.simplify(KP))

# --- 4. Separate numerator/denominator ---
num, den = sp.fraction(KP)
num = sp.factor(num)

# --- 5. Collect terms in aT, a1, a2 for structure ---
den_collected = sp.collect(den, aT)
den_collected = sp.collect(den_collected, a1)
den_collected = sp.collect(den_collected, a2)

# --- 6. Substitute k1 = 1 + a1*r1, k2 = 1 + a2*r2 ---
subs_dict = {1 + a1 * r1: k1, 1 + a2 * r2: k2}
den_compact = den_collected.subs(subs_dict)
num_compact = num.subs(subs_dict)

KP_compact = sp.simplify(num_compact / den_compact)

# --- 7. Print final compact form ---
print("\n=== Final compact K_P ===\n")
print(KP_compact)

# Optional: verify algebraic equivalence
verification = sp.simplify(KP - KP_compact.subs({k1: 1 + a1 * r1, k2: 1 + a2 * r2}))
print("\nVerification (should be 0):", verification)
