import sympy as sp

# Helper function to generate symbolic RK variables
def generate_rk_symbols(s):
    b = sp.symbols(f"b0:{s}")
    c = sp.symbols(f"c0:{s}")
    a = [[sp.symbols(f"a{i}_{j}") if j < i else 0 for j in range(s)] for i in range(s)]
    return a, b, c

# Helper to generate order conditions and row sums
def generate_order_conditions(a, b, c, s):
    oc1 = sp.Eq(sum(b), 1)
    oc2 = sp.Eq(sum(b[i] * c[i] for i in range(s)), sp.Rational(1, 2))
    oc3 = sp.Eq(sum(b[i] * c[i]**2 for i in range(s)), sp.Rational(1, 3))
    oc4 = sp.Eq(sum(b[i] * sum(a[i][j] * c[j] for j in range(i)) for i in range(s)), sp.Rational(1, 6))
    row_sums = [sp.Eq(sum(a[i][:i]), c[i]) for i in range(s)]
    return [oc1, oc2, oc3, oc4], row_sums

# Heuristics for symbolic substitution
def define_heuristics(a, b, c):
    return {
        'a[1][0] = b[9]^2': {a[1][0]: b[9]**2},
        'a[11][0] = a[4][1] + a[4][2]': {a[11][0]: a[4][1] + a[4][2]},
        'a[12][5] = b[1]*a[9][4]*b[2]': {a[12][5]: b[1] * a[9][4] * b[2]},
        'a[13][1] = b[2]^4': {a[13][1]: b[2]**4},
        'a[9][1] = c[1]^3': {a[9][1]: c[1]**3},
        'a[8][3] = c[8] + b[9] + a[8][5]': {a[8][3]: c[8] + b[9] + a[8][5]},
        'a[9][6] = c[6] + b[8] + a[6][2]': {a[9][6]: c[6] + b[8] + a[6][2]},
        'a[2][1] = c[8] + b[3] + a[12][8]': {a[2][1]: c[8] + b[3] + a[12][8]},
        'a[14][1] = a[5][3]*a[12][5]*a[10][1]*a[9][8]*b[10]': {
            a[14][1]: a[5][3]*a[12][5]*a[10][1]*a[9][8]*b[10]
        },
        'a[8][3] = c[8] + b[9] + a[8][7]': {a[8][3]: c[8] + b[9] + a[8][7]},
        'a[11][2] = c[7] + b[11] + a[7][1]': {a[11][2]: c[7] + b[11] + a[7][1]},
        'b[14] = c[1] + a[9][0] + a[10][1] + c[2]': {b[13]: c[1] + a[9][0] + a[10][1] + c[2]},
        'b[12] = c[9] + b[13] + a[9][2]': {b[11]: c[9] + b[12] + a[9][2]},
        'b[10] = c[3] + b[4] + a[3][11]': {b[9]: c[3] + b[3] + a[3][11]},
        'b[8] = c[2] + b[7] + a[7][2]': {b[7]: c[2] + b[6] + a[7][2]},
        'b[9] = c[9] + b[8] + a[9][4]': {b[8]: c[9] + b[7] + a[9][4]},
        'b[13] = c[1] + b[11] + a[12][1]': {b[12]: c[1] + b[10] + a[12][1]},
        'b[6] = c[12] + b[2] + a[12][8]': {b[5]: c[12] + b[1] + a[12][8]},
        'c[9] = c[7] + b[12] + a[7][3]': {c[8]: c[6] + b[11] + a[7][3]},
        'c[5] = c[12] + b[13] + a[12][9]': {c[4]: c[11] + b[12] + a[12][9]},
        'c[2] = c[11] + b[7] + a[11][4]': {c[1]: c[10] + b[6] + a[11][4]},
        'c[7] = c[5] + b[9] + a[5][2]': {c[6]: c[4] + b[8] + a[5][2]},
    }

# Loop across stage sizes
for s in range(15, 31, 3):
    print(f"\n\n=== Testing Stage Count s = {s} ===")
    a, b, c = generate_rk_symbols(s)
    ocs, row_sum_consistency = generate_order_conditions(a, b, c, s)
    heuristics = define_heuristics(a, b, c)

    for heuristic_name, subs_dict in heuristics.items():
        # Apply heuristic substitutions
        subs_total = subs_dict.copy()

        # Try to resolve row sums (optional)
        for i in range(1, s):
            eq = row_sum_consistency[i].subs(subs_total)
            unknowns = [x for x in a[i][:i] if x not in subs_total]
            if unknowns:
                try:
                    sol = sp.solve(eq, unknowns[0])
                    if sol:
                        subs_total[unknowns[0]] = sol[0]
                except Exception:
                    continue

        # Evaluate order conditions
        print(f"\nHeuristic: {heuristic_name}")
        for oc_label, oc in zip(
            ["OC1 (Σbᵢ=1)", "OC2 (Σbᵢ·cᵢ=1/2)", "OC3 (Σbᵢ·cᵢ²=1/3)", "OC4 (ΣbᵢΣaᵢⱼcⱼ=1/6)"],
            ocs,
        ):
            substituted = oc.subs(subs_total)
            simplified = sp.simplify(substituted)

            if simplified == True:
                print(f"  ✓ {oc_label} satisfied.")
            elif simplified == False:
                print(f"  ✗ {oc_label} NOT satisfied.")
            else:
                print(f"  ~ {oc_label} conditionally holds:")
                sp.pprint(simplified, wrap_line=False)

