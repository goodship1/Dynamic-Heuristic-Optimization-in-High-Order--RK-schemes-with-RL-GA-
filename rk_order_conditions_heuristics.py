import sympy as sp

# Number of stages
s = 15

# Define symbolic variables
b = sp.symbols(f"b0:{s}")
c = sp.symbols(f"c0:{s}")
a = [[sp.symbols(f"a{i}_{j}") if j < i else 0 for j in range(s)] for i in range(s)]

# Order conditions (explicit)
oc1 = sp.Eq(sum(b), 1)
oc2 = sp.Eq(sum(b[i]*c[i] for i in range(s)), sp.Rational(1,2))
oc3 = sp.Eq(sum(b[i]*c[i]**2 for i in range(s)), sp.Rational(1,3))
oc4 = sp.Eq(sum(b[i]*sum(a[i][j]*c[j] for j in range(i)) for i in range(s)), sp.Rational(1,6))

# Row sum consistency: sum of a[i,:i] equals c[i]
row_sum_consistency = [sp.Eq(sum(a[i][:i]), c[i]) for i in range(s)]

# Define heuristics as substitution rules (key: substitution variable, value: expression)
heuristics = {
    # a coefficients heuristics (from Table 1)
    'a[1][0] = b[9]^2': {a[1][0]: b[9]**2},
    'a[11][0] = a[4][1] + a[4][2]': {a[11][0]: a[4][1] + a[4][2]},
    'a[12][5] = b[1]*a[9][4]*b[2]': {a[12][5]: b[1] * a[9][4] * b[2]},
    'a[13][1] = b[2]^4': {a[13][1]: b[2]**4},
    'a[9][1] = c[1]^3': {a[9][1]: c[1]**3},
    'a[8][3] = c[8] + b[9] + a[8][5]': {a[8][3]: c[8] + b[9] + a[8][5]},
    'a[9][6] = c[6] + b[8] + a[6][2]': {a[9][6]: c[6] + b[8] + a[6][2]},
    'a[2][1] = c[8] + b[3] + a[12][8]': {a[2][1]: c[8] + b[3] + a[12][8]},
    'a[14][1] = a[5][3]*a[12][5]*a[10][1]*a[9][8]*b[10]': {a[14][1]: a[5][3]*a[12][5]*a[10][1]*a[9][8]*b[10]},
    'a[8][3] = c[8] + b[9] + a[8][7]': {a[8][3]: c[8] + b[9] + a[8][7]},
    'a[11][2] = c[7] + b[11] + a[7][1]': {a[11][2]: c[7] + b[11] + a[7][1]},

    # b coefficients heuristics (from Table 2)
    'b[14] = c[1] + a[9][0] + a[10][1] + c[2]': {b[14-1]: c[1] + a[9][0] + a[10][1] + c[2]},  # b[14] → b[13] (index shift)
    'b[12] = c[9] + b[13] + a[9][2]': {b[12-1]: c[9] + b[13-1] + a[9][2]},                    # indices shifted by -1
    'b[10] = c[3] + b[4] + a[3][11]': {b[10-1]: c[3] + b[4-1] + a[3][11]},
    'b[8] = c[2] + b[7] + a[7][2]': {b[8-1]: c[2] + b[7-1] + a[7][2]},
    'b[9] = c[9] + b[8] + a[9][4]': {b[9-1]: c[9] + b[8-1] + a[9][4]},
    'b[13] = c[1] + b[11] + a[12][1]': {b[13-1]: c[1] + b[11-1] + a[12][1]},
    'b[6] = c[12] + b[2] + a[12][8]': {b[6-1]: c[12] + b[2-1] + a[12][8]},

    # c coefficients heuristics (from Table 3)
    'c[9] = c[7] + b[12] + a[7][3]': {c[9-1]: c[7-1] + b[12-1] + a[7][3]},
    'c[5] = c[12] + b[13] + a[12][9]': {c[5-1]: c[12-1] + b[13-1] + a[12][9]},
    'c[2] = c[11] + b[7] + a[11][4]': {c[2-1]: c[11-1] + b[7-1] + a[11][4]},
    'c[7] = c[5] + b[9] + a[5][2]': {c[7-1]: c[5-1] + b[9-1] + a[5][2]},
}

# Function to print order condition results clearly
def pretty_print_results(name, results):
    print(f"\nTesting heuristic: {name}")
    for oc_name, expr in results.items():
        if expr == True:
            print(f"  [✓] {oc_name} satisfied.")
        elif expr == False:
            print(f"  [✗] {oc_name} NOT satisfied (contradiction).")
        else:
            print(f"  [~] {oc_name} holds conditionally:")
            sp.pprint(expr, wrap_line=False)
            print()

# Loop over each heuristic, test OCs with only that substitution applied
for heuristic_name, subs_dict in heuristics.items():
    # Start with fresh substitutions
    subs_total = subs_dict.copy()

    # Solve row sums if possible (optional - can be omitted if you want pure substitution)
    for i in range(1, s):
        eq = row_sum_consistency[i].subs(subs_total)
        unknowns = [x for x in a[i][:i] if x not in subs_total]
        if unknowns:
            sol = sp.solve(eq, unknowns[0])
            if sol:
                subs_total[unknowns[0]] = sol[0]

    # Check all order conditions with current substitutions
    oc_results = {}
    for oc_label, oc in zip(
        ["OC1 (Σbᵢ=1)", "OC2 (Σbᵢ cᵢ=1/2)", "OC3 (Σbᵢ cᵢ²=1/3)", "OC4 (Σbᵢ Σaᵢⱼ cⱼ=1/6)"],
        [oc1, oc2, oc3, oc4],
    ):
        substituted = oc.subs(subs_total)
        simplified = sp.simplify(substituted)
        oc_results[oc_label] = simplified

    pretty_print_results(heuristic_name, oc_results)
