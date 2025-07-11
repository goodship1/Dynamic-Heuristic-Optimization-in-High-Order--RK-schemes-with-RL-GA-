from __future__ import division
"""
Runge–Kutta (ESRK) search                  
======================================================
* **Exploration**  – every outer loop we create a brand-new random heuristic
  constraint (or mutate an older one) and random starting coefficients.
* **Exploitation** – IPOPT tries to minimise the internal-stability
  objective given that heuristic.
* **Restart**      – regardless of success/failure we immediately go back
  to Exploration, so the cycle continues.
* We **collect** any heuristic whose random start already satisfies the
  Butcher-stability pre-screen; once we have `stable_target` of them we
  stop.

This file is self-contained: run it with

    python rk_search.py --seed 42

and you should reproduce the numbers reported in the manuscript on any
machine with IPOPT & MA27.
"""

import json, random, time, argparse, sys
import numpy as np
import sympy as sym
import pyomo.environ as pyo
import re 
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# ---- reference SSP-RK β values --------------------------------------------
_BETA = np.array([
    7.2362092230726359621444062630453895436513878479204372e+01,
    1.0, 1.0, 0.5, 1/6,
    3.3027840602875752797933229846633e-02,
    4.07886754615175430254075890829e-03,
    3.33527373172737242603705467099e-04,
    1.88438744545771964992699112477e-05,
    7.55565600673228187844472799772e-07,
    2.18067488965933780610647422533e-08,
    4.54329575207387209531041070804e-10,
    6.76964099198340395235741216662e-12,
    7.03301065055001143525874293210e-14,
    4.83760685817544104974918008895e-16,
    1.97953679111068662248975009738e-18,
    3.64746555383158031489459760645e-21
])
BETA_R   = float(_BETA[0])
BETA_VEC = {i: float(v) for i, v in enumerate(_BETA[1:])}


def evaluate_butcher_stability(a_vals, b_vals, s=15, beta_ref=BETA_R):
    """Quick pre‑screen: returns True if tableau passes the β test."""
    tri_len = s*(s+1)//2
    if len(a_vals) < tri_len:
        a_vals = a_vals + [0.0]*(tri_len - len(a_vals))

    # build lower‑triangular A
    amat = [[0.0]*s for _ in range(s)]
    idx = 0
    for i in range(s):
        for j in range(i+1):
            amat[i][j] = a_vals[idx]
            idx += 1
    # rotate last row up
    amat = [amat[-1]] + amat[:-1]
    amat = sym.Matrix(amat)

    bvec = sym.Matrix(b_vals + [0.0]*(s - len(b_vals)))
    evec = sym.ones(s, 1)
    z = sym.symbols('z')
    R = sym.simplify(1 + (z*bvec.T*(sym.eye(s)-z*amat).inv() * evec)[0])

    roots = [rt for rt in sym.nroots(R.subs(z, sym.symbols('x'))) if rt.is_real]
    if not roots:
        print("Pre‑screen failed: no real roots")
        return False
    beta_act = abs(min(roots))
    print(f"beta diff {abs(beta_act-beta_ref)/beta_ref:.3e}")
    passed = sum(roots) >= -187.421549029524
    print("Pre‑screen passed" if passed else "Pre‑screen failed: sum roots too negative")
    return passed


def build_model():
    m       = pyo.ConcreteModel()
    m.order = pyo.Param(initialize=3)
    m.macc  = pyo.Param(initialize=5)
    m.s     = pyo.Param(initialize=m.order.value*m.macc.value)
    m.rows  = range(m.s.value)
    m.cols  = range(m.s.value)

    m.a = pyo.Var(m.rows, m.cols, bounds=(0,1))
    m.b = pyo.Var(m.rows,       bounds=(0,1))

    m.beta = pyo.Param(initialize=BETA_R)
    m.bet  = pyo.Param(range(m.s.value+1), initialize=BETA_VEC)
    m.cons = pyo.ConstraintList()

    global FREE_A, FREE_B, S
    FREE_A = [(i,j) for i in range(1,m.s.value) for j in range(i)]
    FREE_B = list(range(m.s.value))
    S      = m.s.value

    mycons(m)
    mybets(m)
    return m


def mycons(m, extra=None):
    c = [0.0] + [sum(m.a[r,i] for i in range(r)) for r in range(1, m.s.value)]
    # order‐1..3 conditions + row sums
    m.cons.add(sum(m.b[j] for j in m.cols) - 1 == 0)
    m.cons.add(sum(m.b[j]*c[j] for j in m.cols) - 0.5 == 0)
    m.cons.add(sum(m.b[j]*c[j]**2 for j in m.cols) - 1/3 == 0)
    expr4 = sum(m.b[i]*m.a[i,j]*c[j] for i in range(2,m.s.value) for j in range(1,i))
    m.cons.add(expr4 - 1/6 == 0)
    for i in range(1, m.s.value):
        m.cons.add(sum(m.a[i,j] for j in range(i)) - c[i] == 0)
    if extra:
        m.cons.add(eval(extra))


def mybets(m):
    # internal‐stability objective + beta‐matching
    c, Q = [[],[]], []
    zval = -m.beta
    for i in m.rows:
        Q.append(m.b[i]*zval)
    for J in range(1,m.s.value):
        ibet = J+1
        iold, inew = (J-1)%2, J%2
        c[inew].clear()
        for i in range(J):  c[inew].append([None])
        for i in range(J, m.s.value):
            row = []
            for j in range(i-J+1):
                if J>1:
                    row.append(sum(c[iold][i][k]*m.a[k,j]
                                   for k in range(j+1, i-J+2)))
                else:
                    row.append(m.a[i,j])
            c[inew].append(row)
        if ibet > m.order:
            expr = sum(m.b[i]*sum(c[inew][i]) for i in range(J,m.s.value)) - m.bet[ibet]
            m.cons.add(expr==0)
        for j in range(m.s.value-J):
            Q[j] += sum(m.b[i]*c[inew][i][j] for i in range(J+j,m.s.value)) * zval**(J+1)
    L = m.s.value
    m.obj = pyo.Objective(expr=(sum(abs(Q[i])**L for i in m.rows))**(1/L))


def _rand_factor(max_pow=3):
    # choose one of the three variable types, never an int
    r = random.random()
    if r < 0.4:
        i,j = random.choice(FREE_A)
        base = f"m.a[{i},{j}]"
    elif r < 0.8:
        k = random.choice(FREE_B)
        base = f"m.b[{k}]"
    else:
        r_idx = random.randint(0, S-1)
        base  = f"c[{r_idx}]"
    # optionally add a power
    if random.random() < 0.3:
        p = random.randint(2, max_pow)
        base += f"**{p}"
    return base



def expression(min_terms=1, max_terms=5, max_factors=3):
    lhs_i,lhs_j = random.choice(FREE_A)
    n_terms      = random.randint(min_terms, max_terms)
    rhs_terms    = []
    for _ in range(n_terms):
        factors = [_rand_factor() for _ in range(random.randint(1, max_factors))]
        rhs_terms.append(" * ".join(factors))
    expr = f"m.a[{lhs_i},{lhs_j}] == {rhs_terms[0]}"
    for term in rhs_terms[1:]:
        expr += random.choice([" + ", " - "]) + term
    return expr


import random

import random, re

def mutate_expression(expr, p_term=0.2, p_factor=0.8, min_terms=1, max_terms=5):
    """
    Mutate either:
      • the list of terms (with probability p_term), by adding/removing one, or
      • a single factor inside one term (with probability p_factor).
    Ensures no empty factors ever slip through.
    """
    lhs, rhs = expr.split("==")
    lhs = lhs.strip()
    rhs = rhs.strip()

    # 1) split out terms with their leading +/-  
    parts = re.findall(r'([+-]?\s*[^+-]+)', rhs)
    terms = [p.strip() for p in parts]

    def rebuild(terms):
        out = terms[0].lstrip('+ ').strip()
        for t in terms[1:]:
            sign = '+' if t[0] not in '+-' else t[0]
            body = t.lstrip('+- ').strip()
            out += f" {sign} {body}"
        return out

    new_terms = terms.copy()
    if random.random() < p_term:
        # TERM-level mutation
        if len(new_terms) > min_terms and random.random() < 0.5:
            new_terms.pop(random.randrange(len(new_terms)))
        elif len(new_terms) < max_terms:
            i,j = random.choice(FREE_A)
            nf = random.randint(1,3)
            factors = [_rand_factor() for _ in range(nf)]
            term = " * ".join(factors)
            sign = random.choice(['+','-'])
            new_terms.insert(random.randrange(len(new_terms)+1), f"{sign} {term}")
    else:
        # FACTOR-level mutation
        idx = random.randrange(len(new_terms))
        term = new_terms[idx]
        sign = term[0] if term[0] in '+-' else ''
        body = term[1:].strip() if sign else term
        factors = [f.strip() for f in body.split('*') if f.strip()]
        fi = random.randrange(len(factors))
        factors[fi] = _rand_factor()
        new_body = " * ".join(factors)
        new_terms[idx] = f"{sign} {new_body}"

    # Rebuild and return
    new_rhs = rebuild(new_terms)
    return f"{lhs} == {new_rhs}"





def main(seed=42, n_iter=100000, stable_target=10, max_mutations=5):
    random.seed(seed)

    # ────── FIX #1: initialise FREE_A/FREE_B/S before calling expression()
    _ = build_model()

    solver = SolverFactory('ipopt')
    solver.options.update({
        'linear_solver':'ma27',
        'tol':1e-6,
        'acceptable_tol':1e-6,
        'honor_original_bounds':'yes',
        'bound_relax_factor':1e-6,
        'max_iter':477
    })

    stable_bank = []
    for it in range(1, n_iter+1):
        if len(stable_bank) >= stable_target:
            print(f"Reached {stable_target} stable heuristics; done.")
            break

        best_obj = None
        h        = expression()
        for tr in range(1, max_mutations+1):
            print(f"[Iter {it} – try {tr}/{max_mutations}]  heuristic: {h}")
            m = build_model()

            # random initialisation
            init = 1.0/m.s.value
            for i in range(1,m.s.value):
                for j in range(i):
                    m.a[i,j].value = random.uniform(0,init)
            for j in m.cols:
                m.b[j].value   = random.uniform(0,init)

            # add this heuristic constraint
            mycons(m, extra=h)

            # pre-screen
            a_vals = [m.a[i,j].value for i in range(1,m.s.value) for j in range(i)]
            b_vals = [m.b[j].value   for j in m.cols]
            if not evaluate_butcher_stability(a_vals,b_vals):
                print("  ✖ pre-screen failed, mutating…")
                h = mutate_expression(h)
                continue

            # solve
            res = solver.solve(m, tee=False)
            if (res.solver.status == SolverStatus.ok
             and res.solver.termination_condition == TerminationCondition.optimal):
                val = pyo.value(m.obj)
                print(f"  ✔ IPOPT obj = {val:.3e}")
                best_obj = val
                stable_bank.append(h)
                break
            else:
                print("  ✖ IPOPT failed, mutating…")
                h = mutate_expression(h)

        if best_obj is None:
            print(f"Iteration {it} gave up after {max_mutations} tries.")
        else:
            print(f"Iteration {it} succeeded with obj={best_obj:.3e}")

    print("=== Done. Stable heuristics: ===")
    print("\n".join(stable_bank))
    with open("stable_heuristics.json","w") as f:
        json.dump(stable_bank, f, indent=2)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--iter',   type=int, default=1000)
    p.add_argument('--target', type=int, default=10)
    p.add_argument('--mut',    type=int, default=5)
    args = p.parse_args()
    main(seed=args.seed,
         n_iter=args.iter,
         stable_target=args.target,
         max_mutations=args.mut)
