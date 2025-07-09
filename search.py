from __future__ import division
"""
Runge–Kutta (ESRK) search                  
======================================================
* **Exploration**  – every outer loop we create a brand‑new random heuristic
  constraint (or mutate an older one) and random starting coefficients.
* **Exploitation** – IPOPT tries to minimise the internal‑stability
  objective given that heuristic.
* **Restart**      – regardless of success/failure we immediately go back
  to Exploration, so the cycle continues.
* We **collect** any heuristic whose random start already satisfies the
  Butcher‑stability pre‑screen; once we have `stable_target` of them we
  stop.

This file is self‑contained: run it with

    python rk_search.py --seed 42

and you should reproduce the numbers reported in the manuscript on any
machine with IPOPT & MA27.

For the huestrics in the paper a stand alone notebook with results and tableaus 
will be provided this will contain all info needed and notebook to run them 
"""

# ---------------------------------------------------------------------------
# 0.  Imports & global beta data                                                
# ---------------------------------------------------------------------------
import json, random, time, argparse, sys
import numpy as np
import sympy as sym
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# ---- reference SSP‑RK β values --------------------------------------------
_BETA = np.array([
    7.2362092230726359621444062630453895436513878479204372239744118914339008361957274894021123680041692242e+01,
    1.0, 1.0, 0.5, 1/6,
    3.30278406028757527979332298466336725484247797530949148541780288765549097681738463834524909538164545e-02,
    4.0788675461517543025407589082899561446180084150021384558306497480447059168658653272938025859312818e-03,
    3.33527373172737242603705467098764792409315403431559178662909610705332831105918682590799425317677283e-04,
    1.88438744545771964992699112477377463925568623266923890807654445539507743132813283375146121935597446e-05,
    7.55565600673228187844472799772168548435456673497418681132813388498239377614064205083716425567987086e-07,
    2.180674889659337806106474225333623253164152989661516940939984945939720123180192751320554488732457046e-08,
    4.543295752073872095310410708039457390472975268776849051520e-10,
    6.76964099198340395235741216662494203744813306696814739898497900746954960650250467206606180555417237e-12,
    7.03301065055001143525874293209920770968902766712091555585888497900746954960650250467206606180555417237e-14,
    4.83760685817544104974918008895344303724730995799826121736221363185699872131913128162377233612731080e-16,
    1.97953679111068662248975009738027259306277639199446521870174193132300321746057499270873882123310487500e-18,
    3.6474655538315803148945976064548166497173172457990913939216649361282791779749929476922394573365077700e-21
])
BETA_R   = float(_BETA[0])
BETA_VEC = {i: float(v) for i, v in enumerate(_BETA[1:])}

# ---------------------------------------------------------------------------
# 1.  SymPy helper — fixed padding to avoid IndexError -----------------------
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 2.  Pyomo model builder ---------------------------------------------------
# ---------------------------------------------------------------------------




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
    c = [0]
    for r in range(1, m.s.value):
        c.append(sum(m.a[r, i] for i in range(r)))
    # linear order conditions
    m.cons.add(sum(m.b[j] for j in m.cols) - 1 == 0)
    m.cons.add(sum(m.b[j]*c[j] for j in m.cols if j>0) - 0.5 == 0)
    m.cons.add(sum(m.b[j]*c[j]**2 for j in m.cols if j>0) - 1/3 == 0)
    expr4 = sum(m.b[i]*m.a[i,j]*c[j] for i in range(2,m.s.value) for j in range(1,i))
    m.cons.add(expr4 - 1/6 == 0)
    if extra:
        m.cons.add(eval(extra))

def mybets(m):
    c, Q = [[],[]], []
    zval = -m.beta
    for i in m.rows:
        Q.append(m.b[i]*zval)
    for J in range(1,m.s.value):
        ibet = J+1
        iold, inew = (J-1)%2, J%2
        c[inew].clear()
        for i in range(J): c[inew].append([None])
        for i in range(J,m.s.value):
            row=[]
            for j in range(i-J+1):
                if J>1:
                    row.append(sum(c[iold][i][k]*m.a[k,j] for k in range(j+1,i-J+2)))
                else:
                    row.append(m.a[i,j])
            c[inew].append(row)
        if ibet>m.order:
            expr = sum(m.b[i]*sum(c[inew][i]) for i in range(J,m.s.value)) - m.bet[ibet]
            m.cons.add(expr==0)
            print("**beta**", ibet, float(m.bet[ibet]))
        for j in range(m.s.value-J):
            Q[j] += sum(m.b[i]*c[inew][i][j] for i in range(J+j,m.s.value))*zval**(J+1)
    L = m.s.value
    m.obj = pyo.Objective(expr=sum(abs(Q[i])**L for i in m.rows)**(1/L))

# ---------------------------------------------------------------------------
# 4.  Heuristic gen/mutation ------------------------------------------------
# ---------------------------------------------------------------------------

def _rand_factor(max_pow=3):
    """
    Return a random factor, possibly with exponentiation or negation:
      - a[i,j] or a[i,j]**p
      - b[k] or b[k]**p
      - c[r] or c[r]**p
    Each factor may be negated with 30% chance.
    """
    r = random.random()
    if r < 0.4:
        i, j = random.choice(FREE_A)
        p    = random.randint(1, max_pow)
        base = f"m.a[{i},{j}]" + (f"**{p}" if p>1 else "")
    elif r < 0.8:
        k = random.choice(FREE_B)
        p = random.randint(1, max_pow)
        base = f"m.b[{k}]" + (f"**{p}" if p>1 else "")
    else:
        r_idx = random.randint(0, S-1)
        p     = random.randint(1, max_pow)
        base  = f"c[{r_idx}]" + (f"**{p}" if p>1 else "")
    # optional negation
    if random.random() < 0.3:
        return f"-({base})"
    return base


def expression(min_terms=1, max_terms=5, max_factors=3):
    """
    Return a random heuristic constraint composed of up to five terms,
    each term the product of up to three random factors, linked by + or -.
    """
    lhs_i, lhs_j = random.choice(FREE_A)
    n_terms      = random.randint(min_terms, max_terms)
    rhs_terms    = []
    for _ in range(n_terms):
        n_factors = random.randint(1, max_factors)
        factors   = [_rand_factor(max_pow=3) for _ in range(n_factors)]
        rhs_terms.append(" * ".join(factors))

    # build signed expression
    expr = f"m.a[{lhs_i},{lhs_j}] == {rhs_terms[0]}"
    for term in rhs_terms[1:]:
        sign = random.choice([' + ', ' - '])
        expr += sign + term
    return expr


def mutate_expression(expr, p=0.3):
    """
    Randomly tweak a term or factor in the given heuristic expression.
    """
    lhs, rhs = expr.split('==')
    # split by + or - while preserving signs
    tokens = []
    cur    = ''
    for ch in rhs:
        if ch in '+-' and cur.strip():
            tokens.append(cur.strip())
            tokens.append(ch)
            cur = ''
        else:
            cur += ch
    tokens.append(cur.strip())

    # mutate a random factor
    factors = [tok for tok in tokens if '*' in tok or tok.startswith('m.') or tok.startswith('c[')]
    if random.random() < p and factors:
        old = random.choice(factors)
        new = _rand_factor(max_pow=3)
        rhs = rhs.replace(old, new, 1)
    return lhs + '== ' + rhs

# ---------------------------------------------------------------------------
# 5.  Main search loop ------------------------------------------------------
# ---------------------------------------------------------------------------
def main(seed=42,n_iter=100_000,stable_target=10):
    random.seed(seed)
    solver=SolverFactory('ipopt',executable='ipopt')
    solver.options.update({
        'linear_solver':'ma27','tol':1e-6,'acceptable_tol':1e-6,
        'honor_original_bounds':'yes','bound_relax_factor':1e-6,
        'max_iter':             477 
    })

    stable_bank=[]; obj_best=1e10
    for it in range(1, n_iter+1):
        print(f"Iteration {it}: Generating new heuristic")
        if len(stable_bank)>=stable_target:
            print(f"Found {stable_target} stable heuristics → stopping")
            break
        m=build_model(); init=1/m.s.value
        # random init coefficients
        for i in range(1,m.s.value):
            for j in range(i): m.a[i,j].value=random.uniform(0,init)
        for j in m.cols: m.b[j].value=random.uniform(0,init)
        hstr=expression();
        print(f"Trying heuristic: {hstr}")
        mycons(m,hstr)
        a_vals=[m.a[i,j].value for i in range(1,m.s.value) for j in range(i)]
        b_vals=[m.b[j].value for j in m.cols]
        # pre-screen stability
        if evaluate_butcher_stability(a_vals,b_vals):
            stable_bank.append(hstr)
            print(f"✔ Pre‑screen stable #{len(stable_bank)}: {hstr}")
        else:
            print(f"✖ Pre‑screen failed for: {hstr}")
            continue  # skip solver if fails pre-screen
        # attempt optimization
        try:
            res=solver.solve(m,tee=False)
            if res.solver.status==SolverStatus.ok and res.solver.termination_condition==TerminationCondition.optimal:
                val=m.obj()
                print(f"✔ IPOPT optimized obj={val:.3e}")
                if val<obj_best:
                    obj_best=val
                    best_json=json.dumps(pyo.json_loads(pyo.json_save(model=m)))
            else:
                print(f"✖ IPOPT did not converge: status={res.solver.status}")
        except Exception as e:
            print(f"✖ IPOPT failed: {e}")

    print("Best objective value:",obj_best)
    if stable_bank:
        with open('stable_heuristics.json','w') as f: json.dump(stable_bank,f,indent=2)
        print("Stable heuristics saved → stable_heuristics.json")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=412)
    p.add_argument('--iter',type=int,default=100000)
    p.add_argument('--target',type=int,default=10)
    args=p.parse_args(); main(seed=args.seed,n_iter=args.iter,stable_target=args.target)
