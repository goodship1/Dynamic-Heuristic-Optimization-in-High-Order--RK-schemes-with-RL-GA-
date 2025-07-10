# Heuristic Scaling for Explicit Runge–Kutta Schemes

This repository contains a small Sympy‐based driver (`sweep.py`) that symbolically verifies that a collection of **heuristic constraints** (“huestrics”) continue to satisfy the third-order Runge–Kutta order conditions (and row-sum consistency) as you vary the number of stages \(s\).  The script loops over stage counts \(s=15,18,21,24,27,30\) and, for each heuristic:

1. Substitutes the heuristic algebraically into the Butcher‐tableau symbols.
2. Attempts to resolve one row‐sum unknown via Sympy’s `solve`.
3. Tests each of the four order‐conditions:
   - \(\sum_i b_i = 1\)
   - \(\sum_i b_i c_i = \tfrac12\)
   - \(\sum_i b_i c_i^2 = \tfrac13\)
   - \(\sum_i b_i\sum_j a_{ij} c_j = \tfrac16\)

You’ll get a pass/fail report for each condition, for each heuristic, at each stage count.

---

## Repository Contents

- `sweep.py`  
  The main driver.  Generates symbolic \(a_{ij},b_i,c_i\); defines the order conditions and a fixed set of heuristics; then loops over \(s=15\) to \(30\) in steps of 3, reporting which conditions remain satisfied.

- `README.md`  
  This file.

---

## Requirements

- **Python 3.7+**  
- **Sympy** (tested on 1.8+)

You can install Sympy with:

```bash
pip install sympy
