# Numerical Checker for Third-Order Runge–Kutta Heuristic Validation

This folder contains code and data for **numerical validation** of all heuristic constraints used in the paper:

> **Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes using Reinforcement Learning and Genetic Algorithms**

---

## What does this do?

- **Validates each heuristic**: For every heuristic constraint, the code reconstructs the Runge–Kutta coefficients (A, b) and verifies all third-order order conditions for the fifteen-stage explicit scheme.
- **Reports residuals**: For each order condition, the checker reports the computed value, the theoretical target, and the numerical residual (error).
- **One-click verification**: Output makes it immediately clear that all schemes strictly satisfy the third-order accuracy requirements (residuals typically < 1e-10).
- **Checks huestrics** : Verifies that each huestric is  in the Butcher tableau, extra sanity check for reviewers.
- **Stability polynomial**: Each of the huestrics has a stability polynomial that is checked through the stable checker to make sure the huestrics are able to produce the correct stability.

---

## How to use

1. **Run the checker**
   ```bash
   python checker.py
2 . Stable checker This checks that each heuristic is stable.
   ```bash
   python stable_check.py
   
