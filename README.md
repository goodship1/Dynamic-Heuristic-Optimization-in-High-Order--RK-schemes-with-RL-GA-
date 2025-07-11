# Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes

This repository contains all code, data, and validation supporting the paper:

> **Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes using Reinforcement Learning and Genetic Algorithms**

---

## 📄 Overview

This project introduces a novel pipeline for discovering, generating, and validating high-stage third-order Runge–Kutta (RK) schemes with extended stability regions and new heuristic constraints.

---

## 🔬 What’s Included

**1. Symbolic Verification**  
- All heuristics are *symbolically verified* to scale from 15 to 30 stages (increments of 3) at only polynomial cost.
- **How to run:** See `sweep.py` in the `Symbolic_verification` folder.  
- **What it does:** Demonstrates every heuristic passes the *third-order* RK order conditions under symbolic manipulation.

**2. Numerical Validation**  
- For each heuristic, notebooks are provided:
  - Validate the method on 1D/2D Brusselator PDEs
  - Confirm third-order convergence numerically
  - Demonstrate extended stability regions
- Each notebook is named by the heuristic tested and contains code to **numerically verify all third-order order conditions** for that scheme.
- *IPOPT* code for tableaus is included for method generation.

**3. Residual-based Numerical Checker**  
- All heuristics are checked using `checker.py`, which prints for each scheme:
    - Each third-order condition (value, residual, pass/fail)
    - Residuals are all confirmed to be below 1e-10 (see output in appendix or run `checker.py`)
- This ensures full transparency and immediate reproducibility of the third-order validity for every scheme in the repo.

  **4. Coefficients for the schemes that have been found all with raw convergence metrics ODE BRUSS, and PDE BRUSS. are all provided**
  

---

## 🏃‍♂️ Quick Start

1. **Symbolic Verification:**  
   Run  
   ```bash
   python Symbolic_verification/sweep.py
2. **Numerical checker** for heuristics from the paper, this ensures that all order conditions are met to a high tolerance.
     Run
   
   ```bash
   
       python Numerical Checker/checker.py
