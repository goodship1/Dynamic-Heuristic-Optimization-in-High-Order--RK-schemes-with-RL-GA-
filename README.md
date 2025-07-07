# Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes

This repository contains all code and data accompanying the paper:

> **Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes using Reinforcement Learning and Genetic Algorithms**  
>

--- 

## 📄 Overview

We present a novel pipeline for discovering and validating high-stage, third-order Runge–Kutta (RK) methods with extended stability regions:

1. **Heuristic generation** via a hybrid Genetic Algorithm (GA) + Reinforcement Learning (RL) loop  
2. **Feasibility and order-condition enforcement** through IPOPT  
3. **Symbolic verification** that our heuristics scale from 15 to 30 stages (in increments of 3) with only polynomial cost  
4. **Numerical validation** on 1D and 2D Brusselator PDEs, demonstrating clean third-order convergence and enlarged stability regions

---

## 📂 Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── quickstart_convergence.ipynb  # Smoke-test: reproduce main convergence plots
│   └── full_search_demo.ipynb        # Full GA+RL+IPOPT pipeline demo
├── src/
│   ├── heuristics.py                 # GA + RL mutation, selection, fallback logic
│   ├── models.py                     # Pyomo model builders (brusselator_1d, brusselator_ode, etc.)
│   ├── solver.py                     # IPOPT invocation wrappers
│   ├── verify_symbolic.py            # Symbolic scaling tests (15→30 stages)
│   └── convergence.py                # Fixed-step convergence study scripts
├── data/
│   └── precomputed/                  # Precomputed iteration counts, stability data
└── figures/
    ├── stability_contours/           # PNGs of |R(z)|=1 regions
    └── convergence_plots/            # PNGs of log–log error curves
