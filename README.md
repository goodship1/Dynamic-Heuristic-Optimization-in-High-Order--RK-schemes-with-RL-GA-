# Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes

This repository contains all the code and data accompanying the paper:

> **Dynamic Heuristic Optimisation in High-Order Runge–Kutta Schemes using Reinforcement Learning and Genetic Algorithms**  
>

--- 

## 📄 Overview

We present a novel pipeline for discovering and validating high-stage, third-order Runge–Kutta (RK) methods with extended stability regions:

1. **Symbolic verification** that our heuristics scale from 15 to 30 stages (in increments of 3) with only polynomial cost- Run sweep.py in the Symbolic verification file
     In the file Symbolic verification this shows that all the huestrics can be scaled and dont break the third order RK order condtions.

3. **Numerical validation** on 1D and 2D Brusselator PDEs, demonstrating clean third-order convergence and enlarged stability regions for each huestric from the paper, has only the file with the huestric as the file name with a notebook of code to verify the order conditions are met numerically, providing third-order convergence, plus Stability and internal amplification studies, along with the IPOPT code with the huestric, which can be used to generate the tableaus. 

---

