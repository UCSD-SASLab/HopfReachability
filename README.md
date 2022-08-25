# HopfReachability
Code for doing forwards and backwards reachability of optimal 2-player differential games (control vs. disturbance) via Hopf optimization of the Hamilton Jacobi Bellman equation. This method allows for solving the BRS value function in a fast, space-parallelizeable fashion. Based on the work of Yat Tin Chow, Jeremy Darbon and Stan Osher.

Currently, this method has been validated for problems with,
- Linear Dynamics (nonlinear proofs in the works)
- Games satisfying Isaacs' condition (min max = max min)

Beware, if the differential game is nonconvex, ie. if the allowable disturbance exceeds the control set, then the Hopf objective is nonconvex and convergence to the global optimum (true viscosity solution value) is not guaranteed. However, we can check if an optimum is global (p(0) \in \partial J(x^*)) and if false, the optimizer reinitialized. In practice, the global optimum is found within a few guesses on average and does not affect the solution at any other point.

## Current Problem Formulation

Given a linear, time-varying system,
```math
\dot{x} = M(t)x + C_1(t) u + C_2(t) d
```
where control and disturbance are constrained to the time-varying sets,
```math
u \in \{(u-a_u (t))^T Q^{-1}_u (t) (u-a_u (t)) \leq 1\} \quad \& \quad d \in \{(d-a_d (t))^T Q^{-1}_d (t) (d-a_d (t)) \leq 1 \}
```
we will compute the (Backwards) Reachable Set for time T for which all points can be driven to the target with the optimal control despite the worst disturbance.

The target set is defined by a function J for which,
```math
J(x) \leq 0 \iff x \in Target \quad \& \quad J(x) = 0 \iff x \in \partial Target \quad \& \quad J(x) \geq 0 \iff x \notin Target
```

## Code Structure

- Hopf_BRS: fed a system, target and T, (optionally grid and optimization parameters) and makes a grid and performs optimization for points in the grid near the boundary of the target by calling,
- Hopf_cd: does the optimization (coordinate descent currently) and reinitializes to find global optimum and calls,
- Hopf: evaluates the value of the Hopf formula for a given value of x and v.

## Demo

See the demo HopfReachability.ipynb file

## Further Work

We will expand this toolbox to handle problems that include, 
- non-coupled controls
- nonlinear systems (which are still not guaranteed but have conjectured algorithms)
- which only care about finding the optimal control for a single point in space

On the solver side, we will build the ability to 
- utilize the other optimization methods (ADM/PDHG) which have been shown to vastly improve the number of optimization reinitializations 
- manually enter the gradient. 
- autodiff the gradient
- parallelize the grid solving
- render 3D target sets
