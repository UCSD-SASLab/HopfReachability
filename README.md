# HopfReachability
Code for doing forwards and backwards reachability of optimal 2-player differential games (control vs. disturbance) via Hopf optimization of the Hamilton Jacobi Bellman equation. This method allows for solving the BRS value function in a fast, space-parallelizeable fashion. Based on the work of Yat Tin Chow, Jeremy Darbon and Stan Osher.

Currently, this method has been validated for problems with,
- Linear Dynamics (nonlinear proofs in the works, however they involve complex optimization problems)
- Games satisfying Isaacs' condition (min max = max min)

Beware, if the differential game is nonconvex, ie. if the allowable disturbance exceeds the control set, then the Hopf objective is nonconvex and convergence to the global optimum (true viscosity solution value) is not guaranteed. However, we can check if an optimum is global (p(0) \in \partial J(x^*)) and if false, the optimizer reinitialized. In practice, the global optimum is found within a few guesses on average and does not affect the solution at any other point.

Note, **this package and its algorithms are in the early stages of development**. The authors (Will Sharpless & Sylvia Herbert) welcome any criticism or discovery of bugs. If you are interested in helping, we have many ideasto advance this package and look forward to collaboration.

## Current Problem Formulation

Given a linear, time-varying system,
```math
\dot{x} = M(t)x + C_1(t) u + C_2(t) d
```
where control and disturbance are constrained to time-varying sets (we default to ellipses, but inf-norm, 2-norm and 1-norm bounds are all allowable with user-defined functions),
```math
u \in \{(u-a_u (t))^T Q^{-1}_u (t) (u-a_u (t)) \leq 1\} \quad \& \quad d \in \{(d-a_d (t))^T Q^{-1}_d (t) (d-a_d (t)) \leq 1 \}
```
we will compute the (Backwards default) Reachable Set for time T for which all points can be driven to the target with the optimal control despite the worst disturbance.

The target set is defined by an ellipsoidal function J for which,
```math
J(x) \leq 0 \iff x \in Target \quad \& \quad J(x) = 0 \iff x \in \partial Target \quad \& \quad J(x) \geq 0 \iff x \notin Target
```

## Code Structure

- Hopf_BRS: fed a system, target and T, (optionally grid and optimization parameters) and makes a grid and performs optimization for points in the grid near the boundary of the target by calling,
- Hopf_cd or Hopf_admm: do the optimization (coordinate descent or the alternating method of multipliers) and reinitializes to find global optimum and calls,
- Hopf: evaluates the value of the Hopf formula for a given value of x and v.
- Hopf_minT: finds the minimum time such that a given state is reachable and returns the optimal control
- plot_BRS: will produce either scatter (fast) or contour (slow and sometimes misleading) plots

## Demo

See the demo HopfReachability.ipynb file

## Future Work

We will expand this toolbox to handle problems that include, 
- Partnership with linearization techniques such as Koopman lifting
- nonlinear systems (which require generalized Hopf formula which is complex to optimize)
- higher dimensional target spaces (currently 2D + 1 value in any higher dimension)

On the solver side, we will build the ability to 
- utilize the other optimization methods (PDHG) which have been shown to vastly improve the number of optimization reinitializations 
- manually enter the gradient. 
- autodiff the gradient
- parallelize the grid solving
- render 3D target sets
- readily call this toolbox from Python and Matlab
