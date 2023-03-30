# HopfReachability
Code for solving Hamilton-Jacobi reachability and optimal control of 2-player differential games (control vs. disturbance) via optimization of the Hopf cost. This method allows for solving the value function in a space-parallelizeable fashion that avoids the curse of dimensionality. Based on the work of Yat Tin Chow, Jerome Darbon and Stan Osher.

Currently, this method has been validated for problems with,
- Linear Time-Varying Dynamics
- Games satisfying Isaacs' condition (min max = max min)

Beware, if the Hamiltonian is nonconvex, which occurs when the disturbances exceed the control authority, then the Hopf objective is nonconvex and convergence to the global optimum (true viscosity solution value) is not guaranteed. In these settings, we reinitialize the optimization multiple times and in practice, we find that proximal methods converge to the global optimum within one or two guesses. Note, an erroneous value does not affect the solution at any other point (unless warm-starting).

Note, **this package and its algorithms are in the early stages of development**. The authors (Will Sharpless & Sylvia Herbert) welcome any criticism or discovery of bugs. If you are interested in helping, we have many ideas to advance this package and look forward to collaboration.

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
- Hopf_cd/Hopf_admm: do the optimization (coordinate descent or the alternating method of multipliers) and reinitializes to find global optimum and calls,
- Hopf: evaluates the value of the Hopf formula for a given value of x and v.
- Hopf_minT: finds the minimum time such that a given state is reachable and returns the optimal control
- plot_BRS: will produce either scatter (fast) or contour (slow and sometimes misleading) plots, can do 2D or 3D, also can plot value function

## Demo

See the examples files

## Future Work

We will expand this toolbox to handle problems that include, 
- nonlinear systems (which require generalized Hopf formula which is harder to optimize)
- error analysis
- more types of built-in target and constraint geometries

On the solver side, we will build the ability to 
- refine the value function based on Lipschitz bounds
- utilize the other optimization methods (PDHG)
- autodiff the gradient for convex cases
- parallelize the grid solving
- readily call this toolbox from Python and Matlab
