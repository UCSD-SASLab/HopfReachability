# HopfReachability
Code for solving Hamilton-Jacobi reachability and optimal control of 2-player differential games (control vs. disturbance) via optimization of the Hopf cost. This method allows for solving the value function in a space-parallelizeable fashion that avoids the curse of dimensionality. Based on the work of Yat Tin Chow, Jerome Darbon and Stan Osher.

Currently, this solution and algorithm are validated for,
- Linear Time-Varying Dynamics
- Differential Games with unique Nash Equilibria (min max = max min)

Beware, if the Hamiltonian is nonconvex, which occurs when the disturbance set exceeds the control set, then the Hopf objective is nonconvex and convergence to the global optimum (true viscosity solution value) is not guaranteed. In these settings, we reinitialize the optimization multiple times and in practice, we find that proximal methods converge to the global optimum within one or two guesses. Note, an erroneous value does not affect the solution at any other point (unless warm-starting).

Note, **this package and its algorithms are in the early stages of development**. The authors (Will Sharpless, Yat Tin Chow, Sylvia Herbert) welcome any criticism or discovery of bugs. If you are interested in helping, we have many ideas to advance this package and look forward to collaboration.

## Current Problem Formulation

Given a linear, time-varying system,
```math
\dot{x} = Ax + B_1(t) u + B_2(t) d
```
where control and disturbance are constrained to convex sets (ellipses and boxes predefined, but any convex constraint allowed), e.g. 
```math
u \in \big\{ u \in \mathbb{R}^{n_u} \:\: | \:\: (u-c_u (t))^T Q^{-1}_u (t) (u-c_u (t)) \leq 1 \big\} \quad \& \quad d \in \big\{d \in \mathbb{R}^{n_d} \:\: | \:\: (d-c_d (t))^T Q^{-1}_d (t) (d-c_d (t)) \leq 1 \big\}
```
we will compute the Backwards Reachable Set for time T for which all points can be driven to the target with the optimal control despite the worst disturbance.

The target set is defined by an initial Value function $J(x)$ for a convex target set $\mathcal{T}$ (ellipses and boxes predefined) such that,
```math
\begin{cases}
J(x) < 0 \:\: \text{ for } \:\:x \in \mathcal{T} \setminus \partial\mathcal{T} \\
J(x) = 0 \:\: \text{ for }\:\:x \in \partial\mathcal{T} \\
J(x) > 0 \:\: \text{ for } \:\:x \notin \mathcal{T}
\end{cases}
```
where $\partial \mathcal{T}$ is the boundary of $\mathcal{T}$. E.g. 

```math
\mathcal{T} := \big\{x \in \mathbb{R}^{n_x}  \:\: | \:\: (x - c_\mathcal{T})^T Q^{-1}_\mathcal{T} (x - c_\mathcal{T}) \le 1 \big\} \rightarrow J(x) = \frac{1}{2} \big((x - c_\mathcal{T})^T Q^{-1}_\mathcal{T} (x - c_\mathcal{T}) - 1 \big)
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
