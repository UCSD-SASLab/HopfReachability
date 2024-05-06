
using LinearAlgebra, Plots, PlotlyJS, ScatteredInterpolation
plotlyjs()
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, plot_nice, make_grid, make_levelset_fs

## System (3D Example)
A, B₁, B₂ = [0. 1 0.; -2 -3 0.; 0. 0. -1.], 0.5 * [1 0; 0 1; 0. 0.], 0.5 * [2 0; 0 1; 0. 0.]
Q₁, Q₂, c₁, c₂ = 0.1 * 3 * [1 0; 0 1], 0.2 * 2 * [1 0; 0 1], [0. 0.], [0. 0.] # 𝒰 & 𝒟
system = (A, B₁, B₂, Q₁, c₁, Q₂, c₂)

## Target
Q, center, radius = diagm(vcat([1.25^2], ones(size(A)[1]-1))), zero(A[:,1]), 1.
J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.2, 0.8
times = collect(Th : Th : Tf);

## Points to Solve 
bd, res, ϵ = 4, 0.4, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, small_shift=ϵ);

## Hopf Coordinate-Descent Parameters (optional)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

# Hopf ADMM Parameters (optional)
ρ, ρ2, tol, max_its = 1e-1, 1e-1, 1e-5, 3
opt_p_admm = (ρ, ρ2, tol, max_its)

# ADMM-CD Hybrid Parameters (optional)
ρ_grid_vals, hybrid_runs = 1, 3 
opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 3), (0.005, 5, 1e-4, 500, 1, 3, 500), ρ_grid_vals, ρ_grid_vals, hybrid_runs)

solution, run_stats = Hopf_BRS(system, target, times; Xg, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot_scatter = plot_nice(times, solution; A, ϵs=2e-1, interpolate=false, value_fn=true, alpha=0.1)
plot_contour = plot_nice(times, solution; A, ϵc=1e-3, interpolate=true, value_fn=true, alpha=0.5)