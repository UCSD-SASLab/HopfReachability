
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, plot_nice, make_grid, make_levelset_fs, make_set_params
using LinearAlgebra, Plots, PlotlyJS
plotlyjs() # plot_nice uses plotly, not necessary but nice for 3D https://plotly.com/julia/getting-started/

## System (3D Example)
A, B₁, B₂ = [0. 1 0.; -2 -3 0.; 0. 0. -1.], [2 0; 0 1; 0. 0.], [1 0; 0 1; 0. 0.]
max_u, max_d, input_center, input_shapes = 0.4, 0.3, zeros(2), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
Q, center, radius = diagm(vcat([1.25^2], ones(size(A)[1]-1))), zero(A[:,1]), 1.
J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.2, 0.8
times = collect(Th : Th : Tf);

## Points to Solve 
bd, res, ϵ = 4, 0.4, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ);

## Hopf Coordinate-Descent Parameters (optional)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution, run_stats = Hopf_BRS(system, target, times; Xg, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot_scatter = plot_nice(times, solution; ϵs=2e-1, interpolate=false, alpha=0.1);
plot_contour = plot_nice(times, solution; ϵc=1e-3, interpolate=true, alpha=0.5);