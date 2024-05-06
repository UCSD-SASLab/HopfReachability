
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_levelset_fs
using LinearAlgebra, Plots

## System
A, B₁, B₂ = [0. 1; -2 -3], 0.5 * [1 0; 0 1], 0.5 * [2 0; 0 1] # system
Q₁, Q₂, c₁, c₂ = 0.1 * 3 * [1 0; 0 1], 0.2 * 2 * [1 0; 0 1], [0. 0.], [0. 0.] # 𝒰 & 𝒟
system = (A, B₁, B₂, Q₁, c₁, Q₂, c₂)

## Target
Q, center, radius = diagm(vcat([1.25^2], ones(size(A)[1]-1))), zero(A[:,1]), 1.
J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.2, 0.8
times = collect(Th : Th : Tf);

## Point(s) to Solve (here, a grid, but any matrix works)
bd, res, ϵ = 4, 0.25, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, small_shift=ϵ);

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution, run_stats = Hopf_BRS(system, target, times; Xg, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot(solution; labels=vcat("Target", ["t=-$ti" for ti in times]...), grid=true, xigs=xigs, value=true, camera=(30, 15), seriestype=:scatter)