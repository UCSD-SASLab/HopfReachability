
using LinearAlgebra, Plots
# plotlyjs()
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, plot_BRS, make_grid, make_levelset_fs

## System (2D Example)
A, B₁, B₂ = [0. 1; -2 -3], 0.5 * [1 0; 0 1], 0.5 * [2 0; 0 1] # system
Q₁, Q₂, c₁, c₂ = 0.1 * 3 * [1 0; 0 1], 0.2 * 2 * [1 0; 0 1], [0. 0.], [0. 0.] # 𝒰 & 𝒟
system = (A, B₁, B₂, Q₁, c₁, Q₂, c₂)

## System (3D Example)
# A, B₁, B₂ = [0. 1 0.; -2 -3 0.; 0. 0. -1.], 0.5 * [1 0; 0 1; 0. 0.], 0.5 * [2 0; 0 1; 0. 0.]
# Q₁, Q₂, c₁, c₂ = 0.1 * 3 * [1 0; 0 1], 0.2 * 2 * [1 0; 0 1], [0. 0.], [0. 0.] # 𝒰 & 𝒟
# system = (A, B₁, B₂, Q₁, c₁, Q₂, c₂)

## Target
Q, center, radius = diagm(vcat([1.25^2], ones(size(A)[1]-1))), zero(A[:,1]), 1.
J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.2, 0.8
times = collect(Th : Th : Tf);

## Points to Solve 
bd, res, ϵ = 4, 3, .5e-7
Xg, xigs, (lb, ub) = make_grid((bd, res), size(A)[1]; return_all=true, small_shift=ϵ);

## Hopf Coordinate-Descent Parameters (optional)
L, vh, tol, lim, lll, max_runs, max_its = 5, 0.01, 1e-5, 500, 5, 3, 500
opt_p_cd = (vh, L, tol, lim, lll, max_runs, max_its)

# Hopf ADMM Parameters (optional)
ρ, ρ2, tol, max_its = 1e-1, 1e-1, 1e-5, 3
opt_p_admm = (ρ, ρ2, tol, max_its)

# ADMM-CD Hybrid Parameters (optional)
ρ_grid_vals, hybrid_runs = 1, 3 
opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 3), (0.005, 5, 1e-4, 500, 1, 3, 500), ρ_grid_vals, ρ_grid_vals, hybrid_runs)

solution, run_stats = Hopf_BRS(system, target, times; Xg, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

BRS_plot = plot(title="BRS")
labels, colors, alpha, lw = vcat("Target", ["t=-$ti" for ti in T]...), vcat(:black, palette(["red", "blue"], length(T))...), 0.8, 2
for ti=1:length(T)+1; contour!(BRS_plot, xigs..., reshape(solution[2][ti], length(xigs[1]), length(xigs[1]))', levels=[0], color=colors[ti], lw=lw, alpha=alpha, colorbar=false); end
for ti=1:length(T)+1; plot!(BRS_plot, [1e5, 2e5], [1e5, 2e5], color=colors[ti], lw=lw, alpha=alpha, label=labels[ti], xlims=xlims(BRS_plot), ylims=ylims(BRS_plot)); end # contour label workaround
BRS_plot

# plot_scatter = plot_BRS(T, solution...; A, ϵs=2e-1, interpolate=false, value_fn=true, alpha=0.1)
# plot_contour = plot_BRS(T, solution...; A, ϵc=1e-3, interpolate=true, value_fn=true, alpha=0.5)