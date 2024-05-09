
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, plot_nice, make_grid, make_levelset_fs, make_set_params
using LinearAlgebra, Plots, OrdinaryDiffEq

## Times to Solve
th, Th, Tf = 0.05, 0.25, 1.0
T = collect(Th : Th : Tf)

## Time-Varying Systems
A = -[0. float(pi); -float(pi) 0.]
At = [if s <= T[end]/2th; -A; else; A; end; for s=1:Int(T[end]/th)] # can be defined as an array
Af(s) = 2 * (Tf - s) * A                                            # or can be defined as a function

B₁, B₂ = [1. 0; 0 1], [1. 0; 0 1];
B₁t, B₂t = [[1. 0; 0 1] for s=1:Int(T[end]/th)], [[1. 0; 0 1] for s=1:Int(T[end]/th)];
B₁f(s), B₂f(s) = [1. 0; 0 1], [1. 0; 0 1];

max_u, max_d, input_center, input_shapes = 0.75, 0.25, zeros(2), "ball"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) # control set 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # disturbance set

game = "reach"
system   = (A, B₁, B₂, Q₁, c₁, Q₂, c₂)
system_t = (At, B₁, B₂, Q₁, c₁, Q₂, c₂)
system_f = (Af, s -> B₁f(s), s -> B₂f(s), Q₁, c₁, Q₂, c₂)

## Target
center, radius = [-1.; 1.], 0.5
J, Jˢ = make_levelset_fs(center, radius; type="ball")
target = (J, Jˢ, (diagm([1; 1]), center));

## Points to Solve
bd, res, ϵ = 3, 0.1, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ);

## Hopf Coordinate-Descent Parameters (optional)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution, run_stats = Hopf_BRS(system, target, T; th, Xg, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
solution_t, run_stats = Hopf_BRS(system_t, target, T; th, Xg, input_shapes, game, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
solution_f, run_stats = Hopf_BRS(system_f, target, T; th, Xg, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot_contour = plot(solution; xigs=xigs, value=false, title="A - Hopf");
plot_contour_t = plot(solution_t; xigs=xigs, value=false, title="Aₜ - Hopf")
plot_contour_f = plot(solution_f; xigs=xigs, value=false, title="A(t) - Hopf");

### Solve "true" BRS with DP (requires hj_reachability.py)
include(pwd() * "/src/DP_comparison_utils.jl");

dynamics = Linear(A, B₁, B₂; max_u, max_d, input_shapes, game)
dynamics_t = [Linear(At[end:-1:1][tsi], B₁, B₂; max_u, max_d, input_shapes, game) for tsi=1:length(th:th:T[end])]  # t going backwards in DP
dynamics_f = Linear(s -> Af(Tf + s), B₁, B₂; max_u, max_d, input_shapes, game) # t going backwards in DP

Xg_DP, Xg_DP_py, ϕ0Xg_DP, _ = hjr_init(center, diagm(ones(2)), radius; shape="ball", lb=(-bd, -bd), ub=(bd, bd), res=Int(2bd/res)+1);
DP_solution = hjr_solve(Xg_DP_py, ϕ0Xg_DP, [dynamics], T; BRS=true);
DP_solution_t = hjr_solve(Xg_DP_py, ϕ0Xg_DP, [dynamics_t], T; BRS=true, one_shot=false, tv=true, th=th);
DP_solution_f = hjr_solve(Xg_DP_py, ϕ0Xg_DP, [dynamics_f], T; BRS=true);

plot_contour_DP =   plot((fill(Xg_DP, length(T)+1), DP_solution[1]); xigs=xigs, value=false, title="A - DP");
plot_contour_DP_t = plot((fill(Xg_DP, length(T)+1), DP_solution_t[1]); xigs=xigs, value=false, title="Aₜ - DP")
plot_contour_DP_f = plot((fill(Xg_DP, length(T)+1), DP_solution_f[1]); xigs=xigs, value=false, title="A(t) - DP")

comparison_fig = plot(plot_contour, plot_contour_t, plot_contour_f, plot_contour_DP, plot_contour_DP_t, plot_contour_DP_f, layout=(2,3), size=(800,500), legend=:bottomleft)

plot_contour_val = plot(solution_f; xigs=xigs, value=true, title="BRS - Hopf", title_value="Value - Hopf")
plot_contour_DP_f_val = plot((fill(Xg_DP, length(T)+1), DP_solution_f[1]); xigs=xigs, value=true, title="BRS - DP", title_value="Value - DP")

comparison_fig_val = plot(plot_contour_val, plot_contour_DP_f_val, layout=(2,1), size=(600,600), dpi=300, legend=:bottomleft, xlabel="x", ylabel="y", zlabel="V", guidefont=10)