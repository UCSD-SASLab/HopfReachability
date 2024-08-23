
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_target, make_set_params, plot_nice
using LinearAlgebra, Plots
# using PlotlyJS

using PyCall
include(pwd() * "/src/DP_comparison_utils.jl");
include(pwd() * "/src/interp_utils.jl");

using JLD2, Interpolations, ScatteredInterpolation, ImageFiltering

using Interpolations, JLD
path_to_interps = "/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/DRHopf_interps/"
LessLinear2D_interpolations = load(path_to_interps * "LessLinear2D1i_interpolations_res1e-2_r25e-2_c20.jld", "LessLinear2D_interpolations");
# LessLinear2D_interpolations = load(path_to_interps * "LessLinear2D1i_interpolations_res1e-2_r4e-1_el_1_5.jld", "LessLinear2D_interpolations");
V_DP_itp = LessLinear2D_interpolations["g0_m0_a0"]

##

N = 3

## System & Game

A, B₁, B₂ = -0.5*I + hcat(vcat(0, -ones(N-1,1)), zeros(N,N-1)), vcat(zeros(1,N-1), 0.4*I), vcat(zeros(1,N-1), 0.1*I) # system
max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(N-1), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
Q, center, radius = diagm(ones(N)), zeros(N), 0.25
# Q, center, radius = diagm(ones(N)), zeros(N), 0.15
radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
# Q, center, radius = diagm(inv.([1., 5.])), zero(A[:,1]), 0.4
# radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(5) * ones(N-1)))
target = make_target(center, radius_N; Q=Q_N, type="ellipse")
# target = make_target(center, radius; Q, type="ellipse")

## Times to Solve
Th, Tf = 0.25, 1.0
# Th, Tf = 0.025, 1.0
times = collect(Th : Th : Tf);

## Point(s) to Solve (any set works!)
bd, ppd, ϵ = 1.0001, 31, 0*.5e-7
res = 2*bd/(ppd-1)
Xg, xigs, (lb, ub) = make_grid(bd, res, N; return_all=true, shift=ϵ); # solve over grid
Xg_2d, xigs_2d, _ = make_grid(bd, res, 2; return_all=true, shift=ϵ); # solve over grid
Xg_2d_hr, xigs_2d_hr, _ = make_grid(bd, 0.1 * res, 2; return_all=true, shift=ϵ); # solve over grid
Xg_rand = 2bd*rand(N, 1) .- bd .+ ϵ; # solve over random samples

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-5, 100, 1, 1, 1000
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

## Solve
solution, run_stats = Hopf_BRS(system, target, times; Xg, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true);

## Solve Carefully

P_in_f(∇ϕX) = reshape(hcat(∇ϕX[2:end]...), size(∇ϕX[1])..., length(∇ϕX)-1)

vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 0.5, 1e-5, 50, 1, 1, 100
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution_sampled_1, run_stats, opt_data_1, P_final_1 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, opt_tracking=true);

p_path_1, vals_1 = opt_data_1[1][1]
conv_plot = plot(1:length(vals_1), vals_1)

P_in_1 = P_in_f(P_final_1);
solution_sampled_2, run_stats, opt_data_2, P_final_2 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, opt_tracking=true, P_in=P_in_1);

p_path_2, vals_2 = opt_data_2[1][1]
plot!(length(vals_1) .+ (1:length(vals_2)), vals_2)

P_in_2 = P_in_f(P_final_2);
solution_sampled_3, run_stats, opt_data_3, P_final_3 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, opt_tracking=true, P_in=P_in_2);

p_path_3, vals_3 = opt_data_3[1][1]
plot!(length(vals_1) + length(vals_2) .+ (1:length(vals_3)), vals_3)


vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 20, 1e-5, 200, 1, 1, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution_sampled, run_stats, opt_data, P_final = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, opt_tracking=true);

p_path, vals = opt_data[1][1]
plot!(1:length(vals), vals)



p_path

vals