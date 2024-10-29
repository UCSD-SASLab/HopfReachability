
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
# LessLinear2D_interpolations = load(path_to_interps * "LessLinear2D1i_interpolations_res1e-2_r25e-2_c20.jld", "LessLinear2D_interpolations");
LessLinear2D_interpolations = load(path_to_interps * "LessLinear2D1i_interpolations_res1e-2_r15e-2_c20.jld", "LessLinear2D_interpolations");
# LessLinear2D_interpolations = load(path_to_interps * "LessLinear2D1i_interpolations_res1e-2_r4e-1_el_1_5.jld", "LessLinear2D_interpolations");
V_DP_itp = LessLinear2D_interpolations["g0_m0_a0"]

##

N = 3

## System & Game

A, B₁, B₂ = -0.5*I + hcat(vcat(0, -ones(N-1,1)), zeros(N, N-1)), vcat(zeros(1, N-1), 0.4*I), vcat(zeros(1,N-1), 0.1*I) # system
max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(N-1), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
# Q, center, radius = diagm(ones(N)), zeros(N), 0.25
Q, center, radius = diagm(ones(N)), zeros(N), 0.15
radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
# Q, center, radius = diagm(inv.([1., 5.])), zero(A[:,1]), 0.4
# radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(5) * ones(N-1)))
target = make_target(center, radius_N; Q=Q_N, type="ellipse")
# target = make_target(center, radius; Q, type="ellipse")

## Times to Solve
Th, Tf = 0.1, 1.0
# Th, Tf = 0.025, 1.0
times = collect(Th : Th : Tf);

## Point(s) to Solve (any set works!)
bd, ppd, ϵ = 1.0001, 31, 0*.5e-7
res = 2*bd/(ppd-1)
Xg_rand = 2bd*rand(N, 100) .- bd .+ ϵ; # solve over random samples

Xg, xigs, (lb, ub) = make_grid(bd, res, N; return_all=true, shift=ϵ); # solve over grid
Xg_2d, xigs_2d, _ = make_grid(bd, res, 2; return_all=true, shift=ϵ); # solve over grid
Xg_2d_hr, xigs_2d_hr, _ = make_grid(bd, 0.1 * res, 2; return_all=true, shift=ϵ); # solve over grid

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 100, 1, 1, 1000
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

## Solve
solution, run_stats = Hopf_BRS(system, target, times; Xg, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
# solution_sparse, run_stats = Hopf_BRS(system, target, times; Xg, lg=length(xigs[1]), th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=false, check_all=false, printing=true, N=10);
solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true);

## Plot on an xn-xi plane 

solution_grids_2d = convert(Vector{Any}, [Xg_2d for i=1:length(times)+1]);

# xnxi_slice_ix = map(x->x.*vcat(ones(2), zeros(N-2))==x, eachcol(Xg)); # xN / x1
# solution_slice = convert(Vector{Any}, [solution[2][ti][xnxi_slice_ix] for ti=1:length(times)+1]);

xnxij_slice_ix = map(x->x[2]*ones(N-1)==x[2:end], eachcol(Xg)); # xN / (xi=j)
solution_slice = convert(Vector{Any}, [solution[2][ti][xnxij_slice_ix] for ti=1:length(times)+1]);

solution_2d = (solution_grids_2d, solution_slice);
plot_hopf_slice_xnxij = plot(solution_2d; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="Linear BRS - $(N)D Hopf, xn-xi plane", legend=false)

## Compared with Stored

alltimes = vcat(0.,times)
make_tXg(t, X) = vcat(t*ones(size(X,2))', X)

# V_i_itps_DP = [V_DP_itp for i=1:N-1]
# Pis_xi = [vcat(vcat(1, zeros(N-1))', vcat(zeros(i), 1, zeros(N-(i+1)))') for i=1:N-1]
# V_N_DP(t, XgN) = sum(fast_interp(V_i_itps_DP[i], make_tXg(t, Pis_xi[i] * XgN)) for i=1:N-1)

V_N_DP(t, X_N) = sum(fast_interp(V_DP_itp, make_tXg(t, view(X_N,[1, i+1],:))) for i=1:size(X_N,1)-1) # project to N-1 subspaces, getting val from DP interp

function jaccard_solutions_overtime(solution_1, solution_2, Xg_shared; tix=2:length(solution_1[1]))
    n_inter, n_union = 0, 0
    for ti in tix
        BRS_1, BRS_2 = Xg_shared[:, solution_1[2][ti] .< 0], Xg_shared[:, solution_2[2][ti] .< 0]
        BRS_set_1, BRS_set_2 = Set(), Set()
        map(x->push!(BRS_set_1, x), eachcol(BRS_1)); map(x->push!(BRS_set_2, x), eachcol(BRS_2))
        n_inter_ti, n_union_ti = length(intersect(BRS_set_1, BRS_set_2)), length(union(BRS_set_1, BRS_set_2))
        n_inter += n_inter_ti
        n_union += n_union_ti
    end
    return  n_inter / n_union
end

function MSE_solutions_overtime(solution_1, solution_2; tix=1:length(solution_1[1])-1)
    MSE = 0.
    for ti in tix
        @assert solution_1[1][ti] == solution_2[1][ti] "Solution grids don't match!"
        MSE += sum((solution_1[2][ti+1] - solution_2[2][ti+1]).^2) / (length(solution_1[2][ti+1]) * length(tix))
    end
    return MSE
end

function MSE_solution(solution, times; subset=1:length(solution[2][1]), tix=1:length(times))
    MSE, MSEs = 0, 0. * zero(tix)
    for ti in tix
        MSEs[ti] = sum((V_N_DP(times[ti], solution[1][ti+1][:, subset]) - solution[2][ti+1][subset]).^2) / length(solution[2][ti+1][subset]) # subset could be bit array or indexes
        MSE += MSEs[ti] / length(times)
    end
    return MSE, MSEs
end

## Compare on the three planes: xn-xi, xi-xj, xn-xij

slice_ixs = [
    map(x->x.*vcat(ones(2), zeros(N-2))==x, eachcol(Xg)), # xn-xi
    map(x->x.*vcat(zeros(N-2), ones(2))==x, eachcol(Xg)), # xi-xj
    map(x->x[2]*ones(N-1)==x[2:end], eachcol(Xg)), # xn-xij
]
slice_labels = ["xn-xi", "xi-xj", "xn-(xi=xj)"]

slice_plots = []
for i=1:length(slice_ixs)

    slice_ix = slice_ixs[i]
    solution_slice = (solution_grids_2d, convert(Vector{Any}, [solution[2][ti][slice_ix] for ti=1:length(times)+1]));
    plot_hopf_slice = plot(solution_slice; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="$(N)D Hopf, $(slice_labels[i]) plane", legend=false)

    Xg_slice = Xg[:, slice_ix]
    solution_slice_DP_values = convert(Vector{Any}, [V_N_DP(alltimes[ti], Xg_slice) for ti=1:length(times)+1]);
    solution_slice_DP = (solution_grids_2d, solution_slice_DP_values);
    plot_DP = plot(solution_slice_DP; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="True (DP), $(slice_labels[i]) plane", legend=false, alpha=0.5, linestyle=:dash)

    jacc, mse = jaccard_solutions_overtime(solution_slice, solution_slice_DP, Xg_2d), MSE_solutions_overtime(solution_slice, solution_slice_DP)
    println("SLICE $i, $(slice_labels[i]): (Jacc: $jacc), (MSE: $mse)")
    annotate!(plot_hopf_slice, 0., -0.8, ("Jaccard = $(round(jacc, digits=2))", 8, :black, :left))
    annotate!(plot_hopf_slice, 0., -0.9,("     MSE = $(round(mse, digits=2))",  8, :black, :left))
    push!(slice_plots, plot(plot_DP, plot_hopf_slice, layout=(2,1)))
end
plot(slice_plots..., layout=(1,3), size=(750, 500), titlefontsize=8)

## Target Check

i = 1
slice_ix = slice_ixs[i];
solution_target_slice = (solution_grids_2d[1:3], convert(Vector{Any}, [solution[2][ti][slice_ix] for ti=[1,1,1]]));
Xg_slice = Xg[:, slice_ix]
solution_slice_DP_values = convert(Vector{Any}, [V_N_DP(0., Xg_slice) for ti=1:3]);
solution_slice_DP = (solution_grids_2d[1:3], solution_slice_DP_values);

solution_target = (convert(Vector{Any},[solution[1][1], solution[1][1]]), convert(Vector{Any}, [solution[2][1], solution[2][1]]));
solution_target_DP_vals = convert(Vector{Any}, [V_N_DP([0., 0.][ti], Xg) for ti=1:2]);
solution_target_DP = (solution_target[1], solution_target_DP_vals);
MSE_target = MSE_solutions_overtime(solution_target, solution_target_DP);
MSE_target, MSEs_target = MSE_solution(solution_target, [0.])
jacc_target = jaccard_solutions_overtime(solution_target, solution_target_DP, Xg)

println("Targt, all-space: (Jacc: $jacc_target), (MSE: $MSE_target)")
plot_hopf_target_slice = plot(solution_target_slice; interpolate=true, xigs=xigs_2d, value=true, color_range=["black", "black"], title="$(N)D Hopf, $(slice_labels[i]) plane", legend=false);
plot_DP_target_slice = plot(solution_slice_DP; interpolate=true, color_range=["black", "black"], grid=true, xigs=xigs_2d, value=true, title="True (DP), $(slice_labels[i]) plane", legend=false, alpha=0.5);
plot(plot_hopf_target_slice, plot_DP_target_slice, layout=(2,1))

## Total Comparisons

MSE, MSEs = MSE_solution(solution, times)
MSE_sample, MSEs_sample = MSE_solution(solution_sampled, times)

solution_DP_values = convert(Vector{Any}, [V_N_DP(alltimes[ti], Xg) for ti=1:length(times)+1]);
solution_DP = (solution[1], solution_DP_values);
total_jacc = jaccard_solutions_overtime(solution, solution_DP, Xg)

plot(times, MSEs, title="MSE over Solution Time, N=$N", label="Grid, WS: Spatiotemporal"); plot!(times, MSEs_sample, label="Random, WS: Temporal")

# using PlotlyJS
# plot_hopf = plot_nice(times, solution; interpolate=true, alpha=0.1); #using PlotlyJS
# plot_DP = plot_nice(times, solution_DP; interpolate=true, alpha=0.1); #using PlotlyJS

## Profiling
# using ProfileView
# VSCodeServer.@profview solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true);
# VSCodeServer.@profview solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand[:,1:2], th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true)

# using Profile
# Profile.Allocs.@profile solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand[:,1:2], th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true)
# AllocResults = Profile.Allocs.@profile solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true)

function N_dim_test(N, opt_p; sample_total=100, Th=0.1, th=0.025, P_in=nothing, X=nothing)

    A, B₁, B₂ = -0.5*I + hcat(vcat(0, -ones(N-1,1)), zeros(N,N-1)), vcat(zeros(1,N-1), 0.4*I), vcat(zeros(1,N-1), 0.1*I) # system
    max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(N-1), "box"
    Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
    Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
    system_N, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

    ## Target
    Q, center, radius = diagm(ones(N)), zeros(N), 0.15
    # Q, center, radius = diagm(ones(N)), zeros(N), 0.15
    radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(1) * ones(N-1)))
    # Q, center, radius = diagm(inv.([1., 5.])), zero(A[:,1]), 0.4
    # radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), inv(5) * ones(N-1)))
    target_N = make_target(center, radius_N; Q=Q_N, type="ellipse")
    # print("J(X):", target_N[1](X))

    ## Times to Solve
    Tf = 1.
    times = collect(Th : Th : Tf);

    ## Point(s) to Solve (any set works!)
    bd = 1.0001
    X = isnothing(X) ? 2bd*rand(N, sample_total) .- bd : X; # solve over random samples

    ## Solve
    solution_sampled, run_stats, opt_data, P_final = Hopf_BRS(system_N, target_N, times; X=X, th=th, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p, warm=true, warm_pattern="temporal", check_all=true, printing=true, opt_tracking=true, P_in)
    
    ## Score
    MSE_overall, MSE_overtime = MSE_solution(solution_sampled, times)
    exec_time_ppt = run_stats[1] / (size(Xg_rand, 2) * length(times)) # 60000 pt/batch * ppt-rate s/pt * 1/60 min/s = X min/batch
    optimistic_batch_minutes = (1000 * exec_time_ppt) # optimistic bc memory allocation slows us down, only achieved in parallel

    return MSE_overall, MSE_overtime, optimistic_batch_minutes, opt_data, P_final, (solution_sampled, system_N, target_N)
end

## Hopf Coordinate-Descent Parameters
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 100000, 1e-3, 10000, 1, 1, 1000000
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 10^4, 1e-3, 10^4, 1, 1, 10^5
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)
opt_p_cd = (0.01, 5, 1e-3, 50, 4, 5, 500)

N = 7; Th = 0.001; sample_total = 100;
opt_p_cd = (0.01, 1, 1e-3, 300, 1, 1, 300)
MSE_overall, MSE_overtime, optimistic_batch_minutes, P_final, (solution_sampled, system_N, target_N) = N_dim_test(N, opt_p_cd; Th, th=0.001, sample_total); # Xg=Xg_rand
MSE_overall
optimistic_batch_minutes

N = 7; Th = 0.001;
# X = [[0.098,  0.430,  0.206,  0.090,  -0.153, 0.292,  -0.125];;]
X = [0.09762701; 0.43037874; 0.20552675; 0.08976637; -0.1526904; 0.29178822; -0.124825574;;]
opt_p_cd = (0.01, 1, 1e-3, 100, 1, 1, 100)
# opt_p_cd = (0.01, 1, 1e-3, 300, 1, 1, 300)
MSE_overall, MSE_overtime, optimistic_batch_minutes, opt_data, P_final, (solution_X, system_N, target_N) = N_dim_test(N, opt_p_cd; Th, th=0.001, X); # Xg=Xg_rand
MSE_overall
MSE_overtime
optimistic_batch_minutes

N = 15; Th = 0.01; sample_total = 100;
MSE_overall, MSE_overtime, optimistic_batch_minutes, opt_data, P_final, (solution_sampled, system_N, target_N) = N_dim_test(N, opt_p_cd; Th, th=0.001, sample_total); # Xg=Xg_rand
MSE_overall
optimistic_batch_minutes

plot(collect(Th:Th:1.), MSE_overtime, title="MSE over Solution Time, N=$N", label="Random, WS: Temporal")
MSE_overtime

## Iterative Sovles 

P_in_f(∇ϕX) = reshape(hcat(∇ϕX[2:end]...), size(∇ϕX[1])..., length(∇ϕX)-1)

function warmrefine_N_dim_sol(N, opt_p, its; kwargs...)
    P_in = nothing
    MSEs, MSEots, OBM, ODs, P_ins = zeros(its), [], zeros(its), [], []
    for i=1:its
        MSEs[i], MSEoti, OBM[i], ODi, P_final = N_dim_test(N, opt_p; P_in, kwargs...)
        P_in = P_in_f(P_final)
        push!(MSEots, MSEoti); push!(ODs, ODi); push!(P_ins, P_in)
    end
    return MSEs, MSEots, OBM, ODs, P_ins
end

vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

its = 3
N = 3
MSEs, MSEots, OBM, ODs, P_ins = warmrefine_N_dim_sol(N, opt_p_cd, its; Th = 0.025, sample_total = 100);

mse_ot = plot(title="MSEs overtime, N=$N")
for i=1:its
    plot!(MSEots[i], label="iter $i")
end
mse_ot

opt_sample_plot = plot(title="Multi-iter Optim Value")
sample_ix, iter_length = 1, 0
for i=1:its
    p_path, vals = ODs[i][sample_ix][sample_ix]
    println("ITER $i, FIRST:", round.(p_path[:,1],digits=6))
    println("ITER $i, LAST :", round.(p_path[:,end],digits=6))
    plot!(1+iter_length:iter_length + length(vals), vals, label="iter $i")
    iter_length += length(vals)
end
opt_sample_plot


## Iterative test manual

# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 100, 1, 1, 300
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.13 & 253 min/batch
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.88 & 27 min/batch (Th=0.01)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 300, 1, 1, 300 # for N=100, gives MSE=0.85/151 min/batch (Th=0.1), MSE=0.13/253 min/batch
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 1000, 1, 1, 10000
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 10000, 1, 1, 10000
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)
opt_p_cd = (0.01, 1, 1e-3, 300, 1, 1, 300)

# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 10^5, 1e-3, 10^5, 1, 1, 10^6
# opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

Th, Tf = 0.001, 1.0
times = collect(Th : Th : Tf);

solution_sampled_0, run_stats_0, opt_data_0, ∇ϕXsT0 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.001, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, opt_tracking=true);
MSE_overall_0, MSE_overtime_0 = MSE_solution(solution_sampled_0, times)
optimistic_batch_time_0 = 1000 * run_stats_0[1] / (size(Xg_rand, 2) * length(times)) # time (min) to solve 60k pts, if parallelized

# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 20, 1e-3, 1000, 1, 1, 10000
# opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution_sampled_1, run_stats_1, opt_data_1, ∇ϕXsT1 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, P_in=P_in_f(∇ϕXsT0), opt_tracking=true);
# solution_sampled_1, run_stats, _, ∇ϕXsT1 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, P_in=rand(15,1000,10));
MSE_overall_1, MSE_overtime_1 = MSE_solution(solution_sampled_1, times)
optimistic_batch_time_1 = 1000 * run_stats_1[1] / (size(Xg_rand, 2) * length(times)) # time (min) to solve 60k pts, if parallelized

# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 200, 1e-3, 2000, 1, 1, 10000
# opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution_sampled_2, run_stats_2, opt_data_2, ∇ϕXsT2 = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.025, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, warm_pattern="temporal", check_all=true, printing=true, P_in=P_in_f(∇ϕXsT1), opt_tracking=true);
MSE_overall_2, MSE_overtime_2 = MSE_solution(solution_sampled_2, times)

plot(times, MSE_overtime_0, title="MSE over Solution Time, N=$N", label="Iter 0"); plot!(times, MSE_overtime_1, label="Iter 1"); plot!(times, MSE_overtime_2, label="Iter 2")

opt_sample_plot = plot(title="Multi-iter Optim Value")
sample_ix, iter_length = 1, 0
ODs = [opt_data_0, opt_data_1, opt_data_2]
for i=1:its
    p_path, vals = ODs[i][sample_ix][sample_ix]
    println("ITER $i, FIRST:", round.(p_path[:,1],digits=6))
    println("ITER $i, LAST :", round.(p_path[:,end],digits=6))
    plot!(1+iter_length:iter_length + length(vals), vals, label="iter $i")
    iter_length += length(vals)
end
opt_sample_plot


## Retest

# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 1000, 1, 1, 10000
# opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

# MSE_overall, MSE_overtime, optimistic_batch_minutes = N_dim_test(N, opt_p_cd; sample_total=100, Th=0.1, th=0.025);
# MSE_overall, MSE_overtime, optimistic_batch_minutes, P_final, (solution_sampled, system_N, target_N) = N_dim_test(N, opt_p_cd; X=Xg_rand, P_in=P_in_f(∇ϕXsT0));

vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 1, 1e-3, 1000, 1, 1, 1000 # for N=100, gives MSE=0.85/151 min/batch (Th=0.1), MSE=0.13/253 min/batch
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)
MSE_overall, MSE_overtime, optimistic_batch_minutes, P_final, (solution_sampled, system_N, target_N) = N_dim_test(1000, opt_p_cd; sample_total=10, Th=0.01, th=0.01);

MSE_overall, optimistic_batch_minutes
