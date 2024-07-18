
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_levelset_fs, make_set_params, plot_nice
using LinearAlgebra, Plots
# using PlotlyJS

using PyCall
include(pwd() * "/src/DP_comparison_utils.jl");
include(pwd() * "/src/interp_utils.jl");

using JLD2, Interpolations, ScatteredInterpolation, ImageFiltering

N = 3

## System & Game
A, B₁, B₂ = -0.5*I + hcat(vcat(0, -ones(N-1,1)), zeros(N,N-1)), vcat(zeros(1,N-1), 0.4*I), vcat(zeros(1,N-1), 0.1*I) # system
max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(N-1), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
Q, center, radius = diagm(ones(N)), zeros(N), 0.25
radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), ones(N-1)))
J, Jˢ = make_levelset_fs(center, radius_N; Q=Q_N, type="ellipse")
# J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.25, 1.0
times = collect(Th : Th : Tf);

## Point(s) to Solve (any set works!)
bd, ppd, ϵ = 1.0001, 31, 0*.5e-7
res = 2*bd/(ppd-1)
Xg, xigs, (lb, ub) = make_grid(bd, res, N; return_all=true, shift=ϵ); # solve over grid
Xg_2d, xigs_2d, _ = make_grid(bd, res, 2; return_all=true, shift=ϵ); # solve over grid
# Xg_rand = 2bd*rand(2, 500) .- bd .+ ϵ; # solve over random samples

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

## Solve
solution, run_stats = Hopf_BRS(system, target, times; Xg, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
solution_sparse, run_stats = Hopf_BRS(system, target, times; Xg, lg=length(xigs[1]), th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=false, check_all=false, printing=true, N=10);
# solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

## Plot on an xn-xi plane 

solution_grids_2d = convert(Vector{Any}, [Xg_2d for i=1:length(times)+1]);

xnxi_slice_ix = map(x->x.*vcat(ones(2), zeros(N-2))==x, eachcol(Xg)); # x1 / xN
solution_slice = convert(Vector{Any}, [solution[2][ti][xnxi_slice_ix] for ti=1:length(times)+1]);

solution_2d = (solution_grids_2d, solution_slice);
plot_hopf_slice = plot(solution_2d; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="Linear BRS - $(N)D Hopf, xn-xi plane", legend=false)

## Compared with Stored

function compute_jaccard_overtime(solution_1, solution_2, Xg_shared; tix=1:length(solution_1))
    n_inter, n_union = 0, 0
    for ti in tix
        BRS_1, BRS_2 = Xg_shared[:, solution_1[2][ti] .< 0], Xg_shared[:, solution_2[2][ti] .< 0]
        BRS_set_1, BRS_set_2 = Set(), Set()
        map(x->push!(BRS_set_1, x), eachcol(BRS_1))
        map(x->push!(BRS_set_2, x), eachcol(BRS_2))
        n_inter_ti = length(intersect(BRS_set_1, BRS_set_2))
        n_union_ti = length(BRS_set_1) + length(BRS_set_2) - n_inter_ti
        n_inter += n_inter_ti
        n_union += n_union_ti
    end
    return  n_inter / n_union
end

using Interpolations, JLD
LessLinear2D_interpolations = load("LessLinear2D1i_interpolations_res5e-3.jld", "LessLinear2D_interpolations");
V_DP_itp = LessLinear2D_interpolations["g0_m0_a0"]

alltimes = vcat(0.,times)
make_tXg(_t, _Xg) = vcat(_t*ones(size(_Xg,2))', _Xg)

V_i_itps_DP = [V_DP_itp for i=1:N-1]
Pis_xi = [vcat(vcat(1, zeros(N-1))', vcat(zeros(i), 1, zeros(N-(i+1)))') for i=1:N-1]
V_N_itp(V_i_itps, Pis, t, XgN) = sum(fast_interp(V_i_itps[i], make_tXg(t, Pis[i] * XgN)) for i=1:N-1)

## Compare on the xn-xi plane

Xg_xnxi = Xg[:, xnxi_slice_ix]
solution_DP_values = convert(Vector{Any}, [V_N_itp(V_i_itps_DP, Pis_xi, alltimes[ti], Xg_xnxi) for ti=1:length(times)+1]);
solution_DP = (solution_grids_2d, solution_DP_values);
plot_DP = plot(solution_DP; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="Linear BRS - $(N)D/2D DP, xn-xi plane", legend=false, alpha=0.5)

jacc = compute_jaccard_overtime(solution_2d, solution_DP, Xg_2d)
annotate!(plot_hopf_slice, 0.25, -0.75, ("Jaccard = $(round(jacc, digits=2))", 8, :black, :left))
plot(plot_hopf_slice, plot_DP, size=(800,400))

## Compare on an xi-xj plane (xn=0)

xixj_slice_ix = map(x->x.*vcat(zeros(N-2), ones(2))==x, eachcol(Xg)); # x1 / xN
Xg_xixj = Xg[:, xixj_slice_ix]
solution_slice = convert(Vector{Any}, [solution[2][ti][xixj_slice_ix] for ti=1:length(times)+1]);
solution_2d = (solution_grids_2d, solution_slice);
plot_hopf_slice = plot(solution_2d; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="Linear BRS - $(N)D Hopf, xi-xj plane", legend=false)

# solution_DP_values = convert(Vector{Any}, [fast_interp(V_DP_itp, make_tXg(alltimes[ti], Pi * Xg_xixj)) .+ fast_interp(V_DP_itp, make_tXg(alltimes[ti], Pj * Xg_xixj)) for ti=1:length(times)+1]);
solution_DP_values = convert(Vector{Any}, [V_N_itp(V_i_itps_DP, Pis_xi, alltimes[ti], Xg_xixj) for ti=1:length(times)+1]);
solution_DP = (solution_grids_2d, solution_DP_values);
plot_DP = plot(solution_DP; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs_2d, value=false, title="Linear BRS - $(N)D/2D DP, xi-xj plane", legend=false, alpha=0.5)

jacc = compute_jaccard_overtime(solution_2d, solution_DP, Xg_2d)
annotate!(plot_hopf_slice, 0.25, -0.75, ("Jaccard = $(round(jacc, digits=2))", 8, :black, :left))
plot(plot_hopf_slice, plot_DP, size=(800,400))


## Total Jaccard

solution_DP_values = convert(Vector{Any}, [V_N_itp(V_i_itps_DP, Pis_xi, alltimes[ti], Xg) for ti=1:length(times)+1]);
solution_DP = (solution[1], solution_DP_values);
total_jacc = compute_jaccard_overtime(solution, solution_DP, Xg)

using PlotlyJS
plot_hopf = plot_nice(times, solution; interpolate=true, alpha=0.1); #using PlotlyJS
plot_DP = plot_nice(times, solution_DP; interpolate=true, alpha=0.1); #using PlotlyJS

## Target Test

# Q, center, radius = diagm(ones(N)), zeros(N), 0.25

# radius_N, Q_N = sqrt(N-1) * radius, diagm(vcat(1/(N-1), ones(N-1)))
# J_N, Jˢ = make_levelset_fs(center, radius_N; Q=Q_N, type="ellipse")

# # Xg_slice = Xg_xixj # xi - xj
# Xg_slice = vcat(Xg_2d, zeros(N-2, size(Xg_2d, 2))) # xn - xi

# target_N_os = (convert(Vector{Any}, [Xg_2d]), convert(Vector{Any}, [J_N(Xg_slice)]))
# plot_target_N_os = plot(target_N_os; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in 0.]...), xigs=xigs_2d, value=false, title="Target - 4D Hopf, xi-xn plane", legend=false)

# target_N_comb = (convert(Vector{Any}, [Xg_2d]), convert(Vector{Any}, [V_N_itp(V_i_itps_DP, Pis_xi, 0., Xg_slice)]))
# plot_target_N_comb = plot(target_N_comb; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in 0.]...), xigs=xigs_2d, value=false, title="Target - 2Di DP, xi-xn plane", legend=false)

# plot(plot_target_N_os, plot_target_N_comb, size=(800,400))



stop

## Interpolate High-D Solutions

## Test
alltimes = vcat(0., times...)
tXg = zeros(1+size(Xg,1), 0)
for ti=1:length(times)+1
    tXg = hcat(tXg, vcat(zero(solution[1][ti])[[1],:] .+ alltimes[ti], solution[1][ti]))
end

## grid
V_itp_hopf = make_interpolation(solution, alltimes; xigs)
Vg = @time fast_interp(V_itp_hopf, tXg)

# save("lin2d_hopf_interp_linear.jld", "V_itp", V_itp_hopf, "solution", solution, "alltimes", alltimes, "xigs", xigs)
# V_itp_hopf_loaded = load("lin2d_hopf_interp_linear.jld")["V_itp"]
# Vg = @time fast_interp(V_itp_hopf_loaded, tXg)

## Plot
Xg_near = tXg[:, abs.(Vg) .≤ 0.005]
plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false)
scatter!(eachrow(Xg_near[2:end,:])..., alpha=0.5)

x = y = range(-0.95, stop = 0.95, length = 133)
surface(x, y, (x, y) -> V_itp_hopf(y, x, 1.))

### Compare With LessLinear Model with DP (hj_reachability)

ll2d1i_itps = load("LessLinear2D_1i_interpolations_res2e.jld")["LessLinear2D_interpolations"]
V_itp_1 = ll2d1i_itps["g$(gamma1)_m$(mu1)_a$(alpha1)"]

p_sets = [
    [0, 0, 0], ## Linear
    [20, 0, 0], [-20, 0, 0], ## Less Linear Tier 1
    [20, -20, 1], [-20, 20, 1], [20, 20, 1], [-20, -20, 1], ## Less Linear Tier 2
    [20, -20, 20], [-20, 20, -20], [20, 20, 20], [-20, -20, -20], ## Less Linear Tier 3
]

## Test 3D solves of separate Systems

p_set_1, p_set_2 = p_sets[4], p_sets[4]

gamma1, mu1, alpha1 = p_set_1; gamma2, mu2, alpha2 = p_set_2
V_itp_1, V_itp_2 = LessLinear2D_interpolations["g$(gamma1)_m$(mu1)_a$(alpha1)"], LessLinear2D_interpolations["g$(gamma2)_m$(mu2)_a$(alpha2)"]

bd, res, ϵ = 1, 0.05, .5e-7
Xg3, xigs3, (lb3, ub3) = make_grid(bd, res, 3; return_all=true, shift=ϵ); # solve over grid

alltimes = vcat(0., times...)
tXg3 = zeros(1+size(Xg3,1), 0)
for ti=1:length(times)+1
    tXg3 = hcat(tXg3, vcat(zero(Xg3)[[1],:] .+ alltimes[ti], Xg3))
end

P1 = [1 0 0 0; 0 1 0 0; 0 0 0 1]
P2 = [1 0 0 0; 0 0 1 0; 0 0 0 1]

@time Vg3 = fast_interp(V_itp_1, P1 * tXg3) .+ fast_interp(V_itp_2, P2 * tXg3)

Xg3_near = tXg3[:, abs.(Vg3) .≤ 0.1]
# plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false)

# plotlyjs()
gr()
scatter(eachrow(Xg3_near[2:end,:])..., alpha=0.5, xlims=[-1,1], ylims=[-1,1], zlims=[-1,1])

sol_times_3 = convert(Vector{Any}, [Xg3 for i=1:length(times)+1])
sol_vals = [Vg3[1+(i-1)*size(Xg3,2):i*size(Xg3,2)] for i=1:length(times)+1]
plot((sol_times_3, sol_vals); xigs=xigs3, value=false, interpolate=false)


