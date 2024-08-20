
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_target, make_set_params
using LinearAlgebra, Plots

using PyCall
include(pwd() * "/src/DP_comparison_utils.jl");

using JLD2, Interpolations, ScatteredInterpolation

## System & Game
A, B₁, B₂ = [-0.5 0.; -1 -0.5], [0.; 0.4], [0.; 0.1] # system
max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(1), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, reshape(B₁, 2, 1), reshape(B₂, 2, 1), Q₁, c₁, Q₂, c₂), "reach"

## Target
# Q, center, radius = diagm(ones(size(A)[1])), zero(A[:,1]), 0.25
# Q, center, radius = diagm(ones(size(A)[1])), zero(A[:,1]), 0.15
Q, center, radius = diagm(inv.([1., 5.])), zero(A[:,1]), 0.4
target = make_target(center, radius; Q, type="ellipse")

## Times to Solve
Th, Tf = 0.25, 1.0
times = collect(Th : Th : Tf);

## Point(s) to Solve (any set works!)
bd, res, ϵ = 1.0001, 0.015, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ); # solve over grid
# Xg_rand = 2bd*rand(2, 500) .- bd .+ ϵ; # solve over random samples

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
vh, stepsz, tol, stepszstep_its, conv_runs_rqd, max_runs, max_its = 0.01, 2, 1e-3, 50, 3, 3, 200
opt_p_cd = (vh, stepsz, tol, stepszstep_its, conv_runs_rqd, max_runs, max_its)

solution, run_stats = Hopf_BRS(system, target, times; Xg, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
# solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot_hopf = plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false, title="Linear BRS - Hopf", legend=false)
plot_hopf_leg = plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false, title="Linear BRS - Hopf", legend=:bottomleft)
# plot_hopf = plot(solution; interpolate=false, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=true, camera=(30, 15), ϵs=0.001)
# plot_hopf = plot(solution_sampled; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=false, value=true, camera=(30, 15))

## BRT & Smoothing

using ImageFiltering
smooth(X; gv=1.0) = imfilter(X, Kernel.gaussian(gv)) 
resN = length(xigs[1])

solution_BRTs = (copy(solution[1]), copy(solution[2]))
for i=2:length(times)+1; solution_BRTs[2][i] = map(minimum, eachrow(hcat([solution[2][j] for j=1:i]...))); end
for i=2:length(times)+1; solution_BRTs[2][i] = reshape(smooth(reshape(solution_BRTs[2][i], resN, resN); gv=2.0), resN^2); end

plot_hopf_BRTs = plot(solution_BRTs; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false, title="Linear BRT - Hopf", legend=false)

## Interpolate Hopf Solution

include(pwd() * "/src/interp_utils.jl");

## Test
alltimes = vcat(0., times...)
tXg = zeros(1+size(Xg,1), 0)
for ti=1:length(times)+1
    tXg = hcat(tXg, vcat(zero(solution[1][ti])[[1],:] .+ alltimes[ti], solution[1][ti]))
end
# const tXg2 = copy(tXg)

## grid
V_itp_hopf = make_interpolation(solution, alltimes; xigs)
V_itp_hopf = make_interpolation(solution_BRTs, alltimes; xigs)
Vg = @time fast_interp(V_itp_hopf, tXg)

# ## scatter (works but slow)
# V_itp_hopf, fast_interp = make_interpolation(solution, alltimes; xigs, method="scatter")
# Vg = @time fast_interp(V_itp, tXg2)

# save("lin2d_hopf_interp_linear.jld", "V_itp", V_itp_hopf, "solution", solution, "alltimes", alltimes, "xigs", xigs)
# V_itp_hopf_loaded = load("lin2d_hopf_interp_linear.jld")["V_itp"]
# Vg = @time fast_interp(V_itp_hopf_loaded, tXg2)

## Plot
Xg_near = tXg[:, abs.(Vg) .≤ 0.005]
plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false)
scatter!(eachrow(Xg_near[2:end,:])..., alpha=0.5)

x = y = range(-0.95, stop = 0.95, length = 133)
surface(x, y, (x, y) -> V_itp_hopf(y, x, 1.))

### Compare With LessLinear Model with DP (hj_reachability)

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/DP_comparison_files/");
np = pyimport("numpy")
jnp = pyimport("jax.numpy")
hj = pyimport("hj_reachability")
hjr_lesslin = pyimport("l2d_ll2d_hj_reachability")

c = 5
p_sets = [
    [0, 0, 0], ## Linear
    [c, 0, 0], [-c, 0, 0], ## Less Linear Tier 1
    [c, -c, 1], [-c, c, 1], [c, c, 1], [-c, -c, 1], ## Less Linear Tier 2
    [c, -c, c], [-c, c, -c], [c, c, c], [-c, -c, -c], ## Less Linear Tier 3
]

LessLinear2D_interpolations = Dict("hopf" => V_itp_hopf)
LessLinear2D_plots = Dict("hopf" => plot_hopf_leg)

Xg_DP, Xg_DPpy, ϕ0Xg_DP, xigs_DP = hjr_init(center, inv.(Q), radius; shape="ball", lb, ub, res=200, ϵ = 0.5e-7, bc_grad_factor=1.)
solution_times_DP = convert(Vector{Any}, [Xg_DP for i=1:length(times)+1])

for p_set in p_sets

    gamma, mu, alpha = p_set

    lesslin_dynamics_DP = hjr_lesslin.LessLinear2D_1input(gamma=gamma, mu=mu, alpha=alpha)
    DP_solution_BRT = hjr_solve(Xg_DPpy, ϕ0Xg_DP, [lesslin_dynamics_DP], times; BRS=false)

    plot_DP_BRT_ll2d = plot((solution_times_DP, DP_solution_BRT[1]); xigs=xigs_DP, value=false, title="Less (γ=$gamma, μ=$mu, α=$alpha) BRT - DP", labels=vcat("Target", ["t=-$ti" for ti in times]...), legend=false);

    pair_comp_plot = plot(plot_hopf, plot_DP_BRT_ll2d, size=(600, 300), titlefontsize=8, legendfont=6, legend=false)
    # display(pair_comp_plot)

    ## Make Interpolation from DP

    solution_ll2d_BRT_DP = (solution_times_DP, DP_solution_BRT[1]); # BRT has more stable DP solns

    V_itp_DP = make_interpolation(solution_ll2d_BRT_DP, alltimes; xigs=xigs_DP)
    # Vg_DP = @time fast_interp(V_itp_DP, tXg2)

    # save("llin2d_1i_g$(gamma)_m$(mu)_a$(alpha)_DP_interp_linear_res.jld", "V_itp", V_itp_DP, "solution", solution_ll2d_BRT_DP, "alltimes", alltimes, "xigs", xigs_DP)
    LessLinear2D_interpolations["g$(gamma)_m$(mu)_a$(alpha)"] = V_itp_DP
    LessLinear2D_plots["g$(gamma)_m$(mu)_a$(alpha)"] = plot_DP_BRT_ll2d
end
# full_plot = plot(plot_hopf_leg, [LessLinear2D_plots["g$(g)_m$(m)_a$(a)"] for (g,m,a) in p_sets]..., layout=(3,4), size=(800, 600), titlefont=6, legendfontsize=4, xtickfontsize=5, ytickfontsize=5, dpi=300)
full_plot = plot(plot_hopf_BRTs, [LessLinear2D_plots["g$(g)_m$(m)_a$(a)"] for (g,m,a) in p_sets]..., layout=(3,4), size=(800, 600), titlefont=6, legendfontsize=4, xtickfontsize=5, ytickfontsize=5, dpi=300)

# save("LessLinear2D1i_interpolations_res1e-2_r4e-1_el_1_5.jld", "LessLinear2D_interpolations", LessLinear2D_interpolations)
# savefig(full_plot, "LessLinear2d1i_plot_res1e-2_r4e-1_el_1_5.png")

## Exact High-Dimensional Solution (Combined 2D DP Solutions)

N = 10
alltimes = vcat(0., times)
make_tXg(_t, _Xg) = vcat(_t*ones(size(_Xg,2))', _Xg)
Pis_xi = [vcat(vcat(1, zeros(N-1))', vcat(zeros(i), 1, zeros(N-(i+1)))') for i=1:N-1]
V_N_itp(V_i_itps, Pis, t, XgN) = sum(fast_interp(V_i_itps[i], make_tXg(t, Pis[i] * XgN)) for i=1:N-1)

## Full Plot of All - Slice

plots_DP_ND = []
for p_set in p_sets

    gamma, mu, alpha = p_set
    V_i_NL_itps_DP = [LessLinear2D_interpolations["g$(gamma)_m$(mu)_a$(alpha)"] for i=1:N-1]

    XgN = zeros(N,size(Xg,2))

    # xn vs. xi slice, stretches with increasing dimension
    XgN[1:2,:] = Xg 

    # xn vs. (xi=xj) slice, remains constant
    # XgN[1,:], XgN[2:end, :] = Xg[1,:], (Xg[2,:] .* ones(size(Xg,2), N-1))'

    solution_DP_values = convert(Vector{Any}, [V_N_itp(V_i_NL_itps_DP, Pis_xi, alltimes[ti], XgN) for ti=1:length(times)+1]);
    solution_DP = (solution_times_DP, solution_DP_values);

    title = "(γ=$gamma, μ=$mu, α=$alpha) BRT/DP/$(N)D"
    plot_DP = plot(solution_DP; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false, title=title, legend=false)
    push!(plots_DP_ND, plot_DP)
end
full_plot_ND = plot(plot_hopf_BRTs, plots_DP_ND..., layout=(3,4), size=(800, 600), titlefont=6, legendfontsize=4, xtickfontsize=5, ytickfontsize=5, dpi=300)

## Triple Slice View

V_i_NL_itps_DP = [LessLinear2D_interpolations["g-$(c)_m0_a0"] for i=1:N-1]

plots_DP, titles = [], ["$(N)D DP, xn-xi plane", "$(N)D DP, xi-xj plane","$(N)D DP, xn-(xi=xj) plane"]
for i=1:3
    XgN = zeros(N,size(Xg,2))
    if i == 1
        XgN[1:2,:] = Xg
    elseif i == 2
        XgN[2,:] = Xg[1,:]
        XgN[3,:] = Xg[2,:]
    else
        XgN[1,:] = Xg[1,:]
        XgN[2:end, :] = (Xg[2,:] .* ones(size(Xg,2), N-1))'
    end
    solution_DP_values = convert(Vector{Any}, [V_N_itp(V_i_NL_itps_DP, Pis_xi, alltimes[ti], XgN) for ti=1:length(times)+1]);
    solution_DP = (solution_times_DP, solution_DP_values);
    plot_DP = plot(solution_DP; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false, title=titles[i], legend=false, alpha=0.5)
    push!(plots_DP, plot_DP)
end
plot(plots_DP..., layout=(1,3), size=(900,300))

## Test 3D solves of separate Systems

p_set_1, p_set_2 = p_sets[3], p_sets[4]

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


