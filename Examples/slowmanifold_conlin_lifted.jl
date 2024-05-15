

include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, plot_nice

include(pwd() * "/src/cons_lin_utils.jl");
include(pwd() * "/src/DP_comparison_utils.jl"); 
# include(pwd() * "/Zonotoping/cons_lin_utils_NLP_new.jl");

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/DP_comparison_files");
hj_r_setup = pyimport("sm_hj_reachability");
ss_set = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=x->x);

## SlowManifold Example
μ, λ = -0.05, -1.

ψ(x) = x^2
Ψ(x) = vcat(x, ψ(x[1]))

max_u = 0.5; max_d = 0.25;
Q₁ = inv(max_u) * diagm([1., 1.])
Q₂ = inv(max_d) * diagm([1., 1.])
c = [0., 0.]

r = 1.; η = 1/15;
c𝒯 = [0.; 1.25]; 
# c𝒯 = [1.; 1.]; 
c𝒯_aug = Ψ(c𝒯)
d𝒯 = [1., 1.]
Q𝒯 = inv(r) * diagm(d𝒯)
Q𝒯_aug = inv(r) * diagm(vcat(d𝒯, η))
nx = length(c𝒯);
nk = length(c𝒯_aug);

inputs = ((Q₁, c), (Q₂, c))
𝒯target = (nothing, nothing, (Q𝒯, c𝒯))
𝒯target_aug = (nothing, nothing, (Q𝒯_aug, c𝒯_aug))
κ_mats = ([μ 0 0; 0 λ -λ; 0 0 2μ], [1 0; 0 1; 2c𝒯[1] 0], [1 0; 0 1; 2c𝒯[1] 0]) # fixed linear model for testing

(Q𝒯, c𝒯), (Q₁, c₁), (Q₂, c₂) = 𝒯target[3], inputs[1], inputs[2]

# X0 = Hyperrectangle(; low = c𝒯 - diag(inv(Q𝒯)), high = c𝒯 + diag(inv(Q𝒯)))
# X0_aug = Hyperrectangle(; low = c𝒯_aug - diag(inv(Q𝒯_aug)), high = c𝒯_aug + diag(inv(Q𝒯_aug)))
# U = Hyperrectangle(; low = c - diag(inv(Q₁)), high = c + diag(inv(Q₁)))
# D = Hyperrectangle(; low = c - diag(inv(Q₂)), high = c + diag(inv(Q₂)))

function slowmanifold!(dx, x, p, t)
    # ReachabilityAnalysis.jl model style
    dx[1] = μ * x[1] + x[3] + x[5]
    dx[2] = λ * (x[2] - x[1]^2) + x[4] + x[6]
    dx[3] = zero(x[3]) #control
    dx[4] = zero(x[3]) #control
    dx[5] = zero(x[4]) #disturbance
    dx[6] = zero(x[4]) #disturbance
    return dx
end

function slowmanifold_aug!(dx, x, p, t)
    # ReachabilityAnalysis.jl model style
    dx[1] = μ * x[1] + x[4] + x[6]
    dx[2] = λ * (x[2] - x[3]) + x[5] + x[7]
    dx[3] = 2 * (μ * x[3] + x[1] * (x[4] + x[6])) #augmented
    dx[4] = zero(x[4]) #control
    dx[5] = zero(x[5]) #control
    dx[6] = zero(x[6]) #disturbance
    dx[7] = zero(x[7]) #disturbance
    return dx
end




## Solve Error




t = 2.1

# TS Linearization in Original State Space

plotlyjs()
δ̃ˢ, X̃, BRZ, dt, (lin_mat_fs, Gs) = apri_δˢ(slowmanifold!, 𝒯target, inputs, t);
δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(slowmanifold!, 𝒯target, inputs, t; zono_over="U");
δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(slowmanifold!, 𝒯target, inputs, t; zono_over="D");

BRZ_plot = plot(BRZ, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (U & D)", legend=:bottomleft)
plot!(BRZ_plot, BRZu, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (U)")
plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (D)")
scatter!(BRZ_plot, eachrow(hcat(X̃.(dt)...)[1:2,:])..., label="x̃ backsolved w/ trivial ctrl/dist", alpha=0.6, xlims=(-2, 2), ylims=(-1, 5))

error_plot = plot(dt, δ̃ˢ[2].(dt), label="Taylor δˢ for BRZ (U & D), x̃", xlabel="t")
plot!(error_plot, dtU, δ̃ˢU[2].(dtU), label="Taylor δˢ for BRZ (U), x̃")
plot!(error_plot, dtD, δ̃ˢD[2].(dtD), label="Taylor δˢ for BRZ (D), x̃")

plot(BRZ_plot, error_plot)

# TS Linearization in State Augmented Space

δ̃ˢ_aug, X̃_aug, BRZ_aug, dt_aug, (lin_mat_fs_aug, Gs_aug) = apri_δˢ(slowmanifold_aug!, 𝒯target_aug, inputs, t; polyfit=true);
δ̃ˢU_aug, X̃_aug, BRZu_aug, dtU_aug, (_, Gs_aug) = apri_δˢ(slowmanifold_aug!, 𝒯target_aug, inputs, t; zono_over="U", polyfit=true);
δ̃ˢD_aug, X̃_aug, BRZd_aug, dtD_aug, (_, Gs_aug) = apri_δˢ(slowmanifold_aug!, 𝒯target_aug, inputs, t; zono_over="D", polyfit=true);

BRZ_plot_aug = plot(BRZ_aug, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (U & D)", legend=:bottomleft)
plot!(BRZ_plot_aug, BRZu_aug, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (U)")
plot!(BRZ_plot_aug, BRZd_aug, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (D)")
scatter!(BRZ_plot_aug, eachrow(hcat(X̃_aug.(dt_aug)...)[[1,2],:])..., label="x̃ backsolved w/ trivial ctrl/dist", alpha=0.6, xlims=(-2, 2), ylims=(-1, 5))

error_plot_aug = plot(dt_aug, δ̃ˢ_aug[3].(dt_aug), label="Taylor δˢ for BRZ (U & D), x̃", xlabel="t")
plot!(error_plot_aug, dtU_aug, δ̃ˢU_aug[3].(dtU_aug), label="Taylor δˢ for BRZ (U), x̃")
plot!(error_plot_aug, dtD_aug, δ̃ˢD_aug[3].(dtD_aug), label="Taylor δˢ for BRZ (D), x̃")

plot(BRZ_plot_aug, error_plot_aug)

plot(BRZ_plot, error_plot, BRZ_plot_aug, error_plot_aug, layout=(2,2), legend=false, plottitle="")

# Fixed Linear Model w/ Error on Lifted Feasible Only (discrete estimation)

lifted_kwargs = Dict(:error_method=>Lifted_Error_DiscreteAppx, :lin_mat_fs=>lin_mat_fs_aug, :linear_graph=>zeros(length(c𝒯_aug)), :Ψ=>Ψ, :solve_dims=>[3])
# lifted_kwargs = Dict(:error_method=>Lifted_Error_DiscreteAppx, :lin_mat_fs=>κ_mats, :linear_graph=>zeros(length(c𝒯_aug)), :Ψ=>Ψ, :solve_dims=>[3])
δ̃ˢ_man, _, _, _, _ = apri_δˢ(slowmanifold!, 𝒯target, inputs, t; lifted_kwargs...);
δ̃ˢU_man, _, _, _, _ = apri_δˢ(slowmanifold!, 𝒯target, inputs, t; zono_over="U", lifted_kwargs...);
@time δ̃ˢD_man, _, _, _, _ = apri_δˢ(slowmanifold!, 𝒯target, inputs, t; zono_over="D", lifted_kwargs...);

error_plot_man = plot(dt, δ̃ˢ_man[3].(dt), label="Taylor δˢ for BRZ (U & D), x̃", xlabel="t")
plot!(error_plot_man, dtU, δ̃ˢU_man[3].(dtU), label="Taylor δˢ for BRZ (U), x̃")
plot!(error_plot_man, dtD, δ̃ˢD_man[3].(dtD), label="Taylor δˢ for BRZ (D), x̃")

plot(BRZ_plot, error_plot_man)

plot(BRZ_plot, error_plot, BRZ_plot, error_plot_man, BRZ_plot_aug, error_plot_aug, layout=(3,2), legend=false, plottitle="")




## Solve True Value + TS with DP




T = [0.2, 0.5, 1., 1.5, 2.]
# T = collect(0.1:0.1:0.5)

SM_reach = hj_r_setup.SlowManifold(mu=μ, lambduh=λ, max_u=max_u, max_d=max_d)
SM_avoid = hj_r_setup.SlowManifold(mu=μ, lambduh=λ, max_u=max_u, max_d=max_d, control_mode="max", disturbance_mode="min")

A, B₁, B₂, c = lin_mat_fs
E_s(s) = δ̃ˢ[2](-s) * Matrix([0. 1.]') # in forward time
E_sU(s) = δ̃ˢU[2](-s) * Matrix([0. 1.]') # in forward time

# must do iterative solve if tv lin (pycall + jax problem)
SM_LTV_reach = s -> LinearError(A(X̃(-s)), B₁(X̃(-s)), B₂(X̃(-s)), c(X̃(-s)), E_sU(s); max_u=max_u, max_d=max_d, Ushape="box", game="reach")
SM_LTV_avoid = s -> LinearError(A(X̃(-s)), B₁(X̃(-s)), B₂(X̃(-s)), c(X̃(-s)), E_sU(s); max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

dynamics = [SM_reach] #, SM_avoid];
dynamics_linear = [SM_LTV_reach] #, SM_LTV_avoid];
res=100

Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, Q𝒯, r; shape="ball", lb=(-2, -1), ub=(2, 4), res=res);
ϕXgT_DP_dynamics = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics, T; BRS=true, one_shot=true);

th = 0.05
ϕXgT_DP_dynamics_linear = []
for dyni in dynamics_linear
    ϕXgT_DP_dyni = []; push!(ϕXgT_DP_dyni, Matrix(reshape(ϕ0Xg_DP.tolist(), length(ϕ0Xg_DP.tolist()), 1))[:,1]) # target
    hj_r_output = jnp.copy(ϕ0Xg_DP)
    for (tsi, ts) in enumerate(collect(th:th:T[end]))
        dynis = dyni(ts)
        hj_r_output = hj.step(ss_set, dynis, Xg_DP, 0., hj_r_output, -th)
        if ts ∈ T; push!(ϕXgT_DP_dyni, Matrix(reshape(hj_r_output.tolist(), length(hj_r_output.tolist()), 1))[:,1]); end
    end
    push!(ϕXgT_DP_dynamics_linear, ϕXgT_DP_dyni)
end

## Check 

gr()
BRS_plot = plot(); colors = vcat("black", palette(["red", "blue"], length(T))...)
for i=1:length(T)+1; contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][i], res, res)', levels=[0], color=colors[i], lw=2.5, colorbar=false); end

BRS_plot_linear = plot(); whiter_colors = vcat("black", [palette(["white", c], 5)[2] for c in palette(["red", "blue"], length(T))]...)
for i=1:length(T)+1; contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[1][i], res, res)', levels=[0], color=colors[i], lw=2.5, colorbar=false); end

BRS_plots = []
for i=1:length(T)
    pli = plot(title="t=T-$(T[i])")
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][1], res, res)', levels=[0], color="black", lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][i+1], res, res)', levels=[0], color=colors[i+1], lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[1][i+1], res, res)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    plot!(xlabel="x_1", ylabel="x_2")
    push!(BRS_plots, pli)
end
plot(BRS_plots..., layout=(2,3))





## Solve Augmented Value (with DP)





SMa_reach = hj_r_setup.SlowManifoldAug(mu=μ, lambduh=λ, max_u=max_u, max_d=max_d)
SMa_avoid = hj_r_setup.SlowManifoldAug(mu=μ, lambduh=λ, max_u=max_u, max_d=max_d, control_mode="max", disturbance_mode="min")

A_aug, B₁_aug, B₂_aug, c_aug = lin_mat_fs_aug
E_s_aug(s) = δ̃ˢ_aug[3](-s) * Matrix([0. 0. 1.]') # in forward time
E_sU_aug(s) = δ̃ˢU_aug[3](-s) * Matrix([0. 0. 1.]') # in forward time
E_sD_aug(s) = δ̃ˢD_aug[3](-s) * Matrix([0. 0. 1.]') # in forward time
E_sU_man(s) = δ̃ˢU_man[3](-s) * Matrix([0. 0. 1.]') # in forward time
E_sD_man(s) = δ̃ˢD_man[3](-s) * Matrix([0. 0. 1.]') # in forward time

# SMa_LTV_reach = s -> LinearError(A(X̃_aug(s)), B₁(X̃_aug(s)), B₂(X̃_aug(s)), c(X̃_aug(s)), E_sU_aug(s); max_u=max_u, max_d=max_d, Ushape="box", game="reach")
SMa_LTV_reach = s -> LinearError(A_aug(X̃_aug(s)), B₁_aug(X̃_aug(s)), B₂_aug(X̃_aug(s)), c_aug(X̃_aug(s)), E_sU_aug(s); max_u=max_u, max_d=max_d, Ushape="box", game="reach")
SMa_LTV_avoid = s -> LinearError(A_aug(X̃_aug(s)), B₁_aug(X̃_aug(s)), B₂_aug(X̃_aug(s)), c_aug(X̃_aug(s)), E_sD_aug(s); max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

SMm_L_reach = s -> LinearError(κ_mats[1], κ_mats[2], κ_mats[3], zero(c𝒯_aug), E_sU_man(s); max_u=max_u, max_d=max_d, Ushape="box", game="reach")
SMm_L_avoid = s -> LinearError(κ_mats[1], κ_mats[2], κ_mats[3], zero(c𝒯_aug), E_sD_man(s); max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

dynamics_aug = [SMa_reach] #, SM_avoid];
dynamics_aug_linear = [SMa_LTV_reach, SMm_L_reach] #, SM_LTV_avoid]; #FIXME

res2 = 30
Gg, Gg_DP, ϕ0Gg_DP, gig1 = hjr_init(c𝒯_aug, Q𝒯_aug, 1.; shape="ball", lb=(-2, -1, -1.), ub=(2, 4, 4), res=(res2, res2, res2));

ϕGgT_DP_dynamics_aug = hjr_solve(Gg_DP, ϕ0Gg_DP, dynamics_aug, T; BRS=true, one_shot=true);

th = 0.01
ϕGgT_DP_dynamics_aug_linear = []
ϕGgT_DP_dynamics_aug_linear_hjr = []
for dyni in dynamics_aug_linear
    ϕGgT_DP_dyni = []; ϕGgT_DP_dyni_hjr = []; 
    push!(ϕGgT_DP_dyni, Matrix(reshape(ϕ0Gg_DP.tolist(), length(ϕ0Gg_DP.tolist()), 1))[:,1]) # target
    push!(ϕGgT_DP_dyni_hjr, ϕ0Gg_DP)
    hj_r_output = jnp.copy(ϕ0Gg_DP)
    for (tsi, ts) in enumerate(collect(th:th:T[end]))
        dynis = dyni(ts)
        hj_r_output = hj.step(ss_set, dynis, Gg_DP, 0., hj_r_output, -th)
        if ts ∈ T; 
            push!(ϕGgT_DP_dyni, Matrix(reshape(hj_r_output.tolist(), length(hj_r_output.tolist()), 1))[:,1]); 
            push!(ϕGgT_DP_dyni_hjr, hj_r_output)
        end
    end
    push!(ϕGgT_DP_dynamics_aug_linear, ϕGgT_DP_dyni)
    push!(ϕGgT_DP_dynamics_aug_linear_hjr, ϕGgT_DP_dyni_hjr)
end

interp_quad(hjr_vals) = map(xg -> Gg_DP.interpolate(hjr_vals, Ψ(xg)).item(), eachcol(Xg))
ϕGgT_DP_dynamics_aug_linear_quad = [[zeros(length(Xg[1,:])) for _=1:length(T)+1] for _=1:length(dynamics_aug_linear)];
@time for j = 1:length(dynamics_aug_linear)
    for i = 1:length(T)+1
        ϕGgT_DP_dynamics_aug_linear_quad[j][i] = interp_quad(ϕGgT_DP_dynamics_aug_linear_hjr[j][i])
    end
end

## Check

gr()

BRS_plots_dual = []
for i=1:length(T)
    pli = plot(title="t=T-$(T[i])")
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][1], res, res)', levels=[0], color="black", lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][i+1], res, res)', levels=[0], color=colors[i+1], lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕGgT_DP_dynamics_aug_linear_quad[1][i+1], res, res)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    plot!(xlabel="x_1", ylabel="x_2")
    push!(BRS_plots_dual, pli)
end
plot(BRS_plots_dual..., layout=(2,3))

BRS_plots_dual = []
for i=1:length(T)
    pli = plot(title="t=T-$(T[i])")
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][1], res, res)', levels=[0], color="black", lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][i+1], res, res)', levels=[0], color=colors[i+1], lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕGgT_DP_dynamics_aug_linear_quad[2][i+1], res, res)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    plot!(xlabel="x_1", ylabel="x_2")
    push!(BRS_plots_dual, pli)
end
plot(BRS_plots_dual..., layout=(2,3), plottitle="Fixed Augmented Linearization κ")





## Solve Augmented Value with Hopf





# th=0.05; Th=0.25
# T = collect(Th:Th:2.0)
# T = [0.2, 0.5, 1.0, 1.5, 2.0]

# E_aug(s) = diagm(δˢ_dc(s, δ̃ˢ_aug, BRZ_aug)) # in forward time
E_aug(s)    = δ̃ˢ_aug[3](-s)  * diagm([0., 0., 1.]) # in forward time
E_sU_aug_hopf(s) = δ̃ˢU_aug[3](-s) * diagm([0., 0., 1.]) # in forward time
E_sD_aug_hopf(s) = δ̃ˢD_aug[3](-s) * diagm([0., 0., 1.]) # in forward time
E_man_hopf(s) = δ̃ˢD_man[3](-s) * diagm([0., 0., 1.]) # in forward time

# K, L₁, L₂ = κ_mats;
# E_man(s) = diagm(δˢ_dc(s, δ̃ˢ_man, BRZ)) # in forward time
# E_man(s) = δ̃ˢ_man[3](-s) * diagm([0, 0, 1]) # in forward time

# lifted_system_aug_err = (s -> A(X̃_aug(-s)), s -> B₁(X̃_aug(-s)), s -> B₂(X̃_aug(-s)), Q₁, c₁, Q₂, c₁, s -> c(X̃_aug(-s)), E_aug);
lifted_system_aug_err = (s -> A_aug(X̃_aug(-s)), s -> max_u * B₁_aug(X̃_aug(-s)), s -> max_d * B₂_aug(X̃_aug(-s)), diagm(ones(2)), c₁, diagm(ones(2)), c₁, s -> c_aug(X̃_aug(-s)), E_man_hopf);
# lifted_system_aug_err = (s -> A_aug(X̃_aug(-s)), s -> B₁_aug(X̃_aug(-s)), s -> B₂_aug(X̃_aug(-s)), Q₁, c₁, Q₂, c₁, s -> c_aug(X̃_aug(-s)), E_sU_aug_hopf);
# lifted_system_aug_err = (s -> A_aug(X̃_aug(-s)), s -> max_u * B₁_aug(X̃_aug(-s)), s -> max_d * B₂_aug(X̃_aug(-s)), Q₁, c₁, Q₂, c₁, s -> c_aug(X̃_aug(-s)), E_sU_aug_hopf);

# lifted_system_aug_err = (A(X̃_aug(-0.1)), s -> B₁(X̃_aug(-s)), s -> B₂(X̃_aug(-s)), Q₁, c₁, Q₂, c₁, c(X̃_aug(-0.1)), s -> 0 * E_aug(s));
K, L₁, L₂ = κ_mats;
lifted_system_man_err = (K, max_u * L₁, max_d * L₂, diagm(ones(2)), c₁, diagm(ones(2)), c₁, zeros(nk), E_man_hopf);

res3 = 60
Xg_hopf, _, _, xig3 = hjr_init(c𝒯, Q𝒯, 1.; shape="ball", lb=(-2., -1), ub=(2., 4), res=(res3, res3));
# Gg, _, _, gig2 = hjr_init(c𝒯_aug, Q𝒯_aug, 1; shape="ball", lb=(-2, -2, -1.), ub=(2, 2, 4), res=(res2, res2, res2));
Gg_man = hcat(Ψ.(eachcol(Xg_hopf))...)
# Gg_all = hcat(Gg, Gg_man)

J(x::Matrix, Qₓ, cₓ; r=1.0) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2;
Jˢ(v::Vector, Qₓ, cₓ; r=1.0) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2;

# nk = length(c𝒯_aug);
# η = 1/15;
# c𝒯_aug = Ψ(c𝒯)
# Q𝒯_aug = inv(r) * diagm(vcat(d𝒯, inv(η)))
# Q𝒯_aug2 = diagm(vcat(d𝒯,η))

# c𝒯_aug = Ψ(c𝒯) + [0.1; 0.; 0.]
# Q𝒯_aug = inv(1.08r) * diagm(vcat([1.6, 1.0], inv(η)))

lifted_target = (J, Jˢ, (inv(Q𝒯_aug), c𝒯_aug));
gr(); contour(xig3..., reshape(lifted_target[1](Xg_hopf, Q𝒯, c𝒯), res3, res3)', levels=[0], color=:black, lw=2.5, colorbar=false)
contour!(xig3..., reshape(lifted_target[1](Gg_man, lifted_target[3]...), res3, res3)', levels=[0], color=:blue, lw=2.5, colorbar=false, aspect_ratio=:equal)

vh = 0.01; L = 20; tol = 1e-3; step_lim = 400; re_inits = 10; max_runs = 40; max_its = 2000 # working
opt_p_cd = (vh, L, tol, step_lim, re_inits, max_runs, max_its)

# Gg_man_results_reach_cd, _ = Hopf_BRS(lifted_system_aug_err, lifted_target, T; th, Xg=Gg_man, error=true, game="reach", opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true, warm_pattern="previous");
Gg_man2_results_reach_cd, _ = Hopf_BRS(lifted_system_man_err, lifted_target, T; th, Xg=Gg_man, error=true, game="reach", opt_method=Hopf_cd, opt_p=opt_p_cd, warm=false, check_all=true, printing=true, warm_pattern="temporal");

gr()
BRS_plots_dual = []
for i ∈ [0, 2, 3, 4]
    pli = i > 0 ? plot(title=L"t=T-" * latexstring(T[i])) : plot(title=L"t=T", dpi=300)
    hopf_res = i < 3 ? reshape(Gg_man2_results_reach_cd[2][i+1], res3, res3)' : smooth(reshape(Gg_man2_results_reach_cd[2][i+1], res3, res3)', gv=1.0); 
    # hopf_res = i > 0 ? smooth(reshape(Gg_man2_results_reach_cd[2][i+1], res3, res3)', gv=0.8) : reshape(Gg_man2_results_reach_cd[2][i+1], res3, res3)'
    # hopf_res = reshape(Gg_man2_results_reach_cd[2][i+1], res3, res3)'
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][1], res, res)', levels=[0], color="gray", lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[1][i+1], res, res)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    # contour!(xig3..., reshape(Gg_man_results_reach_cd[2][i+1], res3, res3)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    # contour!(xig3..., reshape(Gg_man2_results_reach_cd[2][i+1], res3, res3)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    contour!(xig3..., hopf_res, levels=[0], color=colors[i+1], lw=2.5, colorbar=false)
    # if i == 0; plot!(ylabel=L"x_2", left_margin=15Plots.px); end

    lo, _ = collect(zip(xlims(pli), ylims(pli)))
    locxl = lo .+ ((xlims(pli)[2] - xlims(pli)[1])/2, -0.5)
    locyl = lo .+ (-0.35, (ylims(pli)[2] - ylims(pli)[1])/2)
    annotate!(locxl..., L"x_1", fontsize=16)
    if i == 0
        plot!(xticks=(-2:1:2, [abs(xi)==2 ? latexstring(xi) : "" for xi in -2:1:2]), yticks=(-1:1:4, [abs(xi-1.5)==2.5 ? latexstring(xi) : "" for xi in -1:1:4]), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14)
        annotate!(locyl..., L"x_2", fontsize=16)
    else
        plot!(xticks=(-2:1:2, [abs(xi)==2 ? latexstring(xi) : "" for xi in -2:1:2]), yticks=(-1:1:4, ["" for xi in -1:1:4]), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14)
    end

    push!(BRS_plots_dual, pli)
end
BRS_plots_dual = plot(BRS_plots_dual..., layout=(1,4), size=(900, 270), top_margin = 10Plots.px, bottom_margin = 15Plots.px, dpi=300)




## Full Grid

Gg_results_reach_cd, _ = Hopf_BRS(lifted_system_man_err, lifted_target, T; th, Xg=Gg, error=true, game="reach", opt_method=Hopf_cd, opt_p=opt_p_cd, warm=false, check_all=true, printing=true, warm_pattern="temporal");

plotlyjs()
BRS_contr_plot = plot_nice(T, Gg_results_reach_cd; interpolate=true, ϵs=0.01, markerstrokewidth=0, alpha=0.3);

using ImageFiltering
smooth(X; gv=1.0) = imfilter(X, Kernel.gaussian(gv)) 
smooth3D(X; gv=1.0) = imfilter(X, Kernel.gaussian((gv, gv, gv)))
unpack_n_smooth(results_ϕT, res; gv=1.0) = smooth3D(reshape(results_ϕT, res, res, res); gv=gv)[:]

# Gg_results_reach_cd_smoothed = [unpack_n_smooth(Gg_results_reach_cd[2][i], res2; gv=0.5) for i=1:length(Gg_results_reach_cd[2])];
Gg_results_reach_cd_smoothed = [i > 4 ? unpack_n_smooth(Gg_results_reach_cd[2][i], res2; gv=0.8) : 
                                        unpack_n_smooth(Gg_results_reach_cd[2][i], res2; gv=0.) for i=1:length(Gg_results_reach_cd[2])];
BRS_contr_plot = plot_nice(T[1:5], ([Gg for _=1:6], Gg_results_reach_cd_smoothed[1:6]); interpolate=true, ϵs=0.01, markerstrokewidth=0, alpha=0.4);

# quadvals = reshape(Xg[1,:].^2, res, res)
# quad_trace = PlotlyJS.surface(x = xig1[1], y = xig1[2], z = quadvals, color=:black, opacity=0.1, colorbar=false, usecolormap=false, showscale=false)
add_trace!(BRS_contr_plot[1], quad_trace)
BRS_contr_plot[1]

# PlotlyJS.plot(BRS_contr_plot[1])

trace1 = scatter3d(;x=rw(),y=rw(), z=rw(), mode="lines",
                        marker=attr(color="#1f77b4", size=12, symbol="circle",
                                    line=attr(color="rgb(0,0,0)", width=0)),
                        line=attr(color="#1f77b4", width=1))

#this should give a line which can come from interpolation of the zero level sets of the 3D plots
# note, will need to try interpolating with smooth and non-smooth
# maybe interpolation could be done by hijacking hjr.Grid functionality...

### 3D Plotting

using LaTeXStrings # not rendering ...

xtick_labs = [abs(i) == 2 ? string(i) : "" for i in -2:1:2]; xtick_labs[3] = "g₁" 
ytick_labs = [abs(i-1.5) == 2.5 ? string(i) : "" for i in -1:1:4]; ytick_labs[3] = "g₂" 
# ztick_labs = [abs(i-1.5) == 2.5 ? string(i) : "" for i in -1:1:4]; ztick_labs[3] = "g₃" 
ztick_labs = [ i == 4 ? string(i) : "" for i in -1:1:4]; ztick_labs[3] = "g₃" 

axis_params = attr(xaxis=attr(gridcolor="rgb(230, 230, 230)",
                                zerolinecolor="rgb(230, 230, 230)",
                                showbackground=false, title_standoff = 0,
                                backgroundcolor="rgb(255, 255, 255)", ticklabelposition="inside top",
                                title="", tickvals=-2:1:2, ticktext=xtick_labs),
                   yaxis=attr(gridcolor="rgb(230, 230, 230)",
                                zerolinecolor="rgb(230, 230, 230)",
                                showbackground=false, title_standoff = 0,
                                backgroundcolor="rgb(255, 255, 255)", ticklabelposition="inside top",
                                title="", tickvals=-1:1:4, ticktext=ytick_labs),
                   zaxis=attr(gridcolor="rgb(230, 230, 230)",
                                zerolinecolor="rgb(230, 230, 230)",
                                showbackground=false, title_standoff = 0,
                                backgroundcolor="rgb(255, 255, 255)", ticklabelposition="inside top",
                                title="", tickvals=-1:1:4, ticktext=ztick_labs),
                    aspectratio=attr(x=1, y=1, z=1), aspectmode = "manual", 
                    extra_plot_kwargs = KW(:include_mathjax => "cdn"),
) 

# PlotlyJS.update!(BRS_contr_plot[1], layout)

i = 3
plot_colors = palette(["red", "blue"], 4); alpha=0.3; ϵc = 1e-5;

quadvals = reshape(Xg[1,:].^2, res, res)
quad_trace = PlotlyJS.surface(x = xig1[1], y = xig1[2], z = quadvals, color=:black, opacity=0.1, colorbar=false, usecolormap=false, showscale=false)

# b⁺, ϕ = Gg, Gg_results_reach_cd_smoothed[i+1]
b⁺, ϕ = Gg, Gg_results_reach_cd[2][i+1]

value_trace = PlotlyJS.isosurface(x=b⁺[1,:], y=b⁺[2,:], z=b⁺[3,:], value=ϕ[:], opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
                        name="test", colorscale=[[0, "rgb" * string(plot_colors[i])[13:end]], [1, "rgb" * string(plot_colors[i])[13:end]]],
                        lighting=attr(diffuse=0.9)
)

test_pljs = PlotlyJS.plot([value_trace, quad_trace], layout_nice)
# test_pljs = PlotlyJS.plot([value_trace, quad_trace], layout_nice_noticks)
zomm_out = 1.6;
relayout!(test_pljs, width=400, height=400, scene_camera=attr(eye=attr(x=zomm_out*1.25, y=zomm_out*-1.25, z=zomm_out*0.5)), margin=attr(t=0, r=0, l=30, b=0)); test_pljs

# might fix in interactive mode, https://github.com/microsoft/vscode-jupyter/issues/8131#issuecomment-1589961116
# plotly.offline.init_notebook_mode()
# display(HTML(
#     '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
# ))

using ScatteredInterpolation

itp = ScatteredInterpolation.interpolate(Polyharmonic(), b⁺, ϕ);

res4 = 200
Xg_fine, _, _, xig3 = hjr_init(c𝒯, Q𝒯, 1.; shape="ball", lb=(-1.5, -1), ub=(1.5, 4), res=(res4, 2*res4));
Gg_man_fine = hcat(Ψ.(eachcol(Xg_fine))...);
ϕman = ScatteredInterpolation.evaluate(itp, Gg_man_fine);

# ϵs = 0.0125
ϵs = 0.0025
# ϵs = 0.01

Gg_man_ϕ0 = Gg_man_fine[:, abs.(ϕman) .< ϵs]
Xg_ϕ0 = Xg_fine[:, abs.(ϕman) .< ϵs]

curve_trace = PlotlyJS.scatter3d(;x=Gg_man_ϕ0[1,:], y=Gg_man_ϕ0[2,:], z=Gg_man_ϕ0[3,:], mode="markers", #mode="lines",
                        marker=attr(color="#1f77b4", size=1, symbol="circle",
                                    line=attr(color="rgb(0,0,0)", width=0.)),
                        line=attr(color="#1f77b4", width=1)
)

test_pljs = PlotlyJS.plot([value_trace, quad_trace, curve_trace], layout_nice)

## trace line from scatter 

Gg_man_ϕ0_cp = hcat(copy(Gg_man_ϕ0)); ordered = zero(Gg_man_ϕ0);
# Gg_man_ϕ0_cp = hcat(copy(Gg_man_ϕ0), [-1.2587, -0.61, 1.584], [-1.2587, -0.725, 1.584]); ordered = zero(Gg_man_ϕ0);

maxy_ix = argmax(Gg_man_ϕ0_cp[2,:])
ordered[:,1] = Gg_man_ϕ0_cp[:, maxy_ix]
Gg_man_ϕ0_cp = Gg_man_ϕ0_cp[:, 1:end .!= maxy_ix] # pop

for oix = 1:(size(ordered, 2)-1)
    # print("$oix,")
    min_ix = argmin(map(norm, eachcol(Gg_man_ϕ0_cp .- ordered[:, oix]))) #give me closest, needs low ϵs to reduce "doubling back"
    ordered[:, oix+1] = Gg_man_ϕ0_cp[:,min_ix]
    Gg_man_ϕ0_cp = Gg_man_ϕ0_cp[:, 1:end .!= min_ix] # pop
end

curve_trace_line = PlotlyJS.scatter3d(;x=ordered[1,:], y=ordered[2,:], z=ordered[3,:], mode="lines", #mode="lines",
                        marker=attr(color="#1f77b4", size=12, symbol="circle",
                        line=attr(color="rgb(0,0,0)", width=0.)),
                        line=attr(color="#1f77b4", width=4)
)

test_pljs = PlotlyJS.plot([value_trace, quad_trace, curve_trace_line], layout_nice)

# test_pljs = PlotlyJS.plot([curve_trace_line], layout_nice)
# pt = PlotlyJS.scatter3d(; x=[-1.2587], y=[-0.61],z=[1.584], mode="markers", marker=attr(color="rgb(0.99,0,0)", size=3, symbol="circle", line=attr(color="rgb(0,0,0)", width=1.)))
# add_trace!(test_pljs, pt)
### 3D Plotting (loop)

# plot_colors = [palette(["red", "blue"], 4)...]; 
plot_colors = [RGB{Float64}(0.0,0.0,0.0), palette([RGB(0.99, 0., 0.), RGB(0., 0., 0.99)], 3)...]
# plot_colors = vcat(RGB{Float64}(0.0,0.0,0.0), [palette(["white", c], 5)[3] for c in palette(["red", "blue"], 3)]...);
alpha=0.4; ϵc = 1e-5; zomm_out = 1.6;
# view_2 = (1., -1.25, 0.8)
view_3 = (0.6, -1.25, 1.6)

quadvals = reshape(Xg[1,:].^2, res, res)
quad_trace = PlotlyJS.surface(x = xig1[1], y = xig1[2], z = quadvals, opacity=0.2, colorbar=false, usecolormap=false, showscale=false,
                                colorscale=[[0, "rgb(0.,0.,0.)"], [1, "rgb(0.,0.,0.)"]])

res4 = 200; ϵs = 0.0025
Xg_fine, _, _, xig3 = hjr_init(c𝒯, Q𝒯, 1.; shape="ball", lb=(-1.5, -1), ub=(1.5, 4), res=(res4, 2*res4));
Gg_man_fine = hcat(Ψ.(eachcol(Xg_fine))...);

nice_BRS_3d_plots = Dict()
results = Gg_results_reach_cd[2][[1, 3, 4, 5]] # skip 0.2
Ts = vcat(0., T)[[1, 3, 4, 5]]

for i=1:4

    nice_BRS_3d_plots["$(Ts[i])"] = Dict()
    b⁺, ϕ = Gg, results[i]
    nice_BRS_3d_plots["$(Ts[i])"]["Gg"], nice_BRS_3d_plots["$(Ts[i])"]["ϕGg"] = Gg, results[i]
    color_i = plot_colors[i];

    itp = ScatteredInterpolation.interpolate(Polyharmonic(), b⁺, ϕ);
    ϕman = ScatteredInterpolation.evaluate(itp, Gg_man_fine);
    nice_BRS_3d_plots["$(Ts[i])"]["itp"] = itp

    Gg_man_ϕ0 = Gg_man_fine[:, abs.(ϕman) .< ϵs]
    Xg_ϕ0 = Xg_fine[:, abs.(ϕman) .< ϵs]

    Gg_man_ϕ0_cp = i != 4 ? copy(Gg_man_ϕ0) : hcat(copy(Gg_man_ϕ0), [-1.2587, -0.61, 1.584], [-1.2587, -0.725, 1.584])
    ordered = zero(Gg_man_ϕ0);

    maxy_ix = argmax(Gg_man_ϕ0_cp[2,:])
    ordered[:,1] = Gg_man_ϕ0_cp[:, maxy_ix]
    Gg_man_ϕ0_cp = Gg_man_ϕ0_cp[:, 1:end .!= maxy_ix] # pop

    for oix = 1:(size(ordered, 2)-1)
        min_ix = argmin(map(norm, eachcol(Gg_man_ϕ0_cp .- ordered[:, oix]))) #give me closest, needs low ϵs to reduce "doubling back"
        ordered[:, oix+1] = Gg_man_ϕ0_cp[:,min_ix]
        Gg_man_ϕ0_cp = Gg_man_ϕ0_cp[:, 1:end .!= min_ix] # pop
    end
    if i < 3; ordered = hcat(ordered, ordered[:,1]); end

    value_trace = PlotlyJS.isosurface(x=b⁺[1,:], y=b⁺[2,:], z=b⁺[3,:], value=ϕ[:], opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
                            name="$(Ts[i])", colorscale=[[0, "rgb" * string(color_i)[13:end]], [1, "rgb" * string(color_i)[13:end]]],
    )

    curve_trace_line = PlotlyJS.scatter3d(;x=ordered[1,:], y=ordered[2,:], z=ordered[3,:], mode="lines", alpha=1.0,
                            marker=attr(color="rgb" * string(color_i)[13:end], size=12, symbol="circle",
                            line=attr(color="rgb" * string(color_i)[13:end], width=0.)),
                            line=attr(color="rgb" * string(color_i)[13:end], width=6, alpha=0.8)
    )
    # curve_trace_line = PlotlyJS.scatter3d(;x=ordered[1,:], y=ordered[2,:], z=ordered[3,:], mode="lines", alpha=1.0,
    #                         marker=attr(color="rgb(0.,0.,0.)", size=12, symbol="circle",
    #                         line=attr(color="rgb(0.,0.,0.)", width=0.)),
    #                         line=attr(color="rgb(0.,0.,0.)", width=5, alpha=0.8)
    # )

    nice_BRS_3d_plots["$(Ts[i])"]["traces"] = [value_trace, quad_trace, curve_trace_line];

    test_pljs = PlotlyJS.plot([value_trace, quad_trace, curve_trace_line], layout_nice)
    relayout!(test_pljs, width=400, height=400, scene_camera=attr(eye=attr(x=zomm_out*view_3[1], y=zomm_out*view_3[2], z=zomm_out*view_3[3])), margin=attr(t=0, r=0, l=30, b=0)); 

    nice_BRS_3d_plots["$(Ts[i])"]["plot"] = test_pljs
end

# JLD2.save("SMaug_BRSplots_3d_nicer.jld2", nice_BRS_3d_plots)

# relayout!(nice_BRS_3d_plots[Ts[1]]["plot"], height=4*300, width=4*300)

# PlotlyJS.savefig(nice_BRS_3d_plots[Ts[1]]["plot"], "t0.pdf")

cyl_Ggϕ = J(Gg, diagm([1., 1., Inf]), c𝒯_aug)
aug_target_trace = PlotlyJS.isosurface(x=Gg[1,:], y=Gg[2,:], z=Gg[3,:], value=cyl_Ggϕ[:], opacity=0.3, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
                        name="test", colorscale=[[0, "rgb(0.2,0.35,0.35)"], [1, "rgb(0.2,0.35,0.35)"]],
                        lighting=attr(diffuse=0.9)
)

fullfig = PlotlyJS.make_subplots(rows=1, cols=4, specs=fill(Spec(kind="scene"), 1, 4), horizontal_spacing=0.05)
for i=1:4
    for tri =1:3
        if i==1 && tri == 1
            add_trace!(fullfig, aug_target_trace, row=1, col=i)
        end
        add_trace!(fullfig, nice_BRS_3d_plots["$(Ts[i])"]["traces"][tri], row=1, col=i)
    end
end
relayout!(fullfig, width=1200, height=400, showlegend=false); 
fullfig


view_high = (1.25, -1.25, 0.7); zmh = 1.5;
# view_downtheline = (0.45, -1.25, 1.1); zdl = 1.3;
view_downtheline = (0.65, -1.25, 0.9); zdl = 1.6;
view_downlow = (1.1, -1.25, 0.5); zdlo = 1.3;
view = view_high; zoom = zmh
new_view = attr(eye=attr(x=zoom*view[1], y=zoom*view[2], z=zoom*view[3]))
relayout!(fullfig, scene_camera=new_view, scene2_camera=new_view, scene3_camera=new_view, scene4_camera=new_view); 
fullfig

relayout!(fullfig, font=attr(family="Computer Modern",color=:black, size=12,), scene=axis_params, scene2=axis_params, scene3=axis_params, scene4=axis_params)
relayout!(fullfig, autosize=false, width=1200, height=500)

PlotlyJS.savefig()




