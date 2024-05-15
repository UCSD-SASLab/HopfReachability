
include(pwd() * "/HopfReachability.jl");
include(pwd() * "/src/cons_lin_utils.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, plot_BRS

## VanderPol Example for Testing

r = 0.25; c𝒯 = [0.; 1.]; # c𝒯= [0.; 0.]
max_u = 1.0; max_d = 0.5;
t = 0.4

c = [0.]
Q₁ = inv(max_u) * diagm([1.])
Q₂ = inv(max_d) * diagm([1.])
Q𝒯 = inv(r) * diagm([1., 1.])
nx = length(c𝒯);

inputs = ((Q₁, c), (Q₂, c))
𝒯target = (nothing, nothing, (Q𝒯, c𝒯))

(Q𝒯, c𝒯), (Q₁, c₁), (Q₂, c₂) = 𝒯target[3], inputs[1], inputs[2]

X0 = Hyperrectangle(; low = c𝒯 - diag(inv(Q𝒯)), high = c𝒯 + diag(inv(Q𝒯)))
U = Hyperrectangle(; low = c₁ - diag(inv(Q₁)), high = c₁ + diag(inv(Q₁)))
D = Hyperrectangle(; low = c₂ - diag(inv(Q₂)), high = c₂ + diag(inv(Q₂)))

μ = 1.0
function vanderpol!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = μ * (1.0 - x[1]^2) * x[2] - x[1] + x[3] + x[4]
    dx[3] = zero(x[3]) #control
    dx[4] = zero(x[4]) #disturbance
    return dx
end

## Solve (defualt x̃ is auto from target center)

t = 0.39

δ̃ˢ, X̃, BRZ, dt, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t);
δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U");
δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D");

BRZ_plot = plot(BRZ, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (𝒰 & 𝒟)", legend=:bottomleft)
plot!(BRZ_plot, BRZu, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (𝒰)")
plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (𝒟)")
scatter!(BRZ_plot, eachrow(hcat(X̃.(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃ backsolved w/ trivial ctrl/dist", alpha=0.6)

error_plot = plot(dt, δ̃ˢ[2].(dt), label="Taylor δˢ for BRZ (𝒰 & 𝒟), x̃", xlabel="t")
plot!(error_plot, dtU, δ̃ˢU[2].(dtU), label="Taylor δˢ for BRZ (𝒰), x̃")
plot!(error_plot, dtD, δ̃ˢD[2].(dtD), label="Taylor δˢ for BRZ (𝒟), x̃")

plot(BRZ_plot, error_plot)

## Solve with DP

Th = 0.13
T = collect(Th:Th:t)

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/DP_comparison_files");
hj_r_setup = pyimport("vdp_hj_reachability");

VdP_reach = hj_r_setup.VanderPol(mu=1.0, max_u=max_u, max_d=max_d)
VdP_avoid = hj_r_setup.VanderPol(mu=μ, max_u=max_u, max_d=max_d, control_mode="max", disturbance_mode="min")

A, B₁, B₂, c = lin_mat_fs
B1 = Matrix([0. 1.]'); B2 = Matrix([0. 1.]')
# E_c(s) = sum(ci * (-s)^(d-1) for (d, ci) in enumerate(coeffs(δˢ_TS_Ucarrf))) * Matrix([0. 1.]') # put it in forward time
E_c(s) = δ̃ˢ[2](-s) * Matrix([0. 1.]') # in forward time

dynamics_linear_reach = LinearError(A(X̃(0)), B1, B2, c(X̃(0)), E_c; max_u=max_u, max_d=max_d, Ushape="box", game="reach") # jax cant handle julia fns, so must do iterative solve if tv lin
dynamics_linear_avoid = LinearError(A(X̃(0)), B1, B2, c(X̃(0)), E_c; max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

dynamics_reach = [VdP_reach, dynamics_linear_reach];
dynamics_avoid = [VdP_avoid, dynamics_linear_avoid]; 
res=300

Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(ones(nx)), r; shape="ball", lb=(-2, -2), ub=(2, 2), res=res)
ϕXgT_DP_dynamics_reach = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_reach, T; BRS=true, one_shot=true);
ϕXgT_DP_dynamics_avoid = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_avoid, T; BRS=true, one_shot=true);

## Solve with Hopf

th=0.0325

A, B₁, B₂, c = lin_mat_fs
B1 = Matrix([0. 1.]'); B2 = Matrix([0. 1.]')
Eδc(s) = δ̃ˢ[2](-t) * diagm([0, 1]) # constant error
Eδt(s) = δ̃ˢ[2](-s) * diagm([0, 1]) # tv error
# EδU(s) = δ̃ˢU[2](-s) * diagm([0, 1]) # tv error, ctrl feas
EδD(s) = δ̃ˢD[2](-s) * diagm([0, 1]) # tv error, dist feas

system_errc = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), Eδc);
system_errt = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), Eδt);
# system_errU = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), EδU);
system_errD = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), EδD);

lb = 1.1 .* (-ρ(-[1,0,0,0], BRZ), -ρ(-[0,1,0,0], BRZ))
ub = (ρ([1,0,0,0], BRZ), ρ([0,1,0,0], BRZ))
res2 = 30
Xg, _, _, xig2 = hjr_init(c𝒯, Q𝒯, 1; shape="ball", lb=lb, ub=ub, res=res2);

J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2;
Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2;
target = (J, Jˢ, (diagm(ones(nx)), c𝒯));

# solution, run_stats = Hopf_BRS(system, target, T; th, Xg=Xg[:,1:end], inputshape="box", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
(ϕXgT, ϕXgT_Hopf_errc_reach), _ = Hopf_BRS(system_errc, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errc_avoid),    _ = Hopf_BRS(system_errc, target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errt_reach),    _ = Hopf_BRS(system_errt, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errt_avoid),    _ = Hopf_BRS(system_errt, target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=false, check_all=true, printing=true);
# (_, ϕXgT_Hopf_errU_reach),    _ = Hopf_BRS(system_errU, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errD_reach),    _ = Hopf_BRS(system_errD, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errD_avoid),    _ = Hopf_BRS(system_errD, target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=false, check_all=true, printing=true);
# println("Min Val ϕ(Xg[1:3], t): $(minimum(ϕXgT_Hopf[2]))")
# plotlyjs(); plot_BRS(T, ϕXgT, ϕXgT_Hopf_errD_avoid; interpolate=true, value_fn=true)

## Plot Single BRZ vs. BRS (at t), Constant Error

gr()
pal = palette(:seaborn_colorblind)
colors = [:black, pal[1], pal[2], pal[3], "gray"]
alpha = 0.7; lw=2.5; legend_hfactor=0.7; dpi=300;
tix = 2; ti = T[tix];
dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errc_reach) # true v lin + const err
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errc_avoid)

single_plots = [];
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
    labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-") : fill("", 5)
    title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"

    single_plot = plot(title=title, dpi=dpi);
    contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(single_plot), ylims=ylims(single_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

    plot!(eachrow(hcat(X̃.(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(single_plot), ylims=ylims(single_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[5], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    plot!(BRZ[dtix], vars=(1,2), alpha=alpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot);

    contour!(xig1..., reshape(ϕ_DP[tix+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(single_plot), ylims=ylims(single_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

    contour!(xig2..., reshape(ϕ_Hopf[tix+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(single_plot), ylims=ylims(single_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.25:.5:1.25, (L".25", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    plot!(xlims=(-0.8, .3), ylims=(0.05, 1.5))
    lo, _ = collect(zip(xlims(single_plot), ylims(single_plot)))
    locxl = lo .+ ((xlims(single_plot)[2] - xlims(single_plot)[1])/2, -0.1)
    locyl = lo .+ (-0.1, (ylims(single_plot)[2] - ylims(single_plot)[1])/2)
    annotate!(locxl..., L"x_1", fontsize=16)
    annotate!(locyl..., L"x_2", fontsize=16)

    push!(single_plots, single_plot)
end

single_plots_final = plot(single_plots..., layout=(1,2), legend=(-1.15, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

# single_plots_final = plot(single_plots..., layout=(1,2), legend=(-1.15, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

## Plot Multiple BRZ vs. BRS

gr()
colors = [:black, pal[1], pal[2], pal[3], pal[7]]
alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errc_reach, ϕXgT_Hopf_errt_reach) # true v lin + tv err
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errc_avoid, ϕXgT_Hopf_errt_avoid)

multi_plots = [];
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf, ϕ_Hopf2 = game == "reach" ? reach_comp : avoid_comp
    labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-") : fill("", 6)
    title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"

    multi_plot = plot(title=title, dpi=dpi);
    contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    # plot!(eachrow(hcat(X̃.(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(multi_plot), ylims=ylims(multi_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[5], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
    for (i, ti) in enumerate(T)
        dtix = findfirst(x -> x .+ ti > 0, dt) - 1;

        labels = i == 1 ? labels : fill("", length(labels))
        plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);
        # plot!(BRZ[dtix], vars=(1,2), alpha=alpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, fillalpha = 0.65);

        contour!(xig1..., reshape(ϕ_DP[i+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

        # contour!(xig2..., reshape(ϕ_Hopf[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
        contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
    # plot!(xticks=(0.25:0.5:1.25, (L"0.25", "", L"1.5")), yticks=(-0.25:.5:1.25, (L"0", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    # plot!(xlims=(.2, 1.5), ylims=(-0.6, 1.7))
    lo, _ = collect(zip(xlims(multi_plot), ylims(multi_plot)))
    locxl = lo .+ ((xlims(multi_plot)[2] - xlims(multi_plot)[1])/2, -0.1)
    locyl = lo .+ (-0.1, (ylims(multi_plot)[2] - ylims(multi_plot)[1])/2)
    annotate!(locxl..., L"x_1", fontsize=16)
    annotate!(locyl..., L"x_2", fontsize=16)

    push!(multi_plots, multi_plot)
end

multi_plots_final = plot(multi_plots..., layout=(1,2), legend=(-0.925, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

## Plot Multiple BRZ - U, D

gr()
colors = [:black, pal[1], pal[2], pal[3], pal[5], pal[10]]
alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.9; dpi=300;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errt_reach, ϕXgT_Hopf_errD_reach) #ϕXgT_Hopf_errU_reach) # true v lin + tv err v 𝒮_𝒰
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errt_avoid, ϕXgT_Hopf_errD_avoid)
# labelss = [L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-",  L"\hat{𝒮}_𝒰 / \hat{𝒮}_𝒟", L"ℛ_{δ^*_{𝒰(τ)}} / ℛ_{δ^*_{𝒟(τ)}}^-"]
labelss = [L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-",  L"\hat{𝒮}_𝒟 / \hat{𝒮}_𝒟", L"ℛ_{δ^*_{𝒟(τ)}} / ℛ_{δ^*_{𝒟(τ)}}^-"]

multi_plotsUD = [];
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf, ϕ_Hopf2 = game == "reach" ? reach_comp : avoid_comp
    labels = game == "avoid" ? labelss : fill("", 7)
    title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"
    BRZ2 = game == "reach" ? BRZu : BRZd

    multi_plot = plot(title=title, dpi=dpi);
    contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    # plot!(eachrow(hcat(X̃.(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(multi_plot), ylims=ylims(multi_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[5], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
    for (i, ti) in enumerate(T)
        dtix = findfirst(x -> x .+ ti > 0, dt) - 1;

        labels = i == 1 ? labels : fill("", length(labels))
        plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);
        # plot!(BRZ2[dtix], vars=(1,2), alpha=0., lw=lw, label=labels[6], color=:white, linecolor=colors[6], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);

        contour!(xig1..., reshape(ϕ_DP[i+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

        contour!(xig2..., reshape(ϕ_Hopf[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
        contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[5], lw=lw, alpha=alpha, colorbar=false);
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[7], linecolor=colors[5], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=3)
    plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
    # plot!(xticks=(0.25:0.5:1.25, (L"0.25", "", L"1.5")), yticks=(-0.25:.5:1.25, (L"0", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    # plot!(xlims=(.2, 1.5), ylims=(-0.6, 1.7))
    lo, _ = collect(zip(xlims(multi_plot), ylims(multi_plot)))
    locxl = lo .+ ((xlims(multi_plot)[2] - xlims(multi_plot)[1])/2, -0.1)
    locyl = lo .+ (-0.1, (ylims(multi_plot)[2] - ylims(multi_plot)[1])/2)

    annotate!(locxl..., L"x_1", fontsize=16)
    annotate!(locyl..., L"x_2", fontsize=16)

    push!(multi_plotsUD, multi_plot)
end

multi_plotsUD_final = plot(multi_plotsUD..., layout=(1,2), legend=(-0.7, -.155), bottom_margin=60Plots.px, background_color_legend = nothing, foreground_color_legend = nothing, dpi=dpi,legendfontsize=12)


### New Problem

# r = 0.5; c𝒯 = [-1.; 1.]; # c𝒯= [0.; 0.]
# Q𝒯 = inv(r) * diagm([1., 1.])
# max_u = 1.0; max_d = 0.5;
# J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2;
# Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2;
# 𝒯target = (J, Jˢ, (Q𝒯, c𝒯))
# t = 0.4

# Th = 0.13
# T = collect(Th:Th:t)

### Target Partitioning

ntp = 5; θtp = 2π/ntp

# minimal circle-polygon cover
# ϵ=0.005; ra = ϵ:ϵ:r; l = ϵ:ϵ:2r
# d² = 2 * l^2 * (1 - cos(θtp)) # ntp-polygon length^2
# h = sqrt(l^2 - d²/2)
# a = sqrt(d² * (4 * ra^2 - d²)) / d # dis of nhbr-circ ∩ pts
# f(ra, l) = 0.5*(sqrt(2 * l^2 * (1 - cos(θtp)) * (4 * ra^2 - 2 * l^2 * (1 - cos(θtp)))) / sqrt(2 * l^2 * (1 - cos(θtp)))) + sqrt(l^2 - l^2 * (1 - cos(θtp))) - r

# Gral = hcat(collect.(Iterators.product(ra, l))...)
# fval = ones(size(Gral,2))
# for ii=1:size(Gral,2); try; fval[ii] = f(Gral[:,ii]...); catch y; fval[ii] = 0.2; end; end
# plotlyjs(); scatter(ra, l, reshape(fval, length(ra), length(l))')

# l = 0.154 # regular polygon radius, defining partition centers
# ra = 0.154 # partition radii (optimalish for avoid, using same for reach)

l = r / 2.25 # regular polygon radius, defining partition centers
rs = r / 1.6 # partition radii (optimalish for avoid, using same for reach)
rr, ra = 1rs, 1.375rs

R(θ) = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1];
To(x, y) = [1 0 x; 0 1 y; 0 0 1];
Rot(pr, p, i, θ) = (To(pr...) * R(θ)^(i-1) * To(-pr...) * vcat(p,1))[1:2]
c𝒯rri(i) = Rot(c𝒯, c𝒯 + [0, (r-rr)], i , θtp) # reach centers
c𝒯rai(i) = Rot(c𝒯, c𝒯 + [0, l], i , θtp) # avoid centers

θi = 0.:0.01:2π
scatter(eachrow(c𝒯)..., color=:green)
scatter()
plot!([c𝒯[1] .+ r * cos.(θi)], [c𝒯[2] .+ r * sin.(θi)], lw=2, color=:green)
for i=1:ntp; 
    scatter!(eachrow(c𝒯rri(i))..., color=:blue)
    plot!([c𝒯rri(i)[1] .+ rr * cos.(θi)], [c𝒯rri(i)[2] .+ rr * sin.(θi)], lw=2, color=:blue)
    scatter!(eachrow(c𝒯rai(i))..., color=:red)
    plot!([c𝒯rai(i)[1] .+ ra * cos.(θi)], [c𝒯rai(i)[2] .+ ra * sin.(θi)], lw=2, color=:red)
end
plot!([c𝒯[1] .+ ra * cos.(θi)], [c𝒯[2] .+ ra * sin.(θi)], lw=2, color=:red) #extra avoid to cover middle
plot!(legend=false)

δ̃ˢU_TPr, X̃_TPr, BRZu_TPr, dtU_TPr = [], [], [], []
δ̃ˢD_TPa, X̃_TPa, BRZd_TPa, dtD_TPa = [], [], [], []

for i=1:ntp
    𝒯targetri = (nothing, nothing, (inv(rr) * diagm([1., 1.]), c𝒯rri(i)))    
    # δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetri, inputs, t; zono_over="U");
    δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetri, inputs, t; zono_over="D");
    push!(δ̃ˢU_TPr, δ̃ˢU); push!(X̃_TPr, X̃); push!(BRZu_TPr, BRZu); push!(dtU_TPr, dtU)

    𝒯targetai = (nothing, nothing, (inv(ra) * diagm([1., 1.]), c𝒯rai(i)))
    δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetai, inputs, t; zono_over="D");
    push!(δ̃ˢD_TPa, δ̃ˢD); push!(X̃_TPa, X̃); push!(BRZd_TPa, BRZd); push!(dtD_TPa, dtD)
end
𝒯targetai = (nothing, nothing, (inv(ra) * diagm([1., 1.]), c𝒯)) #extra avoid to cover middle
δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetai, inputs, t; zono_over="D");
push!(δ̃ˢD_TPa, δ̃ˢD); push!(X̃_TPa, X̃); push!(BRZd_TPa, BRZd); push!(dtD_TPa, dtD)

δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U");
δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D");

BRZ_plot_r = plot(BRZu, vars=(1,2), alpha=0.1, lw=3, label="BRZ (𝒰, 𝒯)", legend=:bottomleft); 
BRZ_plot_a = plot(BRZd, vars=(1,2), alpha=0.1, lw=3, label="BRZ (𝒟, 𝒯)", legend=:bottomleft);
for i=1:ntp
    plot!(BRZ_plot_r, BRZu_TPr[i], vars=(1,2), alpha=0.5, lw=1, label="BRZ (𝒰, 𝒯p$i)")
    plot!(BRZ_plot_a, BRZd_TPa[i], vars=(1,2), alpha=0.5, lw=1, label="BRZ (𝒟, 𝒯p$i)")
end
plot!(BRZ_plot_a, BRZd_TPa[ntp+1], vars=(1,2), alpha=0.5, lw=1, label="BRZ (𝒟, 𝒯p$(ntp+1))") #extra avoid to cover middle
plot(BRZ_plot_r, BRZ_plot_a)

### Solve w/ Various x̃ for one target (Linear Ensemble)

nle = 5; nu = 1; nd = 1;
# X̃0ŨD̃ = ([c𝒯, c𝒯 + [r,r], c𝒯 + [-r,r], c𝒯 + [r,-r], c𝒯 - [r,r]], # box
# X̃0ŨD̃ = ([c𝒯, c𝒯 + [r,0], c𝒯 - [r,0], c𝒯 + [0,r], c𝒯 - [0,r]], # circle
# X̃0 = [c𝒯, c𝒯 + [r,r]/sqrt(2), c𝒯 + [-r,r]/sqrt(2), c𝒯 + [r,-r]/sqrt(2), c𝒯 - [r,r]/sqrt(2)]
# X̃0 = [c𝒯rri(i) for i =1:nle]
rle = 1r
X̃0 = [Rot(c𝒯, c𝒯 + [0, rle], i , θtp) for i=1:nle] # reach centers
X̃0ŨD̃_LE = (X̃0, # circle ×
        [(y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu)], 
        [(y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd)])
# δ̃ˢU_LE, X̃_LE, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U", X̃0ŨD̃=X̃0ŨD̃_LE); # trajectories are the same, hence X̃_LE same for both
δ̃ˢU_LE, X̃_LE, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D", X̃0ŨD̃=X̃0ŨD̃_LE); # trajectories are the same, hence X̃_LE same for both
δ̃ˢD_LE, X̃_LE, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D", X̃0ŨD̃=X̃0ŨD̃_LE);
nle = length(X̃0ŨD̃_LE[1])

BRZ_plot = plot(BRZu, vars=(1,2), alpha=0.3, lw=3, label="BRZ (𝒰)", legend=:bottomleft)
plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.3, lw=3, label="BRZ (𝒟)")
for i=1:nle
    scatter!(BRZ_plot, eachrow(hcat(X̃_LE[i].(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃$i", alpha=0.3)
end
plot!()

plot(); for i=1:nle; plot!(dtU, δ̃ˢU_LE[i][2].(dtU)); end; plot!()

## Solve with DP (only needed if true problem changed)

# res = 300
# Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(ones(nx)), r; shape="ball", lb=(-2, -2), ub=(1, 2), res=res)
# ϕXgT_DP_dynamics_reach = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_reach, T; BRS=true, one_shot=true);
# ϕXgT_DP_dynamics_avoid = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_avoid, T; BRS=true, one_shot=true);

## Solve with Hopf

EδU_LE(i, s) = δ̃ˢU_LE[i][2](-s) * diagm([0, 1])
EδD_LE(i, s) = δ̃ˢD_LE[i][2](-s) * diagm([0, 1])
EδU_TP(i, s) = δ̃ˢU_TPr[i][2](-s) * diagm([0, 1])
EδD_TP(i, s) = δ̃ˢD_TPa[i][2](-s) * diagm([0, 1])

system_errU_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδU_LE(i,s));
system_errD_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδD_LE(i,s));
system_errU_TP(i) = (s -> A(X̃_TPr[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPr[i](-s)), s -> EδU_TP(i,s));
system_errD_TP(i) = (s -> A(X̃_TPa[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPa[i](-s)), s -> EδD_TP(i,s));

Jr(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * rr^2;
Jˢr(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * rr^2;
Ja(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * ra^2;
Jˢa(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * ra^2;

targetri(i) = (Jr, Jˢr, (diagm(ones(nx)), c𝒯rri(i)));
targetai(i) = (Ja, Jˢa, (diagm(ones(nx)), c𝒯rai(i)));
targetai_ntpp1 = (Ja, Jˢa, (diagm(ones(nx)), c𝒯));

lb = (1.1 * -ρ(-[1,0,0,0], BRZu), -ρ(-[0,1,0,0], BRZu))
ub = (1.65 * ρ([1,0,0,0], BRZu), 1.25 * ρ([0,1,0,0], BRZu))

res3 = 100
Xg, _, _, xig3 = hjr_init(c𝒯, Q𝒯, r; shape="box", lb=lb, ub=ub, res=res3);

opt_p = (0.01, 2, 1e-4, 1000, 10, 10, 2000)

# (ϕXgT, ϕXgT_Hopf_errU_reachi), _, opt_data = Hopf_BRS(system_errU_LE(1), target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, opt_p=opt_p, warm=true, check_all=true, printing=true, opt_tracking=true);

ϕXgT_Hopf_LE_reach = [];
for i=1:nle
    (ϕXgT, ϕXgT_Hopf_errU_reachi), _ = Hopf_BRS(system_errU_LE(i), target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, opt_p=opt_p, warm=false, check_all=true, printing=true);
    push!(ϕXgT_Hopf_LE_reach, ϕXgT_Hopf_errU_reachi);
end

ϕXgT_Hopf_LE_avoid = [];
for i=1:nle
    (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_LE(i), target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, opt_p=opt_p, warm=false, check_all=true,  printing=true);
    push!(ϕXgT_Hopf_LE_avoid, ϕXgT_Hopf_errD_avoidi);
end

ϕXgT_Hopf_TP_reach = [];
for i=1:ntp
    (ϕXgT, ϕXgT_Hopf_errU_reachi), _ = Hopf_BRS(system_errU_TP(i), targetri(i), T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, opt_p=opt_p, warm=false, check_all=true, printing=true);
    push!(ϕXgT_Hopf_TP_reach, ϕXgT_Hopf_errU_reachi);
end

ϕXgT_Hopf_TP_avoid = [];
for i=1:ntp
    (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_TP(i), targetai(i), T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, opt_p=opt_p, warm=false, check_all=true,  printing=true);
    push!(ϕXgT_Hopf_TP_avoid, ϕXgT_Hopf_errD_avoidi);
end
(_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_TP(ntp+1), targetai_ntpp1, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, opt_p=opt_p, warm=false, check_all=true,  printing=true);
push!(ϕXgT_Hopf_TP_avoid, ϕXgT_Hopf_errD_avoidi);

# plotlyjs()
# plot_BRS(T, fill(Xg, length(T)+1), ϕXgT_Hopf_errD_avoidi[2]; interpolate=true, value_fn=false)

## Plot Linear Ensemble Results

gr()
LE_pal = palette(:oslo10)[2:7][end:-1:1] #palette(:magma)[25:50:end], palette(:oslo10)[2:7], palette(:navia10)[2:7], palette(:lipari10)[1:5]
colors = [:black, pal[1], pal[2], LE_pal, pal[7]]
alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;

tix = 3; ti = T[tix];
dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_LE_reach) # true v lin + const err
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_LE_avoid)

LE_plots = []
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
    # labels = game == "avoid" ? (L"𝒯", L"\tilde{x}_i", L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{𝒰_{i} (τ)}} / ℛ_{δ^*_{𝒟_{i} (τ)}}^-") : fill("", 6)
    labels = game == "avoid" ? (L"𝒯", L"\tilde{x}_i", L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{𝒟_{i} (τ)}} / ℛ_{δ^*_{𝒟_{i} (τ)}}^-") : fill("", 6)
    title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"
    tixg = game == "reach" ? tix : tix-1

    LE_plot = plot(title=title, dpi=dpi);
    contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
   
    contour!(xig1..., reshape(ϕ_DP[tixg+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

    for i=1:nle
        labels_red = i == 1 ? labels[2] : ""
        plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(LE_plot), ylims=ylims(LE_plot), label=labels_red, alpha=0.3, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    for i=1:nle
        labels_red = i == 1 ? labels[6] : ""
        # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(LE_plot), ylims=ylims(LE_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

        # labels = i == 1 ? labels : fill("", length(labels))

        contour!(xig3..., reshape(ϕ_Hopf[i][tixg+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels_red, linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
        # contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.25:.25:1.25, (L".25", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=-1)
    plot!(xlims=(-0.8, .325), ylims=(0.05, 1.4))
    # plot!(xlims = (-2.5, -0.25), ylims = (-1, 3),legendfontsize=12,legend_columns=-1)
    # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    # plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
    # # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(-.25:.5:1.25, (L"-.25", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    # # plot!(xlims=(-0.8, .3), ylims=(-.25, 1.45))
    # lo, _ = collect(zip(xlims(LE_plot), ylims(LE_plot)))
    # locxl = lo .+ ((xlims(LE_plot)[2] - xlims(LE_plot)[1])/2, -0.1)
    # locyl = lo .+ (-0.1, (ylims(LE_plot)[2] - ylims(LE_plot)[1])/2)
    # annotate!(locxl..., L"x_1", fontsize=16)
    # annotate!(locyl..., L"x_2", fontsize=16)

    push!(LE_plots, LE_plot)
end

LE_plots_final = plot(LE_plots..., layout=(1,2), legend=(-1.25, -.165), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

## Plot Target Partition Results

gr()
TP_pal = palette(:acton10)[2:7][end:-1:1] #palette(:seaborn_deep6), palette(:magma)[25:50:end], palette(:oslo10)[2:7], palette(:navia10)[2:7], palette(:lipari10)[1:5], palette(:acton10)[2:6]
colors = [:black, pal[1], pal[2], TP_pal]
alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;

tix = 3; ti = T[tix];
dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_TP_reach) # true v lin + const err
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_TP_avoidd)

using ImageFiltering
smooth(X; gv=0.2) = imfilter(X, Kernel.gaussian(gv))

TP_plots = []
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
    # labels = game == "avoid" ? (L"𝒯", L"\tilde{x}_i", L"\hat{𝒮}", L"ℛ / ℛ^-", L"𝒯_i", L"ℛ_{δ^*_{𝒰_{i} (τ)}} / ℛ_{δ^*_{𝒟_{i} (τ)}}^-") : fill("", 6)
    labels = game == "avoid" ? (L"𝒯", L"\tilde{x}_i", L"\hat{𝒮}", L"ℛ / ℛ^-", L"𝒯_i", L"ℛ_{δ^*_{𝒟_{i} (τ)}} / ℛ_{δ^*_{𝒟_{i} (τ)}}^-") : fill("", 6)
    title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"
    tixg = game == "reach" ? tix : tix-1

    TP_plot = plot(title=title, dpi=dpi);
    contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
   
    contour!(xig1..., reshape(ϕ_DP[tixg+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

    ntpg = game == "reach" ? ntp : ntp
    for i=1:ntpg
        labels_red = i == 1 ? labels[5] : ""
        # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(TP_plot), ylims=ylims(TP_plot), label=labels[2], alpha=0.3, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
        # contour!(xig3..., reshape(ϕ_Hopf[i][1], res3, res3)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false)
        contour!(xig3..., reshape(ϕ_Hopf[i][1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=0.25*alpha, colorbar=false)
        plot!(Ball2(4*ones(2), 0.5), vars=(1,2), label=labels_red, linecolor=cgrad([colors[4][i], "white"], 7)[5], alpha=0.25*alpha, fillcolor=:white, lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    for i=1:ntpg
        labels_red = i == 1 ? labels[6] : ""
        # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(TP_plot), ylims=ylims(TP_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

        # labels = i == 1 ? labels : fill("", length(labels))

        contour!(xig3..., reshape(ϕ_Hopf[i][tixg+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
        # contour!(xig3..., reshape(ϕ_Hopf[i][tixg+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels_red, linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
        # contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    # plot!(xlims = (-2.5, -0.25), ylims = (-1, 3),legendfontsize=12,legend_columns=-1)
    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.25:.25:1.25, (L".25", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=-1)
    plot!(xlims=(-0.8, .33), ylims=(0.05, 1.4))
    # # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(-.25:.5:1.25, (L"-.25", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    # # plot!(xlims=(-0.8, .3), ylims=(-.25, 1.45))
    # lo, _ = collect(zip(xlims(TP_plot), ylims(TP_plot)))
    # locxl = lo .+ ((xlims(TP_plot)[2] - xlims(TP_plot)[1])/2, -0.1)
    # locyl = lo .+ (-0.1, (ylims(TP_plot)[2] - ylims(TP_plot)[1])/2)
    # annotate!(locxl..., L"x_1", fontsize=16)
    # annotate!(locyl..., L"x_2", fontsize=16)

    push!(TP_plots, TP_plot)
end

TP_plots_final = plot(TP_plots..., layout=(1,2), legend=(-1.25, -.165), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

LE_TP_combined_final = plot(LE_plots_final, TP_plots_final, layout=(1,2), size=(1200,400), bottom_margin=50Plots.px, left_margin=-5Plots.px)