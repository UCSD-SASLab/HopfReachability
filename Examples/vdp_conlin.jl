
include(pwd() * "/src/HopfReachability.jl");
include(pwd() * "/src/cons_lin_utils.jl");
include(pwd() * "/src/DP_comparison_utils.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_levelset_fs
using ReachabilityAnalysis, Plots, LaTeXStrings, ImageFiltering

## VanderPol Example

r, max_u, max_d = 0.25, 1.0, 0.5 

c₁, c₂, c𝒯 = [0.], [0.], [0.; 1.]
Q₁, Q₂, Q𝒯 = inv(max_u) * diagm([1.]), inv(max_d) * diagm([1.]), inv(r) * diagm([1., 1.])
inputs, 𝒯target = ((Q₁,c),(Q₂,c)), (0,0,(Q𝒯,c𝒯))
nx = length(c𝒯);
 
μ = 1.0
function vanderpol!(dx, x, p, t) 
    # ReachabilityAnalysis.jl model style
    dx[1] = x[2]
    dx[2] = μ * (1.0 - x[1]^2) * x[2] - x[1] + x[3] + x[4]
    dx[3] = zero(x[3]) #control
    dx[4] = zero(x[4]) #disturbance
    return dx
end

## Solve Feasible Sets

t = 0.39
δ̃ˢ, X̃, BRZ, dt, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t);
δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D");

BRZ_plot = plot(BRZ, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (U & D)", legend=:bottomleft)
plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (D)")
scatter!(BRZ_plot, eachrow(hcat(X̃.(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃ backsolved w/ trivial ctrl/dist", alpha=0.6)

error_plot = plot(dt, δ̃ˢ[2].(dt), label="Taylor δˢ for BRZ (U & D), x̃", xlabel="t")
plot!(error_plot, dtD, δ̃ˢD[2].(dtD), label="Taylor δˢ for BRZ (D), x̃")

plot(BRZ_plot, error_plot)

## Solve HJR with Nonlinear DP

Th = 0.13
T = collect(Th:Th:t)

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/DP_comparison_files");
hj_r_setup = pyimport("vdp_hj_reachability");

VdP_reach = hj_r_setup.VanderPol(mu=1.0, max_u=max_u, max_d=max_d)
VdP_avoid = hj_r_setup.VanderPol(mu=μ, max_u=max_u, max_d=max_d, control_mode="max", disturbance_mode="min")

A, B₁, B₂, c = lin_mat_fs
B1 = Matrix([0. 1.]'); B2 = Matrix([0. 1.]')
E_c(s) = δ̃ˢ[2](-s) * Matrix([0. 1.]') # in forward time

dynamics_linear_reach = LinearError(A(X̃(0)), B1, B2, c(X̃(0)), E_c; max_u=max_u, max_d=max_d, Ushape="box", game="reach") # jax cant handle julia fns, so must do iterative solve if tv lin
dynamics_linear_avoid = LinearError(A(X̃(0)), B1, B2, c(X̃(0)), E_c; max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

dynamics_reach = [VdP_reach, dynamics_linear_reach];
dynamics_avoid = [VdP_avoid, dynamics_linear_avoid]; 
res=300 # DP inaccurate with less

Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(ones(nx)), r; shape="ball", lb=(-2, -2), ub=(2, 2), res=res)
ϕXgT_DP_dynamics_reach = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_reach, T; BRS=true, one_shot=true);
ϕXgT_DP_dynamics_avoid = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_avoid, T; BRS=true, one_shot=true);

## Solve HJR with Linear Hopf + Error

th=0.0325

A, B₁, B₂, c = lin_mat_fs
B1 = Matrix([0. 1.]'); B2 = Matrix([0. 1.]')
Eδc(s) = δ̃ˢ[2](-t) * diagm([0, 1]) # constant error
Eδt(s) = δ̃ˢ[2](-s) * diagm([0, 1]) # tv error
EδD(s) = δ̃ˢD[2](-s) * diagm([0, 1]) # tv error, dist feas

system_errc = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), Eδc);
system_errt = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), Eδt);
system_errD = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), EδD);

lb, ub, res2 = 1.1 .* (-ρ(-[1,0,0,0], BRZ), -ρ(-[0,1,0,0], BRZ)), (ρ([1,0,0,0], BRZ), ρ([0,1,0,0], BRZ)), 50
Xg, _, _, xig2 = hjr_init(c𝒯, Q𝒯, 1; shape="ball", lb=lb, ub=ub, res=res2);

target = (make_levelset_fs(cₓ, r; Q=Qₓ), (Qₓ, cₓ));

vh, stepsz, tol, stepszstep_its, conv_runs_rqd, max_runs, max_its = 0.01, 5, 1e-3, 100, 20, 20, 400
opt_p_cd = (vh, stepsz, tol, stepszstep_its, conv_runs_rqd, max_runs, max_its)

(ϕXgT, ϕXgT_Hopf_errc_reach), _ = Hopf_BRS(system_errc, target, T; th, Xg=Xg, error=true, game="reach", opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errc_avoid),    _ = Hopf_BRS(system_errc, target, T; th, Xg=Xg, error=true, game="avoid", opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errt_reach),    _ = Hopf_BRS(system_errt, target, T; th, Xg=Xg, error=true, game="reach", opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errt_avoid),    _ = Hopf_BRS(system_errt, target, T; th, Xg=Xg, error=true, game="avoid", opt_p=opt_p_cd, warm=false, check_all=true, printing=true);
(_, ϕXgT_Hopf_errD_reach),    _ = Hopf_BRS(system_errD, target, T; th, Xg=Xg, error=true, game="reach", opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
(_, ϕXgT_Hopf_errD_avoid),    _ = Hopf_BRS(system_errD, target, T; th, Xg=Xg, error=true, game="avoid", opt_p=opt_p_cd, warm=false, check_all=true, printing=true);

## Plot Single BRZ vs. BRS (at t) - Constant Error

pal = palette(:seaborn_colorblind)
colors = [:black, pal[1], pal[2], pal[3], "gray"]
alpha = 0.7; lw=2.5; legend_hfactor=0.7; dpi=300;
tix = 2; ti = T[tix];
dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errc_reach) # true v lin + const err
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errc_avoid)
smooth(X; gv=1.0) = imfilter(X, Kernel.gaussian(gv))

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

    contour!(xig2..., smooth(reshape(ϕ_Hopf[tix+1], res2, res2)'), levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
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

## Plot Multiple BRZ vs. BRS - Time-Varying Error

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
    
    for (i, ti) in enumerate(T)
        dtix = findfirst(x -> x .+ ti > 0, dt) - 1;

        labels = i == 1 ? labels : fill("", length(labels))
        plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);

        contour!(xig1..., reshape(ϕ_DP[i+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
        contour!(xig2..., smooth(reshape(ϕ_Hopf2[i+1], res2, res2)'), levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
    plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
    lo, _ = collect(zip(xlims(multi_plot), ylims(multi_plot)))
    locxl = lo .+ ((xlims(multi_plot)[2] - xlims(multi_plot)[1])/2, -0.1)
    locyl = lo .+ (-0.1, (ylims(multi_plot)[2] - ylims(multi_plot)[1])/2)
    annotate!(locxl..., L"x_1", fontsize=16)
    annotate!(locyl..., L"x_2", fontsize=16)

    push!(multi_plots, multi_plot)
end

multi_plots_final = plot(multi_plots..., layout=(1,2), legend=(-0.925, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

## Plot Multiple BRZ - Disturbance Only + TV

gr()
colors = [:black, pal[1], pal[2], pal[3], pal[5], pal[10]]
alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.9; dpi=300;
reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errt_reach, ϕXgT_Hopf_errD_reach)
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errt_avoid, ϕXgT_Hopf_errD_avoid)
labelss = [L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-",  L"\hat{𝒮}_D / \hat{𝒮}_D", L"ℛ_{δ^*_{D(τ)}} / ℛ_{δ^*_{D(τ)}}^-"]

multi_plotsD = [];
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf, ϕ_Hopf2 = game == "reach" ? reach_comp : avoid_comp
    labels = game == "avoid" ? labelss : fill("", 7)
    title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"
    BRZ2 = game == "reach" ? BRZd : BRZd

    multi_plot = plot(title=title, dpi=dpi);
    contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
    plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
    for (i, ti) in enumerate(T)
        dtix = findfirst(x -> x .+ ti > 0, dt) - 1;

        labels = i == 1 ? labels : fill("", length(labels))
        plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);

        contour!(xig1..., reshape(ϕ_DP[i+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

        contour!(xig2..., smooth(reshape(ϕ_Hopf[i+1], res2, res2)'), levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
        contour!(xig2..., smooth(reshape(ϕ_Hopf2[i+1], res2, res2)'), levels=[0], color=colors[5], lw=lw, alpha=alpha, colorbar=false);
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[7], linecolor=colors[5], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=3)
    plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
    lo, _ = collect(zip(xlims(multi_plot), ylims(multi_plot)))
    locxl = lo .+ ((xlims(multi_plot)[2] - xlims(multi_plot)[1])/2, -0.1)
    locyl = lo .+ (-0.1, (ylims(multi_plot)[2] - ylims(multi_plot)[1])/2)

    annotate!(locxl..., L"x_1", fontsize=16)
    annotate!(locyl..., L"x_2", fontsize=16)

    push!(multi_plotsD, multi_plot)
end

multi_plotsD_final = plot(multi_plotsD..., layout=(1,2), legend=(-0.7, -.155), bottom_margin=60Plots.px, background_color_legend = nothing, foreground_color_legend = nothing, dpi=dpi,legendfontsize=12)

### Target Partitioning

ntp = 5; θtp = 2π/ntp

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

δ̃ˢD_TPa, X̃_TPa, BRZd_TPa, dtD_TPa = [], [], [], []

for i=1:ntp
    𝒯targetri = (nothing, nothing, (inv(rr) * diagm([1., 1.]), c𝒯rri(i)))    
    𝒯targetai = (nothing, nothing, (inv(ra) * diagm([1., 1.]), c𝒯rai(i)))
    δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetai, inputs, t; zono_over="D");
    push!(δ̃ˢD_TPa, δ̃ˢD); push!(X̃_TPa, X̃); push!(BRZd_TPa, BRZd); push!(dtD_TPa, dtD)
end
𝒯targetai = (nothing, nothing, (inv(ra) * diagm([1., 1.]), c𝒯)) #extra avoid to cover middle
δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetai, inputs, t; zono_over="D");
push!(δ̃ˢD_TPa, δ̃ˢD); push!(X̃_TPa, X̃); push!(BRZd_TPa, BRZd); push!(dtD_TPa, dtD)

δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D");

BRZ_plot_a = plot(BRZd, vars=(1,2), alpha=0.1, lw=3, label="BRZ (D, 𝒯)", legend=:bottomleft);
for i=1:ntp
    plot!(BRZ_plot_a, BRZd_TPa[i], vars=(1,2), alpha=0.5, lw=1, label="BRZ (D, 𝒯p$i)")
end
plot!(BRZ_plot_a, BRZd_TPa[ntp+1], vars=(1,2), alpha=0.5, lw=1, label="BRZ (D, 𝒯p$(ntp+1))") #extra avoid to cover middle

### Solve w/ Various x̃ for one target (Linear Ensemble)

nle = 5; nu = 1; nd = 1;
rle = 1r
X̃0 = [Rot(c𝒯, c𝒯 + [0, rle], i , θtp) for i=1:nle] # reach centers
X̃0ŨD̃_LE = (X̃0, # circle ×
        [(y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu)], 
        [(y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd)])
δ̃ˢD_LE, X̃_LE, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D", X̃0ŨD̃=X̃0ŨD̃_LE);
nle = length(X̃0ŨD̃_LE[1])

BRZ_plot = plot(BRZd, vars=(1,2), alpha=0.3, lw=3, label="BRZ (U)", legend=:bottomleft)
for i=1:nle
    scatter!(BRZ_plot, eachrow(hcat(X̃_LE[i].(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃$i", alpha=0.3)
end
plot!()

plot(); for i=1:nle; plot!(dtD, δ̃ˢD_LE[i][2].(dtD)); end; plot!()

## Solve with Hopf

EδD_LE(i, s) = δ̃ˢD_LE[i][2](-s) * diagm([0, 1])
EδD_TP(i, s) = δ̃ˢD_TPa[i][2](-s) * diagm([0, 1])

system_errD_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδD_LE(i,s));
system_errD_TP(i) = (s -> A(X̃_TPa[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPa[i](-s)), s -> EδD_TP(i,s));

targetri(i) = (make_levelset_fs(c𝒯rri(i), rr; Q=I(nx))..., (I(nx), c𝒯rri(i)));
targetai(i) = (make_levelset_fs(c𝒯rri(i), ra; Q=I(nx))..., (I(nx), c𝒯rri(i)));
targetai_ntpp1 = (make_levelset_fs(c𝒯, ra; Q=I(nx))..., (diagm(ones(nx)), c𝒯));

lb = (1.1 * -ρ(-[1,0,0,0], BRZu), -ρ(-[0,1,0,0], BRZu))
ub = (1.65 * ρ([1,0,0,0], BRZu), 1.25 * ρ([0,1,0,0], BRZu))

res3 = 100
Xg, _, _, xig3 = hjr_init(c𝒯, Q𝒯, r; shape="box", lb=lb, ub=ub, res=res3);

opt_p = (0.01, 2, 1e-4, 1000, 10, 10, 2000)

ϕXgT_Hopf_LE_reach = [];
for i=1:nle
    (ϕXgT, ϕXgT_Hopf_errD_reachi), _ = Hopf_BRS(system_errD_LE(i), target, T; th, Xg=Xg, error=true, game="reach", opt_p=opt_p_cd, opt_p=opt_p, warm=false, check_all=true, printing=true);
    push!(ϕXgT_Hopf_LE_reach, ϕXgT_Hopf_errD_reachi);
end

ϕXgT_Hopf_LE_avoid = [];
for i=1:nle
    (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_LE(i), target, T; th, Xg=Xg, error=true, game="avoid", opt_p=opt_p_cd, opt_p=opt_p, warm=false, check_all=true,  printing=true);
    push!(ϕXgT_Hopf_LE_avoid, ϕXgT_Hopf_errD_avoidi);
end

ϕXgT_Hopf_TP_reach = [];
for i=1:ntp
    (ϕXgT, ϕXgT_Hopf_errD_reachi), _ = Hopf_BRS(system_errD_TP(i), targetri(i), T; th, Xg=Xg, error=true, game="reach", opt_p=opt_p_cd, opt_p=opt_p, warm=false, check_all=true, printing=true);
    push!(ϕXgT_Hopf_TP_reach, ϕXgT_Hopf_errD_reachi);
end

ϕXgT_Hopf_TP_avoid = [];
for i=1:ntp
    (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_TP(i), targetai(i), T; th, Xg=Xg, error=true, game="avoid", opt_p=opt_p_cd, opt_p=opt_p, warm=false, check_all=true,  printing=true);
    push!(ϕXgT_Hopf_TP_avoid, ϕXgT_Hopf_errD_avoidi);
end
(_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_TP(ntp+1), targetai_ntpp1, T; th, Xg=Xg, error=true, game="avoid", opt_p=opt_p_cd, opt_p=opt_p, warm=false, check_all=true,  printing=true);
push!(ϕXgT_Hopf_TP_avoid, ϕXgT_Hopf_errD_avoidi);

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
    labels = game == "avoid" ? (L"𝒯", L"\tilde{x}_i", L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{D_{i} (τ)}} / ℛ_{δ^*_{D_{i} (τ)}}^-") : fill("", 6)
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

        contour!(xig3..., reshape(ϕ_Hopf[i][tixg+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels_red, linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
      end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.25:.25:1.25, (L".25", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=-1)
    plot!(xlims=(-0.8, .325), ylims=(0.05, 1.4))

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
avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_TP_avoid)

TP_plots = []
for game in ["reach", "avoid"]

    ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
    labels = game == "avoid" ? (L"𝒯", L"\tilde{x}_i", L"\hat{𝒮}", L"ℛ / ℛ^-", L"𝒯_i", L"ℛ_{δ^*_{D_{i} (τ)}} / ℛ_{δ^*_{D_{i} (τ)}}^-") : fill("", 6)
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
        contour!(xig3..., reshape(ϕ_Hopf[i][1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=0.25*alpha, colorbar=false)
        plot!(Ball2(4*ones(2), 0.5), vars=(1,2), label=labels_red, linecolor=cgrad([colors[4][i], "white"], 7)[5], alpha=0.25*alpha, fillcolor=:white, lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    end

    for i=1:ntpg
        labels_red = i == 1 ? labels[6] : ""
        contour!(xig3..., reshape(ϕ_Hopf[i][tixg+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
        plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels_red, linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
      end

    plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.25:.25:1.25, (L".25", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=-1)
    plot!(xlims=(-0.8, .33), ylims=(0.05, 1.4))

    push!(TP_plots, TP_plot)
end

TP_plots_final = plot(TP_plots..., layout=(1,2), legend=(-1.25, -.165), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

LE_TP_combined_final = plot(LE_plots_final, TP_plots_final, layout=(1,2), size=(1200,400), bottom_margin=50Plots.px, left_margin=-5Plots.px)