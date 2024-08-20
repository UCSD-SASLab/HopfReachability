
include(pwd() * "/src/HopfReachability.jl");
include(pwd() * "/src/cons_lin_utils.jl");
include(pwd() * "/src/DP_comparison_utils.jl"); 
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_cd, make_target
using LaTeXStrings, ImageFiltering, JLD2, Plots

## Load Python & Pkgs

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/DP_comparison_files/");
hj_r_setup = pyimport("vdp_hj_reachability");
ss_set = hj.SolverSettings.with_accuracy("medium", hamiltonian_postprocessor=x->x);

model_path = pwd() * "/Examples/lifted_data/gen/models";
pushfirst!(pyimport("sys")."path", model_path);
np = pyimport("numpy");
pk = pyimport("pykoopman"); # requires PyCall.pyversion == v"3.9.12"

models = Dict("pol"=>Dict(), "rbf"=>Dict()); pkdt = 0.1;

## Load Polynomial Lifts

for i = 1:7
    model_name = "VanderPol_poly_deg$i.npy"
    model = np.load(joinpath(model_path, model_name), allow_pickle=true)[1]
    models["pol"][i] = Dict("A" => (model.A[[2:end...,1],[2:end...,1]] - I)/pkdt, "B" => model.B[[2:end...,1], :]/pkdt, "degree" => model.observables._max_degree, "nk" => size(model.A, 2), "nu" => size(model.B, 2), "nx" => size(model.C, 1))
end
using Combinatorics
Poly(x, d) = collect(Iterators.flatten((prod(y) for y in with_replacement_combinations(x, i)) for i = vcat(1:d,0)))
Ψpol(x; d) = Poly(x, d)

## Load Gaussian RBF Lifts

use_rbf_small = true
n_centers = use_rbf_small ? [3, 5, 9] : [9, 16, 25, 36, 49, 64, 81]
if use_rbf_small
    for i = 1:3 # small
        model_name = "VanderPol_rbf_gauss_nc$(n_centers[i])_kw3p0_small.npy"
        model = np.load(joinpath(model_path, model_name), allow_pickle=true)[1]
        models["rbf"][i] = Dict("A" => (model.A - I)/pkdt, "B" => model.B/pkdt, "n_centers" => model.observables.n_centers, "cx" => model.observables.centers, "kw"=> model.observables.kernel_width, "nk" => size(model.A, 2), "nu" => size(model.B, 2), "nx" => size(model.C, 1))
    end
else
    for i = 1:7
        model_name = "VanderPol_rbf_gauss_nc$(n_centers[i])rand_kw1p0.npy"
        model = np.load(joinpath(model_path, model_name), allow_pickle=true)[1]
        models["rbf"][i] = Dict("A" => (model.A - I)/pkdt, "B" => model.B/pkdt, "n_centers" => model.observables.n_centers, "cx" => model.observables.centers, "kw"=> model.observables.kernel_width, "nk" => size(model.A, 2), "nu" => size(model.B, 2), "nx" => size(model.C, 1))
    end
end
Ψrbf(x; cx, kw) = vcat(x, exp.(-(kw^2) * sum(eachrow((x .- cx).^2))))

## VanderPol Example

max_u = 0.25; max_d = 0.;
r = 1.0;

c𝒯, d𝒯, c₁ = [0.; 0.], [1., 1.], [0.]
Q𝒯, Q₁, Q₂ = inv(r) * diagm(d𝒯), inv(max_u) * diagm([1.]), inv(max_d) * diagm([1.])

nx = length(c𝒯);
xlimz, ylimz = (-3.5, 3.5), (-3.5, 3.5)
inputs = ((Q₁, c₁), (Q₂, c₁))
𝒯target = make_target(c𝒯, 1.; Q=Q𝒯)

μ = 1.0
function vanderpol!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = μ * (1.0 - x[1]^2) * x[2] - x[1] + x[3] + x[4]
    dx[3] = zero(x[3]) #control
    dx[4] = zero(x[4]) #disturbance
    return dx
end

## Solve Error (TS Linearization in Original State Space)

t = 1.1
δ̃ˢ, X̃, BRZ, dt, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; polyfit=false);

gr()
BRZ_plot = plot(BRZ, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (U & D)", legend=:bottomleft)
scatter!(BRZ_plot, eachrow(hcat(X̃.(dt)...)[1:2,:])..., label="x̃ backsolved w/ trivial ctrl/dist", alpha=0.6, xlims=xlimz, ylims=ylimz)

error_plot = plot(dt, δ̃ˢ[2,:], label="Taylor δˢ", xlabel="t")
plot(BRZ_plot, error_plot)

## Fixed Linear Model w/ Error on Lifted Feasible Only (Discrete Estimation)

model_mats(i, key) = (models[key][i]["A"], models[key][i]["B"], models[key][i]["B"], zeros(size(models[key][i]["A"], 2)))

error_plot_pol = plot(dt, δ̃ˢ[2,:], label="TS", title="Poly Lifted Errors for BRZ (U & ∅)", xlabel="t", ylabel="δˢ")
error_plot_rbf = plot(dt, δ̃ˢ[2,:], label="TS", title="RBF Lifted Errors for BRZ (U & ∅)", xlabel="t", ylabel="δˢ")
δ̃ˢ_pols, δ̃ˢ_rbfs = Dict(), Dict()

for j=1:7
    lifted_kwargs_pol = Dict(:Ψ => x->Ψpol(x; d=models["pol"][j]["degree"]), :lin_mat_fs=>model_mats(j, "pol"), :linear_graph=>zeros(models["pol"][j]["nk"]), :solve_dims=>collect(1:models["pol"][j]["nk"]), :error_method=>Lifted_Error_DiscreteAppx)    
    δ̃ˢ_pol, _, _, BRZ_pol, _ = apri_δˢ(vanderpol!, 𝒯target, inputs, t; lifted_kwargs_pol..., polyfit=false);
    models["pol"][j]["δˢ"] = δ̃ˢ_pol
    plot!(error_plot_pol, dt, δ̃ˢ_pol[2,:], label="(nk $(models["pol"][j]["nk"]), degree $(models["pol"][j]["degree"]))", lw=2) # plot!(error_plot_pol, dt, map(norm, eachcol(δ̃ˢ_pol)))
end

for j=1:7 #3
    lifted_kwargs_rbf = Dict(:Ψ => x->Ψrbf(x; cx=models["rbf"][j]["cx"], kw=models["rbf"][j]["kw"]), :lin_mat_fs=>model_mats(j, "rbf"), :linear_graph=>zeros(models["rbf"][j]["nk"]), :solve_dims=>collect(1:models["rbf"][j]["nk"]), :error_method=>Lifted_Error_DiscreteAppx)
    δ̃ˢ_rbf, _, _, BRZ_rbf, _ = apri_δˢ(vanderpol!, 𝒯target, inputs, t; lifted_kwargs_rbf..., polyfit=false);
    models["rbf"][j]["δˢ"] = δ̃ˢ_rbf
    plot!(error_plot_rbf, dt, δ̃ˢ_rbf[2,:], label="(nk $(models["rbf"][j]["nk"]), centers $(length(models["rbf"][j]["cx"])))", lw=2) # plot!(error_plot_rbf, dt, map(norm, eachcol(δ̃ˢ_rbf)))
end

plot(BRZ_plot, error_plot_pol, error_plot_rbf, layout=(1,3), plottitle="", size=(1600,500))

## Solve with DP

function δˢ_dc(s, δˢ_disc, solz)
    try
        δˢ_disc[:, end-findfirst(x->x<0, (s - 1e-5) .- vcat(0., high.(tspan.(solz))))] 
    catch y
        δˢ_disc[:, end-findfirst(x->x<0, s .- vcat(0., high.(tspan.(solz))))]
    end
end

Th=0.25
T = collect(Th:Th:1.0)

VdP_reach = hj_r_setup.VanderPol(mu=μ, max_u=max_u, max_d=max_d)
VdP_avoid = hj_r_setup.VanderPol(mu=μ, max_u=max_u, max_d=max_d, control_mode="max", disturbance_mode="min")

A, B₁, B₂, c = lin_mat_fs
E_s(s) = δˢ_dc(s, δ̃ˢ, BRZ) .* Matrix([0. 1.]') # in forward time

# must do iterative solve if tv lin (pycall + jax problem)
VdP_LTV_reach = s -> LinearError(A(X̃(-s)), B₁(X̃(-s)), B₂(X̃(-s)), c(X̃(-s)), E_s(s); max_u=max_u, max_d=max_d, Ushape="box", game="reach")
VdP_LTV_avoid = s -> LinearError(A(X̃(-s)), B₁(X̃(-s)), B₂(X̃(-s)), c(X̃(-s)), E_s(s); max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

dynamics = [VdP_reach, VdP_avoid];
dynamics_linear = [VdP_LTV_reach, VdP_LTV_avoid];

res=100; lbdp=(-4, -4); ubdp=(4, 4)
Xg_res1, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(d𝒯), r; shape="ball", lb=lbdp, ub=ubdp, res=res);
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

game = "avoid"
gix = game == "reach" ? 1 : 2

BRS_plot = plot(); colors = vcat("black", palette(["red", "blue"], length(T))...)
for i=1:length(T)+1; contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][i], res, res)', levels=[0], color=colors[i], lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz); end
display(BRS_plot)

BRS_plot_linear = plot(); whiter_colors = vcat("black", [palette(["white", c], 5)[2] for c in palette(["red", "blue"], length(T))]...)
for i=1:length(T)+1; contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[gix][i], res, res)', levels=[0], color=colors[i], lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz); end
display(BRS_plot_linear)

BRS_plots = []
for i=1:length(T)
    pli = plot(title="t=T-$(T[i])")
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][1], res, res)', levels=[0], color="black", lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][i+1], res, res)', levels=[0], color=colors[i+1], lw=2.5, colorbar=false)
    contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[gix][i+1], res, res)', levels=[0], color=whiter_colors[i+1], lw=2.5, colorbar=false)
    plot!(xlims=xlimz, ylims=ylimz, xlabel="x_1", ylabel="x_2")
    push!(BRS_plots, pli)
end
plot(BRS_plots..., layout=(1,4), size=(1600,500))

## Solve Augmented with Hopf

th=0.05; Th=0.25
T = collect(Th:Th:1.0)
res2 = 100
Xg, _, _, xig2 = hjr_init(c𝒯, Q𝒯, 1; shape="ball", lb=(-3,-4), ub=(3,4), res=res2);

target_check_plot_pol = contour(xig1..., reshape(ϕXgT_DP_dynamics[1][1], res, res)', levels=[0], color=:black, lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz);
target_check_plot_rbf = contour(xig1..., reshape(ϕXgT_DP_dynamics[1][1], res, res)', levels=[0], color=:black, lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz);
target_check_plots = Dict("pol"=>target_check_plot_pol, "rbf"=>target_check_plot_rbf)

for key in ["pol", "rbf"]
    for j = 1:4

        ## Augmented System
        A, B₁, B₂, c = model_mats(j, key); 
        E_s(s) = diagm(δˢ_dc(s, models[key][j]["δˢ"], BRZ)) # in forward time
        lifted_system_err = (A, B₁, B₂, Q₁, c₁, 0*Q₁, c₁, c, E_s);

        ## Points To Solve
        Ψj = key == "pol" ? x -> Ψpol(x; d=models["pol"][j]["degree"]) : x -> Ψrbf(x; cx=models["rbf"][j]["cx"], kw=models["rbf"][j]["kw"]) 
        Gg_man = hcat(Ψj.(eachcol(Xg))...)

        ## Augmented Target
        η = key == "pol" ? 10 : 10; # η = key == "pol" ? 20 : 1.5;
        r_aug = key == "pol" ? 0.9r : 0.9r # r_aug = key == "pol" ? 0.8r : r
        c𝒯_aug = Ψj(c𝒯)
        Q𝒯_aug = inv(r_aug) * diagm(vcat(d𝒯, η*ones(models[key][j]["nk"] - nx)))

        lifted_target = make_target(c𝒯_aug, 1.; Q=Q𝒯_aug);
        gr(); contour!(target_check_plots[key], xig2..., reshape(lifted_target[1](Gg_man), res2, res2)', levels=[0], color=palette(:default)[j+1], lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz)
        
        ## Solve
        vh = 0.01; L = 20; tol = 1e-3; step_lim = 200; re_inits = 5; max_runs = 20; max_its = 1000
        opt_p_cd = (vh, L, tol, step_lim, re_inits, max_runs, max_its)
        results_avoid_cd, _ = Hopf_BRS(lifted_system_err, lifted_target, T; th, Xg=Gg_man, error=true, game="avoid", opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true, warm_pattern="previous");
        models[key][j]["results_avoid_cd"] = results_avoid_cd
    end
end

plot(target_check_plots["pol"], target_check_plots["rbf"])

gr(); 
BRS_plot_pol = plot(); key = "pol"
contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][1], res, res)', levels=[0], color=:black, lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz);
contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][4], res, res)', levels=[0], lw=2.5, color=:blue, colorbar=false, xlims=xlimz, ylims=ylimz, label="True")
contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[gix][4], res, res)', levels=[0], lw=2.5, color=palette(:default)[1], colorbar=false, xlims=xlimz, ylims=ylimz, label="TS")
for j=1:4; contour!(xig2..., reshape(models[key][j]["results_avoid_cd"][2][end], res2, res2)', color=palette(:default)[j+1], levels=[0], lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz, label="(nk $(models["pol"][j]["nk"]), degree $(models["pol"][j]["degree"]))"); end
display(BRS_plot_pol)

BRS_plot_rbf = plot(); key = "rbf"
contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][1], res, res)', levels=[0], color=:black, lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz);
contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][4], res, res)', levels=[0], lw=2.5, color=:blue, colorbar=false, xlims=xlimz, ylims=ylimz, label="True")
contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[gix][4], res, res)', levels=[0], lw=2.5, color=palette(:default)[1], colorbar=false, xlims=xlimz, ylims=ylimz, label="TS")
for j=1:3; contour!(xig2..., reshape(models[key][j]["results_avoid_cd"][2][end], res2, res2)', color=palette(:default)[j+1], levels=[0], lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz, label="(nk $(models["rbf"][j]["nk"]), centers $(length(models["rbf"][j]["cx"])))"); end
display(BRS_plot_rbf)

plot(BRS_plot_pol, BRS_plot_rbf, legend=true)

## Save and/or Load 

# jldsave(pwd() * "/Examples/lifted/results/models.jld2"; models)
models = load(pwd() * "/Examples/lifted_data/results/models_poly_rbfs_refined.jld2")["models"]

## Fancy Single Plot 

gr();
res2 = 100;
dpi=300; lw=2.5; legend_hfactor=2.0; alpha=0.8;
smooth(X; gv=1.0) = imfilter(X, Kernel.gaussian(gv)) 
palettes = Dict("pol" => palette(:blues, 4), "rbf" => palette(:reds,4))
title = L"\textrm{Safe\:\:Envelopes\:\:of\:\:Lifted\:\:Linear\:\:Models}"

BRS_plot_nice = plot(dpi=dpi, title=title); 
contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][1], res, res)', levels=[0], color=:black, lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz, alpha=alpha,);
plot!([100, 100], [110, 110], color=:black, label = L"\textrm{Target}", legend_hfactor=legend_hfactor, alpha=alpha,)

contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][5], res, res)', levels=[0], lw=lw, color=palette(:seaborn_colorblind)[2], colorbar=false, xlims=xlimz, ylims=ylimz, label=L"\text{True}", alpha=alpha,)
plot!([100, 100], [110, 110], color=palette(:seaborn_colorblind)[2], label = L"\textrm{True\:Avoid\:Set}", legend_hfactor=legend_hfactor, lw=lw, alpha=alpha,)

contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[gix][5], res, res)', levels=[0], lw=lw, color=palette(:seaborn_colorblind)[3], colorbar=false, xlims=xlimz, ylims=ylimz, label=L"\text{TS}", alpha=alpha,)
plot!([100, 100], [110, 110], color=palette(:seaborn_colorblind)[3], label = L"\textrm{TS}", legend_hfactor=legend_hfactor, lw=lw, alpha=alpha,)

contour!(xig2..., reshape(models["pol"][1]["results_avoid_cd"][2][end], res2, res2)', color=palette(:seaborn_colorblind)[5], levels=[0], lw=lw, colorbar=false, xlims=xlimz, ylims=ylimz, label = L"\text{DMD}", alpha=alpha,);
plot!([100, 100], [110, 110], color=palette(:seaborn_colorblind)[5], label = L"\textrm{DMD}", legend_hfactor=legend_hfactor, lw=lw, alpha=alpha,)

for key in ["rbf", "pol"]
    for j=2:4; 
        smoothed = key == "pol" ? X -> smooth(X; gv=2.) : X -> X
        jj = key == "pol" ? j : j-1
        Φ = reshape(models[key][jj]["results_avoid_cd"][2][end], res2, res2)'
        contour!(xig2..., smoothed(Φ), 
                color=palettes[key][jj], 
                levels=[0], lw=lw, colorbar=false, xlims=xlimz, ylims=ylimz, alpha=alpha,
                ); 
        plot!([100, 100], [110, 110], color = palettes[key][jj], legend_hfactor=legend_hfactor, alpha=alpha, lw=lw,
              label = key == "pol" ? L"\textrm{Poly}-" * latexstring(models[key][jj]["degree"]) * L" \quad (n_k = " * latexstring(models[key][jj]["nk"]) * L")" : 
                                     L"\textrm{RBF}-" *latexstring(models[key][jj]["nk"]-2)* L" \quad (n_k = " * latexstring(models[key][jj]["nk"]) * L")")
    end
end

plot!(xticks=(-2:1:2, [abs(xi)==2 ? latexstring(xi) : "" for xi in -2:1:2]), yticks=(-3:1:3, [abs(xi)==3 ? latexstring(xi) : "" for xi in -3:1:3]), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14)
plot!(xlims=1.5 .* (-2, 2), ylims= 1.1 .* ylimz .- (0.5, 0.))
lo, _ = collect(zip(xlims(BRS_plot_nice), ylims(BRS_plot_nice)))
locxl = lo .+ ((xlims(BRS_plot_nice)[2] - xlims(BRS_plot_nice)[1])/2, -0.5)
locyl = lo .+ (-0.35, (ylims(BRS_plot_nice)[2] - ylims(BRS_plot_nice)[1])/2)
annotate!(locxl..., L"x_1", fontsize=16)
annotate!(locyl..., L"x_2", fontsize=16)
plot!(legend=(2.5, -1.5), background_color_legend=RGBA(225,225,225,0.6), foreground_color_legend=nothing, legendfontsize=6, legend_columns=3)
plot!(legend=:bottom, size=(400, 400), dpi=300, title="")

savefig(BRS_plot_nice, pwd() * "/Examples/figures/VdPlifted_BRS_plot_single.png")

## Fancy Double Plot

gr();
dpi=300; lw=2.5; legend_hfactor=2.0; alpha=0.8
palettes = Dict("pol" => palette(:blues, 4), "rbf" => palette(:reds,4))
colors = vcat(palette(:seaborn_colorblind)[10], palette(:seaborn_dark)[1], palette(:seaborn_colorblind)[7], palette(:reds, 4)[3]) 
labels = [L"\textrm{Poly}_3 \:\: (n_k = 10)", L"\textrm{Poly}_4 \:\: (n_k = 15)", L"\textrm{RBF}_5 \:\: (n_k = 7)", L"\textrm{RBF}_9 \:\: (n_k = 11)"]
titles = [L"\textrm{Polynomials}", L"\textrm{Radial\:\:Basis\:\:Functions}"]
BRS_plots = []
tix = 4

for i=1:3
    BRS_plot_nice = plot(dpi=dpi); if i != 3; plot!(title=titles[i]); end
    if i != 3; contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][1], res, res)', levels=[0], color=:black, lw=2.5, colorbar=false, xlims=xlimz, ylims=ylimz, alpha=alpha,); end
    if i == 3; plot!([100, 100], [110, 110], color=:black, label = L"\textrm{Target}", legend_hfactor=legend_hfactor, alpha=alpha, lw=lw); end

    if i != 3; contour!(xig1..., reshape(ϕXgT_DP_dynamics[gix][tix+1], res, res)', levels=[0], lw=lw, color=palette(:seaborn_colorblind)[2], colorbar=false, xlims=xlimz, ylims=ylimz, label=L"\text{True}", alpha=alpha,); end
    if i == 3; plot!([100, 100], [110, 110], color=palette(:seaborn_colorblind)[2], label = L"\textrm{True\:\:Avoid\:\:Set}", legend_hfactor=legend_hfactor, lw=lw, alpha=alpha,); end

    if i != 3; contour!(xig1..., reshape(ϕXgT_DP_dynamics_linear[gix][tix+1], res, res)', levels=[0], lw=lw, color=palette(:seaborn_colorblind)[3], colorbar=false, xlims=xlimz, ylims=ylimz, label=L"\text{TS}", alpha=alpha,); end
    if i == 3; plot!([100, 100], [110, 110], color=palette(:seaborn_colorblind)[3], label = L"\textrm{Taylor\:\:Series}", legend_hfactor=legend_hfactor, lw=lw, alpha=alpha,); end

    if i != 3; contour!(xig2..., reshape(models["pol"][1]["results_avoid_cd"][2][tix+1], res2, res2)', color=palette(:seaborn_dark)[6], levels=[0], lw=lw, colorbar=false, xlims=xlimz, ylims=ylimz, label = L"\text{DMD}", alpha=alpha,); end
    if i == 3; plot!([100, 100], [110, 110], color=palette(:seaborn_dark)[6], label = L"\textrm{DMD}", legend_hfactor=legend_hfactor, lw=lw, alpha=alpha,); end

    c = 1;
    for key in ["pol", "rbf"]
        for j=3:4; 
            smoothed = key == "pol" ? X -> smooth(X; gv=2.) : X -> X
            jj = key == "pol" ? j : j-1
            Φ = reshape(models[key][jj]["results_avoid_cd"][2][tix+1], res2, res2)'
            if i == 1 && key == "pol"
                contour!(xig2..., smoothed(Φ), 
                    color = colors[c],  
                    levels=[0], lw=lw, colorbar=false, xlims=xlimz, ylims=ylimz, alpha=alpha,
                    );
            elseif i == 2 && key == "rbf"
                contour!(xig2..., smoothed(Φ), 
                    color = colors[c],  
                    levels=[0], lw=lw, colorbar=false, xlims=xlimz, ylims=ylimz, alpha=alpha,
                    );
            elseif i == 3
                plot!([100, 100], [110, 110], 
                color = colors[c], 
                legend_hfactor=legend_hfactor, alpha=alpha, lw=lw,
                label = labels[c]); 
            end
            c += 1
        end
    end
    
    if i != 3;
        plot!(xlims=1.5 .* (-2, 2), ylims= 1.1 .* ylimz .- (0.5, 0.))
        lo, _ = collect(zip(xlims(BRS_plot_nice), ylims(BRS_plot_nice)))
        locxl = lo .+ ((xlims(BRS_plot_nice)[2] - xlims(BRS_plot_nice)[1])/2, -0.5)
        locyl = lo .+ (-0.35, (ylims(BRS_plot_nice)[2] - ylims(BRS_plot_nice)[1])/2)
        annotate!(locxl..., L"x_1", fontsize=16)
        if i == 1
            plot!(xticks=(-2:1:2, [abs(xi)==2 ? latexstring(xi) : "" for xi in -2:1:2]), yticks=(-3:1:3, [abs(xi)==3 ? latexstring(xi) : "" for xi in -3:1:3]), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14)
            annotate!(locyl..., L"x_2", fontsize=16)
        else
            plot!(xticks=(-2:1:2, [abs(xi)==2 ? latexstring(xi) : "" for xi in -2:1:2]), yticks=(-3:1:3, ["" for xi in -3:1:3]), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14)
        end
        plot!(legend=false)
    else
        plot!(xlims=1.5 .* (-2, 2), ylims= 1.1 .* ylimz .- (0.5, 0.), legend=(0.02, 0.75))
        plot!(show_axis=false, yaxis=false, xaxis=false, grid=false, background_color_legend=RGBA(225,225,225,0.3), foreground_color_legend=nothing, legendfontsize=9, legend_columns=4)
    end

    push!(BRS_plots, BRS_plot_nice)
end

BRS_plot_nice = plot(BRS_plots..., layout=@layout[a b; c{0.08h}], size = (600, 500), legend=(0.04, 0.75), bottom_margin=2Plots.px)

savefig(BRS_plot_nice, pwd() * "/Examples/figures/VdPlifted_BRS_plot_double.png")