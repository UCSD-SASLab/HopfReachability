#   Koopman-Hopf Hamilton Jacobi Reachability
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   wsharpless@ucsd.edu
#   ---------------------


using Pkg
ENV["PYTHON"] = "/Users/willsharpless/miniconda3/envs/pythreeten/bin/python" # need python 3.10 for AutoKoopman 
Pkg.build("PyCall") 

using PyCall, Plots, LinearAlgebra, JLD, LaTeXStrings
home = "/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/"

using LSHFunctions: jaccard

##   Import AutoKoopman
##   ==================

tsm_loc = home * "Code_from_others/AutoKoopman/autoKoopman_codeocean/code"
pushfirst!(pyimport("sys")."path", tsm_loc);
tsm = pyimport("test_saved_models")

filename = home * "Code_from_others/AutoKoopman/autoKoopman_codeocean/experimentsSymbolic_vdp_duff.pickle"

Duffing = tsm.loadedModels(filename=filename)
Duffing.set("Duffing", "poly")

# poly(x) = [x1, x2, x1^2, x1x2, x2^2, x1^3, x1^2x2, x1x2^2, x2^3, x1^4, x1^3x2, x1^2x2^2, x1x2^3, x2^4]

##   Our HopfReachability Pkg
##   ========================

include(home * "HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, plot_BRS, Hopf_minT, Hopf_admm_cd


##   Initialize
##   ==========

nx = 2;
nu = 1; 
disturbance_on_control = false; # applies disturbance only to controlled states (allows us to use same Koopman L)
Max_u, Max_d = 1.0, 0.5; # => u ∈ [-0.1 ,  0.1], d ∈ [-0.1 ,  0.1] 

## System (Koopman)
# ẋ = Kx + Lu + L2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

Kd, Ld = Duffing.get_matrices(); dt = Duffing.curr_dict["samp_period"];
K, L = (Kd - I)/dt, Ld/dt; nk = size(K)[1]; 

nd = disturbance_on_control ? nu : size(L)[2] - nu; #should be nx
Lu, Ld = disturbance_on_control ? (L[:, 1:nu], L[:, 1:nu]) : (L[:, 1:nu], L[:, nu+1:end]);

Q1 = diagm(ones(nu)); Q10 = zero(Q1); a1 = zeros(1, nu); 
Q2 = diagm(ones(nd)); Q20 = zero(Q2); a2 = zeros(1, nd);

system_Koopman = (K, Max_u * Lu, Max_d * Ld, Q1, Q2, a1, a2); # Controlled and Disturbed

## System (Taylor Approx at Target Center)

M = [0 1; 1.0 -0.1]
C1 = Matrix([0. 1]')

nd = disturbance_on_control ? nu : nx;
C2 = disturbance_on_control ? copy(C1) : C2 = diagm(ones(nd));

system_Taylor = (M, Max_u * C1, Max_d * C2, Q1, Q2, a1, a2); # Controlled and Disturbed

## Target
# J(x) = 0 is the boundary of the targetx

cp_Koop = zeros(nk);
cp_Taylor = zeros(nx);

r = 1.0; 

η = 8; Ap_Koop = diagm(vcat(η * ones(nx), 0.125 * ones(nk-nx))); rp_Koop = sqrt(η) * r # Ellipsoidal relaxation, big numbers;

Ap_Taylor = diagm(ones(nx)); rp_Taylor = 1 * r # Ball

J_Koop(g::Vector, A, c; r=rp_Koop) = ((g - c)' * A * (g - c))/2 - 0.5 * r^2;
Js_Koop(v::Vector, A, c; r=rp_Koop) = (v' * inv(A) * v)/2 + c'v + 0.5 * r^2; # Convex Conjugate
J_Koop(g::Matrix, A, c; r=rp_Koop) = diag((g .- c)' * A * (g .- c))/2 .- 0.5 * r^2;
Js_Koop(v::Matrix, A, c; r=rp_Koop) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 * r^2;

target_Koopman = (J_Koop, Js_Koop, (Ap_Koop, cp_Koop))

J_Taylor(g::Vector, A, c; r=rp_Taylor) = ((g - c)' * A * (g - c))/2 - 0.5 * r^2;
Js_Taylor(v::Vector, A, c; r=rp_Taylor) = (v' * inv(A) * v)/2 + c'v + 0.5 * r^2; # Convex Conjugate
J_Taylor(g::Matrix, A, c; r=rp_Taylor) = diag((g .- c)' * A * (g .- c))/2 .- 0.5 * r^2;
Js_Taylor(v::Matrix, A, c; r=rp_Taylor) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 * r^2;

target_Taylor = (J_Taylor, Js_Taylor, (Ap_Taylor, cp_Taylor))

## Lookback Time(s), 

Th = 0.333; 
Th_steps = 6;
Tf = Th_steps * Th;
T = collect(Th : Th : Tf);
th = 0.066;

## Grid Definition

lb, ub, N = -2.5 * r, 2.5 * r, 50
xig = collect(lb : (ub - lb) / 99 : ub); lg = length(xig)
Xg = hcat(collect.(Iterators.product([xig for i in 1:nx]...))...)[end:-1:1,:]
Gg = Matrix(Duffing.lift_data(Xg')');

## Test Target

# ϕ0 = J(Xg, Ap[1:nx,1:nx], cp[1:nx]);
# plot_BRS([0.0], [Xg, Xg], [ϕ0, ϕ0]; M=K, ϵs=1e-3, interpolate=false, value_fn=true, alpha=0.5)

## ADMM-CD Hybrid Parameters
opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.01, 10, 1e-5, 100, 1, 4), 1, 1, 3)

##   Solve
##   =====

# ## Koopman
solution_Koop, run_stats_Koop = Hopf_BRS(system_Koopman, target_Koopman, T; Xg=Gg, th, opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, warm=true, printing=true);
GT, ϕGT = solution_Koop;

# @save "solution_Koop_res100_later_dall.jld" solution_Koop, run_stats_Koop
# (solution_Koop, run_stats_Koop) = load("solution_Koop_res100_later.jld")["(solution_Koop, run_stats_Koop)"]

## Plot the Values given their first two-coordinates
BRS_plots = plot_BRS(T, [Xg for i=1:length(T)+1], solution_Koop[2]; M=K, ϵs=3e-1, interpolate=false, value_fn=false, alpha=0.25)

# ## Taylor
solution_Taylor, run_stats_Taylor = Hopf_BRS(system_Taylor, target_Taylor, T; Xg, th, opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, warm=true, printing=true);
XT, ϕXT = solution_Taylor;

# @save "solution_Taylor_res100_later_dall.jld" solution_Taylor, run_stats_Taylor
# (solution_Taylor, run_stats_Taylor) = load("solution_Taylor_res100_later.jld")["(solution_Taylor, run_stats_Taylor)"]

## Plot the Values given their first two-coordinates
BRS_plots = plot_BRS(T, [Xg for i=1:length(T)+1], solution_Taylor[2]; M, ϵs=5e-2, interpolate=true, value_fn=false, alpha=0.5)

## Get the Ground Truth from hj_reachability.py (nans when run in julia, smh jax)

hj_r_loc = home * "HL_fastHJR/Linearizations"
pushfirst!(pyimport("sys")."path", hj_r_loc);
np = pyimport("numpy")
jnp = pyimport("jax.numpy")
hj = pyimport("hj_reachability")
hj_r_duffing = pyimport("duffing_hj_reachability")

println("Do your grid parameters match?")
println("GRID RES: ", length(hj_r_duffing.grid.coordinate_vectors[1].tolist()), " in py vs $lg in julia")
println("UPPER: ",    maximum(hj_r_duffing.grid.states.tolist()), " in py vs $ub in julia")
println("LOWER: ",    minimum(hj_r_duffing.grid.states.tolist()), " in py vs $lb in julia")

ϕ0 = Matrix(reshape(hj_r_duffing.values.tolist()', length(hj_r_duffing.values.tolist()), 1))[:,1]
ϕ_33_dall = Matrix(reshape(hj_r_duffing.target_values_33_dall.tolist()', length(hj_r_duffing.target_values_33_dall.tolist()), 1))[:,1]
ϕ_66_dall = Matrix(reshape(hj_r_duffing.target_values_66_dall.tolist()', length(hj_r_duffing.target_values_66_dall.tolist()), 1))[:,1]
ϕ_99_dall = Matrix(reshape(hj_r_duffing.target_values_99_dall.tolist()', length(hj_r_duffing.target_values_99_dall.tolist()), 1))[:,1]
ϕ_132_dall = Matrix(reshape(hj_r_duffing.target_values_132_dall.tolist()', length(hj_r_duffing.target_values_132_dall.tolist()), 1))[:,1]
ϕ_165_dall = Matrix(reshape(hj_r_duffing.target_values_165_dall.tolist()', length(hj_r_duffing.target_values_165_dall.tolist()), 1))[:,1]
ϕ_198_dall = Matrix(reshape(hj_r_duffing.target_values_198_dall.tolist()', length(hj_r_duffing.target_values_198_dall.tolist()), 1))[:,1]
ϕ_231_dall = Matrix(reshape(hj_r_duffing.target_values_231_dall.tolist()', length(hj_r_duffing.target_values_231_dall.tolist()), 1))[:,1]

ϕ_33_dc   = Matrix(reshape(hj_r_duffing.target_values_33_dc.tolist()', length(hj_r_duffing.target_values_33_dc.tolist()), 1))[:,1]
ϕ_66_dc   = Matrix(reshape(hj_r_duffing.target_values_66_dc.tolist()', length(hj_r_duffing.target_values_66_dc.tolist()), 1))[:,1]
ϕ_99_dc   = Matrix(reshape(hj_r_duffing.target_values_99_dc.tolist()', length(hj_r_duffing.target_values_99_dc.tolist()), 1))[:,1]
ϕ_132_dc   = Matrix(reshape(hj_r_duffing.target_values_132_dc.tolist()', length(hj_r_duffing.target_values_132_dc.tolist()), 1))[:,1]
ϕ_165_dc   = Matrix(reshape(hj_r_duffing.target_values_165_dc.tolist()', length(hj_r_duffing.target_values_165_dc.tolist()), 1))[:,1]
ϕ_198_dc   = Matrix(reshape(hj_r_duffing.target_values_198_dc.tolist()', length(hj_r_duffing.target_values_198_dc.tolist()), 1))[:,1]
ϕ_231_dc   = Matrix(reshape(hj_r_duffing.target_values_231_dc.tolist()', length(hj_r_duffing.target_values_231_dc.tolist()), 1))[:,1]

ϕT_dall = [ϕ0, ϕ_33_dall, ϕ_66_dall, ϕ_99_dall, ϕ_132_dall, ϕ_165_dall, ϕ_198_dall, ϕ_231_dall]
ϕT_dc   = [ϕ0, ϕ_33_dc, ϕ_66_dc, ϕ_99_dc, ϕ_132_dc, ϕ_165_dc, ϕ_198_dc, ϕ_231_dc]

ϕT = disturbance_on_control ? ϕT_dc : ϕT_dall

BRS_plots = plot_BRS(T, [Xg for i=1:length(T)+1], ϕT; M, ϵs=5e-2, interpolate=true, value_fn=false, alpha=0.5)

## Compare the Sets
## ================

Koop_jaccards, Taylor_jaccards = [], []
# ϵs = 5e-1
ϕGT = solution_Koop[2]
ϕXT = solution_Taylor[2]
for i=1:length(solution_Koop[1])

    BRS_HJ_R   = Xg[:, ϕT[i]  .< 0]
    BRS_Koop   = Xg[:, ϕGT[i] .< 0]
    BRS_Taylor = Xg[:, ϕXT[i] .< 0]

    BRS_HJ_R_set, BRS_Koop_set, BRS_Taylor_set = Set(), Set(), Set()
    map(x->push!(BRS_HJ_R_set, x),   eachcol(BRS_HJ_R))
    map(x->push!(BRS_Koop_set, x),   eachcol(BRS_Koop))
    map(x->push!(BRS_Taylor_set, x), eachcol(BRS_Taylor))

    push!(Koop_jaccards,   jaccard(BRS_HJ_R_set, BRS_Koop_set))
    push!(Taylor_jaccards, jaccard(BRS_HJ_R_set, BRS_Taylor_set))

end

Koop_jaccards
Taylor_jaccards

time_pt = 6

BRS_plots = plot_BRS_pretty(T[1:3], [Xg for i=1:length(T)], [ϕT[1], ϕT[time_pt], ϕGT[time_pt], ϕXT[time_pt]]; 
                        M, ϵs=[1e-1, 4e-1, 2e-1, 4e-1], ϵc=2e-3, interpolate=false, value_fn=false, alpha=0.1, 
                        latex_title=true, input_labels=[L"\mathcal{T}", L"\mathcal{R}(\mathcal{T}, t)", L"\mathcal{R}(\mathcal{T})", L"\mathcal{R}(\widetilde{\mathcal{T}}_\mathcal{G}, t)", L"\mathcal{R}(\widetilde{\mathcal{T}}_\mathcal{G}, t)", L"\mathcal{R}(\mathcal{T}_{\text{Taylor}}, t)", L"\mathcal{R}(\mathcal{T}_{\text{Taylor}}, t)"])


using LinearAlgebra, StatsBase, ScatteredInterpolation
using Plots, ImageFiltering, TickTock, Suppressor, PlotlyJS
# plotly()

function plot_BRS_pretty(T, B⁺T, ϕB⁺T; M=nothing, simple_problem=true, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, 
    zplot=false, interpolate=false, inter_method=Polyharmonic(), pal_colors=[:red, :blue], alpha=0.5, 
    title=nothing, value_fn=false, nx=size(B⁺T[1])[1], xlims=[-2, 2], ylims=[-2, 2], latex_title=false, input_labels=nothing)

    if nx > 2 && value_fn; println("4D plots are not supported yet, can't plot Value fn"); value_fn = false; end

    Xplot = isnothing(title) ? Plots.plot(title=latex_title ? L"\text{BRS}: \:\: \phi(X, t) = 0" : "BRS: Φ(X, t) = 0", extra_plot_kwargs = KW(:include_mathjax => "cdn")) : Plots.plot(title=title, extra_plot_kwargs = KW(:include_mathjax => "cdn"))
    if zplot; Zplot = Plots.plot(title="BRS: Φ(Z, T) = 0"); end

    # plot!(Xplot, xlabel=L"x_1", xlabel=L"x_2")
    annotate!(Xplot, -2, -2, L"\mathcal{X}")

    plots = zplot ? [Xplot, Zplot] : [Xplot]
    if value_fn; vfn_plots = zplot ? [Plots.plot(title="Value: Φ(X, T)"), Plots.plot(title="Value: Φ(Z, T)")] : [Plots.plot(title=latex_title ? L"V: \phi (X, T)" : "Value: Φ(X, T)")]; end

    B⁺Tc, ϕB⁺Tc = copy(B⁺T), copy(ϕB⁺T)
    
    ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
    Jlabels = "J(⋅, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
    labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

    # Tcolors = length(T) > 1 ? palette(pal_colors, length(T)) : [pal_colors[2]]
    # B0colors = length(T) > 1 ? palette([:black, :gray], length(T)) : [:black]
    Tcolors = palette(:seaborn_colorblind)
    B0colors = palette([:black, :gray], length(T))
    plot_colors = collect(Iterators.flatten(zip(B0colors, Tcolors)))

    ## Zipping Target to Plot Variation in Z-space over Time (already done in moving problems)
    if simple_problem && (length(T) > 1)
        for i = 3 : 2 : 2*length(T)
            insert!(B⁺Tc, i, B⁺T[1])
            insert!(ϕB⁺Tc, i, ϕB⁺T[1])
        end
    end

    if nx > 2 && interpolate; plotly_pl = zplot ? [Array{GenericTrace{Dict{Symbol, Any}},1}(), Array{GenericTrace{Dict{Symbol, Any}},1}()] : [Array{GenericTrace{Dict{Symbol, Any}},1}()]; end

    for (j, i) in enumerate(1 : 2 : 2*length(T))        
        B⁺0, B⁺, ϕB⁺0, ϕB⁺ = B⁺Tc[i], B⁺Tc[i+1], ϕB⁺Tc[i], ϕB⁺Tc[i+1]
        Bs = zplot ? [B⁺0, B⁺, exp(-T[j] * M) * B⁺0, exp(-T[j] * M) * B⁺] : [B⁺0, B⁺]

        for (bi, b⁺) in enumerate(Bs)
            if simple_problem && bi == 1 && i !== 1; continue; end

            ϕ = bi % 2 == 1 ? ϕB⁺0 : ϕB⁺
            label = simple_problem && i == 1 && bi == 1 ? "J(⋅)" : labels[i + (bi + 1) % 2]
            label = isnothing(input_labels) ? label : simple_problem && i == 1 && bi == 1 ? input_labels[1] : input_labels[i + (bi + 1) % 2]

            ## Plot Scatter
            if interpolate == false

                ## Find Boundary in Near-Boundary
                ϵss = length(ϵs) > 1 ? ϵs[j] : ϵs
                b = b⁺[:, abs.(ϕ) .< ϵss]

                scatter!(plots[Int(bi > 2) + 1], [b[i,:] for i=1:nx]..., label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                # scatter!(plots[Int(bi > 2) + 1], b[1,:], b[2,:], label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                
                if value_fn
                    scatter!(vfn_plots[Int(bi > 2) + 1], b⁺[1,:], b⁺[2,:], ϕ, label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha, xlims, ylims)
                    # scatter!(vfn_plots[Int(bi > 2) + 1], b[1,:], b[2,:], ϕ, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], label=label)
                end
            
            ## Plot Interpolation
            else 

                if nx == 2
                    contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] for i=1:nx]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label, linewidth=3)

                    if value_fn

                        ## Make Grid
                        xig = [collect(minimum(b⁺[i,:]) : cres : maximum(b⁺[i,:])) for i=1:nx]
                        G = hcat(collect.(Iterators.product(xig...))...)'
                        
                        ## Construct Interpolationb (Should skip this in the future and just use Plotly's built in one for contour)
                        itp = ScatteredInterpolation.interpolate(inter_method, b⁺, ϕ)
                        itpd = evaluate(itp, G')
                        iϕG = reshape(itpd, length(xig[1]), length(xig[2]))'
                        
                        surface!(vfn_plots[Int(bi > 2) + 1], xig..., iϕG, colorbar=false, color=plot_colors[i + (bi + 1) % 2], label=label, alpha=alpha)
                    end
            
                else
                    # isosurface!(plots[Int(bi > 2) + 1], xig..., iϕG, isomin=-ϵc, isomax=ϵc, surface_count=2, lc=plot_colors[i + (bi + 1) % 2], alpha=0.5)
                    pl = isosurface(x=b⁺[1,:], y=b⁺[2,:], z=b⁺[3,:], value=ϕ[:], opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
                        name=label, colorscale=[[0, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]], [1, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]]])

                    push!(plotly_pl[Int(bi > 2) + 1], pl)
                end
            end
        end
    end

    if value_fn
        Xplot = Plots.plot(vfn_plots[1], Xplot)
        if zplot; Zplot = Plots.plot(vfn_plots[2], Zplot); end
    end

    if nx > 2 && interpolate 
        Xplot = PlotlyJS.plot(plotly_pl[1], Layout(title= latex_title ? L"BRS of T, in X" : "BRS of T, in X"));
        if zplot; Zplot = PlotlyJS.plot(plotly_pl[2], Layout(title="BRS of T, in X")); end
    end

    display(Xplot); 
    plots = [Xplot]
    if zplot; display(Zplot); plots = [Xplot, Zplot]; end

    return plots
end