
### Hopf Reachability of Textbook Koopman Systems (Exact Linearization)
# wsharpless@ucsd.edu

using LinearAlgebra, StatsBase, ScatteredInterpolation
using Plots, PlotlyJS, LaTeXStrings, JLD
plotlyjs() 

include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, plot_BRS, Hopf

using LSHFunctions: jaccard
using Polyhedra
include("LoewnerJohnEllipsoids.jl")

using Pkg
ENV["PYTHON"] = "/Users/willsharpless/miniconda3/envs/pythreeten/bin/python" # need python 3.10 for AutoKoopman 
Pkg.build("PyCall") 
using PyCall

########################################################################################
## 2D, Autonomous
########################################################################################

μ, λ = -0.05, -1.

## System: ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
dim = 2
M = [λ -λ;
     0 2μ]
B, C = [0. 1.; 2μ 0.], [0. 1.; 2μ 0.]
Q, Q2 = zeros(dim,dim), zeros(dim,dim) #Autonomous
a1, a2 = zeros(1, dim), zeros(1, dim)

system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.05
Th = 0.2
Tf = 1.0
T = collect(Th : Th : Tf)

## Target: J(x) = 0 is the boundary of the target
Ap = diagm(inv.(cat([1, 1], 0.5*ones(dim - 2), dims=1)).^2)
cp = [5.; 5.]
J(x::Vector, A, c) = ((x - c)' * A * (x - c))/2 - 0.5 #don't need yet
Js(v::Vector, A, c) = (v' * inv(A) * v)/2 + c'v + 0.5
J(x::Matrix, A, c) = diag((x .- c)' * A * (x .- c))/2 .- 0.5
Js(v::Matrix, A, c) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 #don't need yet
target = (J, Js, (Ap, cp))

## Grid Parameters (optional, deafult here)
bd = (2, 8)
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 5
max_runs = 10
max_its = 2000
opt_p_cd = (vh, L, tol, lim, lll, max_runs, max_its)

# Hopf ADMM Parameters (default)
ρ, ρ2 = 1e-4, 1e-4
tol = 1e-5
max_its = 10
opt_p_admm = (ρ, ρ2, tol, max_its)

# ADMM-CD Hybrid
ρ_grid_vals = 1 
hybrid_runs = 3
opt_p_admm_cd = (opt_p_admm, opt_p_cd, ρ_grid_vals, ρ_grid_vals, hybrid_runs)

## Run the solver
solution, run_stats = Hopf_BRS(system, target, T;   th, grid_p,
                                                    opt_method = Hopf_cd,
                                                    opt_p = opt_p_cd,
                                                    warm = true,
                                                    printing = true);

plot_scatter = plot_BRS(T, solution...; M, ϵs=0.05, interpolate=false, value_fn=true, alpha=0.3)
plot_contour = plot_BRS(T, solution...; M, ϵc=0.001, interpolate=true, value_fn=true, alpha=0.5)

########################################################################################
## 2D, Controlled & Disturbed
########################################################################################

# plts = []
# pal_colors_list = [["blue", "blue"], ["red", "red"], ["green", "green"], ["yellow", "yellow"]]
# T = [0.2]

# vh = 0.05
# L = 5
# tol = ϵ
# lim = 500
# lll = 5
# max_runs = 10
# max_its = 2000
# opt_p_cd = (vh, L, tol, lim, lll, max_runs, max_its)

# ## Functions just for merging the plots
# function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
#      append!(sp1.series_list, sp2.series_list)
#      Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
#      Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
#      Plots.expand_extrema!(sp1[:zaxis], zlims(sp2))
#      return sp1
#  end
 
#  function merge_series!(plt, plts...)
#      for (i, sp) in enumerate(plt.subplots)
#          for other_plt in plts
#              if i in eachindex(other_plt.subplots)
#                  merge_series!(sp, other_plt[i])
#              end
#          end
#      end
#      return plt
#  end

# for (ri, r) in enumerate([0.01, 1., 5., 10.])

#      Q, Q2 = r^2 * diagm(ones(dim)), r^2 * 0.5 * diagm(ones(dim)) #Controlled and Disturbed
#      system = (M, B, C, Q, Q2, a1, a2)

#      solution, run_stats = Hopf_BRS(system, target, T;  
#                                                        opt_method = Hopf_cd, 
#                                                        th,
#                                                        grid_p,
#                                                     #    opt_p = opt_p_admm,
#                                                        warm=true,
#                                                        check_all=true,
#                                                        printing=true);
#      B⁺T, ϕB⁺T = solution;

#     #  plot_contour = plot_BRS(T, B⁺T, ϕB⁺T; M, cres=0.01, interpolate=true, pal_colors = pal_colors_list[ri]);
#     #  plot_scatter = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵs=0.1, interpolate=false, value_fn=true, alpha=0.5)
#      plot_contour = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵc=0.001, interpolate=true, value_fn=false, alpha=0.5, pal_colors = pal_colors_list[ri])
#      push!(plts, plot_contour)
# end

# plot_contours = merge_series!([plts[i][1] for i in eachindex(plts)]...)

########################################################################################
## 3D, Autonomous & Controlled
########################################################################################

max_u, max_d = 0.5, 0.25
cx = [0., 1.25]
lbc, ubc = -2., 2. # relative to cx

## System: ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

M = [ μ  0  0;
      0  λ -λ;
      0  0 2μ]
B = [ 1 0; 
      0 1;
      2μ*cx[1] 0]
C = copy(B)
 
dim_in = size(B)[2]
Q_auto, Q2_auto = zeros(dim_in, dim_in), zeros(dim_in, dim_in) #Autonomous
Q, Q2 = I + zeros(dim_in, dim_in), I + zeros(dim_in, dim_in) # Controlled
a1, a2 = zeros(1, dim_in), zeros(1, dim_in)

system_auto = (M, max_u*B, max_d*C, Q_auto, Q2_auto, a1, a2)
system = (M, max_u*B, max_d*C, Q, Q2, a1, a2)

## Time
th = 0.05
T = [0.2, 0.5, 1.0, 1.5, 2.0]

## Targets: J(x) = 0 is the boundary of the target

c1p = cx[1]
rp = 1.;

cg = vcat(cx, c1p^2) # lifted center
 
# Augmented Target (cylindrical ellipse)

η = 15; 
Ap_cyle = diagm(vcat(η * ones(2), 1/η)); rp_cyle = sqrt(η) * rp # ellipsoidal relaxation, big numbers;

J_cyle(g::Vector, A, c; r=rp_cyle) = ((g - c)' * A * (g - c))/2 - 0.5 * r^2;
Js_cyle(v::Vector, A, c; r=rp_cyle) = (v' * inv(A) * v)/2 + c'v + 0.5 * r^2; # Convex Conjugate
J_cyle(g::Matrix, A, c; r=rp_cyle) = diag((g .- c)' * A * (g .- c))/2 .- 0.5 * r^2;
Js_cyle(v::Matrix, A, c; r=rp_cyle) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 * r^2;

target_cyle = (J_cyle, Js_cyle, (Ap_cyle, cg))

# Ball-Ellipse over Lifted Target 

Ap_ball = diagm(inv.(cat([1., 1., (c1p + rp)^2 - c1p], dims=1)).^2);

J_ball(g::Vector, A, c; r=rp) = ((g - c)' * A * (g - c))/2 - 0.5 * r^2;
Js_ball(v::Vector, A, c; r=rp) = (v' * inv(A) * v)/2 + c'v + 0.5 * r^2; # Convex Conjugate
J_ball(g::Matrix, A, c; r=rp) = diag((g .- c)' * A * (g .- c))/2 .- 0.5 * r^2;
Js_ball(v::Matrix, A, c; r=rp) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 * r^2;

target_ball = (J_ball, Js_ball, (Ap_ball, cg))

# Custom Tight Ellipse over Lifted Target

c_tgte = [c1p; cx[2]; c1p^2 + rp^2] # tight ellipse center

a1_Es = rp^2 * sqrt(1 + 4c1p^2); a2_Es = 1.0;
A_Es = diagm([a1_Es, 1., a2_Es].^2);
W_Es = [1 0 -2c1p;  0 sqrt(1 + 4c1p^2) 0; 2c1p 0 1] / sqrt(1 + 4c1p^2);
Ap_tgte = inv(W_Es)' * inv(A_Es) * inv(W_Es)

target_tgte = (J_ball, Js_ball, (Ap_tgte, c_tgte))

# Loewner-John Inner & Outer Ellipses of Lifted Target (takes a min)

ϵ = 0.5e-7; N = 10 + ϵ
x1g = collect(cx[1] + lbc : 1/N : cx[1] + ubc) .+ ϵ
x2g = collect(cx[2] + lbc : 1/N : cx[2] + ubc) .+ ϵ
Xg = hcat(collect.(Iterators.product(x1g, x2g))...) # Grid for target

Xgϕ = diag((Xg .- cx)' * (Xg .- cx))/2 .- 0.5 * rp^2;
Xgc = Xg[:, abs.(Xgϕ) .< 0.05] # Find boundary of target
Ggc = vcat(Xgc, Xgc[1,:]'.^2) # Lift target

hc = hrep(polyhedron(convexhull([Ggc[:,i] for i=1:size(Ggc)[2]]...))) # h-representation of lifted target boundary hull
constraints = [LoewnerJohnEllipsoids.LinearConstraint(hs.a, hs.β) for hs in allhalfspaces(hc)] # poly conversion

ljoe = LoewnerJohnEllipsoids.outer_ellipsoid([eachcol(Ggc)...]) # Solve Outer ellipsoid of lifted boundary
ljie = LoewnerJohnEllipsoids.inner_ellipsoid(LoewnerJohnEllipsoids.Polyhedron(constraints)) # Solve Inner ellipsoid of lifted boundary hull

A_ljoe, A_ljie = round.(ljoe.P, digits=3), round.(ljie.P, digits=3)
c_ljoe, c_ljie = round.(ljoe.d, digits=3), round.(ljie.d, digits=3)

target_oute = (J_ball, Js_ball, (A_ljoe, c_ljoe))
target_inne = (J_ball, Js_ball, (A_ljie, c_ljie))

## Grid Parameters (optional, deafult here)
ϵ = 0.5e-7; N = 3 + ϵ
x1g = collect(cx[1] + lbc : 1/N : cx[1] + ubc) .+ ϵ
x2g = collect(cx[2] + lbc : 1/N : cx[2] + ubc) .+ ϵ
gg = collect(0. : 1/N : (cx[2] + ubc)^2) .+ ϵ
Gg = hcat(collect.(Iterators.product(x1g, x2g, gg))...)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
fd_step = 5e-3; step_sz_inv = 10; tol = 5e-7 #ϵ
min_converged_runs = 5; max_runs = 20; max_its_per_run = 1500; step_step = 1500;
opt_p_cd = (fd_step, step_sz_inv, tol, step_step, min_converged_runs, max_runs, max_its_per_run)

opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.005, 10, 1e-5, 100, 1, 4, 500), 1, 1, 3)

## Run the solver
# solution_ball, run_stats = Hopf_BRS(system, target_ball, T;
#                                                 Xg=Gg, th,
#                                                 opt_method = Hopf_admm_cd,
#                                                 opt_p = opt_p_admm_cd,
#                                                 warm = true,
#                                                 check_all = true,
#                                                 printing = true);

# solution_cyle, run_stats = Hopf_BRS(system, target_cyle, T;
#                                                 Xg=Gg, th,
#                                                 opt_method = Hopf_admm_cd,
#                                                 opt_p = opt_p_admm_cd,
#                                                 warm = true,
#                                                 check_all = true,
#                                                 printing = true);

# solution_tgte, run_stats = Hopf_BRS(system, target_tgte, T;
#                                                 Xg=Gg, th,
#                                                 opt_method = Hopf_admm_cd,
#                                                 opt_p = opt_p_admm_cd,
#                                                 warm = true,
#                                                 check_all = true,
#                                                 printing = true);

# solution_oute, run_stats = Hopf_BRS(system, target_oute, T;
#                                                 Xg=Gg, th,
#                                                 opt_method = Hopf_admm_cd,
#                                                 opt_p = opt_p_admm_cd,
#                                                 warm = true,
#                                                 check_all = true,
#                                                 printing = true);

# solution_inne, run_stats = Hopf_BRS(system, target_inne, T;
#                                                 Xg=Gg, th,
#                                                 opt_method = Hopf_admm_cd,
#                                                 opt_p = opt_p_admm_cd,
#                                                 warm = true,
#                                                 check_all = true,
#                                                 printing = true);

# plot_scatter = plot_BRS(T, solution_ball...; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_contour = plot_BRS(T, solution_ball...; M, ϵc=0.001, interpolate=true, alpha=0.5)

# plot_scatter = plot_BRS(T, solution_cyle...; M, ϵs=2., interpolate=false, alpha=0.1)
# plot_contour = plot_BRS(T, solution_cyle...; M, ϵc=0.001, interpolate=true, alpha=0.5);

# plot_scatter = plot_BRS(T, solution_tgte...; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_contour = plot_BRS(T, solution_tgte...; M, ϵc=0.01, interpolate=true, alpha=0.5);

# plot_scatter = plot_BRS(T, solution_oute...; M, ϵs=0.5, interpolate=false, alpha=0.1)
# plot_contour = plot_BRS(T, solution_oute...; M, ϵc=0.01, interpolate=true, alpha=0.5);

# plot_scatter = plot_BRS(T, solution_inne...; M, ϵs=0.5, interpolate=false, alpha=0.1)
# plot_contour = plot_BRS(T, solution_inne...; M, ϵc=0.1, interpolate=true, alpha=0.5);

## Attempt to combine plots

# target_d = Gg[:, abs.(J_ball(Gg, Ap_ball, cg)) .< 0.1]
# # border_nxt = (I + T[1] * M) * border
# T = [0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0]

# target_dbef = copy(target_d)
# label = 0.
# scatter(eachrow(border)..., label="t=0.")

# target_dbef = inv(I + 0.2 * M) * target_dbef
# label = label - 0.2
# scatter!(eachrow(target_dbef)..., label=string(label))

# scatter(eachrow(border)..., label="t=0.");
# # scatter!(eachrow(border_nxt)..., label="t=0.2")
# scatter!(eachrow(target_dbef[9])...)

# scatter!(eachrow(border_bef1)..., label="t=-0.2")
# scatter!(eachrow(border_bef2)..., label="t=-0.8")
# scatter!(eachrow(border_bef3)..., label="t=-2.")

# scatter!(eachrow(Gg)..., label="grid", alpha=0.3)

# alld = hcat(border, border_nxt, border_bef, Gg)
# x1g = collect(0. : 0.1 : 4.);
# x2g = collect(-1. : 0.1 : 4.);
# surface!(x1g, x2g, (x1, x2)->x1^2, colorbar=false, color="green", alpha=0.5, label="Real Surface")
# plot!(xlims=[0, 4], ylims=[-1,4], zlims=[-1,16])

# surface!(x1g, x2g, (x1, x2)->x2, colorbar=false, color="dark green", alpha=0.5, label="Stable Surface")


## Attempt to modify Plotly sync plots (doesnt work, need to pull generic_trace from sync)
# ϵc = 0.01
# pl = isosurface(x=Gg[1,:], y=Gg[2,:], z=(Gg[1,:].^2 - Gg[3,:]), value=Gg, opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
# name="True States", colorscale=[[0, "rgb(1, 0, 0)"], [1, "rgb(1, 0, 0)"]])

# push!(plot_contour, pl)

# PlotlyJS.plot(plot_contour[1], Layout(title="BRS of T, in X"))

########################################################################################
## 3D but Lifted States Only (i.e on the Quadratic)
########################################################################################

ϵ = 0.5e-7; res = 100
x1g = collect(cx[1] + lbc : (ubc-lbc)/(res-1) : cx[1] + ubc) .+ ϵ; lg1 = length(x1g) # == res, for comparing to DP
x2g = collect(cx[2] + lbc : (ubc-lbc)/(res-1) : cx[2] + ubc) .+ ϵ; lg2 = length(x2g)
Xg = hcat(collect.(Iterators.product(x1g, x2g))...)
Gg = vcat(Xg, Xg[1,:]'.^2)

T = [0.2, 0.5, 1.0, 1.5, 2.0]
Xgs = [Xg for i=1:length(T)+1] # for plotting

# Facile Params
opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.005, 10, 1e-5, 100, 1, 4, 500), 1, 1, 3)

## Run the solver
solution_ball_quad, run_stats_ball = Hopf_BRS(system, target_ball, T;
                                                Xg=Gg, th,
                                                opt_method = Hopf_admm_cd,
                                                opt_p = opt_p_admm_cd,
                                                warm = true,
                                                check_all = true,
                                                printing = true);

solution_tgte_quad, run_stats_tgte = Hopf_BRS(system, target_tgte, T;
                                                Xg=Gg, th,
                                                opt_method = Hopf_admm_cd,
                                                opt_p = opt_p_admm_cd,
                                                warm = true,
                                                check_all = true,
                                                printing = true);

solution_cyle_quad, run_stats_cyle = Hopf_BRS(system, target_cyle, T;
                                                Xg=Gg, th,
                                                opt_method = Hopf_admm_cd,
                                                opt_p = opt_p_admm_cd,
                                                warm = true,
                                                check_all = true,
                                                printing = true)

solution_oute_quad, run_stats_oute = Hopf_BRS(system, target_oute, T;
                                                Xg=Gg, th,
                                                opt_method = Hopf_admm_cd,
                                                opt_p = opt_p_admm_cd,
                                                warm = true,
                                                check_all = true,
                                                printing = true);

solution_inne_quad, run_stats_inne = Hopf_BRS(system, target_inne, T;
                                                Xg=Gg, th,
                                                opt_method = Hopf_admm_cd,
                                                opt_p = opt_p_admm_cd,
                                                warm = false,
                                                check_all = true,
                                                printing = true);

## Plotting Solution on quadtratic surface

# plot_scatter_ball_quad = plot_BRS(T, solution_ball_quad...; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_scatter_cyle_quad = plot_BRS(T, solution_cyle_quad...; M, ϵs=2., interpolate=false, alpha=0.1)
# plot_scatter_tgte_quad = plot_BRS(T, solution_tgte_quad...; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_scatter_oute_quad = plot_BRS(T, solution_oute_quad...; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_scatter_inne_quad = plot_BRS(T, solution_inne_quad...; M, ϵs=0.1, interpolate=false, alpha=0.1)

## Plotting in 2D by Projection

ϕXT_KH = Dict(
    "Ball" => solution_ball_quad[2],
    "CylE" => solution_cyle_quad[2],
    "TgtE" => solution_tgte_quad[2],
    "OutE" => solution_oute_quad[2],
    "InnE" => solution_inne_quad[2],
)

run_stats_pp = Dict(
    "Ball" => run_stats_ball[2],
    "CylE" => run_stats_cyle[2],
    "TgtE" => run_stats_tgte[2],
    "OutE" => run_stats_oute[2],
    "InnE" => run_stats_inne[2],
)

# ϕXT_KH_auto = Dict(
#     "Ball" => solution_ball_quad[2],
#     "CylE" => solution_cyle_quad_auto[2],
#     "TgtE" => solution_tgte_quad[2],
#     "OutE" => solution_oute_quad_auto[2],
#     "InnE" => solution_inne_quad_auto[2],
# )

# plot_scatter_ball_2d = plot_BRS(T, Xgs, ϕXT_KH["Ball"]; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_scatter_tgte_2d = plot_BRS(T, Xgs, ϕXT_KH["TgtE"]; M, ϵs=0.035, interpolate=false, alpha=0.1)
# plot_scatter_cyle_2d = plot_BRS(T, Xgs, ϕXT_KH["CylE"]; M, ϵs=1.0, interpolate=false, alpha=0.1)
# plot_scatter_oute_2d = plot_BRS(T, Xgs, ϕXT_KH["OutE"]; M, ϵs=0.1, interpolate=false, alpha=0.1)
# plot_scatter_inne_2d = plot_BRS(T, Xgs, ϕXT_KH["InnE"]; M, ϵs=0.1, interpolate=false, alpha=0.1)

plot_contour_ball_2d = plot_BRS(T, Xgs, ϕXT_KH["Ball"]; M, ϵc=0.001, interpolate=true, alpha=0.5)
plot_contour_tgte_2d = plot_BRS(T, Xgs, ϕXT_KH["TgtE"]; M, ϵc=0.001, interpolate=true, alpha=0.5)
plot_contour_cyle_2d = plot_BRS(T, Xgs, ϕXT_KH["CylE"]; M, ϵc=0.001, interpolate=true, alpha=0.5)
plot_contour_oute_2d = plot_BRS(T, Xgs, ϕXT_KH["OutE"]; M, ϵc=0.001, interpolate=true, alpha=0.5)
plot_contour_inne_2d = plot_BRS(T, Xgs, ϕXT_KH["InnE"]; M, ϵc=0.001, interpolate=true, alpha=0.5)

# save("KH_Toy_results_c01p25_up5-dp25_T5_fine---.jld", "max_u_d", (max_u, max_d), "T", T, "Xg", Xg, "ϕXT_KH", ϕXT_KH)

########################################################################################
## Comparison with Taylor Lin (2D)
########################################################################################

## System: ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

M = [ μ  0;
      -2λ*cx[1] λ]
B = [ 1 0; 
      0 1]
C = copy(B)
 
dim_in = size(B)[2]
Q_auto, Q2_auto = zeros(dim_in, dim_in), zeros(dim_in, dim_in) #Autonomous
Q, Q2 = I + zeros(dim_in, dim_in), I + zeros(dim_in, dim_in) # Controlled
a1, a2 = zeros(1, dim_in), zeros(1, dim_in)

system_auto_taylor = (M, max_u*B, max_d*C, Q_auto, Q2_auto, a1, a2)
system_taylor = (M, max_u*B, max_d*C, Q, Q2, a1, a2)

## Targets: J(x) = 0 is the boundary of the target

rp = 1.;
Ap_taylor = diagm(inv.(cat([1., 1.], dims=1)).^2);

J_taylor(g::Vector, A, c; r=rp) = ((g - c)' * A * (g - c))/2 - 0.5 * r^2;
Js_taylor(v::Vector, A, c; r=rp) = (v' * inv(A) * v)/2 + c'v + 0.5 * r^2; # Convex Conjugate
J_taylor(g::Matrix, A, c; r=rp) = diag((g .- c)' * A * (g .- c))/2 .- 0.5 * r^2;
Js_taylor(v::Matrix, A, c; r=rp) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 * r^2;

target_taylor = (J_taylor, Js_taylor, (Ap_taylor, cx))

## Run the solver
solution_taylor, run_stats_taylor = Hopf_BRS(system_taylor, target_taylor, T;
                                                Xg, th,
                                                opt_method = Hopf_admm_cd,
                                                opt_p = opt_p_admm_cd,
                                                warm = true,
                                                check_all = true,
                                                printing = true);

plot_scatter = plot_BRS(T, solution_taylor...; M, ϵs=0.1, interpolate=false, alpha=0.1)
plot_contour = plot_BRS(T, solution_taylor...; M, ϵc=0.001, interpolate=true, alpha=0.5)

ϕXT_KH["Taylor"] = solution_taylor[2]
run_stats_pp["Taylor"] = run_stats_taylor[2]

########################################################################################
## Comparison with hj_reachability
########################################################################################

## Get the "True" BRS from hj_reachability.py

hj_r_loc = pwd() * "/Linearizations"
pushfirst!(pyimport("sys")."path", hj_r_loc);
np = pyimport("numpy")
jnp = pyimport("jax.numpy")
hj = pyimport("hj_reachability")
hj_r_toy = pyimport("toy_hj_reachability")

DP_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([cx[1] + lbc, cx[2] + lbc]),
                                                                              np.array([cx[1] + ubc, cx[2] + ubc])),
                                                                             (lg1, lg2)) #lg has to be even

DP_values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(np.subtract(DP_grid.states, np.array(cx)))), axis=-1)) - rp^2) * 0.5

dynamics = hj_r_toy.SlowManifold(mu=μ, lambduh=λ, max_u=max_u, max_d=max_d)

println("Do your grid parameters match?")
println("GRID RES: ", length(DP_grid.coordinate_vectors[1].tolist()), " in py vs $lg1 in julia")
println("UPPER: ",    maximum(DP_grid.states.tolist()), " in py vs $(cx[2]+ubc) in julia")
println("LOWER: ",    minimum(DP_grid.states.tolist()), " in py vs $(cx[1]+lbc) in julia")

## Solve BRS with DP

ϕXT_DP = []
push!(ϕXT_DP, Matrix(reshape(DP_values.tolist(), length(DP_values.tolist()), 1))[:,1])
for ts in T
    hj_r_output = hj.step(hj_r_toy.solver_settings, dynamics, DP_grid, 0., DP_values, -ts)
    push!(ϕXT_DP, Matrix(reshape(hj_r_output.tolist(), length(hj_r_output.tolist()), 1))[:,1])
end

BRS_plots = plot_BRS(T, Xgs, ϕXT_DP; M, ϵs=5e-2, interpolate=true, value_fn=false, alpha=0.5)

## Compare the Sets Quantitatively

jaccards, false_in_perc, false_ex_perc = Dict(), Dict(), Dict()
grid_total = size(Xg)[2]

for i=1:length(T)+1

    BRS_DP = Xg[:, ϕXT_DP[i]  .< 0]

    BRS_DP_set = Set()
    map(x->push!(BRS_DP_set, x), eachcol(BRS_DP))

    for key in keys(ϕXT_KH)

        BRS_KH_k = Xg[:, ϕXT_KH[key][i] .< 0]

        BRS_KH_k_set = Set()
        map(x->push!(BRS_KH_k_set, x), eachcol(BRS_KH_k))

        # Count False Inclusion
        false_in_perc_c = 0.
        for x in BRS_KH_k_set
            if x ∈ BRS_DP_set
                continue
            else
                false_in_perc_c += 100/grid_total
            end
        end

        # Count False Inclusion
        false_ex_perc_c = 0.
        for x in BRS_DP_set
            if x ∈ BRS_KH_k_set
                continue
            else
                false_ex_perc_c += 100/grid_total
            end
        end

        # Store Data
        if i == 1
            jaccards[key] = [jaccard(BRS_DP_set, BRS_KH_k_set)]
            false_ex_perc[key]  = [false_ex_perc_c]
            false_in_perc[key]  = [false_in_perc_c]
        else
            push!(jaccards[key], jaccard(BRS_DP_set, BRS_KH_k_set))
            push!(false_ex_perc[key],  false_ex_perc_c)
            push!(false_in_perc[key],  false_in_perc_c)
        end

    end
end

jaccards
false_in_perc
false_ex_perc
run_stats_pp

# save("KH_Toy_results_c01p25_up5-dp25_T5_fine_taylor_L0--.jld", 
#     "max_u_d", (max_u, max_d), 
#     "T", T, 
#     "Xg", Xg, 
#     "ϕXT_KH", ϕXT_KH, 
#     "ϕXT_DP", ϕXT_DP, 
#     "jaccards", jaccards, 
#     "false_ex_perc", false_ex_perc, 
#     "false_in_perc", false_in_perc, 
#     "run_stats_pp", run_stats_pp)

########################################################################################
## Plotting
########################################################################################

load_data = JLD.load("KH_Toy_results_c01p25_up5-dp25_T5_fine.jld") # also "KH_Toy_results_c01p25_up5-dp25_T5_fine_taylor.jld", "KH_Toy_results_c01p25_up5-dp25_T5_fine_taylor_L0.jld"
max_u, max_d = load_data["max_u_d"] 
T = load_data["T"]
Xg = load_data["Xg"]
ϕXT_KH = load_data["ϕXT_KH"] 
ϕXT_DP = load_data["ϕXT_DP"] 
jaccards = load_data["jaccards"] 
false_ex_perc = load_data["false_ex_perc"] 
false_in_perc = load_data["false_in_perc"] 
run_stats_pp = load_data["run_stats_pp"]

## Combine all Contour Plots

titles_long = Dict(
      "DP"     => L"\mathcal{R}(\mathcal{T}, t) \:\:\:", 
      "Taylor" => L"\mathcal{R}(\mathcal{T}_{Taylor}, t) \:\:\:",
      "Ball"   => L"\mathcal{R}(\mathcal{E}_{Ball}, t),\:\: \mathcal{E}_{Ball} \subseteq \widetilde{\mathcal{T}_\mathcal{G}}\:\:\:",
      "TgtE"   => L"\mathcal{R}(\mathcal{E}_{Tgt}, t),\:\: \mathcal{E}_{Tgt} \supseteq \mathcal{T}_\mathcal{G}\:\:\:",
      "CylE"   => L"\mathcal{R}(\mathcal{E}_{Cyl}, t),\:\: \mathcal{E}_{Cyl} \approx \widetilde{\mathcal{T}_\mathcal{G}}\:\:\:",
      "OutE"   => L"\mathcal{R}(\mathcal{E}_{O},t),\:\: \mathcal{E}_{O} \supseteq Conv(\mathcal{T}_\mathcal{G})\:\:\:",
      "InnE"   => L"\mathcal{R}(\mathcal{E}_{I},t),\:\: \mathcal{E}_{I} \subseteq Conv(\mathcal{T}_\mathcal{G})\:\:\:"
)

t_labels = [L"t=0.",
            L"t=0.2", "",
            L"t=0.5", "",
            L"t=1", "",
            L"t=1.5", "",
            L"t=2", ]

# titles_short = Dict(
#       "DP"   => L"S:=\mathcal{T}", 
#       "Ball" => L"S:=\mathcal{E}_{Ball} \subseteq \widetilde{\mathcal{T}_\mathcal{G}}",
#       "TgtE" => L"S:=\mathcal{E}_{Tgt}  \supseteq \mathcal{T}_\mathcal{G}",
#       "CylE" => L"S:=\mathcal{E}_{Cyl}  \approx \widetilde{\mathcal{T}_\mathcal{G}}",
#       "OutE" => L"S:=\mathcal{E}_{O}    \supseteq Conv(\mathcal{T}_\mathcal{G})",
#       "InnE" => L"S:=\mathcal{E}_{I}    \subseteq Conv(\mathcal{T}_\mathcal{G})"
# )
# sup_title = L"\mathcal{R}(\mathcal{S}, t)"

titles_shortest = Dict(
      "DP"     => L"\mathcal{R}(\mathcal{T}, t)", 
      "Taylor" => L"\mathcal{R}(\mathcal{T}_{Taylor}, t)",
      "Ball"   => L"\mathcal{R}(\mathcal{E}_{Ball}, t)",
      "TgtE"   => L"\mathcal{R}(\mathcal{E}_{Tgt}, t)",
      "CylE"   => L"\mathcal{R}(\mathcal{E}_{Cyl}, t)",
      "OutE"   => L"\mathcal{R}(\mathcal{E}_{O}, t)",
      "InnE"   => L"\mathcal{R}(\mathcal{E}_{I}, t)"
)

titles = titles_long

# ord_keys = ["Ball", "OutE", "CylE", "TgtE", "InnE"] 
ord_keys = ["CylE", "OutE", "Taylor", "Ball", "InnE"] 
Xgs = [Xg for i=1:length(T)+1] # for plotting
 
pal_colors = ["red", "blue"];
pal_colors = [palette([color, "white"], 10)[3] for color in ["red", "blue"]];
# pal_colors = palette(:tol_bright)[2:-1:1];
# pal_colors = [palette(:seaborn_muted)[4], palette(:seaborn_muted)[1]];
# palett = palette(pal_colors, 5)

pal_colors_d = Dict(
    "DP"     => pal_colors,
    "CylE"   => pal_colors,
    "OutE"   => pal_colors,
    "Taylor" => pal_colors,
    "Ball"   => pal_colors,
    "InnE"   => pal_colors,
)

# pal_colors_d = Dict(
#     "DP"     => palette(:tab20)[2:-1:1],
#     "CylE"   => palette(:tab20)[4:-1:3],
#     "OutE"   => palette(:tab20)[6:-1:5],
#     "Taylor" => palette(:tab20)[14:-1:13], # [20:-1:19]
#     "Ball"   => palette(:tab20)[10:-1:9],
#     "InnE"   => palette(:tab20)[8:-1:7],
# )

contour_plots = [];
plot_contour_DP = plot_BRS_pretty(T, Xgs, ϕXT_DP; M, ϵs=2e-2, ϵc=1e-5, interpolate=true, alpha=0.9, title=titles["DP"], pal_colors=pal_colors_d["DP"], input_labels=t_labels, thickening=true);
# Plots.savefig(plot_contour_DP[1], "toy_BRS_legend_muted----.html")
push!(contour_plots, plot_contour_DP[1])
for key in ord_keys
    pal_colors = 
    plot_contour_KH = plot_BRS_pretty(T, Xgs, ϕXT_KH[key]; M, ϵs=2e-2, ϵc=1e-6, interpolate=true, alpha=0.9, title=titles[key], pal_colors=pal_colors_d[key], thickening=true);
    push!(contour_plots, plot_contour_KH[1]);
end
contour_plot = Plots.plot(contour_plots..., layout=(2,3), extra_plot_kwargs = KW(:include_mathjax => "cdn"), size=(1000,600))
# Plots.savefig(contour_plot, "toy_test_all_fine_taylor.html")

## Plotting the Quantitative Measures (Jaccard, False Included, False Excluded, Run Time per Point)

jplot, fiplot, feplot, rtplot = Plots.plot(ylabel=L"JI"), Plots.plot(ylabel=L"FI\%"), Plots.plot(ylabel=L"FE\%", xlabel=L"t\:\:(s)"), Plots.plot(ylabel=L"ms/pt", xlabel=L"t\:\:(s)")
lw = 2; markersize=5.; alpha=0.55; T0 = vcat(0, T); xlims=[-0.025, 2.025]; 

ord_keys2 = ["Ball", "CylE", "InnE", "OutE", "Taylor"] 
colors = palette(:grays1)[1:5]
# colors = palette(:tab10)[[5,2,4,3,7]] #tab10 color matched
markers = [:circ, :utriangle, :rect, :diamond, :xcross]

for (ki, key) in enumerate(ord_keys2)
    plot!(jplot,  T0, jaccards[key],         label="", lw=lw, linestyle=:dot, alpha=alpha, xticks=(T0, ["" for i=1:6]), yticks=[0.1, 0.5, 1], color=colors[ki], marker=markers[ki], markersize=markersize, xlims=xlims, ylims=[-0.025, 1.05]);
    plot!(fiplot, T0, false_in_perc[key],    label="", lw=lw, linestyle=:dot, alpha=alpha, xticks=(T0, ["" for i=1:6]), yticks=[1, 3.5, 7],   color=colors[ki], marker=markers[ki], markersize=markersize, xlims=xlims, ylims=[-0.25, 7.0]);
    plot!(feplot, T0, false_ex_perc[key],    label="", lw=lw, linestyle=:dot, alpha=alpha, xticks=T0, yticks=[10, 30, 60],  color=colors[ki], marker=markers[ki], markersize=markersize, xlims=xlims, ylims=[-2.5, 65]);
    plot!(rtplot, T, 1000*run_stats_pp[key], label=titles_shortest[key], lw=lw, linestyle=:dot, alpha=alpha, xticks=T0, yticks=[10, 30, 60], color=colors[ki], marker=markers[ki], markersize=markersize, xlims=xlims, ylims=[-2.5, 65]);
    # plot!(rtplot, T, run_stats_pp[key], label=titles_shortest[key], lw=lw, linestyle=:dot, alpha=alpha, xticks=T0, yticks=([1e-3, 1e-2, 10^(-1.2)], ["1e-3", "1e-2", "1e-1.2"]), color=colors[key], marker=markers[key], markersize=2., xlims=xlims, ylims=[1e-4 - 5e-5, 10^(-1.2)]);
end

blankspace = Plots.plot(grid = false, axis = false, bordercolor = "white", xticks=nothing, yticks=nothing, bottom_margin = -50Plots.px); 
qBRS_plot_1 = Plots.plot(jplot, rtplot, layout=(2,1), bottom_margin = -8Plots.px);
qBRS_plot_2 = Plots.plot(fiplot, feplot, layout=(2,1), bottom_margin = -8Plots.px);
# qBRS_plot_sbs = Plots.plot(blankspace, qBRS_plot_1, qBRS_plot_2, layout=(1,3), right_margin=10Plots.px, legend=:outerbottomleft)
# qBRS_plot_sbs = Plots.plot(blankspace, qBRS_plot_1, blankspace, qBRS_plot_2, blankspace, layout=@layout([a{0.1w} b c{0.1w} d e{0.1w}]), legend=:outerbottomright, right_margin=10Plots.px,)
qBRS_plot_sbs = Plots.plot(qBRS_plot_1, qBRS_plot_2, blankspace, layout=@layout([a b c{0.1w}]), legend=:outerbottomright, right_margin=10Plots.px,)
# Plots.savefig(qBRS_plot_1, "SlowManifold_qBRS_legend.svg")

## Combining all into one mega plot (jesus)

# mega = Plots.plot(contour_plot, blankspace, qBRS_plot, layout=@layout([a; b{0.1h}; c{0.3h}]))
# mega_s = Plots.plot(contour_plot, blankspace, qBRS_plot_s, layout=@layout([a; b{0.1h}; c{0.3h}]))
mega_sbs = Plots.plot(contour_plot, blankspace, qBRS_plot_sbs, layout=@layout([a; b{0.12h}; c{0.3h}]), legend=:bottomright)

# Plots.savefig(mega_sbs, "SlowManifold_mega_sbs_7_grays1_thic_taylor_resize.html")

## Print Averages for Table 

for key in ord_keys

    mean_j   = mean(jaccards[key][2:end])
    mean_fip = mean(false_in_perc[key][2:end])
    mean_fep = mean(false_ex_perc[key][2:end])
    mean_tc  = mean(1000*run_stats_pp[key][2:end])

    println()
    println(key)
    println("mean_j   :", mean_j) 
    println("mean_fip :", mean_fip)
    println("mean_fep :", mean_fep)
    println("mean_tc  :", mean_tc)

end

## Combined Plotting 

# load("KH_Toy_results_c01p25_up5-dp25_T5.jld")
# ("max_u_d", (max_u, max_d), "T", T, "Xg", Xg, "ϕXT_KH", ϕXT_KH, "ϕXT_DP", ϕXT_DP, "jaccards", jaccards)

# time_pt = 3
# fake_T = collect(1:6)
# ord_keys = ["Ball", "CylE", "TgtE", "OutE", "InnE"]

# input_labels = [L"\mathcal{T}", 
#                 L"\mathcal{R}(\mathcal{T}, t)", "", 
#                 L"\mathcal{R}(\mathcal{E}_{Ball}, t),\:\: \mathcal{E}_{Ball} \subseteq \widetilde{\mathcal{T}_\mathcal{G}}", "",
#                 L"\mathcal{R}(\mathcal{E}_{Cyl}, t),\:\: \mathcal{E}_{Cyl} \approx \widetilde{\mathcal{T}_\mathcal{G}}", "",
#                 L"\mathcal{R}(\mathcal{E}_{Tgt}, t),\:\: \mathcal{E}_{Tgt} \supset \mathcal{T}_\mathcal{G}", "",
#                 L"\mathcal{R}(\mathcal{E}_{O},t),\:\: \mathcal{E}_{O} \supset \mathcal{T}_\mathcal{G}", "",
#                 L"\mathcal{R}(\mathcal{E}_{I},t),\:\: \mathcal{E}_{I} \subset Conv(\mathcal{T}_\mathcal{G})", "",]

# BRS_plots = plot_BRS_pretty(fake_T, [Xg for i=1:length(fake_T)+1], [ϕXT_DP[1], ϕXT_DP[time_pt], [ϕXT_KH[key][time_pt] for key in ord_keys]...]; 
#                     M, ϵs=[0.05, 0.05, 0.2, 0.025, 0.025, 0.05], interpolate=true, value_fn=false, alpha=0.7, 
#                     latex_title=true, legend=true, legendfontsize=6, input_labels=input_labels)

# Plots.savefig(BRS_plots[1], "toy_test.html")

# BRS_plots_tp = []
# for (tpi, time_pt) in enumerate([2, 4, 6])

#     legend = time_pt == 2 ? true : false
#     title = [L"t=0.66s", L"t=1.33s", L"t=2s"][tpi]
#     # title = [L"\text{BRS}: \:\: \phi(X, t=0.66 s) = 0", L"\text{BRS}: \:\: \phi(X, t=1.33s) = 0", L"\text{BRS}: \:\: \phi(X, t=2s) = 0"][tpi]

#     BRS_plots = plot_BRS_pretty(T[1:3], [Xg for i=1:length(T)], [ϕT[1], ϕT[time_pt], ϕXT[time_pt], ϕGT[time_pt]]; 
#             M, ϵs=[1e-1, time_pt/3 * 1e-1, 5e-1, 1e-1], ϵc=2e-3, interpolate=false, value_fn=false, alpha=0.7, 
#             title, latex_title=true, legend, legendfontsize=12, input_labels=[L"\mathcal{T}", L"\mathcal{R}(\mathcal{T}, t)", L"\mathcal{R}(\mathcal{T})", L"\mathcal{R}(\mathcal{T}_{\text{Taylor}}, t)", L"\mathcal{R}(\mathcal{T}_{\text{Taylor}}, t)", L"\mathcal{R}(\widetilde{\mathcal{T}_\mathcal{G}}, t)", L"\mathcal{R}(\widetilde{\mathcal{T}_\mathcal{G}}, t)"])

#     Plots.savefig(BRS_plots[1], "later_test_tp$time_pt.html")

#     push!(BRS_plots_tp, BRS_plots[1])
# end

# BRS_pannel = Plots.plot(BRS_plots_tp..., layout=(3,1), size=(400, 1200), extra_plot_kwargs = KW(:include_mathjax => "cdn"))
# Plots.savefig(BRS_pannel, "BRS_pannel_nc_sp.html")
function plot_BRS_pretty(T, B⁺T, ϕB⁺T; M=nothing, simple_problem=true, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, 
    zplot=false, interpolate=false, inter_method=Polyharmonic(), pal_colors=[:red, :blue], alpha=0.5, 
    title=nothing, value_fn=false, nx=size(B⁺T[1])[1], xlims=[cx[1]+lbc, cx[1]+ubc], ylims=[cx[2]+lbc, cx[2]+ubc+0.1], latex_title=false, 
    legend=true, input_labels=nothing, legendfontsize=12, thickening=false)

    if nx > 2 && value_fn; println("4D plots are not supported yet, can't plot Value fn"); value_fn = false; end

    Xplot = isnothing(title) ? Plots.plot(title=latex_title ? L"\text{BRS}: \:\: \phi(X, t) = 0" : "BRS: Φ(X, t) = 0", extra_plot_kwargs = KW(:include_mathjax => "cdn")) : Plots.plot(title=title, extra_plot_kwargs = KW(:include_mathjax => "cdn"))
    if zplot; Zplot = Plots.plot(title="BRS: Φ(Z, T) = 0"); end

    # plot!(Xplot, xlabel=L"x_1", xlabel=L"x_2")
    annotate!(Xplot, xlims[1]+0.15, ylims[1]+0.3, text(L"\mathcal{X}", 15))

    plots = zplot ? [Xplot, Zplot] : [Xplot]
    if value_fn; vfn_plots = zplot ? [Plots.plot(title="Value: Φ(X, T)"), Plots.plot(title="Value: Φ(Z, T)")] : [Plots.plot(title=latex_title ? L"V: \phi (X, T)" : "Value: Φ(X, T)")]; end

    B⁺Tc, ϕB⁺Tc = copy(B⁺T), copy(ϕB⁺T)
    
    ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
    Jlabels = "J(⋅, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
    labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

    Tcolors = length(T) > 1 ? palette(pal_colors, length(T)) : [pal_colors[2]]
    # B0colors = length(T) > 1 ? palette([:black, :gray], length(T)) : [:black]
    # Tcolors = palette(:seaborn_colorblind)
    # Tcolors = [palette(:seaborn_colorblind)[i] for i in [1,3,2]]
    # Tcolors = palette(:tab10)[1:length(T)]

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
                println(label, " --- ", ϵss, j)

                scatter!(plots[Int(bi > 2) + 1], [b[i,:] for i=1:nx]..., label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha)
                # scatter!(plots[Int(bi > 2) + 1], b[1,:], b[2,:], label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                
                if value_fn
                    scatter!(vfn_plots[Int(bi > 2) + 1], b⁺[1,:], b⁺[2,:], ϕ, label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha, xlims, ylims)
                    # scatter!(vfn_plots[Int(bi > 2) + 1], b[1,:], b[2,:], ϕ, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], label=label)
                end
            
            ## Plot Interpolation
            else 

                if nx == 2
                    contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] for i=1:nx]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label, linewidth=3, alpha=alpha)
                    if thickening
                        for wc in [[1, 1], [1, -1], [-1, -1], [-1, -1]]
                            wcϵ = 0.005
                            contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] .+ wcϵ*wc[i] for i=1:nx]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label, linewidth=3, alpha=alpha)
                        end
                    end

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

    plot!(Xplot, xlims=xlims, ylims=ylims, legendfontsize=legendfontsize, legend=legend)

    display(Xplot); 
    plots = [Xplot]
    if zplot; display(Zplot); plots = [Xplot, Zplot]; end

    return plots
end

plot_contour_DP = plot_BRS_pretty(T, Xgs, ϕXT_KH["TgtE"]; M, ϵc=1e-6, interpolate=true, alpha=0.7, title=titles["DP"], pal_colors=pal_colors)

########################################################################################
## Tight Ellipse test
########################################################################################

c = 2.; r = 1.

ϵ = 0.5e-7; N = 10 + ϵ
x1g = collect(-10. : 1/N : 10.);

scatter();
contour!(x1g, x2g, (x,y) -> x^2 - y, lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="black")
contour!(x1g, x2g, (x,y) -> 2 * c * x + (r^2 - c^2) - y, lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="orange")
contour!(x1g, x2g, (x,y) -> (x - c)^2 - r, lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="green")

# scatter!(x1g, 2 * c * x1g .+ (r^2 - c^2))
scatter!([c-r, c+r], [0, 0], label="Target Bounds")
scatter!([c-r, c+r], [(c-r)^2, (c+r)^2], label="Lifted Target Bounds")
# scatter!([c-r+2r, c+r], [(c-r)^2, (c+r)^2 - 4*c*r], label="")
scatter!([c], [0], label="Target Center")
scatter!([c], [c^2], label="Lifted Target Center")

x2g = collect(-5 : 1/N : 16.)
cgog = [c; c^2];
cg = [c; c^2 + r^2];

# scatter!(eachrow(cg)..., label="")

a = r^2 * sqrt(1+4*c^2); # long enough to contain the two points
a2 = 0.5 #user choice rn
A = diagm([a, a2].^2); ϵc = 0.001
# contour!(x1g, x2g, (x,y) -> (vcat(x,y) - cg)' * inv(A) * (vcat(x,y) - cg) - 1, lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="blue", xlims=[-2.5, 7.5], ylims=[-0.1, 10])

W = [1 -2*c; 2*c 1] / sqrt(1 + 4c^2)
# contour!(x1g, x2g, (x,y) -> W[1,:] * vcat(x,0), lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="blue", xlims=[-0.1, 4], ylims=[-0.1, 16])
# scatter!(eachrow(W * vcat(x1g', zero(x1g)'))..., color="blue")

contour!(x1g, x2g, (x,y) -> (vcat(x,y) - cg)' * inv(W)' * inv(A) * inv(W) * (vcat(x,y) - cg) - 1, lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="red", xlims=[-2.5, 7.5], ylims=[-0.1, 10], label="Tight-Bound Ellipse")

A2 = diagm([r, 10].^2);
contour!(x1g, x2g, (x,y) -> (vcat(x,y) - cgog)' * inv(A2) * (vcat(x,y) - cgog) - 1, lw=2, cbar=false, levels=-ϵc:ϵc:ϵc, color="blue", xlims=[-2.5, 7.5], ylims=[-0.1, 10], label="Augmented-Target Ellipse")

c = 2.; r = 1.;
cg = [c; c^2 + r^2]
a = r^2 * sqrt(1 + 4c^2)
a2 = 0.5; #user choice rn
A = diagm([a, a2].^2)
W = [1 -2*c; 2*c 1] / sqrt(1 + 4c^2)
A_Et = inv(W)' * inv(A) * inv(W)

# in 3d
# W = [1 0 -2*c; 0 sqrt(1 + 4c^2) 0; 2*c 0 1] / sqrt(1 + 4c^2)
# A = diagm([a, 1, a2].^2)
# A_Et = inv(W)' * inv(A) * inv(W)

########################################################################################
## Loewner-John Ellipsoid Demonstration
########################################################################################

# Target Definition
c1p = 2.;
rp = 1.0;
cp = [c1p; c1p; c1p^2] # lifted center

# True-space Gridding
ϵ = 0.5e-7; N = 10 + ϵ
x1g = collect(0. : 1/N : 4.) .+ ϵ;
x2g = collect(0. : 1/N : 4.) .+ ϵ;
Xg = hcat(collect.(Iterators.product(x1g, x2g))...)

# Find target boundary
Xgϕ = diag((Xg .- cp[1:2])' * (Xg .- cp[1:2]))/2 .- 0.5 * rp^2;
Xgc = Xg[:, abs.(Xgϕ) .< 0.05]
scatter(eachrow(Xgc)...)

# Lift target boundary
Ggc = vcat(Xgc, Xgc[1,:]'.^2)
scatter(eachrow(Ggc)...)

# Solve Outer ellipse
ljoe = LoewnerJohnEllipsoids.outer_ellipsoid([eachcol(Ggc)...])

# Koopman-space Gridding
gg = collect(0. : 1/N : 16) .+ ϵ;
Gg = hcat(collect.(Iterators.product(x1g, x2g, gg))...)

# Find Outer boundary in Koopman Space
c_ljoe, A_ljoe = ljoe.d, ljoe.P
ϕ_outer = diag((Gg .- c_ljoe)' * A_ljoe * (Gg .- c_ljoe))/2 .- 0.5;
Ggc2 = Gg[:, abs.(ϕ_outer) .< 0.3]
scatter(eachrow(Ggc)..., label="Lifted Target Boundary")
scatter!(eachrow(Ggc2)..., alpha=0.6, label="Outer Ellipse")

# Defind Hull of Lifted Target and Solve Inner Ellipse
hc = hrep(polyhedron(convexhull([Ggc[:,i] for i=1:size(Ggc)[2]]...))) # h-rep of lifted target boundary hull
constraints = [LoewnerJohnEllipsoids.LinearConstraint(hs.a, hs.β) for hs in allhalfspaces(hc)] # poly conversion
ljie = LoewnerJohnEllipsoids.inner_ellipsoid(LoewnerJohnEllipsoids.Polyhedron(constraints)) # Solve inner ellipse
c_ljie, A_ljie = ljie.d, ljie.P

# Find Inner boundary in Koopman Space
ϕi = diag((Gg .- c_ljie)' * A_ljie * (Gg .- c_ljie))/2 .- 0.5;
Ggc_inner = Gg[:, ϕi .< 0.3]
scatter!(eachrow(Ggc_inner)..., alpha=0.6, label="Inner Ellipse")

########################################################################################
## Observe the Flows of this system
########################################################################################

# using DifferentialEquations
# gr()

# function Kf(g, p, t)
#     μ, λ, α = p
#     # K = [μ 0 0; 0 λ -λ; 0 0 2μ]
#     # K = [μ 0 0; 0 λ -λ*(1 + α*sin(2*g[1])); 0 0 2μ]
#     K = [μ α -α; α λ -λ; α α 2μ]
#     K * g
# end

# μ, λ, α = -0.05, -1., 0.05
# g0 = [5; 5; 25]
# tspan = (0, 100)
# p = [μ, λ, α]
# prob = ODEProblem(Kf, g0, tspan, p)
# sol = solve(prob)
# plot(sol, idxs = (1, 2, 3))

# X0 = (5 * rand(2, 10) .- 5/2) .+ 5 
# # x1g = collect(-2.5 : 1/5 : 2.5); x2g = collect(-3 : 1/5 : 5);
# # X0 = hcat(collect.(Iterators.product(x1g, x2g))...)[end:-1:1,:]
# G0 = vcat(X0, X0[1,:]'.^2)
# prob = ODEProblem(Kf, G0, tspan, p)
# sol = solve(prob); sol_arr = Array(sol)

# ## Plotting

# True, Koop = plot(title=L"True"), plot(title=L"Lifted")
# for i=1:size(sol_arr)[2]
#     plot!(Koop, eachcol(sol_arr[:,i,:]')..., label="")
#     plot!(True, eachcol(sol_arr[1:2,i,:]')..., label="")
# end
# plot!(True, xlabel=L"x_1", ylabel=L"x_2")
# plot!(Koop, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"g_3")
# plot(True, Koop, layout=@layout[a{0.4w} b])

# x1g = collect(minimum(sol_arr[1,:,:])-0.1 : 1/5 : maximum(sol_arr[1,:,:])+0.1) 
# x2g = collect(minimum(sol_arr[2,:,:])-0.1 : 1/5 : maximum(sol_arr[2,:,:])+0.1)
# # Xg = hcat(collect.(Iterators.product(x1g, x2g))...)[end:-1:1,:]
# sKoop = surface(Koop, x1g, x2g, (x1, x2) -> x1^2, opacity=0.3, color=:red, colorbar=false)
# surface!(sKoop, x1g, x2g, (x1, x2) -> x2, opacity=0.3, color=:black, colorbar=false)

# eqg = x1g.^2 .* (1 .+ α*sin.(2*x1g))
# sTrue = plot(True, x1g, eqg, color="black", linewidth=2, label="")
# plot!(sKoop, x1g, eqg, eqg, color="black", linewidth=2, label="")

# plot(sTrue, sKoop, layout=@layout[a{0.4w} b])
