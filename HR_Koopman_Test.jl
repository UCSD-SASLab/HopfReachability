
### Hopf Reachability of Textbook Koopman Systems
# wsharpless@ucsd.edu

using LinearAlgebra, Plots, JLD
push!(LOAD_PATH,"/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachabilityv2: Hopf_BRS, intH_ytc17, preH_ytc17, plot_BRS

########################################################################################
## Textbook Koopman System (Exact Linearizaton) - Autonomous
########################################################################################

μ, λ = -0.05, -1.

## System
# ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
dim = 2
M = [λ -λ;
     0 2μ]
B, C = [0. 1.; 2μ 0.], [0. 1.; 2μ 0.]
Q, Q2 = zeros(dim,dim), zeros(dim,dim)
a1, a2 = zeros(1, dim), zeros(1, dim)

system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.05
Th = 1.0
Tf = 1.0
T = collect(Th : Th : Tf)

## Target
# J(x) = 0 is the boundary of the target
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
lll = 20
max_runs = 20
opt_p = (vh, L, tol, lim, lll, max_runs)

## Run the solver
solution, averagetime = Hopf_BRS(system, target, intH_ytc17, T; 
                                                    preH=preH_ytc17,
                                                    th,
                                                    grid_p,
                                                    opt_p,
                                                    check_all=true,
                                                    printing=true);
B⁺T, ϕB⁺T = solution;

# using JLD
# save("KHR_test.jld", "solution", solution)
# B⁺T, ϕB⁺T = load("HR_v2_solution.jld", "solution");

plot = plot_BRS(T, B⁺T, ϕB⁺T; M, cres=0.01, contour=true);


########################################################################################
## Textbook Koopman System (Exact Linearizaton) - Controlled
########################################################################################

plts = []
pal_colors_list = [[:blue :blue], [:red :red], [:green :green], [:yellow :yellow]]

function merge_series!(sp1::Plots.Subplot, sp2::Plots.Subplot)
     append!(sp1.series_list, sp2.series_list)
     Plots.expand_extrema!(sp1[:xaxis], xlims(sp2))
     Plots.expand_extrema!(sp1[:yaxis], ylims(sp2))
     Plots.expand_extrema!(sp1[:zaxis], zlims(sp2))
     return sp1
 end
 
 function merge_series!(plt, plts...)
     for (i, sp) in enumerate(plt.subplots)
         for other_plt in plts
             if i in eachindex(other_plt.subplots)
                 merge_series!(sp, other_plt[i])
             end
         end
     end
     return plt
 end

for (ri, r) in enumerate([0., 1., 5., 10.])

     Q, Q2 = r^2 * diagm(ones(dim)), r^2 * 0.5 * diagm(ones(dim))
     system = (M, B, C, Q, Q2, a1, a2)

     solution, averagetime = Hopf_BRS(system, target, intH_ytc17, T; 
                                                       preH=preH_ytc17,
                                                       th,
                                                       grid_p,
                                                       opt_p,
                                                       check_all=true,
                                                       printing=true);
     B⁺T, ϕB⁺T = solution;

     plot = plot_BRS(T, B⁺T, ϕB⁺T; M, cres=0.01, contour=true, pal_colors = pal_colors_list[ri]);
     push!(plts, plot)
end

plot = merge_series!([plts[i][1] for i in eachindex(plts)]...)


