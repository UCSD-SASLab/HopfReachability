
### Hopf Reachability of Textbook Koopman Systems (Exact Linearization)
# wsharpless@ucsd.edu

using LinearAlgebra, Plots, JLD
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, plot_nice, Hopf


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

## Points to Solve (optional, deafult here)
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
opt_p_cd = (vh, L, tol, lim, lll, max_runs)

# Hopf ADMM Parameters (default)
ρ, ρ2 = 1e-4, 1e-4
tol = 1e-5
max_its = 10
opt_p_admm = (ρ, ρ2, tol, max_its)

## Run the solver
solution, run_stats = Hopf_BRS(system, target, T;
                                                    opt_method = Hopf_cd,
                                                    th,
                                                    grid_p,
                                                    opt_p = opt_p_cd,
                                                    warm=false,
                                                    check_all=true,
                                                    printing=true);

B⁺T, ϕB⁺T = solution;

# save("KHR_test.jld", "solution", solution)
# B⁺T, ϕB⁺T = load("KHR_solution.jld", "solution");

plot_scatter = plot_nice(T, solution; M, ϵs=0.1, interpolate=false, value_fn=true, alpha=0.1)
plot_contour = plot_nice(T, solution; M, ϵc=0.001, interpolate=true, value_fn=true, alpha=0.5)


########################################################################################
## 2D, Controlled & Disturbed
########################################################################################

plts = []
pal_colors_list = [["blue", "blue"], ["red", "red"], ["green", "green"], ["yellow", "yellow"]]

## Functions just for merging the plots
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

for (ri, r) in enumerate([0.01, 1., 5., 10.])

     Q, Q2 = r^2 * diagm(ones(dim)), r^2 * 0.5 * diagm(ones(dim)) #Controlled and Disturbed
     system = (M, B, C, Q, Q2, a1, a2)

     solution, run_stats = Hopf_BRS(system, target, T;
                                                       opt_method = Hopf_cd, 
                                                       th,
                                                       grid_p,
                                                    #    opt_p = opt_p_admm,
                                                       warm=true,
                                                       check_all=true,
                                                       printing=true);
     B⁺T, ϕB⁺T = solution;

    #  plot_contour = plot_nice(T, solution; M, cres=0.01, interpolate=true, pal_colors = pal_colors_list[ri]);
    #  plot_scatter = plot_nice(T, solution; M, ϵs=0.1, interpolate=false, value_fn=true, alpha=0.5)
     plot_contour = plot_nice(T, solution; M, ϵc=0.001, interpolate=true, value_fn=false, alpha=0.5, pal_colors = pal_colors_list[ri])
     push!(plts, plot_contour)
end

plot_contours = merge_series!([plts[i][1] for i in eachindex(plts)]...)



########################################################################################
## 3D, Autonomous
########################################################################################

## System: ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
dim = 3
M = [ λ -λ/2 -λ/2;
      0 2μ  0;
      0  0 2μ]
B = [ 0  0  1; 
     2μ  0  0;
      0  2μ 0]
C = copy(B)

Q, Q2 = zeros(dim, dim), zeros(dim, dim) #Autonomous
a1, a2 = zeros(1, dim), zeros(1, dim)
system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.05
Th = 0.2
Tf = 1.0
T = collect(Th : Th : Tf)

## Target: J(x) = 0 is the boundary of the target
Ap = diagm(inv.(cat([1, 1, 1], dims=1)).^2)
cp = [5.; 5.; 5.]
J(x::Vector, A, c) = ((x - c)' * A * (x - c))/2 - 0.5 #don't need yet
Js(v::Vector, A, c) = (v' * inv(A) * v)/2 + c'v + 0.5
J(x::Matrix, A, c) = diag((x .- c)' * A * (x .- c))/2 .- 0.5
Js(v::Matrix, A, c) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 #don't need yet
target = (J, Js, (Ap, cp))

## Points to Solve (optional, deafult here)
bd = (2, 8)
ϵ = 0.5e-7
N = 3 + ϵ
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
solution, run_stats = Hopf_BRS(system, target, T;
                                                    opt_method = Hopf_cd,
                                                    th,
                                                    grid_p,
                                                    opt_p,
                                                    warm = false,
                                                    check_all = true,
                                                    printing = true);
B⁺T, ϕB⁺T = solution;

# plot = plot_nice(T, solution; M, cres=0.01, contour=true);

plot_scatter = plot_nice(T, solution; M, ϵs=0.1, interpolate=false, alpha=0.1)
plot_contour = plot_nice(T, solution; M, ϵc=0.001, interpolate=true, alpha=0.2);



########################################################################################
## Speed Test - ND, Controlled & Disturbed
########################################################################################

dims = collect(2:2:20)
run_stats_dims = []

r = 1. 
th = 0.05
T = [1.0]

bd = (2, 8)
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N)

for dim in dims
    println("\nFor ", dim, "...")

    M = vcat(vcat(λ, -λ/(dim-1)*ones(dim-1))', hcat(zeros(dim-1), 2μ * diagm(ones(dim-1))))
    B = vcat(vcat(zeros(dim-1), 1)', hcat(2μ * diagm(ones(dim-1)), zeros(dim-1)))
    C = copy(B)

    Q, Q2 = r^2 * diagm(ones(dim)), r^2 * 0.5 * diagm(ones(dim)) #Controlled and Disturbed
    a1, a2 = zeros(1, dim), zeros(1, dim)
    system = (M, B, C, Q, Q2, a1, a2)

    Ap = diagm(inv.(ones(dim)).^2)
    cp = 5 * ones(dim)
    target = (J, Js, (Ap, cp))

    ## Run the solver
    solution, run_stats = Hopf_BRS(system, target, T;
                                                        opt_method = Hopf_cd,
                                                        th,
                                                        grid_p,
                                                        opt_p = opt_p_cd,
                                                        check_all = true,
                                                        sampling = true,
                                                        samples = 1000,
                                                        printing = true);
    
    push!(run_stats_dims, run_stats)
end

plot(dims, [run_stats_dims[i][2][1] for i=1:length(dims)], yerror=[run_stats_dims[i][3][1] for i=1:length(dims)], ylabel="Seconds", xaxis="Dimension", title="Computation Time per Point", label="HopfReachability.jl")
