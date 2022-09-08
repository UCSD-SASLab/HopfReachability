
### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, Plots
push!(LOAD_PATH,"/Users/willsharpless/Documents/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachabilityv2: Hopf_BRS, intH_ytc17, preH_ytc17, boundary, plot_BRS

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
dim = 2
M = [0. 1; -2 -3]
C = 0.5 * I
C2 = 0.5 * [2 0; 0 1]
Q = 0.1 * diagm(0 => 3*ones(dim), -1 => 1*ones(dim-1), 1 => 1*ones(dim-1))
Q2 = 0.2 * diagm(0 => 2*ones(dim), -1 => 1*ones(dim-1), 1 => 1*ones(dim-1))
a1 = [0.5 0.75]
a2 = -[0.5 0]
system = (M, C, C2, Q, Q2, a1, a2)

## Time
th = 0.02
Th = 0.1
Tf = 0.7
T = collect(Th : Th : Tf)

## Target
# J(x) = 0 is the boundary of the target
A = [exp(-Ti * M)' * diagm(inv.(cat([2.5, 1], 0.5*ones(dim - 2), dims=1)).^2) * exp(-Ti * M) for Ti in T]
J(x::Vector, Ap) = (x' * Ap * x)/2 .- 0.5
Js(x::Vector, Ap) = (x' * inv(Ap) * x)/2 .+ 0.5
J(x::Matrix, Ap) = diag(x' * Ap * x)/2 .- 0.5
Js(x::Matrix, Ap) = diag(x' * inv(Ap) * x)/2 .+ 0.5
target = (J, Js, A)

"""
IF J(x, A) := (x' * Ap * x)/2 - 0.5 => Js(v, A) = (v' * inv(A) * v)/2 + 0.5

= J(exp(T * M) * z, A) = (z' * exp(T * M)' * A * exp(T * M) * z)/2 - 0.5

= J(z, Az) if Az := exp(T * M)' * A * exp(T * M)

Then similary Js(v, A) (for x) = Js(v, Az) (for z)
"""

## Grid Parameters (optional, deafult here)
bd = 3
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N)

## Back Computing Xg 
zig = collect(-bd : 1/N : bd); lg = length(zig)
Zg = hcat(collect.(Iterators.product(zig .- ϵ, zig .+ ϵ))...)[end:-1:1,:]
Xg = []; for Ti in T; push!(Xg, exp(Ti * M) * Zg); end

# Xg = Xg[end:-1:1,:]
# kkk = 2

# plot()
# # scatter!(Xg[1,:], Xg[2,:], alpha=0.4)
# # solution = J(vcat(Xg, zeros(kkk-2, lg^2)), exp(0.1 * M') * A[1] * exp(0.1 * M))
# # solution_sq = reshape(solution', lg, lg) 
# # contour!(xig .+ ϵ, xig .- ϵ, solution_sq, levels=[-ϵ, ϵ], colorbar = false)

# for (Tix, Ti) in enumerate(T)
#     solution = J(vcat(Xg[Tix], zeros(kkk-2, lg^2)), A[Tix])
#     solution_sq = reshape(solution', lg, lg) 
#     contour!(zig .+ ϵ, zig .- ϵ, solution_sq, levels=[-ϵ, ϵ], colorbar = false)

# #     Xg_actually = exp(Ti * M) * Xg
# #     scatter!(Xg_actually[1,:], Xg_actually[2,:], alpha=0.4)
 
#     # ZgT[Tix] = exp(-Ti * M) * Xg[Tix]
#     # scatter!(ZgT[Tix][1,:], ZgT[Tix][2,:], alpha=0.4)
# end
# plot!()

## Hopf Coordinate-Descent Parameters (optional, deafult here)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 20
opt_p = (vh, L, tol, lim, lll)

## Run the solver
solution, averagetime = Hopf_BRS(system, target, intH_ytc17, T; 
                                                    Xg,
                                                    preH=preH_ytc17,
                                                    grid_p,
                                                    opt_p,
                                                    printing=true,
                                                    plotting=true,
                                                    zplot=true);
B⁺T, ϕB⁺T = solution;

using JLD
# B⁺T, ϕB⁺T = load("HR_v2_solution.jld", "solution");
# plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem = false, ϵplot = 0.1, zplot=true)

# using JLD
# save("HR_v2_solution_3.jld", "solution", solution)

# B⁺T, ϕB⁺T = load("HR_v2_solution.jld", "solution");
plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem = false, ϵ = 0.1, zplot=true)

