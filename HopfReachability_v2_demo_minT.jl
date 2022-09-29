
### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, Plots, JLD
# push!(LOAD_PATH,"/Users/willsharpless/Documents/Herbert/Koop_HJR/HL_fastHJR");
push!(LOAD_PATH, "/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR")
using HopfReachabilityv2: Hopf_BRS, intH_ytc17, preH_ytc17, plot_BRS, Hopf_minT, HJoc_ytc17 

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
dim = 10
# M = [0. 1; -2 -3]
# M = [0. 1 0; -2 -3 1; 0 2 -1]
M = rand(dim,dim)
C = 0.5 * I
# C2 = 0.5 * [2 0; 0 1]
C2 = 0.2 * I
Q = 0.1 * diagm(0 => 3*ones(dim), -1 => 1*ones(dim-1), 1 => 1*ones(dim-1))
Q2 = 0.2 * diagm(0 => 2*ones(dim), -1 => 1*ones(dim-1), 1 => 1*ones(dim-1))
# a1 = [0.5 0.75 0.5]
a1 = rand(1, dim)
# a2 = -[0.5 0 0.1]
a2 = rand(1, dim)
system = (M, C, C2, Q, Q2, a1, a2)

## Time
th = 0.02
Th = 0.1
Tf = 0.4
T = collect(Th : Th : Tf)

## Target
# J(x) = 0 is the boundary of the target
# A = [exp(-Ti * M)' * diagm(inv.(cat([2.5, 1], 0.5*ones(dim - 2), dims=1)).^2) * exp(-Ti * M) for Ti in T]
A = diagm(inv.(cat([2.5, 1], 0.5*ones(dim - 2), dims=1)).^2)
J(x::Vector, Ap) = (x' * Ap * x)/2 .- 0.5
Js(x::Vector, Ap) = (x' * inv(Ap) * x)/2 .+ 0.5
J(x::Matrix, Ap) = diag(x' * Ap * x)/2 .- 0.5
Js(x::Matrix, Ap) = diag(x' * inv(Ap) * x)/2 .+ 0.5
target = (J, Js, A)

## Grid Parameters (optional, deafult here)
bd = (2., 8.)
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 5 # min convergent runs
max_runs = 20
opt_p = (vh, L, tol, lim, lll, max_runs)

## Run the BRS solver
println("Computing BRS,")
solution, averagetime = Hopf_BRS(system, target, intH_ytc17, T; 
                                                    preH=preH_ytc17,
                                                    grid_p,
                                                    opt_p,
                                                    printing=true);
B⁺T, ϕB⁺T = solution;

# save("HR_v2_solution_sp.jld", "solution", solution)

# solution = load("HR_v2_solution_sp.jld", "solution");
# B⁺T, ϕB⁺T = solution;

plots = plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem = true, cres=0.05, zplot=true, contour=true);




# scatter(B⁺T[5][1,:], B⁺T[5][2,:], color=(ϕB⁺T[5]))
# ## Find a valid x for checking minT
# z = [-2.4, 0.75]
# scatter!([z[1]], [z[2]], markersize = 2, label="z") 
# tm = 0.6
# # tp = 0.65 
# tp = 0.5
# x = exp(tm * M) * z
# zp = exp(-tp * M) * x
# scatter!([zp[1]], [zp[2]], markersize = 2, label="zp") 

# ## Run the minT solver
# println("For x, computing minimum reachable time and corresponding control,")
# solution, uˢ, Tˢ = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, x; 
#                                                     preH=preH_ytc17, 
#                                                     time_p=(0.02, 0.1, 1), 
#                                                     opt_p=(0.01, 5, 0.5e-7, 500, 20),
#                                                     printing=true)


# ## debug

# intH = intH_ytc17
# HJoc = HJoc_ytc17
# preH = preH_ytc17
# time_p = (0.02, 0.1, 1)
# opt_p = (0.01, 5, 0.5e-7, 500, 20)
# T_init = nothing 

# J, Jˢ, A = target
# th, Th, maxT = time_p
# tvec = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)

# if !(typeof(A) <: Matrix || typeof(A) <: Vector)
#     error("A needs to be either a matrix (still target) or array of matrices (time-varying target)")
# elseif typeof(A) <: Matrix
#     moving_target = false
# else 
#     moving_target = true
# end

# ## Initialize solutions
# solution, dϕdz = Inf, 0
# v_init = nothing 

# ## Precomputing some of H for efficiency
# Hmats = preH(system, tvec)

# ## Loop Over Time Frames until ϕ(z, T) > 0
# Thi = 0; Tˢ = tvec[end]
# tix = length(tvec)