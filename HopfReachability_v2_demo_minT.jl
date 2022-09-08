
### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, Plots
push!(LOAD_PATH,"/Users/willsharpless/Documents/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachabilityv2: Hopf_BRS, intH_ytc17, preH_ytc17, boundary, Hopf_minT, HJoc_ytc17 

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

## Lookback Time(s)
Th = 0.1
Tf = 0.3
T = collect(Th : Th : Tf)

## Target
# J(x) = 0 is the boundary of the target
A = [exp(-Ti * M)' * diagm(inv.(cat([2.5, 1], 0.5*ones(dim - 2), dims=1)).^2) * exp(-Ti * M) for Ti in T]
J  = (x, Ap) -> (x' * Ap * x)/2 - 0.5
Js  = (x, Ap) -> (x' * inv(Ap) * x)/2 + 0.5
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
th = 0.02
grid_p = (bd, ϵ, N, th)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 20
opt_p = (vh, L, tol, lim, lll)

## Run the BRS solver
println("Computing BRS,")
solution, averagetime = Hopf_BRS(system, target, intH_ytc17, T; 
                                                    preH=preH_ytc17,
                                                    grid_p,
                                                    opt_p,
                                                    plotting=true,
                                                    printing=true);
plot!(title="BRS for T, Z-Space")

## Find a valid x for checking minT
z = [-2.4, 0.75]
scatter!([z[1]], [z[2]], markersize = 2, label="z") 
tm = 0.6
# tp = 0.65 
tp = 0.5
x = exp(tm * M) * z
zp = exp(-tp * M) * x
scatter!([zp[1]], [zp[2]], markersize = 2, label="zp") 

## Run the minT solver
println("For x, computing minimum reachable time and corresponding control,")
solution, uˢ, Tˢ = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, x; 
                                                    preH=preH_ytc17, 
                                                    time_p=(0.02, 0.1, 1), 
                                                    opt_p=(0.01, 5, 0.5e-7, 500, 20),
                                                    printing=true)


## debug

intH = intH_ytc17
HJoc = HJoc_ytc17
preH = preH_ytc17
time_p = (0.02, 0.1, 1)
opt_p = (0.01, 5, 0.5e-7, 500, 20)
T_init = nothing 

J, Jˢ, A = target
th, Th, maxT = time_p
tvec = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)

if !(typeof(A) <: Matrix || typeof(A) <: Vector)
    error("A needs to be either a matrix (still target) or array of matrices (time-varying target)")
elseif typeof(A) <: Matrix
    moving_target = false
else 
    moving_target = true
end

## Initialize solutions
solution, dϕdz = Inf, 0
v_init = nothing 

## Precomputing some of H for efficiency
Hmats = preH(system, tvec)

## Loop Over Time Frames until ϕ(z, T) > 0
Thi = 0; Tˢ = tvec[end]
tix = length(tvec)