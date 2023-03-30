
using LinearAlgebra, Plots
plotlyjs()
include("/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, Hopf

## Initialize (2D Example)
M = [0. 1; -2 -3]
B = 0.5 * [1 0; 0 1]
C = 0.5 * [2 0; 0 1]
Q = 0.1 * 3 * [1 0; 0 1]
Q2 = 0.2 * 2 * [1 0; 0 1]
a1 = 0*[0.5 0.75]
a2 = -0*[0.5 0]
system = (M, B, C, Q, Q2, a1, a2)

## Initialize (3D Example)
# M = [0. 1 0.; -2 -3 0.; 0. 0. -1.]
# B = 0.5 * [1 0; 0 1; 0. 0.]
# C = 0.5 * [2 0; 0 1; 0. 0.]
# Q = 0.1 * 3 * [1 0; 0 1]
# Q2 = 0.2 * 2 * [1 0; 0 1]
# a1 = 0*[0.5 0.75]
# a2 = -0*[0.5 0]
# system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.02
Th = 0.2
Tf = 0.8
T = collect(Th : Th : Tf)

## Target: J(x) = 0 is the boundary of the target
Ap = diagm([1.5, 1]) # Ap = diagm([1.5, 1, 1.])
cp = [0.; 0.] # cp = [0.; 0.; 0.]
r = 2.0
J(x::Vector, A, c) = ((x - c)' * inv(A) * (x - c))/2 - 0.5 * r^2 #don't need yet
Js(v::Vector, A, c) = (v' * A * v)/2 + c'v + 0.5 * r^2
J(x::Matrix, A, c) = diag((x .- c)' * inv(A) * (x .- c))/2 .- 0.5 * r^2
Js(v::Matrix, A, c) = diag(v' * A * v)/2 + (c'v)' .+ 0.5 * r^2 #don't need yet
target = (J, Js, (Ap, cp))

## Point Target (Indicator Set)
# ap = -1.
# cp = [0.; 0.]
# J(x::Vector, a, c) = x == c ? a : Inf
# Js(v::Vector, a, c) = v'c - a
# J(x::Matrix, a, c) = map(x -> x == c ? a : Inf, eachcol(x))
# Js(v::Matrix, a, c) = v'c .- a
# target = (J, Js, (ap, cp))

## Affine Target
# ap = [1.; 1.]
# cp = 0.
# J(x::Vector, a, c) = a'x - c #don't need yet
# Js(v::Vector, a, c) = v == a ? c : 10
# J(x::Matrix, a, c) = (a'x)' .- c
# Js(v::Matrix, a, c) = map(v -> v == a ? c : 10, eachcol(v))
# target = (J, Js, (ap, cp))

## Grid Parameters
bd = 3. # (-3, 3) for ellipse
ϵ = 0.5e-7
N = 3 + ϵ
grid_p = (bd, N)

## Hopf ADMM Parameters (default)
ρ, ρ2 = 1e-4, 1e-4
tol = 1e-5
max_its = 10
opt_p_admm = (ρ, ρ2, tol, max_its)

## Hopf CD Parameters (default)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 20
max_runs = 40
opt_p_cd = (vh, L, tol, lim, lll, max_runs)

solution, run_stats = Hopf_BRS(system, target, intH_ytc17, T;
                                                    opt_method=Hopf_cd,
                                                    preH=preH_ytc17,
                                                    th,
                                                    grid_p,
                                                    opt_p=opt_p_cd,
                                                    warm=false,
                                                    check_all=true,
                                                    printing=true);
B⁺T, ϕB⁺T = solution;

plot_scatter = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵs=0.1, interpolate=false, value_fn=true, alpha=0.5)
plot_contour = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵc=0.01, interpolate=true, value_fn=true, alpha=0.5)
