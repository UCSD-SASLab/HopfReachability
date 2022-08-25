
### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, Plots
push!(LOAD_PATH,"/Users/willsharpless/Documents/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachability

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
const dim = 2
const M = [0. 1; -2 -3]
const C = 0.5 * I
const C2 = 0.5 * [2 0; 0 1]
const Q = 0.1 * diagm(0 => 3*ones(dim), -1 => 1*ones(dim-1), 1 => 1*ones(dim-1))
const Q2 = 0.2 * diagm(0 => 2*ones(dim), -1 => 1*ones(dim-1), 1 => 1*ones(dim-1))
const a1 = [0.5 0.75]
const a2 = -[0.5 0]
const system = (M, C, C2, Q, Q2, a1, a2)

## Target
# J(x) = 0 is the boundary of the target
const aJ = cat([2.5, 1], 0.5*ones(dim - 2), dims=1)
const J  = x -> sum(abs2, x./aJ)/2 - 0.5
const Js = v -> sum(abs2, v.*aJ)/2 + 0.5
const target = (J, Js)

## Lookback Time(s)
const Th = 0.1
const Tf = 0.7
const T = collect(Th : Th : Tf)

## Grid Parameters (optional, deafult here)
const bd = 3
const ϵ = 0.5e-7
const N = 10 + ϵ
const th = 0.02
const grid_p = (bd, ϵ, N, th)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
const vh = 0.01
const L = 5
const tol = ϵ
const lim = 500
const lll = 20
const opt_p = (vh, L, tol, lim, lll)

## Run the solver
solution, averagetime = HopfReachability.Hopf_BRS(system, target, T; plotting=true);
plot!(title="BRS for T ="*string(T))