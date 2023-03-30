#   Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   wsharpless@ucsd.edu
#   ---------------------

using LinearAlgebra, Plots
plotlyjs()

## Comparison (Stochastic MPC, 2P MPC, 2P LQR)
include("Control_Comparison_fn.jl")

## Our HopfReachability Pkg
include("/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, plot_BRS, Hopf_minT

#   Initialize
#   ==========

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

nx, nu, nd = 2, 2, 2
M = [0. 1; -2 -3]
C = 0.5 * [1 0; 0 1]
C2 = 0.5 * [2 0; 0 1]
Q = 0.1 * diagm(0 => 3*ones(nu), -1 => 1*ones(nu-1), 1 => 1*ones(nu-1))
Q2 = 0.2 * diagm(0 => 2*ones(nd), -1 => 1*ones(nd-1), 1 => 1*ones(nd-1))
a1 = [0. 0.]
a2 = [0. 0.]
system = (M, C, C2, Q, Q2, a1, a2);

## Target
# J(x) = 0 is the boundary of the target

Ap = diagm(ones(nx))
cp = zeros(nx)

J(x::Vector, A, c) = ((x - c)' * A * (x - c))/2 - 0.5
Js(v::Vector, A, c) = (v' * inv(A) * v)/2 + c'v + 0.5 # Convex Conjugate

J(x::Matrix, A, c) = diag((x .- c)' * A * (x .- c))/2 .- 0.5
Js(v::Matrix, A, c) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5

target = (J, Js, (Ap, cp));

## Lookback Time(s), 
# each time solved independently! solving multiple for comparison to Dynamic Programming

th = 0.05
Th = 0.4
Tf = 2.0
T = collect(Th : Th : Tf);

## Grid Parameters
bd = (-3, 3)
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N);


#  Solve the BRS (Coordinate Descent)
#   ============

solution, run_stats = Hopf_BRS(system, target, T; 
                                th,
                                grid_p,
                                warm=false, # warm starts the optimizer => 10x speed, risky if H non-convex
                                check_all=true, # checks all points in grid (once); if false, just convolution of boundary
                                printing=true);

plot_BRS(T, solution...; M, cres=0.1, interpolate=true)


#   Test Control, Compare Solution of 1st step
#   ==========================================

x0 = [-2.3, 2.6]; steps = 1; 

## Controllers
ctrls = Dict("Hopf" => x -> Hopf_minT(system, target, x; time_p = (0.05, 0.4, 2.0)),
             "MPCs" => x -> MPC_stochastic(system, target, x; H=10, N_T=20),
             "MPCg" => x -> MPC_game(system, target, x; H=1, its=1));

## Simulate
Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps);

## Plot Game Surface
pair_strats = [(Us[c][:, 1], Ds[c][:, 1]) for c in keys(ctrls)]; pair_labels = collect(keys(ctrls));
p1_strats = [Us[c][:, 1] for c in keys(ctrls)]; p1_labels = collect(keys(ctrls));

_, _, Tˢ, _, dϕdz = Hopf_minT(system, target, x0; time_p = (0.05, 0.4, 2.0));

game_plot = plot_game(system, pair_strats, pair_labels; p=dϕdz, t=Tˢ, p1_strats, p1_labels)

#   Compare Hopf Against Random Disturbance
#   =======================================

x0 = [-2.3, 2.6]; steps = 35; 

hopf_ctrl = Dict("Hopf" => x -> Hopf_minT(system, target, x; time_p = (0.05, 0.4, 2.0)))

Xs, Us, Ds = roll_out(system, target, hopf_ctrl, x0, steps; printing=false);

for ni in string.(1:5)
    Xs_rand, Us_rand, Ds_rand = roll_out(system, target, hopf_ctrl, x0, steps; random_d=true)
    Xs["Hopf_rand_" * ni] = Xs_rand["Hopf"]
    Ds["Hopf_rand_" * ni] = Ds_rand["Hopf"]
end

plot_sim(system, target, Xs; grid_p, title="Hopf: Worst vs. Random Disturbance")

plot_inputs(system, Us, "u")
plot_inputs(system, Ds, "d")


#   Compare Controllers (with a random disturbance)
#   ===============================================

Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps; random_d=true, random_same=true, plotting=true, title="Controller Comparison: Random Disturbance")

plot_inputs(system, Us, "u")
plot_inputs(system, Ds, "d")

#   Compare Controllers (against worst disturbance)
#   ===============================================

Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps; worst_d=true, plotting=true, title="Controller Comparison: Worst Disturbance");

plot_inputs(system, Us, "u")
plot_inputs(system, Ds, "d")
