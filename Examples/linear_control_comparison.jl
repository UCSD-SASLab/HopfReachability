
include(pwd() * "/src/HopfReachability.jl");
include(pwd() * "/src/control_utils.jl");
using .HopfReachability: Hopf_BRS, plot_nice, Hopf_minT, Hopf_cd, make_set_params, make_grid, make_target
using LinearAlgebra, Plots

## System
nx, nu, nd = 2, 2, 2
A, B₁, B₂ = [0. 1; -2 -3], 0.5*[1 0; 0 1], 0.5*[1 0; 0 1] # system
max_u, max_d, input_center, input_shapes = 1.0, 0.5, zeros(nu), "ball"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
Q, center, radius = diagm(ones(nx)), zeros(nx), 1.
target = make_target(center, radius; Q, type="ellipse")

## Lookback Time(s), 
th, Th, Tf = 0.05, 0.4, 2.0
T = collect(Th : Th : Tf);

## Points to Solve
bd, res, ϵ = 3, 0.1, .5e-7
grid_p = (bd, res)

## Params
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

## Solve the BRS (Coordinate Descent)
solution, run_stats = Hopf_BRS(system, target, T; 
                                th,
                                grid_p,
                                opt_method=Hopf_cd,
                                opt_p=opt_p_cd,
                                warm=true, # warm starts the optimizer => 10x speed, risky if H non-convex
                                check_all=true, # checks all points in grid (once); if false, just convolution of boundary
                                printing=true);

# plot_nice(T, solution; cres=0.1, interpolate=true) #FIXME or phase out
plot(solution; grid_p, value_alpha=0.5)
 
#   Test Control, Compare Solution of 1st step
#   ==========================================

x0 = [2.5, 1.5]; steps = 1; 

## Controllers
ctrls = Dict("Hopf" => x -> Hopf_minT(system, target, x; input_shapes, time_p = (0.05, 0.2, 2.0)),
             "MPCs" => x -> MPC_stochastic(system, target, x; H=10, N_T=20),
             "MPCg" => x -> MPC_game(system, target, x; H=1, its=1));

## Simulate
Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps);

## Plot Game Surface
pair_strats = [(Us[c][:, 1], Ds[c][:, 1]) for c in keys(ctrls)]; pair_labels = collect(keys(ctrls));
p1_strats = [Us[c][:, 1] for c in keys(ctrls)]; p1_labels = collect(keys(ctrls));

_, _, Tˢ, _, dϕdz = Hopf_minT(system, target, x0; input_shapes, time_p = (0.05, 0.2, 2.0))

game_plot = plot_game(system, pair_strats, pair_labels; p=dϕdz, t=Tˢ, p1_strats, p1_labels)

#   Compare Hopf Against Random Disturbance
#   =======================================

hopf_ctrl = Dict("Hopf" => x -> Hopf_minT(system, target, x; input_shapes, time_p = (0.05, 0.4, 2.0)))
steps = Int(ceil(hopf_ctrl["Hopf"](x0)[3] / th)) + 1;

Xs, Us, Ds = roll_out(system, target, hopf_ctrl, x0, steps; printing=false);

for ni in string.(1:5)
    Xs_rand, Us_rand, Ds_rand = roll_out(system, target, hopf_ctrl, x0, steps; random_d=true)
    Xs["Hopf_rand_" * ni] = Xs_rand["Hopf"]
    Ds["Hopf_rand_" * ni] = Ds_rand["Hopf"]
end

plot_sim(system, target, Xs; grid_p, title="Hopf: Worst vs. Random Disturbance", scale=0.3)

plot_inputs((Q₁, c₁, max_u), Us; title="Control Set 𝒰")
plot_inputs((Q₂, c₂, max_d), Ds; title="Disturbance Set 𝒟")

#   Compare Controllers (with a random disturbance)
#   ===============================================

Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps; random_d=true, random_same=true, plotting=true, title="Controller Comparison: Random Disturbance");

plot_inputs((Q₁, c₁, max_u), Us; title="Control Set 𝒰")
plot_inputs((Q₂, c₂, max_d), Ds; title="Disturbance Set 𝒟")

#   Compare Controllers (against worst disturbance)
#   ===============================================

Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps; same_d="Hopf", plotting=true, title="Controller Comparison: Worst Disturbance");

plot_inputs((Q₁, c₁, max_u), Us; title="Control Set 𝒰")
plot_inputs((Q₂, c₂, max_d), Ds; title="Disturbance Set 𝒟")
