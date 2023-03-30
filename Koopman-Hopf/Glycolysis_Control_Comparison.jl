#   Koopman-Hopf Hamilton Jacobi Reachability
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   wsharpless@ucsd.edu
#   ---------------------

using Pkg
ENV["PYTHON"] = "/Users/willsharpless/miniconda3/bin/python"
Pkg.build("PyCall") 

using PyCall, Plots, LinearAlgebra, Suppressor, LaTeXStrings, JLD, Dates, StatsBase
plotlyjs()
home = "/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/"

##   Import PyKoopman
##   ==================

np = pyimport("numpy")
pk = pyimport("pykoopman") # requires PyCall.pyversion == v"3.9.12"

pkG_loc = home * "Traditional_K"
pushfirst!(pyimport("sys")."path", pkG_loc);
pk_Glycolysis_utils = pyimport("pk_Glycolysis_DMDc")

## DMDc (Lowest n_k (= n_x = 10) and mild accuracy)

train_ix = np.arange(0, 8000)
Glycolysis = pk_Glycolysis_utils.fit_DMDc(pk_Glycolysis_utils.Xfull, train_ix)

##   Our HopfReachability Pkg
##   ========================

## Comparison (Stochastic MPC, 2P MPC, 2P LQR)
include(home * "HL_fastHJR/Comparison/Control_Comparison_fn.jl");

include(home * "HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, plot_BRS, Hopf_minT, Hopf_admm_cd, Hopf_cd

#   Initialize
#   ==========

nx = 10;
nu = 3; 
dt = 0.05;
ATP_goal = 2.5 

disturbance_on_control = true; # applies disturbance only to controlled states (allows us to use same Koopman L)
nd = disturbance_on_control ? nu : size(L)[2] - nu;
Max_u, Max_d = 0.1, 0.025; # => u ∈ [-0.1 ,  0.1], d ∈ [-0.1 ,  0.1] 

include(home * "HL_fastHJR/Linearizations/Glycolysis_true.jl") #takes a while because DiffEq slow

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

Kd, Ld = Glycolysis.state_transition_matrix, Glycolysis.control_matrix; nk = size(Kd)[1]; 
for i in [7, 9, 10]; Kd[i, :] = [k == i ? 1 : 0 for k=1:nk]; end # fix control and disturbed states

K, L = (Kd - I)/dt, Ld/dt; 
Lu, Ld = disturbance_on_control ? (L[:, 1:nu], L[:, 1:nu]) : (L[:, 1:nu], L[:, nu+1:end]);

Q1 = diagm(ones(nu)); Q10 = zero(Q1); a1 = zeros(1, nu); 
Q2 = diagm(ones(nd)); Q20 = zero(Q2); a2 = zeros(1, nd);

system = (K, Max_u * Lu, Max_d * Ld, Q1, Q2, a1, a2); # Controlled and Disturbed

## Target
# J(x) = 0 is the boundary of the target

target_center = [0.19,  0.378,  0.04,  0.088,  0.146,  ATP_goal,  0.29,  0.037,  1.0,  4.0] # from auto evo
target_dim_priority = [i != 6 ? 0.5 : 5.0 for i=1:nk]

Ap = diagm(target_dim_priority);
cp = target_center;
rp = 1.0

J(g::Vector, A, c; r=rp) = ((g - c)' * A * (g - c))/2 - 0.5 * r^2;
Js(v::Vector, A, c; r=rp) = (v' * inv(A) * v)/2 + c'v + 0.5 * r^2; # Convex Conjugate
J(g::Matrix, A, c; r=rp) = diag((g .- c)' * A * (g .- c))/2 .- 0.5 * r^2;
Js(v::Matrix, A, c; r=rp) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 * r^2;

target = (J, Js, (Ap, cp));

## Solver Params

# Hopf Coordinate Descent
opt_p_cd = (0.01, 5, 1e-5, 500, 20, 20);

# Hopf ADMM-CD Hybrid Solver
opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.01, 10, 1e-5, 100, 1, 4), 1, 1, 3);

## True Nonlinear System
F = Glcyolysis_SciPySolveIVP # Used to generate data (Radau = RadauIIA5)

## min_T controller parameters

Max_u, Max_d = 0.1, 0.025;
system = (K, Max_u * Lu, Max_d * Ld, Q1, Q2, a1, a2); # Controlled and Disturbed
x0 = [0.15, 0.19, 0.04, 0.1, 0.08, 0.14, 0.34, 0.05, 1., 4.]; # from Lutz paper

## Autonomous
auto = Dict("Auto" => x -> (zeros(nu), zeros(nd)))
Xs_auto, _, _ = roll_out(system, target, auto, x0, 200; th=0.05, F)

## Some initial points to try from auto evolution
## note, because evolved in true system controller cannot guarantee if reaching or not 
# x0 = Xs_auto["Auto"][:, 1]  # first time point ("reachable" but fails)
# x0 = Xs_auto["Auto"][:, 13]  # mid time point ("unreachable" but works)
# x0 = Xs_auto["Auto"][:, 16]  # mid time point ("reachable" and works)
x0 = Xs_auto["Auto"][:, 100] # stablized time point ("unreachable" but works)

#   Test Control, Compare Solution of 1st step
#   ==========================================

steps = 1; 

time_p = (0.025, 1.0, 5.0)
th = 0.05;

## Controllers
ctrls = Dict(
             L"\text{Hopf}"   => x -> Hopf_minT(system, target, x; opt_method=Hopf_cd, opt_p=opt_p_cd, time_p, warm=true),
             L"\text{MPCs}"   => x -> MPC_stochastic(system, target, x; H=10, N_T=20),
             L"\text{MPCg}"   => x -> MPC_game(system, target, x; H=1, its=1),
             L"\text{Auto}"   => x -> (zeros(nu), zeros(nd)) #d not used
             );

## Simulate
Xs, Us, Ds, Ts = roll_out(system, target, ctrls, x0, steps; th, F);
plot_sim_sbs(Xs; th)

plot_sim_sbs(Xs; th)


#   Compare Controllers (with a random disturbance)
#   ===============================================

steps = Int(1. / th) * 3

@time Xs, Us, Ds, Ts = roll_out(system, target, ctrls, x0, steps; random_d=true, random_same=true, printing=false, F)

plot_sim_sbs(Xs; th, pal=:seaborn_colorblind)

U_plot = plot_sim_sbs(Us; th, labels=["u1 (glucose_ext)" "u2 (NAD+/NADH)" "u3 (ATP/ADP)"], dim_splits=[[1,2,3]]);
D_plot = plot_sim_sbs(Ds; th, labels=["d1 (glucose_ext)" "d2 (NAD+/NADH)" "d3 (ATP/ADP)"], dim_splits=[[1,2,3]]);
plot(U_plot, D_plot, layout=(2,1))


## Iterate
runs = 50
Xs_runs = []

ctrl_stats = Dict(
             L"\text{Hopf}"  => Dict(L"\text{Success_History}" => zeros(runs), L"\text{Mean_Comp_Time}" => zeros(runs), L"\text{Std_Comp_Time}" => zeros(runs), L"\text{Maximum_ATP}" => zeros(runs)),
             L"\text{MPCs}"  => Dict(L"\text{Success_History}" => zeros(runs), L"\text{Mean_Comp_Time}" => zeros(runs), L"\text{Std_Comp_Time}" => zeros(runs), L"\text{Maximum_ATP}" => zeros(runs)),
             L"\text{MPCg}"  => Dict(L"\text{Success_History}" => zeros(runs), L"\text{Mean_Comp_Time}" => zeros(runs), L"\text{Std_Comp_Time}" => zeros(runs), L"\text{Maximum_ATP}" => zeros(runs)),
             L"\text{Auto}"  => Dict(L"\text{Success_History}" => zeros(runs), L"\text{Mean_Comp_Time}" => zeros(runs), L"\text{Std_Comp_Time}" => zeros(runs), L"\text{Maximum_ATP}" => zeros(runs))
             );

for it=1:runs

    θ = rand(nk)

    ## Initial Point sampled from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0119821
    x0 = [rand() * 1.45 + 0.15, rand() * 2.0 + 0.1, rand() * 0.1 + 0.15, rand() * 0.1 + 0.15, rand() * 0.1 + 0.15, rand() * 0.1 + 0.15,  rand() * 0.5 + 0.15, rand()*0.05 * + 0.05, rand()*0.1 + 1, rand()*0.1 + 4.]

    ## Time Check
    if it % 5 == 0; 
        println("Computing run $it at $(Dates.format(now(), "HH:MM"))..."); 
    end

    ## Run
    Xs, Us, Ds, Ts = roll_out(system, target, ctrls, x0, steps; random_d=true, random_same=true, printing=false, F)

    ## Store Data
    for c in keys(ctrls)

        ctrl_stats[c][L"\text{Maximum_ATP}"][it] = maximum(Xs[c][6,:])
        success = maximum(Xs[c][6,:]) >= ATP_goal ? 1 : 0
        ctrl_stats[c][L"\text{Success_History}"][it] = success

        ctrl_stats[c][L"\text{Mean_Comp_Time}"][it], ctrl_stats[c][L"\text{Std_Comp_Time}"][it] = mean(Ts[c]), std(Ts[c])

        if it % 5 == 0; println("    $c has $(sum(ctrl_stats[c][L"\text{Success_History}"])) successes"); end

    end

    ## Store and Save
    push!(Xs_runs, Xs)
    # @save "Xs_50runs_ATP2p5_2" Xs_runs
    # @save "ctrl_stats_50runs_ATP2p5_2" ctrl_stats
end
