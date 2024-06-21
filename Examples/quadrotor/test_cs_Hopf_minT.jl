
include(pwd() * "/src/HopfReachability.jl")
include(pwd() * "/Examples/quadrotor/quadrotor_utils.jl");
include(pwd() * "/src/control_utils.jl");
include(pwd() * "/src/cons_lin_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations, ControlSystemsBase
using .HopfReachability: Hopf_minT, Hopf_BRS, Hopf_cd, Hopf_admm_cd, make_levelset_fs, make_set_params

## Initialize 

nx, nx_model = 12, 7;
x0, x0_m = zeros(nx), zeros(nx_model);
x_hover = [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.];

tf, dt = 8., 5e-4;
g = 9.81;

#		x	y 	z 		vx 	vy  vz 		psi
A7 = [ 	0	0	0	1	0	0	0;		# x
        0	0	0	0	1	0	0;		# y
        0	0	0	0	0	1	0;		# z
        0	0	0	0	0	0	0;		# vx_b
        0	0	0	0	0	0	0;		# vy_b
        0	0	0	0	0	0	0;		# vz_b
        0	0	0	0	0	0	0];		# psi

#		ph  th  w   T-g                 (ie R P Y T)
B7 = [	0	0	0	0;	    # x
        0	0	0	0;		# y
        0	0	0	0;		# z
        0   -g 	0	0;		# vx_b (HAD TO CHANGE THIS TO -g TO WORK)
        -g 	0	0	0;		# vy_b
        0 	0 	0	1;		# vz_b
        0	0	1	0];		# psi

## Controls

max_rpy, max_rel_thrust = 0.5 * π/6, 1 # RPY in ±π/6, Thrust in ±1 (+ 12), could also try ±2.5
max_u = [max_rpy, max_rpy, max_rpy, max_rel_thrust];
Qu, cu = make_set_params(zeros(4), max_u; type="box");
Qd, cd = zero(Qu), zero(cu);
u_target = [0; 0; 0; 12]
c̃ = B7 * u_target; # assuming f(x̃) - Ax̃ == 0...
game, input_shapes = "reach", "box"

system = (A7, B7, B7, Qu, cu, Qd, cd, c̃)

## Target Definition

# q = 2e-1
# Q = diagm((q * [1,1,20,1,1,20,1e-6]).^2);

r𝒯 = 1e-1;
c𝒯 = x_hover[1:nx_model];
Q𝒯 = diagm(inv.([1,1,2,1,1,2,1]).^2);                  # in crazyflie_clean          
J, Jˢ = make_levelset_fs(c𝒯, r𝒯; Q=Q𝒯);

target = (J, Jˢ, (Q𝒯, c𝒯));

## Solve
th, Th, Tf = 2e-2, 0.4, 4.;
time_p = (th, Th, Tf + 1e-5);
refines = 0;
# opt_p = (0.01, 2, 1e-5, 500, 20, 20, 2000)
# opt_p = (0.01, 20, 1e-5, 500, 2, 5, 2000)
opt_p = (0.01, 20, 1e-5, 500, 5, 10, 500)
warm = true
@time uˢ, _, Tˢ, ϕ = Hopf_minT(system, target, x0_m; game, input_shapes, refines, time_p, opt_p, warm, printing=true)
uˢ, Tˢ

## gather opt data
times = collect(Th:Th:Tˢ+Th)
# opt_p = (0.01, 20, 1e-5, 500, 2, 5, 2000)
# opt_p = (0.01, 20, 1e-5, 200, 5, 10, 800)
# opt_p = (0.01, 20, 1e-5, 100, 5, 10, 500)
# opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 10), opt_p, 1, 1, 3)
opt_p = (0.01, 20, 1e-5, 100, 2, 2, 500)
opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 2), opt_p, 1, 1, 2)
# @time solution, run_stats, opt_data = Hopf_BRS(system, target, times; Xg=[x0_m;;], th, game, input_shapes, opt_p, warm, printing=true, opt_tracking=true);
@time solution, run_stats, opt_data = Hopf_BRS(system, target, times; Xg=[x0_m;;], th, opt_method=Hopf_admm_cd, game, input_shapes, opt_p=opt_p_admm_cd, warm, printing=true, opt_tracking=true);

plot(title=L"\textrm{Convergence}", ylabel=L"\textrm{Hopf\:value}", xlabel=L"\textrm{Iteration}");
for i=1:length(times)
    plot!(-opt_data[i][1][2], label="t=-$(round(times[i], digits=2))")
end
plot!()

opt_p_cd = (0.01, 20, 1e-5, 100, 2, 2, 500)
opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 2), opt_p_cd, 1, 1, 2)
# @time uˢ, _, Tˢ, ϕ = Hopf_minT(system, target, x0_m; opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, game, input_shapes, refines, time_p, warm, printing=false)
uˢ, _, Tˢ, ϕ = Hopf_minT(system, target, x0_m; opt_p=opt_p_cd, game, input_shapes, refines, time_p, warm, printing=false)

opt_methods = [Hopf_cd, Hopf_cd, Hopf_cd, Hopf_admm_cd, Hopf_admm_cd, Hopf_admm_cd]
opt_ps = [(0.01, 20, 1e-5, 100, 2, 2, 500), # spread in both -> increasing runs
        (0.01, 20, 1e-5, 500, 5, 10, 500), # same as 5, keeping
        (0.01, 10, 1e-5, 20, 2, 2, 100), #
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 100, 2, 2, 500), 1, 1, 2), # tightest oc spread (4)
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 500, 5, 10, 500), 1, 1, 2), # same as 2
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 10, 1e-5, 20, 2, 2, 100), 1, 1, 2)] # gives most oc's (12), bad -> increasing runs

opt_ps = [(0.01, 20, 1e-5, 100, 5, 10, 500), # still terrible
        (0.01, 20, 1e-5, 500, 5, 10, 500), # same as 5 again
        (0.01, 20, 1e-5,  20, 5, 10, 100), # still terrible
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 100, 5, 10, 500), 1, 1, 2), # got bigger in oc and grad spread???
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 500, 2, 2, 500), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 20, 5, 10, 100), 1, 1, 2) # still terrible
]

opt_methods = [Hopf_cd, Hopf_cd, Hopf_admm_cd, Hopf_admm_cd]

opt_ps = [(0.01, 20, 1e-5, 500, 20, 20, 2000),
        (0.01, 50, 1e-5, 2000, 20, 20, 2000),
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 100, 20, 20, 500), 1, 1, 2), # still bigger
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-5, 500, 2, 2, 500), 1, 1, 2),
]

opt_methods = [Hopf_cd, Hopf_cd, Hopf_admm_cd, Hopf_admm_cd]

opt_ps = [
        (0.01, 50, 1e-5, 2000, 20, 20, 2000),
        (0.01, 50, 1e-5, 2000, 5, 5, 2000),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-5, 100, 5, 10, 500), 1, 1, 3), # still bigger
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-5, 500, 2, 2, 500), 1, 1, 3),
]
## Scoring wins^^:
# 0.522
# 0.572
# 0.034
# 0.702 # looks like ADMM really makes a difference!

opt_p_cd = (0.01, 50, 1e-5, 2000, 5, 5, 2000)
opt_p_cd = (0.01, 20, 1e-5, 2000, 5, 5, 2000)
opt_p_admm_cd = ((1e-0, 1e-0, 1e-4, 3), (0.01, 10, 1e-4, 500, 2, 2, 500), 1, 1, 5)

@time uˢ, _, Tˢ, ϕ = Hopf_minT(system, target, x0_m1; opt_p=opt_p_cd, game, input_shapes, refines, time_p, warm, printing=false)
@time uˢ, _, Tˢ, ϕ = Hopf_minT(system, target, x0_m1; opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, game, input_shapes, refines, time_p, warm, printing=false)

time_p = (0.05, 0.1, 2.)
opt_methods = [Hopf_admm_cd, Hopf_admm_cd, Hopf_admm_cd, Hopf_admm_cd, Hopf_admm_cd, Hopf_admm_cd]
opt_ps = [
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 20, 1e-3, 500, 2, 2, 500), 1, 1, 2),
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 20, 1e-3, 500, 2, 2, 500), 1, 1, 5),
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 20, 1e-3, 500, 5, 5, 500), 1, 1, 2),
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 20, 1e-3, 500, 5, 5, 500), 1, 1, 5),
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 10, 1e-3, 500, 5, 5, 500), 1, 1, 2),
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 100, 1e-3, 500, 5, 5, 500), 1, 1, 2),
]
# scores:
#  0.56
#  0.672
#  0.554
#  0.64
#  0.666
#  0.584
# avg_times:
#  0.08645787759000001
#  0.379083470368
#  0.20242854589599996
#  0.9318736206560008
#  0.1963249820560001
#  0.21529995047200004

time_p = (0.1, 0.3, 2.)
opt_ps = [
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 20, 1e-3, 500, 2, 2, 500), 1, 1, 5),
        ((1e-1, 1e-1, 1e-3, 3), (0.01, 10, 1e-3, 500, 5, 5, 500), 1, 1, 2),
        ((1e-1, 1e-1, 1e-3, 3), (0.01,  5, 1e-3, 500, 5, 5, 500), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 250, 2, 2, 250), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-5, 500, 2, 2, 500), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-3, 500, 2, 2, 500), 1, 1, 3),
]
opt_methods = [Hopf_admm_cd for _=1:length(opt_ps)]
# scores
#  0.58
#  0.59
#  0.386
#  0.8
#  0.594
#  0.596

# avg_times
#  0.12173984998999995
#  0.06018544100400002
#  0.06007985575399996
#  0.013632399362000013
#  0.05923982417800002
#  0.05769147230400001

time_p = (0.1, 0.3, 2.)
opt_ps = [
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 250, 2, 2, 250), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-3, 250, 2, 2, 250), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 125, 2, 2, 125), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 250, 5, 5, 250), 1, 1, 3),
]
opt_methods = [Hopf_admm_cd for _=1:length(opt_ps)]
# scores
#  0.802
#  0.856
#  0.713
#  0.888
# avg_times
#  0.014155537441000007
#  0.013974188238999988
#  0.00795441278599999
#  0.061987594804

time_p = (0.1, 0.3, 2.)
opt_ps = [
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 250, 2, 2, 250), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-3, 250, 2, 2, 250), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 10, 1e-3, 250, 2, 2, 250), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 20, 1e-3, 250, 2, 2, 250), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 125, 4, 4, 125), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-3, 125, 2, 2, 125), 1, 1, 3),
]; opt_methods = [Hopf_admm_cd for _=1:length(opt_ps)]
# scores
#  0.802
#  0.858
#  0.892
#  0.834
#  0.891
#  0.557
# avg_times
#  0.013759927344999992
#  0.013660725019999976
#  0.026231868523999984
#  0.02701254940300003
#  0.02661034726199997
#  0.01577293410900001
# == hz
#  72.67480233922707
#  73.20255685814264
#  38.121569536119125
#  37.01983048993293
#  37.57936678368783
#  63.39974497385377

time_p = (0.1, 0.3, 2.)
opt_ps = [
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 12, 1e-3, 125, 2, 2, 125), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 125, 2, 2, 125), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 20, 1e-3, 250, 2, 2, 250), 1, 1, 2),
        ((1e-1, 1e-1, 1e-5, 2), (0.01, 10, 1e-3, 250, 2, 2, 250), 1, 1, 3),
        ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 125, 4, 4, 125), 1, 1, 3),
]; opt_methods = [Hopf_admm_cd for _=1:length(opt_ps)]

# scores (neg points):
#  0.851
#  0.871
#  0.902
#  0.881
#  0.869

# hz
#  64.90331433484039
#  67.30162040791669
#  74.04974784879404
#  38.58952246806928
#  37.90936669688696

using TickTock, Suppressor 

n_samples, n_psets = 1000, length(opt_methods)
U, ∇ϕ = zeros(4, n_samples, n_psets), zeros(7, n_samples, n_psets)
avg_times = zeros(n_psets)
x0_m1 = -[1., 1., 0., 0., 0., 0., 0.]
for j=1:n_psets
    opt_method, opt_p = opt_methods[j], opt_ps[j]
    for i=1:n_samples
        @suppress begin; tick(); end # timing
        uˢ, _, Tˢ, ϕ, dϕdz = Hopf_minT(system, target, x0_m1; opt_method=opt_method, opt_p=opt_p, game, input_shapes, refines, time_p, warm, printing=false); 
        avg_times[j] += tok()/n_samples
        U[:,i,j], ∇ϕ[:,i,j] = uˢ, dϕdz
    end
end
scores = [sum([x == [-0.5*π/6, -0.5*π/6, 0.5*π/6, 1.] || x == [-0.5*π/6, -0.5*π/6, -0.5*π/6, 1.] for x in eachcol(U[:,:,i])]) for i=1:n_psets]/n_samples # correct answer has positive roll, pitch and thrust
hz = inv.(avg_times)

using MultivariateStats

Mu = MultivariateStats.fit(PCA, reshape(U, 4, Int(n_samples*n_psets)); maxoutdim=3)
Mϕ = MultivariateStats.fit(PCA, reshape(∇ϕ, 7, Int(n_samples*n_psets)); maxoutdim=3)

U_PC, ∇ϕ_PC = zeros(3, n_samples, n_psets), zeros(3, n_samples, n_psets)
upca_plot, ϕpca_plot = plot(title="Optimal U - PCA", xlabel="pc 1", ylabel="pc 2", zlabel="pc 3"), plot(title="Optimal ∇ϕ - PCA", xlabel="pc 1", ylabel="pc 2", zlabel="pc 3")
for j=1:n_psets
    U_PC[:,:,j] = predict(Mu, U[:,:,j])
    ∇ϕ_PC[:,:,j] = predict(Mϕ, ∇ϕ[:,:,j])
    scatter!(upca_plot, eachrow(U_PC[:,:,j])..., alpha=0.5, label="PSET $j")
    scatter!(ϕpca_plot, eachrow(∇ϕ_PC[:,:,j])..., alpha=0.5, label="PSET $j")
end
plot(upca_plot, ϕpca_plot, size=(1000, 500))










## SIMULATION 

controller_kwargs = Dict(
    :game => "reach", 
    :input_shapes => "box",
    :opt_p => ((1e-1, 1e-1, 1e-5, 3), (0.01, 10, 1e-3, 125, 2, 2, 125), 1, 1, 3), #(0.01, 20, 1e-5, 500, 5, 10, 500),                  
    :opt_method => Hopf_admm_cd,
    :refines => 0,
    :warm => true
)

# time_p = (0.1, 0.3, 2.)
Hopf_ctrl_longhzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (.02, 0.4, 4.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)
Hopf_ctrl_medmhzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.1, 0.3, 2.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)
Hopf_ctrl_shrthzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (.05, .25, 1.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)

@time Hopf_ctrl_longhzn(x0, 0.)
@time Hopf_ctrl_medmhzn(x0, 0.)
@time Hopf_ctrl_shrthzn(x0, 0.)

# ctrl_thrust_cs = (x,t)->[0., 0., 0., 4.925e4]
# sol_test, U_test = cs_solve_loop(x0, ctrl_thrust_cs, tf);

hz = 200
@time sol_mdhz, U_mdhz = cs_solve_loop(x0, Hopf_ctrl_medmhzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4], ll_ctrl_name="pid");
# @time sol_shhz, U_shhz = cs_solve_loop(x0, Hopf_ctrl_shrthzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4], ll_ctrl_name="mellinger");
# @time sol_lghz, U_lghz = cs_solve_loop(x0, Hopf_ctrl_longhzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4]);

plot_kwargs = Dict(:legend => :right,); #:xlims => (-10, 10), :ylims => (-10, 10), :zlims => (0, 1.5), :legend => :right, :backend=>gr)
flight_plot(sol_mdhz, U_mdhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Test}", plot_kwargs...)
# flight_plot(sol_shhz, U_shhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Sht\:Hzn\:Test}", plot_kwargs...)
# flight_plot(sol_lghz, U_lghz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Lng\:Hzn\:Test}", plot_kwargs...)

flight_gif(sol_mdhz, U_mdhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Test}", plot_kwargs...)
# flight_gif(sol_shhz, U_shhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Sht\:Hzn\:Test}", plot_kwargs...)


function Hopf_ctrl_tvarhzn(x, t) # crude atm
    if t < 4.0
        cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.1, 0.5, 2.0 + 1e-5), controller_kwargs...)[1])
    else
        cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.05, 0.1, 2.0 + 1e-5), controller_kwargs...)[1])
    end
end

function Hopf_ctrl_midhzn_patch(x, t)
    u_hopf = Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.05, 0.5, 2.0 + 1e-5), controller_kwargs...)[1]
    u = t < 4 ? u_target + u_hopf : u_target + u_hopf .* max.(abs.(x[[5,4,9,6]] - x_hover[[5,4,9,6]]), 10)/10
    return cs_control(u)
end


hz = 200
@time sol_tvhz, U_tvhz = cs_solve_loop(x0, Hopf_ctrl_tvarhzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4], ll_ctrl_name="mellinger");
flight_plot(sol_tvhz, U_tvhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:TV\:Hzn\:Test}", plot_kwargs...)

hz = 200
@time sol_mdhzp, U_mdhzp = cs_solve_loop(x0, Hopf_ctrl_midhzn_patch, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4], ll_ctrl_name="mellinger");
flight_plot(sol_mdhzp, U_mdhzp; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Patch}", plot_kwargs...)

K_cfc7 = [0.0 0.2 0.0 0.0 0.2 0.0 0.0;
    0.2 0.0 0.0 0.2 0.0 0.0 0.0; 
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;  
    0.0 0.0 -5.4772 0.0 0.0 -5.5637 0.0]

xref_default = t->[0., 0., 1., 0., 0., 0., 0.]

function LQR7(x::Vector, t::Number; K7::Matrix=K_cfc7, u_target::Vector=[0.0, 0.0, 0.0, 12], xref::Function=xref_default)
    u = K7 * (x[[1:6..., 9]] - xref(t)) + u_target
    return u
end

r𝒯 = 1e-1;
c𝒯 = x_hover[1:nx_model];
# Q𝒯 = diagm(inv.([50,50,50,10,10,5,1]).^2); #works with 200Hz and 0.9 smoothing
Q𝒯 = diagm(inv.([1,1,2,10,10,1,1]).^2);
J, Jˢ = make_levelset_fs(c𝒯, r𝒯; Q=Q𝒯);
target = (J, Jˢ, (Q𝒯, c𝒯));

function Hopf_ctrl_midhzn_LQR_hover(x, t)
    # if 10*(t - floor(t)) % 1 == 0.; println("t: $t, ‖x - x_target‖:$(norm(x[1:3] - x_hover[1:3]))"); end
    u = norm(x[1:3] - x_hover[1:3]) > 0.2 ? u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.1, 0.5, 2.0 + 1e-5), controller_kwargs...)[1] : LQR7(x,t)
    return cs_control(u)
end

hz = 200
@time sol_mdhz_lqr, U_mdhz_lqr = cs_solve_loop(x0, Hopf_ctrl_midhzn_LQR_hover, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.3, smooth_ix=[1,2,3,4], ll_ctrl_name="pid");
flight_plot(sol_mdhz_lqr, U_mdhz_lqr; plot_title=L"\textrm{CrazySwarm - Hopf+LQR - \:Mid\:Hzn}", plot_kwargs...)
flight_gif(sol_mdhz_lqr, U_mdhz_lqr; plot_title=L"\textrm{CrazySwarm - Hopf+LQR - \:Mid\:Hzn}", plot_kwargs...)

r𝒯 = 1e-1;
c𝒯 = x_hover[1:nx_model];
# Q𝒯 = diagm(inv.([1,1,2,10,10,1,1]).^2);
Q𝒯 = diagm(inv.([1,1,2,1,1,1,1]).^2);
J, Jˢ = make_levelset_fs(c𝒯, r𝒯; Q=Q𝒯);
target_fn = x_ref -> (make_levelset_fs(x_ref, r𝒯; Q=Q𝒯)..., (Q𝒯, x_ref));

function Hopf_ctrl_midhzn_LQR_7D(x, t; xref_fn::Function=xref_default)
    # if 10*(t - floor(t)) % 1 == 0.; println("t: $t, ‖x - x_target‖:$(norm(x[1:3] - x_hover[1:3]))"); end
    u = norm(x[1:3] - x_hover[1:3]) > 0.2 ? u_target + Hopf_minT(system, target_fn(xref_fn(t)), x[[1:6..., 9]]; time_p = (0.1, 0.5, 2.0 + 1e-5), controller_kwargs...)[1] : LQR7(x,t; xref=xref_fn)
    return cs_control(u)
end

wp_list = [[1., 1., 1.], [1., -1., 1.], [-1., -1., 1.], [-1., 1., 1.]]
track_waypoints(t; delay=2.5) = t < 5 ? [0., 0., 1., 0., 0., 0., 0.] : ((t - 5)/delay < length(wp_list) ? vcat(wp_list[1+Int(floor((t - 5)/delay))], zeros(4)) : [0., 0., 1., 0., 0., 0., 0.])

hz = 100
tf = length(wp_list)*2.5 + 8
@time sol_mdhz_lqr, U_mdhz_lqr = cs_solve_loop(x0, (x,t)->Hopf_ctrl_midhzn_LQR_7D(x,t;xref_fn=track_waypoints), tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.7, smooth_ix=[1,2,3,4], ll_ctrl_name="pid");
flight_plot(sol_mdhz_lqr, U_mdhz_lqr; plot_title=L"\textrm{CrazySwarm - Hopf+LQR - \:Mid\:Hzn}", plot_kwargs...)
flight_gif(sol_mdhz_lqr, U_mdhz_lqr; plot_title=L"\textrm{CrazySwarm - Hopf+LQR - \:Mid\:Hzn}", plot_kwargs...)