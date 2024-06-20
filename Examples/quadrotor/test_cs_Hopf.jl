
include(pwd() * "/src/HopfReachability.jl")
include(pwd() * "/Examples/quadrotor/quadrotor_utils.jl");
include(pwd() * "/src/control_utils.jl");
include(pwd() * "/src/cons_lin_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations, ControlSystemsBase
using .HopfReachability: Hopf_minT, Hopf_BRS, make_levelset_fs, make_set_params

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
        0   -g 	0	0;		# vx_b
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
# Q𝒯 = diagm((q𝒯 * [1,1,20,1,1,20,0.5]).^2);
# Q𝒯 = diagm(inv.([1,1,10,1,1,10,0.5]).^2);
Q𝒯 = diagm(inv.([1,1,2,20,20,10,10]).^2);
J, Jˢ = make_levelset_fs(c𝒯, r𝒯; Q=Q𝒯);

target = (J, Jˢ, (Q𝒯, c𝒯));

## Solve
th, Th, Tf = 2e-2, 0.4, 4.;
time_p = (th, Th, Tf + 1e-5);
refines = 0;
# opt_p = (0.01, 2, 1e-5, 500, 20, 20, 2000)
opt_p = (0.01, 20, 1e-5, 500, 2, 5, 2000)
warm = true
@time uˢ, _, Tˢ, ϕ = Hopf_minT(system, target, x0_m; game, input_shapes, refines, time_p, opt_p, warm, printing=false)
uˢ, Tˢ

## gather opt data
times = collect(Th:Th:Tˢ+Th)
opt_p = (0.01, 20, 1e-5, 500, 2, 5, 2000)
@time solution, run_stats, opt_data = Hopf_BRS(system, target, times; Xg=[x0_m;;], th, game, input_shapes, opt_p, warm, printing=true, opt_tracking=true);

plot(title=L"\textrm{Convergence}", ylabel=L"\textrm{Hopf\:value}", xlabel=L"\textrm{Iteration}");
for i=1:length(times)
    plot!(-opt_data[i][1][2], label="t=-$(round(times[i], digits=2))")
end
plot!()

controller_kwargs = Dict(
    :game => "reach",
    :input_shapes => "box",
    :opt_p => (0.01, 20, 1e-5, 500, 5, 10, 500),                  
    :refines => 0,
    :warm => true
)

Hopf_ctrl_longhzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (.02, 0.4, t+4.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)

Hopf_ctrl_medmhzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.1, 0.5, t+2.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)

Hopf_ctrl_shrthzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (.05, .25, t+1.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)

@time for i=1:10; Hopf_ctrl_longhzn(x0, 0.); end
@time for i=1:10; Hopf_ctrl_medmhzn(x0, 0.); end
@time for i=1:10; Hopf_ctrl_shrthzn(x0, 0.); end

# ctrl_thrust_cs = (x,t)->[0., 0., 0., 4.925e4]
# sol_test, U_test = cs_solve_loop(x0, ctrl_thrust_cs, tf);

@time sol_mdhz, U_mdhz = cs_solve_loop(x0, Hopf_ctrl_medmhzn, tf, ctrl_dt=5e-3, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4]);
@time sol_shhz, U_shhz = cs_solve_loop(x0, Hopf_ctrl_shrthzn, tf, ctrl_dt=5e-3, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4]);
# @time sol_lghz, U_lghz = cs_solve_loop(x0, Hopf_ctrl_longhzn, tf, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3]);

plot_kwargs = Dict(:legend => :right,) #:xlims => (-10, 10), :ylims => (-10, 10), :zlims => (0, 1.5), :legend => :right, :backend=>gr)
flight_plot(sol_mdhz, U_mdhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Test}", plot_kwargs...)
flight_plot(sol_shhz, U_shhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Sht\:Hzn\:Test}", plot_kwargs...)
# flight_plot(sol_lghz, U_lghz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Lng\:Hzn\:Test}", plot_kwargs...)

flight_gif(sol_mdhz, U_mdhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Test}", plot_kwargs...)
flight_gif(sol_shhz, U_shhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Sht\:Hzn\:Test}", plot_kwargs...)

stop
















## LQR7 with Crazyswarm sim model

#		    x	y  z 	  vx vy vz 		psi
K_cfc7 = [  0. .2 0.      0. .2 0.      0.;
            .2 0. 0.      .2 0. 0.      0.; 
            0. 0. 0.      0. 0. 0.      0.;  
            0. 0. -5.4772 0. 0. -5.5637 0.]

xref_default = t-> x_hover[1:7]

function LQR7(x::Vector, t::Number; K7::Matrix=K_cfc7, u_target::Vector=[0.0, 0.0, 0.0, 12], xref::Function=xref_default)
    control = K7 * (x[[1:6..., 9]] - xref(t)) + u_target
    return vcat(clamp.(control[1:3], -π/6, π/6)*180/π, clamp(4096 * control[4], 0., 60000))
end

tf = 8.
cs_sol = cs_solve(x0, LQR7, tf);
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7:\:Test}")

cs_sol = cs_solve(x0, (x,t) -> LQR7(x,t; xref=t->[1,0,1,0,0,0,0]), tf)
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7:\:Test\:2}")

## LQR7 Way-Pointing

wp_list = [[1., 1., 1.], [1., -1., 1.], [-1., -1., 1.], [-1., 1., 1.]]
track_waypoints(t; delay=2.5) = t < 5 ? [0., 0., 1., 0., 0., 0., 0.] : ((t - 5)/delay < length(wp_list) ? vcat(wp_list[1+Int(floor((t - 5)/delay))], zeros(4)) : [0., 0., 1., 0., 0., 0., 0.])

tf = length(wp_list)*2.5 + 8
cs_sol = cs_solve(x0, (x,t) -> LQR7(x,t; xref=track_waypoints), tf)
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7\:Waypoints}")
# flight_gif(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7\:Waypoints}")

## LQR7 Fig 8 

tf_fig8 = 8 + ceil(2pi)
fig8(t) = t < 5 ? [0., 0., 1., 0., 0., 0., 0.] : (t < 5 + 2pi ? [sin(t-5), sin(t-5)*cos(t-5), 1., 0., 0., 0., 0.] : [0., 0., 1., 0., 0., 0., 0.])
cs_sol = cs_solve(x0, (x,t) -> LQR7(x,t; xref=fig8), tf_fig8)
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7\:Fig8}")
# flight_gif(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm\:(LL=PID) - LQR7\:Fig8}")

## LQR7 freefall

tf, h = 8., 10.
x0_flip = [0,0,h,0,0,0,0,π,0,0,0,0]
cs_sol = cs_solve(x0_flip, (x,t) -> LQR7(x,t; xref=t->[0,0,h,0,0,0,0]), tf)
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7\:Freefall}", backend=gr, xlims=(-2,2), ylims=(-2,2), body_times=[0., 0.25, 0.5, 0.75, 1., 2., 3.], rotor_alpha=0.7, camera=(45,10))
# flight_gif(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7\:Freefall}", xlims=(-2,2), ylims=(-2,2), dt=0.05)

## MPC ?

## LQR-Koopman

model_path = pwd() * "/Examples/quadrotor/lifted_data/gen";
pushfirst!(pyimport("sys")."path", model_path);
np = pyimport("numpy");
# pk = pyimport("pykoopman"); # requires PyCall.pyversion == v"3.9.12"

Ψrbf(x; cx, kw) = vcat(x, exp.(-(kw^2) * sum(eachrow((x .- cx).^2))))

n_centers = [10, 100] # [10, 15, 50, 100]
models = Dict("rbf"=>Dict()); pkdt = 0.05996;
for ci in n_centers
    model_name = "nb_1_cb_1_obs_RBF_$(ci)_reg_EDMDc.npy" # "nb_3_cb_2_obs_RBF_$(ci)_reg_EDMDc.npy"
    model = np.load(joinpath(model_path, model_name), allow_pickle=true)[1]
    models["rbf"][ci] = Dict("A" => (model.A - I)/pkdt, "B" => model.B/pkdt, 
                            "Ad" => model.A, "Bd" => model.B, 
                            "n_centers" => model.observables.n_centers, 
                            "cx" => model.observables.centers, "kw"=> model.observables.kernel_width, 
                            "nk" => size(model.A, 2), "nu" => size(model.B, 2), "nx" => size(model.C, 1))
end

function LQR(x::Vector, t::Number; K::Matrix=I, u_target::Vector=[0.0, 0.0, 0.0, 12], xref::Function=xref_default)
    control = K * (x - xref(t)) + u_target
    # control = (K * (x - xref(t)))[1:4] + u_target
    # control = (K * (x - xref(t)))[5:8] + u_target
    return vcat(clamp.(control[1:3], -π/6, π/6)*180/π, clamp(4096 * control[4], 0., 60000))
end

nk = 10
Ψrbfk(x) = Ψrbf(x; cx=models["rbf"][nk]["cx"], kw=models["rbf"][nk]["kw"])
xref_default_rbfk(t) = Ψrbfk(xref_default(t))
LQR7_rbf(x, t; K) = LQR(Ψrbfk(x[[1:6..., 9]]), t; K, xref=xref_default_rbfk)
A, B = models["rbf"][nk]["A"], models["rbf"][nk]["B"] # continuous, works but underactuated
# A, B = models["rbf"][nk]["Ad"], models["rbf"][nk]["Bd"] # discrete - original Δt, works but underactuated (identical)
# A, B = 5e-4 * (models["rbf"][nk]["A"] + I), 5e-4 * models["rbf"][nk]["B"] # discrete - integration Δt, doesn't stablize

sparsify, min_val = true, 1e-10
As, Bs = copy(A), copy(B)
if sparsify; As[abs.(A) .< min_val] .= 0; Bs[abs.(B) .< min_val] .= 0; end

AsP, BsP = As[1:7, 1:7], Bs[1:7, :]

# q, r, ϵ = 1, 1, 1 # works for 10d
# q, r, ϵ = 1e-8, 1e-2, 1e-4 # sort-of works for 10d
# Qk = q * diagm(vcat([1,1,2,1,1,2,10].^2, ϵ * ones(nk)));
# # Qk = q * diagm(vcat([1,1,1,1,1,1,1].^2, ϵ * ones(nk)));
# R  = r * diagm([1,1,1,1].^2);
# # K_rbfk = lqr(Discrete, As, Bs, Qk, R)
# K_rbfk = lqr(Continuous, As, Bs, Qk, R)

q, r, ϵ, rpy_lim = 1e-1, 1, 1e-1, 1
Q = diagm((q * [1,1,20,1,1,20,1e-6]).^2);
# Qk = diagm(vcat((q * [1,1,20,1,1,20,1e-6]).^2, ϵ * ones(nk)));
R  = r^2 * diagm([1, 1, 1, 1])
# R  = r^2 * diagm([rpy_lim, rpy_lim, rpy_lim, 1e4])

# K_rbfk = -lqr(Continuous, As, Bs, Qk, R)
K_rbf_dmd = -lqr(Continuous, AsP, BsP, Q, R)

control = K_rbfk * (Ψrbfk(x0[1:7]) - Ψrbfk(x_hover[1:7])) + [0.0, 0.0, 0.0, 12]
vcat(clamp.(control[1:3], -π/6, π/6)*180/π, clamp(4096 * control[4], 0., 60000))

LQR7_rbfk(x, t) = LQR7_rbf(x, t; K=K_rbfk)
# LQR7_rbfk(x, t) = LQR7_rbf(x, t; K=K_rbfk) .* [0., 0., 0., 1.]
# LQR7_rbfk(x, t) = LQR7_rbf(x, t; K=K_rbfk) .* [0.05, 0.05, 0.05, 1.]
# @time LQR7_rbfk(x0, 0.)
# @time LQR7_rbfk(0.9 * x_hover, 0.)
@time LQR7(x0, 0., K7=K_rbf_dmd)
@time LQR7(0.9 * x_hover, 0., K7=K_rbf_dmd)

tf, ll_ctrl_name = 4., "pid"
cs_sol = cs_solve(0.99 * x_hover, LQR7_rbfk, tf; ll_ctrl_name);
# cs_sol = cs_solve(x_hover, LQR7_rbfk, tf);
flight_plot(cs_sol, LQR7_rbfk; plot_title=L"\textrm{CrazySwarm - SA7-RBF10-KLQR:\:Test}", backend=gr, zlims=(0, 1.5))
flight_gif(cs_sol, LQR7_rbfk; plot_title=L"\textrm{CrazySwarm(PID) - SA7-RBF10-KLQR:\:Test}", zlims=(0, 2)) #, fname="SA7-RBF100-KLQR_hover.gif")

## for comparison

# pushfirst!(pyimport("sys")."path", "/Users/willsharpless/crazyflie-firmware/build");
# cs_SIL_dyn = pyimport("crazyswarm_SIL_dynamics")

tf = 4.
cs_sol = cs_solve(x0, LQR7, tf);
# cs_sol = cs_solve(x_hover, LQR7, tf);
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7:\:Test}")
# flight_gif(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7:\:Test}", fname="LQR_test.gif")



### MPC

K_cfc7m = -lqr(Continuous, A, B, Q, R)
K_cfc7ms = copy(K_cfc7m); K_cfc7ms[abs.(K_cfc7ms) .< 1e-5] .= 0;
K_cfc7ms

cs_sol = cs_solve(x0, LQR7, tf; ll_ctrl_name="mellinger");
cs_sol = cs_solve(x0, (x,t) -> LQR7(x,t; K7=K_cfc7ms), tf; ll_ctrl_name="mellinger");
cs_sol = cs_solve(x0, (x,t) -> LQR7(x,t; K7=K_cfc7ms, xref=fig8), tf_fig8; ll_ctrl_name="pid");

flight_plot(cs_sol, (x,t) -> LQR7(x,t; K7=K_cfc7, xref=fig8); plot_title=L"\textrm{CrazySwarm - LQR7:\:Test}")
flight_gif(cs_sol, (x,t) -> LQR7(x,t; K7=K_cfc7, xref=fig8); plot_title=L"\textrm{CrazySwarm - LQR7:\:Test}")