
include(pwd() * "/Examples/quadrotor/quadrotor_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations, ControlSystemsBase

## Initialize 

nx = 12
x0 = zeros(nx)

## Thrust / Roll test with Crazyswarm sim model

tf, dt = 8., 5e-4
ctrl_thrust_cs = (x,t)->[0., 0., 0., 4.925e4] # RPY can't be so easily controlled tho... try 8.18512e-4
cs_sol = cs_solve(x0, ctrl_thrust_cs, tf; dt)
flight_plot(cs_sol, ctrl_thrust_cs; plot_title=L"\textrm{CrazySwarm - Thrust}")

ctrl_roll_cs = (x,t)->[1., 0., 0., 4.925e4] # RPY dyneeds dt ≤ 1e-4
cs_sol = cs_solve(x0, ctrl_roll_cs, tf; dt)
flight_plot(cs_sol, ctrl_roll_cs; plot_title=L"\textrm{CrazySwarm - Roll}")

## LQR7 with Crazyswarm sim model

K_cfc7 = [0.0 0.2 0.0 0.0 0.2 0.0 0.0;
    0.2 0.0 0.0 0.2 0.0 0.0 0.0; 
    0.0 0.0 0.0 0.0 0.0 0.0 0.0;  
    0.0 0.0 -5.4772 0.0 0.0 -5.5637 0.0]

xref_default = t->[0., 0., 1., 0., 0., 0., 0.]

function LQR7(x::Vector, t::Number; K7::Matrix=K_cfc7, u_target::Vector=[0.0, 0.0, 0.0, 12], xref::Function=xref_default)
    control = K7 * (x[[1:6..., 9]] - xref(t)) + u_target
    return vcat(clamp.(control[1:3], -π/6, π/6)*180/π, 4096 * control[4])
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

tf = 8 + ceil(2pi)
fig8(t) = t < 5 ? [0., 0., 1., 0., 0., 0., 0.] : (t < 5 + 2pi ? [sin(t-5), sin(t-5)*cos(t-5), 1., 0., 0., 0., 0.] : [0., 0., 1., 0., 0., 0., 0.])
cs_sol = cs_solve(x0, (x,t) -> LQR7(x,t; xref=fig8), tf)
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

n_centers = [10, 15, 50, 100]
models = Dict("rbf"=>Dict()); pkdt = 0.05996;
for ci in n_centers
    model_name = "nb_3_cb_2_obs_RBF_$(ci)_reg_EDMDc.npy"
    model = np.load(joinpath(model_path, model_name), allow_pickle=true)[1]
    models["rbf"][ci] = Dict("A" => (model.A - I)/pkdt, "B" => model.B/pkdt, "n_centers" => model.observables.n_centers, "cx" => model.observables.centers, "kw"=> model.observables.kernel_width, "nk" => size(model.A, 2), "nu" => size(model.B, 2), "nx" => size(model.C, 1))
end
Ψrbf(x; cx, kw) = vcat(x, exp.(-(kw^2) * sum(eachrow((x .- cx).^2))))

function LQR(x::Vector, t::Number; K::Matrix=I, u_target::Vector=[0.0, 0.0, 0.0, 12], xref::Function=xref_default)
    # control = K * (x - xref(t)) + u_target
    control = (K * (x - xref(t)))[1:4] + u_target
    # control = (K * (x - xref(t)))[5:8] + u_target
    return vcat(clamp.(control[1:3], -π/6, π/6)*180/π, 4096 * control[4])
end

A, B = models["rbf"][10]["A"], models["rbf"][10]["B"]
q, r = 0.1, 0.1
Q10 = diagm(vcat([1,1,2,1,1,2,10].^2, 0.1*ones(10)))
K_rbf10 = -lqr(Discrete, A, B, Q10, r^2*I)

Ψrbf_10(x) = Ψrbf(x; cx=models["rbf"][10]["cx"], kw=models["rbf"][10]["kw"])
xref_default_rbf10(t) = Ψrbf_10(xref_default(t))
LQR7_rbf10(x ,t) = LQR(Ψrbf_10(x[[1:6..., 9]]), t; K=K_rbf10, xref=xref_default_rbf10)

@time LQR7_rbf10(x0, 0.)

tf = 8.
cs_sol = cs_solve(x0, LQR7, tf);
flight_plot(cs_sol, LQR7; plot_title=L"\textrm{CrazySwarm - LQR7:\:Test}")

cs_sol = cs_solve(x0, LQR7_rbf10, tf);
flight_plot(cs_sol, LQR7_rbf10; plot_title=L"\textrm{CrazySwarm - RBF10-LQR7:\:Test}")

