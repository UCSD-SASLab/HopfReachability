
include(pwd() * "/Examples/quadrotor/quadrotor_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations

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

function LQR7(x, t; u_target=[0.0, 0.0, 0.0, 12], xref=t->[0., 0., 1., 0., 0., 0., 0.])
    K_matrix = [0.0 0.2 0.0 0.0 0.2 0.0 0.0;
            0.2 0.0 0.0 0.2 0.0 0.0 0.0; 
            0.0 0.0 0.0 0.0 0.0 0.0 0.0;  
            0.0 0.0 -5.4772 0.0 0.0 -5.5637 0.0]
    control = K_matrix * (x[[1:6..., 9]] - xref(t)) + u_target
    return vcat(clamp.(control[1:3], -np.pi/6, np.pi/6)*180/π, 4096 * control[4])
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

## LQR-Koopman

# model_path = pwd() * "/Examples/quadrotor/lifted_data/gen/models";
# pushfirst!(pyimport("sys")."path", model_path);
# np = pyimport("numpy");
# pk = pyimport("pykoopman"); # requires PyCall.pyversion == v"3.9.12"

# models = Dict("rbf"=>Dict()); pkdt = 0.1;
# use_rbf_small = true
# n_centers = use_rbf_small ? [3, 5, 9] : [9, 16, 25, 36, 49, 64, 81]
# for i = 1:7
#     model_name = "VanderPol_rbf_gauss_nc$(n_centers[i])rand_kw1p0.npy"
#     model = np.load(joinpath(model_path, model_name), allow_pickle=true)[1]
#     models["rbf"][i] = Dict("A" => (model.A - I)/pkdt, "B" => model.B/pkdt, "n_centers" => model.observables.n_centers, "cx" => model.observables.centers, "kw"=> model.observables.kernel_width, "nk" => size(model.A, 2), "nu" => size(model.B, 2), "nx" => size(model.C, 1))
# end
# Ψrbf(x; cx, kw) = vcat(x, exp.(-(kw^2) * sum(eachrow((x .- cx).^2))))

