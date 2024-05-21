
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