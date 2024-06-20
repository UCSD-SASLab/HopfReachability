
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
# Q𝒯 = diagm((q𝒯 * [1,1,20,1,1,20,0.5]).^2);
# Q𝒯 = diagm(inv.([1,1,10,1,1,10,0.5]).^2);
# Q𝒯 = diagm(inv.([1,1,2,1,1,2,1]).^2);                  # in crazyflie_clean
# Q𝒯 = diagm(inv.([1,1,2,20,20,10,10]).^2);              # better?
Q𝒯 = diagm(inv.([1,1,2,50,50,5,1]).^2);
# Q𝒯 = diagm(inv.([1,1,2,200,200,100,10]).^2);          
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

Hopf_ctrl_longhzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (.02, 0.4, 4.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)
Hopf_ctrl_medmhzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (0.1, 0.5, 2.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)
Hopf_ctrl_shrthzn(x, t) = cs_control(u_target + Hopf_minT(system, target, x[[1:6..., 9]]; time_p = (.05, .25, 1.0 + 1e-5), controller_kwargs...)[1]) # FIXME for TV systems (time_p)

@time for i=1:10; Hopf_ctrl_longhzn(x0, 0.); end
@time for i=1:10; Hopf_ctrl_medmhzn(x0, 0.); end
@time for i=1:10; Hopf_ctrl_shrthzn(x0, 0.); end

# ctrl_thrust_cs = (x,t)->[0., 0., 0., 4.925e4]
# sol_test, U_test = cs_solve_loop(x0, ctrl_thrust_cs, tf);

hz = 200
@time sol_mdhz, U_mdhz = cs_solve_loop(x0, Hopf_ctrl_medmhzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4], ll_ctrl_name="mellinger");
# @time sol_shhz, U_shhz = cs_solve_loop(x0, Hopf_ctrl_shrthzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4], ll_ctrl_name="mellinger");
# @time sol_lghz, U_lghz = cs_solve_loop(x0, Hopf_ctrl_longhzn, tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.9, smooth_ix=[1,2,3,4]);

plot_kwargs = Dict(:legend => :right,) #:xlims => (-10, 10), :ylims => (-10, 10), :zlims => (0, 1.5), :legend => :right, :backend=>gr)
flight_plot(sol_mdhz, U_mdhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Test}", plot_kwargs...)
# flight_plot(sol_shhz, U_shhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Sht\:Hzn\:Test}", plot_kwargs...)
# flight_plot(sol_lghz, U_lghz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Lng\:Hzn\:Test}", plot_kwargs...)

flight_gif(sol_mdhz, U_mdhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Mid\:Hzn\:Test}", plot_kwargs...)
flight_gif(sol_shhz, U_shhz; plot_title=L"\textrm{CrazySwarm - Hopf - \:Sht\:Hzn\:Test}", plot_kwargs...)


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
    u = norm(x[1:3] - x_hover[1:3]) > 0. ? u_target + Hopf_minT(system, target_fn(xref_fn(t)), x[[1:6..., 9]]; time_p = (0.1, 0.5, 2.0 + 1e-5), controller_kwargs...)[1] : LQR7(x,t; xref=xref_fn)
    return cs_control(u)
end

wp_list = [[1., 1., 1.], [1., -1., 1.], [-1., -1., 1.], [-1., 1., 1.]]
track_waypoints(t; delay=2.5) = t < 5 ? [0., 0., 1., 0., 0., 0., 0.] : ((t - 5)/delay < length(wp_list) ? vcat(wp_list[1+Int(floor((t - 5)/delay))], zeros(4)) : [0., 0., 1., 0., 0., 0., 0.])

hz = 100
tf = length(wp_list)*2.5 + 8
@time sol_mdhz_lqr, U_mdhz_lqr = cs_solve_loop(x0, (x,t)->Hopf_ctrl_midhzn_LQR_7D(x,t;xref_fn=track_waypoints), tf, ctrl_dt=1/hz, smoothing=true, sm_ratio=0.7, smooth_ix=[1,2,3,4], ll_ctrl_name="pid");
flight_plot(sol_mdhz_lqr, U_mdhz_lqr; plot_title=L"\textrm{CrazySwarm - Hopf+LQR - \:Mid\:Hzn}", plot_kwargs...)
flight_gif(sol_mdhz_lqr, U_mdhz_lqr; plot_title=L"\textrm{CrazySwarm - Hopf+LQR - \:Mid\:Hzn}", plot_kwargs...)