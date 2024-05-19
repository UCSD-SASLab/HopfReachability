
using LinearAlgebra, Plots, DifferentialEquations, LaTeXStrings

nx = 12
function quadrotor_12D!(dx, x, p, t)
    params, controls = p
    m, g, Ixx, Iyy, Izz = params
    F, τx, τy, τz = controls(x, t)
      
    vx, vy, vz = x[4:6]
    ϕ, θ, ψ = x[7:9]
    wx, wy, wz = x[10:12]

    dx[1] = vx
    dx[2] = vy
    dx[3] = vz

    dx[4] = (cos(ϕ)*sin(θ)*cos(ψ) + sin(ϕ)*sin(ψ)) * F/m
    dx[5] = (cos(ϕ)*sin(θ)*sin(ψ) - sin(ϕ)*cos(ψ)) * F/m
    dx[6] = (cos(ϕ)*cos(θ)) * F/m - g
    
    dx[7] = wx + wy*sin(ϕ)*tan(θ) + wz*cos(ϕ)*tan(θ)
    dx[8] = wy*cos(ϕ) - wz*sin(ϕ)
    dx[9] = wy*sin(ϕ)/cos(θ) + wz*cos(ϕ)/cos(θ)
    
    dx[10] = (τx - (Izz - Iyy) * wy * wz) / Ixx
    dx[11] = (τy - (Ixx - Izz) * wx * wz) / Iyy
    dx[12] = (τz - (Iyy - Ixx) * wx * wy) / Izz 
end

R_ZYX(ϕ,θ,ψ) = [cos(θ)*cos(ψ) cos(θ)*sin(ψ) -sin(θ);
    sin(ϕ)*sin(θ)*cos(ψ)-cos(ϕ)*sin(ψ) sin(ϕ)*sin(θ)*sin(ψ)+cos(ϕ)*cos(ψ) sin(ϕ)*cos(θ);
    cos(ϕ)*sin(θ)*cos(ψ)+sin(ϕ)*sin(ψ) cos(ϕ)*sin(θ)*sin(ψ)-sin(ϕ)*cos(ψ) cos(ϕ)*cos(θ)]'

function plot_quadrotor!(traj_plot, x; r = 0.1, ph = 0.025, base_alpha=0.8, rotor_alpha=0.2, scale=1, kwargs...)
    xyz, rpy = x[1:3], x[7:9]
    colors = ["green", "red", "blue", "blue"]
    r, ph = scale * r, scale * ph
    base = vcat(r * [-1 1 1 -1 -1; 1 1 -1 -1 1], zeros(5)')
    coffs = [[1.5r, 1.5r, ph], [1.5r, -1.5r, ph], [-1.5r, -1.5r, ph], [-1.5r, 1.5r, ph]]
    for i=1:4
        pipe = R_ZYX(rpy...) * base[:, i:i+1] .+ xyz
        plot!(traj_plot, pipe[1,:], pipe[2,:], pipe[3,:]; st=:line, lw=2, color=:black, alpha=base_alpha, label="", kwargs...)
    end
    for i=1:4
        prop = R_ZYX(rpy...) * (vcat(1.05r * vcat(cos.(0:1/100:2π+1e-1)', sin.(0:1/100:2π+1e-1)'), zero(0:1/100:2π+1e-1)') .+ (coffs[i])) .+ xyz
        plot!(traj_plot, eachrow(prop)...; st=:line, color=colors[i], alpha=rotor_alpha, label="", kwargs...)
    end
    return traj_plot
end

function flight_plot(sol, ctrl_law; traj_path=true, backend=gr, xyz_only=false, base_alpha=0.6, rotor_alpha=0.6, scale=1, traj_alpha=0.6, camera=(45,15),
    fig_size=(1000,500), body_times=nothing, vline=true, bd=nothing, xlims=nothing, ylims=nothing, zlims=nothing, plot_title=L"\textrm{Flight}", kwargs...)
    
    backend()
    bd = isnothing(bd) ? 1.2maximum(abs.(sol[1:3,:])) : bd
    body_times = isnothing(body_times) ? [0., 1sol.t[end]/4, sol.t[end]/2, 3sol.t[end]/4, sol.t[end]] : body_times
    xlims_3d = isnothing(xlims) ? (-bd, bd) : xlims; ylims_3d = isnothing(ylims) ? (-bd, bd) : ylims; zlims_3d = isnothing(zlims) ? (0, bd) : zlims
    s3d_plot = plot(xlabel=L"x", ylabel=L"y", zlabel=L"z", camera=camera);
    if traj_path; plot!(sol, vars=(1,2,3), lw=2, label=["" "" ""], xlims=xlims_3d, ylims=ylims_3d, zlims=zlims_3d, alpha=traj_alpha); else; plot!(xlims=xlims_3d, ylims=ylims_3d, zlims=zlims_3d); end
    for t ∈ body_times; plot_quadrotor!(s3d_plot, sol(t); base_alpha=base_alpha, rotor_alpha=rotor_alpha, scale=scale); end 
    if xyz_only; 
        plot!(title=plot_title)
        return s3d_plot; 
    end

    v_plot = plot(sol, vars=(0,4:6), lw=2, ylabel=L"v_i", label=[L"v_x" L"v_y" L"v_z"], xlabel="", xlims=(0., sol.t[end]+0.1))
    ω_plot = plot(sol, vars=(0,10:12), lw=2, ylabel=L"ω_i", label=[L"ω_ϕ" L"ω_θ" L"ω_ψ"], xlabel="", xlims=(0., sol.t[end]+0.1))

    tt = 0.:0.01:sol.t[end]
    ctrls_tt = hcat([ctrl_law(x,t) for (x,t) in zip(sol.(tt), tt)]...)
    thrust_plot = plot(tt, ctrls_tt[4,:], ylabel=L"u_{thrust}", label="", xlims=(0., sol.t[end]+0.1)); 
    urpy_plot = plot(tt, ctrls_tt[1,:], ylabel=L"u_{rpy}", label=L"u_ϕ", xlims=(0., sol.t[end]+0.1)); 
    plot!(tt, ctrls_tt[2,:], label=L"u_θ"); plot!(tt, ctrls_tt[3,:], label=L"u_ψ", xlabel=L"t"); 

    if vline; for t ∈ body_times, pl ∈ [v_plot, ω_plot, thrust_plot, urpy_plot]; vline!(pl, [t], lw=1.5, label="", color="black"); end; end

    plot(s3d_plot, v_plot, ω_plot, thrust_plot, urpy_plot; layout=@layout[A{0.5w} [B{0.3h}; C{0.3h}; D{0.2h}; E{0.2h}]], plot_title=plot_title, size=fig_size, dpi=300, legend=:left, kwargs...)
end

function flight_gif(sol, ctrl_law; fname=nothing, fps=20, loop=0, dt=0.1, traj_path=false, kwargs...)
    fname = isnothing(fname) ? "test_$(time()).gif" : fname
    anim = @animate for tt=0.:dt:sol.t[end]
        flight_plot(sol, ctrl_law; traj_path=traj_path, body_times=[tt], kwargs...)
    end
    gif(anim, fname; fps, loop)
end

## Initialize

test_params = 1.0, 9.81, 0.1, 0.1, 0.2
cf_params = 0.034, 9.81, 16.571710e-6, 16.655602e-6, 29.261652e-6 # kg, m s^-2, kg m^2, kg m^2, kg m^2
params = cf_params

ut, ur = t -> 9.81*params[1] + 5e-3, t -> 1e-7 * sin(t)
zero_f = t -> 0.
sine_thrust = function (t); t < 2π ? 9.81*params[1] + 0.01 * sin(t) : 9.81*params[1] - 0.01 * sin(t); end

ctrls_thrust = (x,t) -> [sine_thrust(t), zero_f(t), zero_f(t), zero_f(t)]
ctrl_pitch = (x,t) -> [ut(t), zero_f(t), ur(t), zero_f(t)]
ctrl_yaw = (x,t) -> [ut(t), zero_f(t), zero_f(t), 10*ur(t)]
ctrl_roll = (x,t) -> [ut(t), ur(t), zero_f(t), zero_f(t)]

## Thrust Test

x0 = zeros(nx)
tspan = (0, 4π)
p = [params, ctrls_thrust]

prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_thrust = DifferentialEquations.solve(prob)

flight_plot(sol_thrust, ctrls_thrust; plot_title=L"\textrm{Thrust}")

# flight_gif(sol_thrust, ctrls_thrust; plot_title=L"\textrm{Thrust}", fname="test.gif")

## Positive Roll

tspan = (0, 2π)
p = [params, ctrl_roll]

prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_roll = DifferentialEquations.solve(prob)

flight_plot(sol_roll, ctrl_roll; plot_title=L"\textrm{Roll}")

## Positive Pitch

p = [params, ctrl_pitch]
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_pitch = DifferentialEquations.solve(prob)

flight_plot(sol_pitch, ctrl_pitch; plot_title=L"\textrm{Pitch}")

## Positive Yaw

p = [params, ctrl_yaw]
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_yaw = DifferentialEquations.solve(prob)

flight_plot(sol_yaw, ctrl_yaw; plot_title=L"\textrm{Yaw}")

### Comparison with Crazyswarm (Rowan) model

using PyCall

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/quadrotor/");
pushfirst!(pyimport("sys")."path", "/Users/willsharpless/crazyflie-firmware/build");
# hj_r_setup = pyimport("vdp_hj_reachability");
np, rowan = pyimport("numpy"), pyimport("rowan")
cs_model, cs_types, cs_SIL = pyimport("crazyswarm_model"), pyimport("sim_data_types"), pyimport("crazyflie_sil");

function cs_solve(x0, ut, tf; dt=5e-4, ll_ctrl_name = "mellinger")
    
    x0_cs = cs_types.State(pos=x0[1:3], vel=x0[4:6], quat=rowan.from_euler(x0[7:9]...), omega=x0[10:12])
    steps = length(0:dt:tf)
    X = zeros(12, steps)
    X[:,1] = x0
    
    ## Solve with SIL Crazyswarm model
    model = cs_model.Quadrotor(x0_cs)
    uav = cs_SIL.CrazyflieSIL("name", x0_cs, ll_ctrl_name, model.time, initialState=x0_cs) # usually self.backend.time

    for (tix, ti) in enumerate(0:dt:tf)
        uav.cmdVelLegacy(ut(model.fullstate(), ti)...)            # sets mode & set_point
        action = uav.executeController()      # calls controller, powerDist, pwm_to_rpm
        model.step(action, dt)                # evolves 13D model, uses FE+rowan
        uav.setState(model.state)             # updates internal
        X[:,tix] = model.fullstate()
    end

    return X
end

function make_sol(cs_X, tf; dt=5e-3)

    x0 = cs_X[:,1]
    tspan = (0, tf)
    params = [0.034, 9.81, 16.571710e-6, 16.655602e-6, 29.261652e-6] # filler
    p = [params, (x,t)->zeros(4)]

    prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
    cs_sol = DifferentialEquations.solve(prob; saveat=dt)

    @assert length(cs_sol) == size(cs_X, 2) "arr lens dont match: $(length(cs_sol))≠$(size(cs_X, 2))"
    for i=1:length(cs_sol); cs_sol[i] = cs_X[:,i]; end

    return cs_sol
end

## Thrust / Roll test with Crazyswarm sim model

tf, dt = 10., 5e-3
ctrl_thrust_cs = (x,t)->[0., 0., 0., 4.925e4] # RPY can't be so easily controlled tho... try 8.18512e-4
cs_sol = make_sol(cs_solve(x0, ctrl_thrust_cs, tf; dt), tf);
flight_plot(cs_sol, ctrl_thrust_cs; plot_title=L"\textrm{CrazySwarm - Thrust}")

tf, dt = 10., 5e-3
ctrl_thrust_cs = (x,t)->[8.18511e-4, 0., 0., 4.925e4] # RPY can't be so easily controlled tho... try 8.18512e-4 vs. 8.18511e-4
cs_sol = make_sol(cs_solve(x0, ctrl_thrust_cs, tf; dt), tf);
flight_plot(cs_sol, ctrl_thrust_cs; plot_title=L"\textrm{CrazySwarm - Thrust}", backend=gr)

## LQR test with Crazyswarm sim model

K_matrix = [0.0 0.2 0.0 0.0 0.2 0.0 0.0;
            0.2 0.0 0.0 0.2 0.0 0.0 0.0; 
            0.0 0.0 0.0 0.0 0.0 0.0 0.0;  
            0.0 0.0 -5.4772 0.0 0.0 -5.5637 0.0]

function LQR(x, t; u_target=[0.0, 0.0, 0.0, 12], xref=t->[0., 0., 1., 0., 0., 0., 0.])
    control = K_matrix * (x[[1:6..., 9]] - xref(t)) + u_target
    return vcat(clamp.(control[1:3], -np.pi/6, np.pi/6)*180/π, 4096 * control[4])
end

tf, dt = 10., 5e-3
cs_sol = make_sol(cs_solve(x0, LQR, tf; dt), tf);
flight_plot(cs_sol, LQR; plot_title=L"\textrm{CrazySwarm - LQR}", backend=plotly)
flight_gif(cs_sol, LQR; plot_title=L"\textrm{CrazySwarm - LQR}", fname="LQR_quadsim_csport.gif")

tf, dt = 15., 5e-3
xref_2 = t->[0., 1., 1., 0., 0., 0., 0.]
cs_sol = make_sol(cs_solve(x0, (x,t) -> LQR(x,t; xref=xref_2), tf; dt), tf);
flight_plot(cs_sol, LQR; plot_title=L"\textrm{CrazySwarm - LQR Test 2}")

# tf, dt = 15., 5e-3
# fqy = 0.05
# fig8(t) = t < 5 ? [0., 0., 1., 0., 0., 0., 0.] : [sin(fqy*(t-5)), sin(fqy*(t-5))*cos(fqy*(t-5)), 1., 0., 0., 0., 0.]
# cs_sol = make_sol(cs_solve(x0, (x,t) -> LQR(x,t; xref=fig8), tf; dt), tf);
# flight_plot(cs_sol, LQR; plot_title=L"\textrm{CrazySwarm - LQR Fig8}")
# flight_gif(cs_sol, LQR; plot_title=L"\textrm{CrazySwarm - LQR Fig8}", fname="LQR_fig8_quadsim_csport.gif")

### model for controller (7D? 12D?)

### linearize model for controller

### make controller(s) (LQR, MPC, load LQR matrix from cfclean)

### simulate 
## - go to waypoint
## - track figure 8 
## - invert and stabilize (reach upright at initial point/ any point?)
## Score with quantitative metrics (oc time, problem solved)