
using LinearAlgebra, Plots, DifferentialEquations, LaTeXStrings

nx = 12
function quadrotor_12D!(du, u, p, t)
    
    m, g, Ixx, Iyy, Izz, controls = p
    F, τx, τy, τz = controls
      
    vx, vy, vz = u[4:6]
    ϕ, θ, ψ = u[7:9]
    wx, wy, wz = u[10:12]

    du[1] = vx
    du[2] = vy
    du[3] = vz

    du[4] = (cos(ϕ)*sin(θ)*cos(ψ) + sin(ϕ)*sin(ψ)) * F(t)/m
    du[5] = (cos(ϕ)*sin(θ)*sin(ψ) - sin(ϕ)*cos(ψ)) * F(t)/m
    du[6] = (cos(ϕ)*cos(θ)) * F(t)/m - g
    
    du[7] = wx + wy*sin(ϕ)*tan(θ) + wz*cos(ϕ)*tan(θ)
    du[8] = wy*cos(ϕ) - wz*sin(ϕ)
    du[9] = wy*sin(ϕ)/cos(θ) + wz*cos(ϕ)/cos(θ)
    
    du[10] = (τx(t) - (Izz - Iyy) * wy * wz) / Ixx
    du[11] = (τy(t) - (Ixx - Izz) * wx * wz) / Iyy
    du[12] = (τz(t) - (Iyy - Ixx) * wx * wy) / Izz 
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

function flight_plot(sol, ctrl_law; traj_path=true, backend=gr, xyz_only=false, base_alpha=0.6, rotor_alpha=0.6, scale=2, traj_alpha=0.6, camera=(45,15),
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
    thrust_plot = plot(tt, ctrl_law[1].(tt), ylabel=L"u_{thrust}", label="", xlims=(0., sol.t[end]+0.1)); 
    urpy_plot = plot(tt, ctrl_law[2].(tt), ylabel=L"u_{rpy}", label=L"u_ϕ", xlims=(0., sol.t[end]+0.1)); 
    plot!(tt, ctrl_law[3].(tt), label=L"u_θ"); plot!(tt, ctrl_law[4].(tt), label=L"u_ψ", xlabel=L"t"); 

    if vline; for t ∈ body_times, pl ∈ [v_plot, ω_plot, thrust_plot, urpy_plot]; vline!(pl, [t], lw=1.5, label="", color="black"); end; end

    plot(s3d_plot, v_plot, ω_plot, thrust_plot, urpy_plot; layout=@layout[A{0.5w} [B{0.3h}; C{0.3h}; D{0.2h}; E{0.2h}]], plot_title=plot_title, size=fig_size, dpi=300, legend=:left, kwargs...)
end

function flight_gif(sol, ctrl_law; fname=nothing, fps=20, loop=0, dt=0.1, traj_path=false, kwargs...)
    fname = isnothing(fname) ? "test_$(time()).gif" : fname
    anim = @animate for tt=1:dt:sol.t[end]
        flight_plot(sol, ctrl_law; traj_path=traj_path, body_times=[tt], kwargs...)
    end
    gif(anim, fname; fps, loop)
end

## Thrust Test

x0 = zeros(nx)
tspan = (0, 4π)

sine_thrust = function (t)
    t < 2π ? 9.81 + sin(t) : 9.81 - sin(t)
end
zero_f = t -> 0.
ctrls_thrust = [sine_thrust, zero_f, zero_f, zero_f]

p = [1.0, 9.81, 0.1, 0.1, 0.2, ctrls_thrust]  # Initial thrust and torques are set to zero

prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_thrust = DifferentialEquations.solve(prob)

flight_plot(sol_thrust, ctrls_thrust; plot_title=L"\textrm{Thrust}")

# flight_gif(sol_thrust, ctrls_thrust; plot_title=L"\textrm{Thrust}", fname="test.gif")

## Positive Roll

tspan = (0, 2π)
ut, ur = t -> 10, t -> 0.00125 * sin(t)
ctrl_roll = [ut, ur, zero_f, zero_f]
p = [1.0, 9.81, 0.1, 0.1, 0.2, ctrl_roll]  # Initial thrust and torques are set to zero

prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_roll = DifferentialEquations.solve(prob)

flight_plot(sol_roll, ctrl_roll; plot_title=L"\textrm{Roll}")

## Positive Pitch

ctrl_pitch = [ut, zero_f, ur, zero_f]
p = [1.0, 9.81, 0.1, 0.1, 0.2, ctrl_pitch]  # Initial thrust and torques are set to zero
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_pitch = DifferentialEquations.solve(prob)

flight_plot(sol_pitch, ctrl_pitch; plot_title=L"\textrm{Pitch}")

## Positive Yaw

ctrl_yaw = [ut, zero_f, zero_f, t -> 10*ur(t)]
p = [1.0, 9.81, 0.1, 0.1, 0.2, ctrl_yaw]  # Initial thrust and torques are set to zero
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_yaw = DifferentialEquations.solve(prob)

flight_plot(sol_yaw, ctrl_yaw; plot_title=L"\textrm{Yaw}")

### model for controller (7D? 12D?)



### linearize model for controller

### make controller(s) (LQR, MPC, load LQR matrix from cfclean)

### simulate 
## - go to waypoint
## - track figure 8 
## - invert and stabilize (reach upright at initial point/ any point?)
## Score with quantitative metrics (oc time, problem solved)
