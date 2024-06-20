
using LinearAlgebra, Plots, DifferentialEquations, LaTeXStrings

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

function flight_plot(sol, ctrl_law; traj_path=true, backend=gr, xyz_only=false, degrees=false, base_alpha=0.6, rotor_alpha=0.6, scale=1, traj_alpha=0.6, camera=(45,15),
    fig_size=(1000,500), body_times=nothing, vline=true, bd=nothing, xlims=nothing, ylims=nothing, zlims=nothing, plot_title=L"\textrm{Flight}", kwargs...)
    
    backend()
    bd = isnothing(bd) ? 1.2maximum(abs.(sol[1:3,:])) : bd
    body_times = isnothing(body_times) ? [0., 1sol.t[end]/4, sol.t[end]/2, 3sol.t[end]/4, sol.t[end]] : body_times
    xlims_3d = isnothing(xlims) ? (-bd, bd) : xlims; ylims_3d = isnothing(ylims) ? (-bd, bd) : ylims; zlims_3d = isnothing(zlims) ? (0, bd) : zlims
    s3d_plot = plot(xlabel=L"x", ylabel=L"y", zlabel=L"z", camera=camera);
    if traj_path; plot!(sol, idxs=(1,2,3), lw=2, label=["" "" ""], xlims=xlims_3d, ylims=ylims_3d, zlims=zlims_3d, alpha=traj_alpha); else; plot!(xlims=xlims_3d, ylims=ylims_3d, zlims=zlims_3d); end
    for t ∈ body_times; plot_quadrotor!(s3d_plot, sol(t); base_alpha=base_alpha, rotor_alpha=rotor_alpha, scale=scale, degrees=degrees); end 
    if xyz_only; 
        plot!(title=plot_title)
        return s3d_plot; 
    end

    v_plot = plot(sol, idxs=(0,4:6), lw=2, ylabel=L"v_i", label=[L"v_x" L"v_y" L"v_z"], xlabel="", xlims=(0., sol.t[end]+0.1))
    ω_plot = plot(sol, idxs=(0,10:12), lw=2, ylabel=L"ω_i", label=[L"ω_ϕ" L"ω_θ" L"ω_ψ"], xlabel="", xlims=(0., sol.t[end]+0.1))

    tt, ctrls_tt = nothing, nothing
    if typeof(ctrl_law) <: Function
        tt = 0.:0.01:sol.t[end]
        ctrls_tt = hcat([ctrl_law(x,t) for (x,t) in zip(sol.(tt), tt)]...)
    else typeof(ctrl_law) <: Array{Float64, 2}
        tt = range(0., sol.t[end], size(ctrl_law,2))
        ctrls_tt = ctrl_law
    end
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

R_ZYX(ϕ,θ,ψ) = [cos(θ)*cos(ψ) cos(θ)*sin(ψ) -sin(θ);
    sin(ϕ)*sin(θ)*cos(ψ)-cos(ϕ)*sin(ψ) sin(ϕ)*sin(θ)*sin(ψ)+cos(ϕ)*cos(ψ) sin(ϕ)*cos(θ);
    cos(ϕ)*sin(θ)*cos(ψ)+sin(ϕ)*sin(ψ) cos(ϕ)*sin(θ)*sin(ψ)-sin(ϕ)*cos(ψ) cos(ϕ)*cos(θ)]'

function plot_quadrotor!(traj_plot, x; r = 0.1, ph = 0.025, base_alpha=0.8, rotor_alpha=0.2, scale=1, degrees=false, kwargs...)
    xyz = x[1:3]
    rpy = degrees ? π/180 * x[7:9] : x[7:9]
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

### Crazyswarm (Rowan) model

using PyCall

pushfirst!(pyimport("sys")."path", pwd() * "/Examples/quadrotor/");
pushfirst!(pyimport("sys")."path", "/Users/willsharpless/crazyflie-firmware/build");
cs_SIL_dyn = pyimport("crazyswarm_SIL_dynamics")

function cs_control(u)
    # convert from 
    return vcat(clamp.(u[1:3], -π/6, π/6)*180/π, clamp(4096 * u[4], 0., 60000))
end

function sol_wrap(cs_X, tf; dt=5e-4)
    # wrap in Julia DiffEq solution structure for plotting
    x0, tspan = cs_X[:,1], (0, tf)
    p = [[0.034, 9.81, 0.1, 0.1, 0.1], (x,t)->zeros(4)]
    prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
    cs_sol = DifferentialEquations.solve(prob; saveat=dt) # dummy solution, overwritten with crazyswarm trajectory
    @assert length(cs_sol) == size(cs_X, 2) "arr lens dont match: $(length(cs_sol))≠$(size(cs_X, 2))"
    for i=1:length(cs_sol); cs_sol[i] = cs_X[:,i]; end
    return cs_sol
end

function cs_solve(x0, ut, tf; dt=5e-4, ll_ctrl_name="pid")
    # solve crazyswarm simulation
    cs_SIL_dyn = pyimport("crazyswarm_SIL_dynamics")
    cs_X = cs_SIL_dyn.simulate(x0, ut, tf; dt, ll_ctrl_name)
    return sol_wrap(cs_X, tf; dt)
end

np, rowan = pyimport("numpy"), pyimport("rowan")
# cs_model, cs_types, cs_SIL = pyimport("crazyswarm_model"), pyimport("sim_data_types"), pyimport("crazyflie_sil");
cs_SIL_dyn = pyimport("crazyswarm_SIL_dynamics")
function cs_solve_loop(x0, ut, tf; dt=5e-4, ctrl_dt=1e-2, ll_ctrl_name = "pid", smoothing=false, sm_ratio=0.8, smooth_ix=[1,2,3], print_ctrl_t=false)
    
    x0_cs = cs_SIL_dyn.State(pos=x0[1:3], vel=x0[4:6], quat=rowan.from_euler(x0[7:9]...), omega=x0[10:12])
    steps = length(0:dt:tf)
    X, U = zeros(12, steps), zeros(4, steps)
    X[:,1] = x0
    u_prev = zeros(4)

    ## Solve with SIL Crazyswarm model
    model = cs_SIL_dyn.Quadrotor(x0_cs)
    uav = cs_SIL_dyn.CrazyflieSIL("name", x0_cs, ll_ctrl_name, model.time, initialState=x0_cs) # usually self.backend.time

    max_ctrl_t = 0.
    for (tix, ti) in enumerate(0:dt:tf)

        # Solve new control at frequency 1/ctrl_dt
        ctrl_t = @elapsed begin
            uˢ = (ti/dt) % Int(ctrl_dt/dt) ≈ 0 ? ut(model.fullstate(), ti) : u_prev
            uˢ[smooth_ix] = smoothing ? (1-sm_ratio) * uˢ[smooth_ix] + sm_ratio * u_prev[smooth_ix] : uˢ[smooth_ix]
            u_prev = uˢ
        end
        max_ctrl_t = max(max_ctrl_t, ctrl_t)

        # Evolve sim at frequency 1/dt
        uav.cmdVelLegacy(uˢ...)               # sets mode & set_point
        action = uav.executeController()      # calls controller, powerDist, pwm_to_rpm
        model.step(action, dt)                # evolves 13D model, uses FE+rowan
        uav.setState(model.state)             # updates internal

        X[:,tix], U[:,tix] = model.fullstate(), uˢ
    end
    if print_ctrl_t; println("MAX CTRL TIME: $max_ctrl_t s")

    return sol_wrap(X, tf; dt), U
end