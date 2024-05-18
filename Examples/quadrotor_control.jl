
include(pwd() * "/src/lin_utils.jl");
include(pwd() * "/src/control_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations

### i/c, inputs, model params

### model for simulation (12D?)

using DifferentialEquations

nx = 12
function quadrotor_12D!(du, u, p, t)
    
    m, g, Ixx, Iyy, Izz, F, τx, τy, τz = p   # Mass, gravity, moments of inertia, thrust, and torques
      
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

## Thrust Test

thrust = function (t)
    t < 2π ? 9.81 + sin(t) : 9.81 - sin(t)
end
zero_f = t -> 0.
p = [1.0, 9.81, 0.1, 0.1, 0.2, thrust, zero_f, zero_f, zero_f]  # Initial thrust and torques are set to zero

x0 = zeros(nx)
tspan = (0, 4π)
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol = DifferentialEquations.solve(prob)
s3d_plot = plot(sol, vars=(1,2,3), lw=2);
states_plot = plot(sol, lw=2, ylabel="x_i");

tt = tspan[1]:0.01:tspan[2]
ctrls_plot = plot(tt, thrust.(tt), ylabel="u_1", label="thrust");

plot(s3d_plot, states_plot, ctrls_plot, layout=@layout[A{0.65w} [B{0.8h}; C{0.2h}]], plot_title="+Thrust", size=(1000,550))

## Positive Roll

using LaTeXStrings

plotlyjs()
tspan = (0, 2π)
ut, ur = t -> 10, t -> 0.00125 * sin(t)
ctrl_law = [ut, ur, zero_f, zero_f]
p = [1.0, 9.81, 0.1, 0.1, 0.2, ctrl_law...]  # Initial thrust and torques are set to zero
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol = DifferentialEquations.solve(prob)

gr()
plotly()
bd = 1.2maximum(abs.(sol[1:3,:]))
s3d_plot = plot(sol, vars=(1,2,3), lw=2, xlims=(-bd, bd), ylims=(-bd, bd), zlims=(0, bd), xlabel=L"x", ylabel=L"y", zlabel=L"z", label=["" "" ""]);
for t ∈ [0., tspan[2]/2, 3tspan[2]/4, tspan[2]]; plot_quadrotor!(s3d_plot, sol(t); scale=2); end 
s3d_plot

v_plot = plot(sol, vars=(0,4:6), lw=2, ylabel=L"v_i", label=[L"v_x" L"v_y" L"v_z"], xlabel="")
rpy_plot = plot(sol, vars=(0,10:12), lw=2, ylabel=L"ω_i", label=[L"ω_ϕ" L"ω_θ" L"ω_ψ"], xlabel="")

tt = tspan[1]:0.01:tspan[2]
thrust_plot = plot(tt, ctrl_law[1].(tt), ylabel=L"u_{thrust}", label=""); 
urpy_plot = plot(tt, ctrl_law[2].(tt), ylabel=L"u_{rpy}", label=L"u_ϕ"); 
plot!(tt, ctrl_law[3].(tt), label=L"u_θ"); plot!(tt, ctrl_law[4].(tt), label=L"u_ψ", xlabel=L"t"); 

plot(s3d_plot, v_plot, rpy_plot, thrust_plot, urpy_plot, layout=@layout[A{0.5w} [B{0.3h}; C{0.3h}; D{0.2h}; E{0.2h}]], plot_title=L"+\textrm{Thrust}/+\textrm{Roll}", size=(1000,500), dpi=300)

## Positive Pitch

plotlyjs()
tspan = (0, 2π)
ut, ur = t -> 11, t -> 0.00125*t
p = [1.0, 9.81, 0.1, 0.1, 0.2, ut, zero_f, ur, zero_f]  # Initial thrust and torques are set to zero
# p = [1.0, 9.81, 0.1, 0.1, 0.2, ut, zero_f, zero_f, ur]  # Initial thrust and torques are set to zero
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol = DifferentialEquations.solve(prob)

s3d_plot = plot(sol, vars=(1,2,3), lw=2, xlims=(-2, 2), ylims=(-2, 2), zlims=(0, 3));
states_plot = plot(sol, vars=(0,4:6), lw=2, ylabel="x_i");

tt = tspan[1]:0.01:tspan[2]
thrust_plot = plot(tt, ut.(tt), ylabel="u_1", label="thrust"); 
rpy_plot = plot(tt, ur.(tt), ylabel="u_3", label="pitch");

plot(s3d_plot, states_plot, thrust_plot, rpy_plot, layout=@layout[A{0.7w} [B{0.6h}; C{0.2h}; D{0.2h}]], plot_title="+Thrust/+Pitch", size=(1000,500))

## Positive Yaw

plotlyjs()
tspan = (0, 2π)
ut, ur = t -> 11, t -> 0.00125*t
p = [1.0, 9.81, 0.1, 0.1, 0.2, ut, zero_f, zero_f, ur]  # Initial thrust and torques are set to zero
# p = [1.0, 9.81, 0.1, 0.1, 0.2, ut, zero_f, zero_f, ur]  # Initial thrust and torques are set to zero
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol = DifferentialEquations.solve(prob)

s3d_plot = plot(sol, vars=(1,2,3), lw=2, xlims=(-2, 2), ylims=(-2, 2), zlims=(0, 3));
states_plot = plot(sol, vars=(0,4:6), lw=2, ylabel="x_i");

tt = tspan[1]:0.01:tspan[2]
thrust_plot = plot(tt, ut.(tt), ylabel="u_1", label="thrust"); 
rpy_plot = plot(tt, ur.(tt), ylabel="u_3", label="pitch");

plot(s3d_plot, states_plot, thrust_plot, rpy_plot, layout=@layout[A{0.7w} [B{0.6h}; C{0.2h}; D{0.2h}]], plot_title="+Thrust/+Yaw", size=(1000,500))

R_ZYX(ϕ,θ,ψ) = [cos(θ)*cos(ψ) cos(θ)*sin(ψ) -sin(θ);
    sin(ϕ)*sin(θ)*cos(ψ)-cos(ϕ)*sin(ψ) sin(ϕ)*sin(θ)*sin(ψ)+cos(ϕ)*cos(ψ) sin(ϕ)*cos(θ);
    cos(ϕ)*sin(θ)*cos(ψ)+sin(ϕ)*sin(ψ) cos(ϕ)*sin(θ)*sin(ψ)-sin(ϕ)*cos(ψ) cos(ϕ)*cos(θ)]'

# R_ZYX(a,b,c) = [cos(a)*cos(b) cos(a)*sin(b)*sin(c)-cos(a)*sin(b) cos(a)*sin(b)*cos(c)-sin(a)*sin(c); 
#     sin(a)*cos(b) sin(a)*sin(b)*sin(c)+cos(a)*cos(c) sin(a)*sin(b)*cos(c)-cos(a)*sin(c); 
#     -sin(b) cos(b)*sin(c) cos(b)*cos(c)]'

# rpy = [0., 0., 0.] 
# plot(title="rpy = $(round.(rpy, digits=2))");
# ; c = ones(3)

traj_plot = plot(title="test");
plot_quadrotor!(traj_plot, [0.5, 0.5, 1., 0., 0., 0., 0., 0., 0.], xlims=(-1,1), ylims=(-1,1), zlims=(0, 1.5))

function plot_quadrotor!(traj_plot, x; r = 0.1, ph = 0.025, base_alpha=0.8, rotor_alpha=0.2, scale=1, kwargs...)
    xyz, rpy = x[1:3], x[7:9]
    colors = ["green", "red", "blue", "blue"]
    r, ph = scale * r, scale * ph
    base = vcat(r * [-1 1 1 -1 -1; 1 1 -1 -1 1], zeros(5)')
    coffs = [[1.5r, 1.5r, ph], [1.5r, -1.5r, ph], [-1.5r, -1.5r, ph], [-1.5r, 1.5r, ph]]
    for i=1:4
        prop = R_ZYX(rpy...) * (vcat(1.05r * vcat(cos.(0:1/100:2π+1e-1)', sin.(0:1/100:2π+1e-1)'), zero(0:1/100:2π+1e-1)') .+ (coffs[i])) .+ xyz
        pipe = R_ZYX(rpy...) * base[:, i:i+1] .+ xyz
        plot!(traj_plot, eachrow(prop)...; st=:line, color=colors[i], alpha=rotor_alpha, label="", kwargs...)
        plot!(traj_plot, pipe[1,:], pipe[2,:], pipe[3,:]; st=:line, lw=2, color=:black, alpha=base_alpha, label="", kwargs...)
    end
    return traj_plot
end

why
### model for controller (7D? 12D?)



### linearize model for controller

### make controller(s) (LQR, MPC, load LQR matrix from cfclean)

### simulate 
## - go to waypoint
## - track figure 8 
## - invert and stabilize (reach upright at initial point/ any point?)
## Score with quantitative metrics (oc time, problem solved)

### NEW SCRIPT TIME ####

### try solcing oc w/ hopf (Hopf_minT)
## timing & parameter refinement pt 1

# using .HopfReachability: Hopf_BRS, Hopf_cd, make_levelset_fs
# include(pwd() * "/src/HopfReachability.jl");

### simulate hopf controller
## timing & parameter optimization pt 2
## score

### MAYBE compute lin error w/ RA.jl (historgram?) for LQR-fig8-sim segment

# include(pwd() * "/src/cons_lin_utils.jl");

### simulate hopf controller w error (levels) + MPCg
## score

### load koopman models, trajectory data, compute lifted error on trajectory set

### MAYBE compute max error on lifted feasible set 

### simulate koopman-hopf controller w error (levels) + MPCg
## timing & parameter optimization pt 3
## score 

## now go call from ros/rospy and see timing

## more complicated experiments (P-E, Dodge incoming projectile)? 
## active learning of KO in sim (ProximalAlgorithms.jl or Flux.jl)? this probably calls for comparison w/ Ian's learning
# https://github.com/FluxML/model-zoo/blob/da4156b4a9fb0d5907dcb6e21d0e78c72b6122e0/other/diffeq/neural_ode.jl