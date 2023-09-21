
using LinearAlgebra, Plots
plotlyjs()
include("/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, intH_box, preH_box, intH_ball, preH_ball, plot_BRS, Hopf

## Time
th = 0.05
Th = 0.25
Tf = 1.0
T = collect(Th : Th : Tf)

## Initialize (2D Example)
max_u, max_d = 1.0, 0.5

A = - [0. float(pi); -float(pi) 0.]
At = [if s < 11; A; else; -A; end; for s=1:Int(T[end]/th)]
Af(s) = - 2s * [0. float(pi); -float(pi) 0.]

B₁, B₂ = [1. 0; 0 1], [1. 0; 0 1];
B₁t, B₂t = [[1. 0; 0 1] for s=1:Int(T[end]/th)], [[1. 0; 0 1] for s=1:Int(T[end]/th)];
B₁f(s), B₂f(s) = [1. 0; 0 1], [1. 0; 0 1];

inputshape = "Ball"
Q₁ = [1. 0; 0 1]; Q₂ = [1. 0; 0 1];
c₁ = [0. 0.]; c₂ = [0. 0.];

system   = (A, max_u * B₁, max_d * B₂, Q₁, Q₂, c₁, c₂)
system_t = (At, max_u * B₁, max_d * B₂, Q₁, Q₂, c₁, c₂)
system_f = (Af, s -> max_u * B₁f(s), s -> max_d * B₂f(s), Q₁, Q₂, c₁, c₂)

## Target: J(x) = 0 is the boundary of the target
Qₓ = diagm([4; 1])
cₓ = zero(A[:,1])
r = 1.0
J(x::Vector, Qₓ, cₓ) = ((x - cₓ)' * inv(Qₓ) * (x - cₓ))/2 - 0.5 * r^2 #don't need yet
Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2
J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2
Jˢ(v::Matrix, Qₓ, cₓ) = diag(v' * Qₓ * v)/2 + (cₓ'v)' .+ 0.5 * r^2 #don't need yet
target = (J, Jˢ, (Qₓ, cₓ))

## Automatic Grid Parameters (can also define matrix of points Xg)
ϵ = 0.5e-7; res = 100; lbc, ubc = -3., 3.;
x1g = collect(cₓ[1] + lbc : (ubc-lbc)/(res-1) : cₓ[1] + ubc) .+ ϵ; lg1 = length(x1g); # == res, for comparing to DP
x2g = collect(cₓ[2] + lbc : (ubc-lbc)/(res-1) : cₓ[2] + ubc) .+ ϵ; lg2 = length(x2g);
Xg = hcat(collect.(Iterators.product(x1g, x2g))...);

## Hopf Coordinate-Descent Parameters (optional)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 5
max_runs = 3
max_its = 500
opt_p_cd = (vh, L, tol, lim, lll, max_runs, max_its)

# Hopf ADMM Parameters (optional)
ρ, ρ2 = 1e-1, 1e-1
tol = 1e-5
max_its = 3
opt_p_admm = (ρ, ρ2, tol, max_its)

# ADMM-CD Hybrid Parameters (optional)
ρ_grid_vals = 1 
hybrid_runs = 3
opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 3), (0.005, 5, 1e-4, 500, 1, 3, 500), ρ_grid_vals, ρ_grid_vals, hybrid_runs)

solution, run_stats = Hopf_BRS(system, target, T; th, Xg, inputshape, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
solution_t, run_stats = Hopf_BRS(system_t, target, T; th, Xg, inputshape, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
solution_f, run_stats = Hopf_BRS(system_f, target, T; th, Xg, inputshape, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

# plot_scatter = plot_BRS(T, solution...; A, ϵs=1e-1, interpolate=false, value_fn=false, alpha=0.1)
plot_contour = plot_BRS(T, solution...; ϵc=1e-3, interpolate=true, value_fn=false, alpha=0.5, title="A - Hopf")
plot_contour_t = plot_BRS(T, solution_t...; ϵc=1e-3, interpolate=true, value_fn=false, alpha=0.5, title="At[:] - Hopf")
plot_contour_f = plot_BRS(T, solution_f...; ϵc=1e-3, interpolate=true, value_fn=false, alpha=0.5, title="A(t) - Hopf")

### Get the "True" BRS from hj_reachability.py

## For hj_reachability
using Pkg
using PyCall

hj_r_loc = pwd() * "/Linearizations"
pushfirst!(pyimport("sys")."path", hj_r_loc);
np = pyimport("numpy")
jnp = pyimport("jax.numpy")
jax = pyimport("jax.experimental.host_callback")
jnp = pyimport("jax.numpy")
hj = pyimport("hj_reachability")

@pydef mutable struct Linear <: hj.ControlAndDisturbanceAffineDynamics
    function __init__(self, A, B, C; Ushape="ball", Dshape="ball", max_u=1.0, max_d=0.5, control_mode="min", disturbance_mode="max", control_space=nothing, disturbance_space=nothing)
        """
        Linear (Potentially Time-Varying) Dynamics Class
        A, B, C: linear parameters, can be constant Matrix{Float64}, or fn of t (Float64) -> Mt (Matrix{Float64})
        """
        self.n_x = typeof(A) <: Function ? size(A(0.0))[2] : size(A)[1]
        self.n_u = typeof(B) <: Function ? size(B(0.0))[2] : size(B)[2]
        self.n_d = typeof(C) <: Function ? size(C(0.0))[2] : size(C)[2]

        if isnothing(control_space)
            if Ushape == "box"
                control_space = hj.sets.Box(jnp.zeros(self.n_u), max_u)
            elseif Ushape == "ball"
                control_space = hj.sets.Ball(jnp.zeros(self.n_u), max_u)
            end
        end

        if isnothing(disturbance_space)
            if Dshape == "box"
                disturbance_space = hj.sets.Box(jnp.zeros(self.n_d), max_d)
            elseif Dshape == "ball"
                disturbance_space = hj.sets.Ball(jnp.zeros(self.n_d), max_d)
            end
        end

        pybuiltin(:super)(Linear, self).__init__(control_mode, disturbance_mode, control_space, disturbance_space) #(Linear, self)
        
        ## Store Linear Matrices, Perhaps fn's of t
        self.A = typeof(A) <: Function ? A : jnp.array(A)
        self.B = typeof(B) <: Function ? B : jnp.array(B)
        self.C = typeof(C) <: Function ? C : jnp.array(C)
    end

    function open_loop_dynamics(self, x, t)
        At = typeof(self.A) <: Function ? jnp.array(self.A(t)) : self.A
        return jnp.matmul(At, x)
    end

    function control_jacobian(self, x, t)
       Bt = typeof(self.B) <: Function ? jnp.array(self.B(t)) : self.B
        return Bt
    end

    function disturbance_jacobian(self, x, t)
        Ct = typeof(self.C) <: Function ? jnp.array(self.C(t)) : self.C
        return Ct
    end
end

DP_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([cₓ[1] + lbc, cₓ[2] + lbc]),
                                                                              np.array([cₓ[1] + ubc, cₓ[2] + ubc])),
                                                                             (lg1, lg2)) #lg has to be even

DP_values = (jnp.array(np.sum(np.multiply(inv.(diag(Qₓ)), np.square(np.subtract(DP_grid.states, np.array(cₓ)))), axis=-1)) - r^2) * 0.5
BRS = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor = x -> x)

dynamics = Linear(A, B₁, B₂; max_u, max_d)
dynamics_f = Linear(s -> Af(-s), B₁, B₂; max_u, max_d) # t going backwards in DP

### Solve BRS with DP

ϕXT_DP, ϕXT_DP_t, ϕXT_DP_f = [], [], []
push!(ϕXT_DP, Matrix(reshape(DP_values.tolist(), length(DP_values.tolist()), 1))[:,1])
push!(ϕXT_DP_t, Matrix(reshape(DP_values.tolist(), length(DP_values.tolist()), 1))[:,1])
push!(ϕXT_DP_f, Matrix(reshape(DP_values.tolist(), length(DP_values.tolist()), 1))[:,1])

## One-shot (Constant or Fn of t)
for ts in T
    hj_r_output = hj.step(BRS, dynamics, DP_grid, 0., DP_values, -ts)
    hj_r_output_f = hj.step(BRS, dynamics_f, DP_grid, 0., DP_values, -ts)
    push!(ϕXT_DP, Matrix(reshape(hj_r_output.tolist(), length(hj_r_output.tolist()), 1))[:,1])
    push!(ϕXT_DP_f, Matrix(reshape(hj_r_output_f.tolist(), length(hj_r_output.tolist()), 1))[:,1])
end

## Reinitializing (Array of t)
values = jnp.copy(DP_values)
for (tsi, ts) in enumerate(collect(th:th:T[end]))
    dynamics_t = Linear(At[tsi], B₁, B₂; max_u, max_d)
    hj_r_output_t = hj.step(BRS, dynamics_t, DP_grid, 0., values, -th)
    values = jnp.copy(hj_r_output_t)
    if ts ∈ T; push!(ϕXT_DP_t, Matrix(reshape(hj_r_output_t.tolist(), length(hj_r_output_t.tolist()), 1))[:,1]); end
end

Xgs = [Xg for i=1:length(T)+1] # for plotting
BRS_plots = plot_BRS(T, Xgs, ϕXT_DP; ϵs=5e-2, interpolate=true, value_fn=false, alpha=0.5, title="A - DP")
BRS_plot_t = plot_BRS(T, Xgs, ϕXT_DP_t; ϵs=5e-2, interpolate=true, value_fn=false, alpha=0.5, title="At[:] - DP")
BRS_plots_f = plot_BRS(T, Xgs, ϕXT_DP_f; ϵs=5e-2, interpolate=true, value_fn=false, alpha=0.5, title="A(t) - DP")

plot(plot_contour[1], plot_contour_t[1], plot_contour_f[1], 
    BRS_plots[1], BRS_plot_t[1], BRS_plots_f[1], layout=(2,3))