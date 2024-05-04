
using LinearAlgebra, Polynomials, DifferentialEquations, ForwardDiff
using ReachabilityAnalysis, Polyhedra
using PyCall
using Plots, LaTeXStrings

# plotlyjs()

## Solve backwards zonotopes in one-shot w/ ReachabilityAnalysis.jl
function solve_BRZ_fp(f, X0, U, D, t; zono_over="UD", alg=TMJets(; maxsteps=1_000, abstol=1e-10, orderT=6, orderQ=4))
    
    # BRZ of U & D, BRZ just U, BRZ just D, auto BRZ
    Us = occursin("U", zono_over) ? U : Hyperrectangle(; low=zero(low(U)), high=zero(high(U)))
    Ds = occursin("D", zono_over) ? D : Hyperrectangle(; low=zero(low(D)), high=zero(high(D)))
    X̂0 = X0 × Us × Ds;

    sys = InitialValueProblem(BlackBoxContinuousSystem(f, length(low(X̂0))), X̂0)
    sol = solve(sys, tspan=(t, 0.); alg=alg);

    return overapproximate(sol, Zonotope)
end

## Backwards integrate x0 to t (w/ u/d state fb fn) via DifferentialEquations.jl
function roll_back(f, x0, uxt, dxt, t; th=0.05, alg=Tsit5())
    
    nx, nt = length(x0), Int(round(t / th))
    dt, sols = zeros(nt), [] # storing interpolations

    x̃s = float(vcat(x0, uxt(x0, 0.0), dxt(x0, 0.0)))
    for ti=1:nt
        sol = solve(ODEProblem(f, x̃s, (-th*(ti-1), -th*ti)), alg, reltol = 1e-12, abstol = 1e-12)
        x̃s = float(vcat(sol[end][1:nx], uxt(sol[end][1:nx], th*ti), dxt(sol[end][1:nx], th*ti)))
        dt[ti] = th*ti; push!(sols, sol)
    end

    x̃(s) = sols[findfirst(x->x>=0, (s .+ dt))](s) #stitched interpolations
    return x̃
end

## Make Jacobian and residual fns of f via ForwardDiff.jl
function linearize(f, nx, nu, nd; solve_lg=true)

    g(x) = f(zero(x), x, 0.0, 0.0)
    A(x̃) = ForwardDiff.jacobian(g,x̃)[1:nx, 1:nx]
    B₁(x̃) = ForwardDiff.jacobian(g,x̃)[1:nx, nx+1:nx+nu]
    B₂(x̃) = ForwardDiff.jacobian(g,x̃)[1:nx, nx+nu+1:nx+nu+nd]
    # c(x̃) = g(x̃)[1:nx] - A(x̃) * x̃[1:nx] - B₁(x̃) * x̃[nx+1:nx+nu] - B₂(x̃) * x̃[nx+nu+1:nx+nu+nd]
    c(x̃) = (g(x̃) - ForwardDiff.jacobian(g,x̃) * x̃)[1:nx]
    
    linear_graph = falses(nx, nx+nu+nd)
    if solve_lg
        for i=1:nx, j=1:nx+nu+nd; linear_graph[i,j] = ForwardDiff.jacobian(g, Inf*((1:nx+nu+nd).==j))[i,:] == ForwardDiff.jacobian(g,zeros(nx+nu+nd))[i,:]; end
    end

    return (A, B₁, B₂, c), linear_graph
end

## Make Hessian fns of f via ForwardDiff.jl
function hessians(f; solve_dims=solve_dims, affine_inputs=false)
    Gs = []
    for i in solve_dims
        gi(x) = f(zero(x), x, 0.0, 0.0)[i]
        Gi(ξ) = ForwardDiff.hessian(gi, ξ)
        push!(Gs, Gi)
    end
    return Gs
end

## Compute maximum Taylor Series error for f to xl, over a cvx set
function TSerror(f, xl, shape; solve_dims=nothing, Gs=nothing, pnorm=Inf, linear_graph=nothing, lin_mat_fs=nothing, extra_args=nothing)

    nx = isnothing(linear_graph) ? length(xl) : size(linear_graph)[1]; δˢ = zeros(nx)
    linear_graph = isnothing(linear_graph) ? falses(nx, ReachabilityAnalysis.dim(shape)) : linear_graph
    solve_dims = isnothing(solve_dims) ? findall(!all, eachrow(linear_graph)) : solve_dims
    Gs = isnothing(Gs) ? hessians(f, solve_dims=solve_dims) : Gs

    for (i, xi) in enumerate(solve_dims)
        # v_dists = map(x -> norm(x, pnorm), eachcol(shape[.!linear_graph[xi,:], :] .- xl[.!linear_graph[xi,:]]))
        # max_v_dists2 = maximum(v_dists).^2
        # v_norms = map(x -> opnorm(Gs[i](x - xl), pnorm), eachcol(shape))
        # δˢ[xi] = 0.5 * maximum(v_norms) .* max_v_dists2

        max_v_hess, max_v_norm = 0., 0.
        for v in vertices_list(shape)
            v_hess = opnorm(Gs[i](v - xl), pnorm)
            v_norm = norm(v[.!linear_graph[xi,:],:] .- xl[.!linear_graph[xi,:]], pnorm)
            max_v_hess, max_v_norm = max(max_v_hess, v_hess), max(max_v_norm, v_norm)
        end

        δˢ[xi] = 0.5 * max_v_hess .* max_v_norm^2
    end

    return δˢ
end

## Compute maximum Taylor Series inf-error for L96 to xl, over cvx set
function TSerror_fast_pInf_L96(f, xl, shape; solve_dims=nothing, Gs=nothing, linear_graph=nothing, lin_mat_fs=nothing, pnorm=Inf, extra_args=nothing)

    N = isnothing(linear_graph) ? length(xl) : size(linear_graph)[1]; δˢ = zeros(N)
    linear_graph = isnothing(linear_graph) ? falses(N, ReachabilityAnalysis.dim(shape)) : linear_graph

    for (i, xi) in enumerate(1:N)
        ei = vcat(zeros(i-1), 1, zeros(size(linear_graph,2)-i))
        max_v_norm = norm([-ρ(-ei, shape), ρ(ei, shape)] .- xl[xi], Inf)
        δˢ[xi] = max_v_norm^2 # L96 opnorm(Hess)(⋅) = 2
    end

    return δˢ
end

function TSerror_Inf_MultiDubins(f, xl, shape; solve_dims=nothing, Gs=nothing, linear_graph=nothing, lin_mat_fs=nothing, pnorm=Inf, extra_args=nothing)
    N = dim_xh_d; δˢ = zeros(N)

    for xi in 0:dim_x:dim_x*(n_ag-1)
        max_norm_agent = 0
        for i = 1:dim_x, pm in [1, -1]; max_norm_agent = max(max_norm_agent, abs(pm * ρ(pm * eix(xi + i), shape) - xl[xi + i])); end
        for pm in [1, -1]; max_norm_agent = max(max_norm_agent, abs(pm * ρ(pm * eix(dim_xh_d), shape) - xl[dim_xh_d])); end

        θi_hi, θi_lo = ρ(eix(xi + 3), shape), -ρ(-eix(xi + 3), shape)
        if abs(θi_hi - θi_lo) ≥ π/2
            max_sinξθi, max_cosξθi = 1, 1
        elseif sign(cos(θi_hi)) != sign(cos(θi_lo)) || sign(sin(θi_hi)) != sign(sin(θi_lo))
            max_sinξθi = sign(cos(θi_hi)) == sign(cos(θi_lo)) ? max(abs(sin(θi_hi)), abs(sin(θi_lo))) : 1.
            max_cosξθi = sign(sin(θi_hi)) == sign(sin(θi_lo)) ? max(abs(cos(θi_hi)), abs(cos(θi_lo))) : 1.
        else
            max_cosξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
            max_sinξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
        end

        δˢ[xi + 1] = 0.5 * max(abs(v_b*max_cosξθi), 1) * max_norm_agent^2 # MultiDubins opnorm(Hess)(⋅) = 
        δˢ[xi + 2] = 0.5 * max(abs(v_b*max_sinξθi), 1) * max_norm_agent^2 # MultiDubins opnorm(Hess)(⋅) = 
    end
    return δˢ
end

## Make Jacobian and residual fns of f via ForwardDiff.jl
function lift_f(f, Ψ, nx, n)

    DₓΨ(x) = ForwardDiff.jacobian(Ψ,x)
    fg(gud) = DₓΨ(gud[1:nx]) * f(zeros(n), vcat(gud[1:nx], gud[end-n+nx+1:end]), 0., 0.)[1:nx]

    return fg
end

## Approx. maximum error between nonlinear and linear model via discretization of feas set
function Lifted_Error_DiscreteAppx(fg, xl, shape; solve_dims=nothing, Gs=nothing, linear_graph=nothing, lin_mat_fs=nothing, pnorm=Inf, extra_args=nothing, res=5, Ψ=nothing)

    Ng, Nu, Nd = size(lin_mat_fs[1], 2), size(lin_mat_fs[2], 2), size(lin_mat_fs[3], 2);
    Nx, δˢ = length(xl) - (Nu + Nd), zeros(Ng)
    κ(gud) = lin_mat_fs[1] * gud[1:Ng] + hcat(lin_mat_fs[2], lin_mat_fs[3]) * gud[Ng+1:end] #assumes lin mats correspond to model of interest

    ## Make Grid from Box around Feasible Sets (in XUD)
    xgis = []
    for i in 1:(Nx + Nu + Nd)
        ei = vcat(zeros(i-1), 1, zeros(Nx+Nu+Nd-i))
        bds_i = [-ρ(-ei, shape), ρ(ei, shape)]
        xgi = i ≤ Nx ? collect(bds_i[1] : (bds_i[2] - bds_i[1])/(res-2) : bds_i[2] + 1e-5) : bds_i # u or d
        push!(xgis, xgi)        
    end
    Sg = hcat(collect.(Iterators.product(xgis...))...)

    ## Lift Grid and Find Max Error
    GSg = vcat(hcat(map(Ψ, eachcol(Sg[1:Nx, :]))...), Sg[Nx+1:end, :])
    Δ = hcat(map(gud -> abs.(fg(gud) - κ(gud)), eachcol(GSg))...)

    for i in solve_dims
        δˢ[i] = maximum(Δ[i, :])
        # δˢ[i] = sum(Δ[i, :])/size(Δ,2) # check mean
    end

    return δˢ
end

# ## Compute maximum error between nonlinear and linear model via NL Opt over cvx feas set
# function Lifted_Error_NLP(f, xl, shape; solve_dims=nothing, Gs=nothing, linear_graph=nothing, lin_mat_fs=nothing, pnorm=Inf, extra_args=nothing)

#     N = isnothing(linear_graph) ? length(xl) : size(linear_graph)[1]; δˢ = zeros(N)
#     linear_graph = isnothing(linear_graph) ? falses(N, ReachabilityAnalysis.dim(shape)) : linear_graph

#     for (i, xi) in enumerate(1:N)
#         ei = vcat(zeros(i-1), 1, zeros(size(linear_graph,2)-i))
#         max_v_norm = norm([-ρ(-ei, shape), ρ(ei, shape)] .- xl[xi], Inf)
#         δˢ[xi] = max_v_norm^2 # L96 opnorm(Hess)(⋅) = 2
#     end

#     return δˢ
# end

## Computes err bound for each BRZ as a function of t
function compute_δˢ_BRZ(f, BRZ, xl; linear_graph=nothing, solve_dims=nothing, error_method=TSerror, Gs=nothing, 
                                    pnorm=Inf, polyfit=true, polydeg=4, lin_mat_fs=nothing, extra_args=nothing) # G=x->[x[2] x[1]; x[1] 0.]
    
    nx = isnothing(linear_graph) ? size(xl(0.0))[1] : size(linear_graph)[1]
    linear_graph = isnothing(linear_graph) ? falses(nx, length(set(BRZ[1]).center)) : linear_graph
    solve_dims = isnothing(solve_dims) ? findall(!all, eachrow(linear_graph)) : solve_dims
    δˢ = zeros(nx, length(BRZ)); dt = zeros(length(BRZ));

    for ti in eachindex(BRZ)
        dt[ti] = high(tspan(BRZ[1])) - low(tspan(BRZ[ti]))
        # set = hcat(vertices_list(set(BRZ[ti]))...) # implicitly defines \hat{𝒮}
        BRZi = set(BRZ[ti])
        xlud = vcat(xl(-dt[ti]), xl(-dt[ti])[1]^2, zeros(4))
        lin_mat_fs_ti = typeof(lin_mat_fs[1]) <: Function ? (lin_mat_fs[1](xlud), lin_mat_fs[2](xlud), lin_mat_fs[3](xlud), lin_mat_fs[4](xlud)) : lin_mat_fs
        δˢ[:,ti] = error_method(f, xl(-dt[ti]), BRZi; solve_dims=solve_dims, Gs=Gs, pnorm=pnorm, lin_mat_fs=lin_mat_fs_ti, linear_graph=linear_graph, extra_args=extra_args)
    end
    
    if polyfit
        δˢpf = [fit(dt[1:end-1], δˢ[i, 1:end-1], polydeg) for i=1:nx]
        δˢf = [s -> sum(ci * (-s)^(d-1) for (d, ci) in enumerate(coeffs(δˢpf[i]))) for i=1:nx]
        return δˢf, -dt[end:-1:1]
    else
        return δˢ[:, end:-1:1], -dt[end:-1:1]
    end
end

## Full A Priori Algorithm
function apri_δˢ(f, target, inputs, t; X̃0ŨD̃=nothing, zono_over="UD", pnorm=Inf, polyfit=true, lin_mat_fs=nothing,
                                        error_method=TSerror, Gs=nothing, solve_dims=nothing, linear_graph=nothing, Ψ=nothing)
    
    (Q𝒯, c𝒯), (Q₁, c₁), (Q₂, c₂) = target[3], inputs[1], inputs[2]
    nx, nu, nd = length(c𝒯), length(c₁), length(c₂)
    X̃0, Ũ, D̃ = isnothing(X̃0ŨD̃) ? ([c𝒯], [(y,s) -> zeros(nu)], [(y,s) -> zeros(nd)]) : X̃0ŨD̃
    δ̃ˢ, X̃, dt = [], [], nothing
    solve_lg = isnothing(linear_graph) ? true : false
    lin_mat_fs, linear_graph = isnothing(lin_mat_fs) ? linearize(f, nx, nu, nd; solve_lg=solve_lg) : (lin_mat_fs, linear_graph)

    ## Precompute Hessian for f Dims to Solve Error if TSerror
    solve_dims = isnothing(solve_dims) ? findall(!all, eachrow(linear_graph)) : solve_dims
    Gs = error_method == TSerror ? (isnothing(Gs) ? hessians(f; solve_dims=solve_dims) : Gs) : Gs

    ## Over-approximate set of all trajctories
    X0 = Hyperrectangle(; low = c𝒯 - diag(inv(Q𝒯)), high = c𝒯 + diag(inv(Q𝒯)))
    U = isfinite(maximum(diag(Q₁))) ? Hyperrectangle(; low = c₁ - diag(inv(Q₁)), high = c₁ + diag(inv(Q₁))) : Hyperrectangle(; low = c₁, high = c₁)
    D = isfinite(maximum(diag(Q₂))) ? Hyperrectangle(; low = c₂ - diag(inv(Q₂)), high = c₂ + diag(inv(Q₂))) : Hyperrectangle(; low = c₂, high = c₂)
    BRZ = solve_BRZ_fp(f, X0, U, D, t; zono_over=zono_over) # zono_over = "UD", "U", "D"

    ## If Ψ defined, then lifting
    fg = isnothing(Ψ) ? f : lift_f(f, Ψ, nx, nx+nu+nd)
    error_method_wrapped(f, xl, shape; kwargs...) = isnothing(Ψ) ? error_method(f, xl, shape; kwargs...) : error_method(fg, xl, shape; Ψ=Ψ, kwargs...)

    ## Iterate Through Linearization Trajectories
    for ĩ in axes(X̃0,1)

        ## Backwards solve trajectory for linearization
        # x̃ = isnothing(Ψ) ? roll_back(f, X̃0[ĩ], Ũ[ĩ], D̃[ĩ], t) : s -> c𝒯
        x̃ = typeof(lin_mat_fs[1]) <: Function ? roll_back(f, X̃0[ĩ], Ũ[ĩ], D̃[ĩ], t) : s -> vcat(X̃0[ĩ], zeros(nu), zeros(nd))

        ## Solve Error for Linearization
        δˢ, dt = compute_δˢ_BRZ(f, BRZ, x̃; error_method=error_method_wrapped, solve_dims=solve_dims, Gs=Gs, pnorm=pnorm, polyfit=polyfit, lin_mat_fs=lin_mat_fs, linear_graph=linear_graph);
        push!(δ̃ˢ, δˢ); push!(X̃, x̃);
    end

    δ̃ˢ, X̃ = length(X̃0) == 1 ? δ̃ˢ[1] : δ̃ˢ, length(X̃0) == 1 ? X̃[1] : X̃
    return δ̃ˢ, X̃, BRZ[end:-1:1], dt, (lin_mat_fs, Gs)
end

### hj_reachability.py Utilities

np = pyimport("numpy")
jnp = pyimport("jax.numpy")
hj = pyimport("hj_reachability")

function hjr_init(c𝒯, Q𝒯, r; shape="box", lb=nothing, ub=nothing, res=100, ϵ = 0.5e-7, stretch=3)
    # given lb, ub, res, c𝒯, Q, T_shape

    dim = length(c𝒯)
    lb = isnothing(lb) ? stretch .* [c𝒯[i] - Q𝒯[i,i] for i=1:dim] : lb
    ub = isnothing(ub) ? stretch .* [c𝒯[i] + Q𝒯[i,i] for i=1:dim] : ub
    res = typeof(res) <: Number ? fill(res, dim) : res

    # x1g = collect(lb[1] : (ub[1]-lb[1])/(res-1) : ub[1] + ϵ);
    # x2g = collect(lb[2] : (ub[2]-lb[2])/(res-1) : ub[2] + ϵ); # lb=(-2, -2), ub=(1, 2)
    xgs = [collect(lb[i] : (ub[i]-lb[i])/(res[i]-1) : ub[i] + ϵ) for i=1:dim]
    Xg = hcat(collect.(Iterators.product(xgs...))...);

    Xg_DP = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array(lb),
                                                                              np.array(ub)),
                                                                             res) #lg has to be even
    
    if shape == "box"                                                                       
        ϕ0Xg_DP = (jnp.array(np.max(np.abs(np.multiply(diag(Q𝒯), np.subtract(Xg_DP.states, np.array(c𝒯)))), axis=-1)) - r)
    else
        ϕ0Xg_DP = (jnp.array(np.sum(np.multiply(diag(Q𝒯), np.square(np.subtract(Xg_DP.states, np.array(c𝒯)))), axis=-1)) - r^2) * 0.5
    end

    return Xg, Xg_DP, ϕ0Xg_DP, xgs
end

function hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics, T; BRS=true, one_shot=true, num_sets_tube=4, th=0.01, tv=false)
    # given Xg_DP, target_values ϕ0Xg_DP, dynamics, times T, soln=tube/set, num_sets
    ϕXgT_DP_dynamics = []

    ss_tube = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    backwards_reachable_set(x) = x;
    ss_set = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=backwards_reachable_set)
    ss = BRS ? ss_set : ss_tube

    if one_shot
        for dyni in dynamics
            ϕXgT_DP_dyni = []
            push!(ϕXgT_DP_dyni, Matrix(reshape(ϕ0Xg_DP.tolist(), length(ϕ0Xg_DP.tolist()), 1))[:,1]) # target

            for ts in T
                hj_r_output = hj.step(ss, dyni, Xg_DP, 0., ϕ0Xg_DP, -ts)
                push!(ϕXgT_DP_dyni, Matrix(reshape(hj_r_output.tolist(), length(hj_r_output.tolist()), 1))[:,1])
            end
            push!(ϕXgT_DP_dynamics, ϕXgT_DP_dyni)
        end
    else #iterative
        for dyni in dynamics
            ϕXgT_DP_dyni = []
            push!(ϕXgT_DP_dyni, Matrix(reshape(ϕ0Xg_DP.tolist(), length(ϕ0Xg_DP.tolist()), 1))[:,1]) # target

            values = jnp.copy(ϕ0Xg_DP)
            for (tsi, ts) in enumerate(collect(th:th:T[end]))
                dynis = tv ? dyni[tsi] : dyni
                hj_r_output = hj.step(ss, dynis, Xg_DP, 0., values, -th)
                values = hj_r_output
                if ts ∈ T; 
                    push!(ϕXgT_DP_dyni, Matrix(reshape(hj_r_output.tolist(), length(hj_r_output.tolist()), 1))[:,1])
                end
            end
            push!(ϕXgT_DP_dynamics, ϕXgT_DP_dyni)
        end
    end
    
    return ϕXgT_DP_dynamics
end

@pydef mutable struct LinearError <: hj.ControlAndDisturbanceAffineDynamics
    function __init__(self, A, B1, B2, c, E; Ushape="box", max_u=1.0, max_d=0.5, max_e=1.0, game="reach", control_space=nothing, disturbance_space=nothing)
        """
        Linear (Potentially Time-Varying) Dynamics Class with Error
        A, B1, B2, c, E: linear parameters, s.t. dxdt = A(t)x + B1(t)u + B2(t)d + c(t) + E(t)e,
        - can be constant: Matrix{Float}, or fn of t: Float -> Matrix{Float}
        To fit the existing code, D & E must be inf-norm bounded (box).
        """
        self.n_x = typeof(A) <: Function ? size(A(0.0))[2] : size(A)[1]
        self.n_u = typeof(B1) <: Function ? size(B1(0.0))[2] : size(B1)[2]
        self.n_d = typeof(B2) <: Function ? size(B2(0.0))[2] : size(B2)[2]
        self.n_e = typeof(E) <: Function ? size(E(0.0))[2] : size(E)[2]
        control_mode, disturbance_mode = game == "reach" ? ("min", "max") : ("max", "min")

        if isnothing(control_space)
            if Ushape == "box"
                ub = max_u*ones(self.n_u);
                control_space = hj.sets.Box(lo=-ub, hi=ub)
            elseif Ushape == "ball"
                control_space = hj.sets.Ball(jnp.zeros(self.n_u), max_u)
            end
        end

        if isnothing(disturbance_space)
            deb = vcat(max_d*ones(self.n_d), max_e*ones(self.n_e))
            disturbance_space = hj.sets.Box(lo=-deb, hi=deb)
        end

        pybuiltin(:super)(LinearError, self).__init__(control_mode, disturbance_mode, control_space, disturbance_space) #(Linear, self)
        
        ## Store Linear Matrices, Perhaps fn's of t
        self.A = typeof(A) <: Function ? A : jnp.array(A)
        self.B1 = typeof(B1) <: Function ? B1 : jnp.array(B1)
        self.B2 = typeof(B2) <: Function ? B2 : jnp.array(B2)
        self.c = typeof(c) <: Function ? c : jnp.array(c)
        self.E = typeof(E) <: Function ? E : jnp.array(E)
    end

    function open_loop_dynamics(self, x, t)
        At = typeof(self.A) <: Function ? jnp.array(self.A(t)) : self.A
        ct = typeof(self.c) <: Function ? jnp.array(self.c(t)) : self.c
        return jnp.matmul(At, x) + ct
    end

    function control_jacobian(self, x, t)
        B1t = typeof(self.B1) <: Function ? jnp.array(self.B1(t)) : self.B1
        return B1t
    end

    function disturbance_jacobian(self, x, t)
        B2t = typeof(self.B2) <: Function ? jnp.array(self.B2(t)) : self.B2
        Et = typeof(self.E) <: Function ? jnp.array(self.E(t)) : self.E
        return jnp.concatenate((B2t, Et), 1)
    end
end

# ## VanderPol Example for Testing

# r = 0.25; c𝒯 = [0.; 1.]; # c𝒯= [0.; 0.]
# max_u = 1.0; max_d = 0.5;
# t = 0.4

# c = [0.]
# Q₁ = inv(max_u) * diagm([1.])
# Q₂ = inv(max_d) * diagm([1.])
# Q𝒯 = inv(r) * diagm([1., 1.])
# nx = length(c𝒯);

# inputs = ((Q₁, c), (Q₂, c))
# 𝒯target = (nothing, nothing, (Q𝒯, c𝒯))

# (Q𝒯, c𝒯), (Q₁, c₁), (Q₂, c₂) = 𝒯target[3], inputs[1], inputs[2]

# X0 = Hyperrectangle(; low = c𝒯 - diag(inv(Q𝒯)), high = c𝒯 + diag(inv(Q𝒯)))
# U = Hyperrectangle(; low = c₁ - diag(inv(Q₁)), high = c₁ + diag(inv(Q₁)))
# D = Hyperrectangle(; low = c₂ - diag(inv(Q₂)), high = c₂ + diag(inv(Q₂)))

# μ = 1.0
# function vanderpol!(dx, x, p, t)
#     dx[1] = x[2]
#     dx[2] = μ * (1.0 - x[1]^2) * x[2] - x[1] + x[3] + x[4]
#     dx[3] = zero(x[3]) #control
#     dx[4] = zero(x[4]) #disturbance
#     return dx
# end

# ## Solve (defualt x̃ is auto from target center)

# t = 0.39

# δ̃ˢ, X̃, BRZ, dt, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t);
# δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U");
# δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D");

# BRZ_plot = plot(BRZ, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (𝒰 & 𝒟)", legend=:bottomleft)
# plot!(BRZ_plot, BRZu, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (𝒰)")
# plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.6, lw=0.2, label="BRZ (𝒟)")
# scatter!(BRZ_plot, eachrow(hcat(X̃.(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃ backsolved w/ trivial ctrl/dist", alpha=0.6)

# error_plot = plot(dt, δ̃ˢ[2].(dt), label="Taylor δˢ for BRZ (𝒰 & 𝒟), x̃", xlabel="t")
# plot!(error_plot, dtU, δ̃ˢU[2].(dtU), label="Taylor δˢ for BRZ (𝒰), x̃")
# plot!(error_plot, dtD, δ̃ˢD[2].(dtD), label="Taylor δˢ for BRZ (𝒟), x̃")

# plot(BRZ_plot, error_plot)

# ## Solve with DP

# Th = 0.13
# T = collect(Th:Th:t)

# pushfirst!(pyimport("sys")."path", pwd() * "/Zonotoping");
# hj_r_setup = pyimport("VdP_hj_reachability");

# VdP_reach = hj_r_setup.VanderPol(mu=1.0, max_u=max_u, max_d=max_d)
# VdP_avoid = hj_r_setup.VanderPol(mu=μ, max_u=max_u, max_d=max_d, control_mode="max", disturbance_mode="min")

# A, B₁, B₂, c = lin_mat_fs
# B1 = Matrix([0. 1.]'); B2 = Matrix([0. 1.]')
# # E_c(s) = sum(ci * (-s)^(d-1) for (d, ci) in enumerate(coeffs(δˢ_TS_Ucarrf))) * Matrix([0. 1.]') # put it in forward time
# E_c(s) = δ̃ˢ[2](-s) * Matrix([0. 1.]') # in forward time

# dynamics_linear_reach = LinearError(A(X̃(0)), B1, B2, c(X̃(0)), E_c; max_u=max_u, max_d=max_d, Ushape="box", game="reach") # jax cant handle julia fns, so must do iterative solve if tv lin
# dynamics_linear_avoid = LinearError(A(X̃(0)), B1, B2, c(X̃(0)), E_c; max_u=max_u, max_d=max_d, Ushape="box", game="avoid")

# dynamics_reach = [VdP_reach, dynamics_linear_reach];
# dynamics_avoid = [VdP_avoid, dynamics_linear_avoid]; 
# res=300

# Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(ones(nx)), r; shape="ball", lb=(-2, -2), ub=(1, 2), res=res)
# ϕXgT_DP_dynamics_reach = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_reach, T; BRS=true, one_shot=true);
# ϕXgT_DP_dynamics_avoid = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_avoid, T; BRS=true, one_shot=true);

# ## Solve with Hopf

# include(pwd() * "/HopfReachability.jl");
# using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_cd, plot_BRS

# th=0.0325

# A, B₁, B₂, c = lin_mat_fs
# B1 = Matrix([0. 1.]'); B2 = Matrix([0. 1.]')
# Eδc(s) = δ̃ˢ[2](-t) * diagm([0, 1]) # constant error
# Eδt(s) = δ̃ˢ[2](-s) * diagm([0, 1]) # tv error
# EδU(s) = δ̃ˢU[2](-s) * diagm([0, 1]) # tv error, ctrl feas
# EδD(s) = δ̃ˢD[2](-s) * diagm([0, 1]) # tv error, dist feas

# system_errc = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), Eδc);
# system_errt = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), Eδt);
# system_errU = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), EδU);
# system_errD = (s -> A(X̃(-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃(-s)), EδD);

# lb = 1.1 .* (-ρ(-[1,0,0,0], BRZ), -ρ(-[0,1,0,0], BRZ))
# ub = (ρ([1,0,0,0], BRZ), ρ([0,1,0,0], BRZ))
# res2 = 100
# Xg, _, _, xig2 = hjr_init(c𝒯, Q𝒯, 1; shape="box", lb=lb, ub=ub, res=res2);

# J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2;
# Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2;
# target = (J, Jˢ, (diagm(ones(nx)), c𝒯));

# # solution, run_stats = Hopf_BRS(system, target, T; th, Xg=Xg[:,1:end], inputshape="box", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# (ϕXgT, ϕXgT_Hopf_errc_reach), _ = Hopf_BRS(system_errc, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# (_, ϕXgT_Hopf_errc_avoid),    _ = Hopf_BRS(system_errc, target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# (_, ϕXgT_Hopf_errt_reach),    _ = Hopf_BRS(system_errt, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# (_, ϕXgT_Hopf_errt_avoid),    _ = Hopf_BRS(system_errt, target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=false, check_all=true, printing=true);
# (_, ϕXgT_Hopf_errU_reach),    _ = Hopf_BRS(system_errU, target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# (_, ϕXgT_Hopf_errD_avoid),    _ = Hopf_BRS(system_errD, target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=false, check_all=true, printing=true);
# # println("Min Val ϕ(Xg[1:3], t): $(minimum(ϕXgT_Hopf[2]))")
# # plotlyjs(); plot_BRS(T, ϕXgT, ϕXgT_Hopf_errD_avoid; interpolate=true, value_fn=true)

# ## Plot Single BRZ vs. BRS (at t), Constant Error

# gr()
# pal = palette(:seaborn_colorblind)
# colors = [:black, pal[1], pal[2], pal[3], "gray"]
# alpha = 0.7; lw=2.5; legend_hfactor=0.7; dpi=300;
# tix = 2; ti = T[tix];
# dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
# reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errc_reach) # true v lin + const err
# avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errc_avoid)

# single_plots = [];
# for game in ["reach", "avoid"]

#     ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
#     labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-") : fill("", 5)
#     title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"

#     single_plot = plot(title=title, dpi=dpi);
#     contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(single_plot), ylims=ylims(single_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#     plot!(eachrow(hcat(X̃.(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(single_plot), ylims=ylims(single_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[5], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     plot!(BRZ[dtix], vars=(1,2), alpha=alpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot);

#     contour!(xig1..., reshape(ϕ_DP[tix+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(single_plot), ylims=ylims(single_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#     contour!(xig2..., reshape(ϕ_Hopf[tix+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(single_plot), ylims=ylims(single_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#     plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.25:.5:1.25, (L".25", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     plot!(xlims=(-0.8, .3), ylims=(0.05, 1.5))
#     lo, _ = collect(zip(xlims(single_plot), ylims(single_plot)))
#     locxl = lo .+ ((xlims(single_plot)[2] - xlims(single_plot)[1])/2, -0.1)
#     locyl = lo .+ (-0.1, (ylims(single_plot)[2] - ylims(single_plot)[1])/2)
#     annotate!(locxl..., L"x_1", fontsize=16)
#     annotate!(locyl..., L"x_2", fontsize=16)

#     push!(single_plots, single_plot)
# end

# single_plots_final = plot(single_plots..., layout=(1,2), legend=(-1.15, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)


# ## Plot Multiple BRZ vs. BRS

# gr()
# colors = [:black, pal[1], pal[2], pal[3], pal[7]]
# alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;
# reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errc_reach, ϕXgT_Hopf_errt_reach) # true v lin + tv err
# avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errc_avoid, ϕXgT_Hopf_errt_avoid)

# multi_plots = [];
# for game in ["reach", "avoid"]

#     ϕ_DP, ϕ_Hopf, ϕ_Hopf2 = game == "reach" ? reach_comp : avoid_comp
#     labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-") : fill("", 6)
#     title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"

#     multi_plot = plot(title=title, dpi=dpi);
#     contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     # plot!(eachrow(hcat(X̃.(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(multi_plot), ylims=ylims(multi_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[5], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
#     for (i, ti) in enumerate(T)
#         dtix = findfirst(x -> x .+ ti > 0, dt) - 1;

#         labels = i == 1 ? labels : fill("", length(labels))
#         plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);
#         # plot!(BRZ[dtix], vars=(1,2), alpha=alpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, fillalpha = 0.65);

#         contour!(xig1..., reshape(ϕ_DP[i+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#         # contour!(xig2..., reshape(ϕ_Hopf[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#         # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
#         contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
#     # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(-.25:.5:1.25, (L"-.25", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # plot!(xlims=(-0.8, .3), ylims=(-.25, 1.45))
#     lo, _ = collect(zip(xlims(multi_plot), ylims(multi_plot)))
#     locxl = lo .+ ((xlims(multi_plot)[2] - xlims(multi_plot)[1])/2, -0.1)
#     locyl = lo .+ (-0.1, (ylims(multi_plot)[2] - ylims(multi_plot)[1])/2)
#     annotate!(locxl..., L"x_1", fontsize=16)
#     annotate!(locyl..., L"x_2", fontsize=16)

#     push!(multi_plots, multi_plot)
# end

# multi_plots_final = plot(multi_plots..., layout=(1,2), legend=(-0.925, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)


# ## Plot Multiple BRZ - U, D

# gr()
# colors = [:black, pal[1], pal[2], pal[3], pal[5], pal[10]]
# alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.9; dpi=300;
# reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_errt_reach, ϕXgT_Hopf_errU_reach) # true v lin + tv err v 𝒮_𝒰
# avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_errt_avoid, ϕXgT_Hopf_errD_avoid)
# labelss = [L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-",  L"\hat{𝒮}_𝒰 / \hat{𝒮}_𝒟", L"ℛ_{δ^*_{𝒰(τ)}} / ℛ_{δ^*_{𝒟(τ)}}^-"]

# multi_plotsUD = [];
# for game in ["reach", "avoid"]

#     ϕ_DP, ϕ_Hopf, ϕ_Hopf2 = game == "reach" ? reach_comp : avoid_comp
#     labels = game == "avoid" ? labelss : fill("", 7)
#     title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"
#     BRZ2 = game == "reach" ? BRZu : BRZd


#     multi_plot = plot(title=title, dpi=dpi);
#     contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     # plot!(eachrow(hcat(X̃.(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(multi_plot), ylims=ylims(multi_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[5], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
#     for (i, ti) in enumerate(T)
#         dtix = findfirst(x -> x .+ ti > 0, dt) - 1;

#         labels = i == 1 ? labels : fill("", length(labels))
#         plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);
#         plot!(BRZ2[dtix], vars=(1,2), alpha=0., lw=lw, label=labels[6], color=:white, linecolor=colors[6], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);

#         contour!(xig1..., reshape(ϕ_DP[i+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#         contour!(xig2..., reshape(ϕ_Hopf[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
#         contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[5], lw=lw, alpha=alpha, colorbar=false);
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[7], linecolor=colors[5], alpha=0., lw=lw, xlims=xlims(multi_plot), ylims=ylims(multi_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=14,legend_columns=3)
#     plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))

#     lo, _ = collect(zip(xlims(multi_plot), ylims(multi_plot)))
#     locxl = lo .+ ((xlims(multi_plot)[2] - xlims(multi_plot)[1])/2, -0.1)
#     locyl = lo .+ (-0.1, (ylims(multi_plot)[2] - ylims(multi_plot)[1])/2)

#     annotate!(locxl..., L"x_1", fontsize=16)
#     annotate!(locyl..., L"x_2", fontsize=16)

#     push!(multi_plotsUD, multi_plot)
# end

# multi_plotsUD_final = plot(multi_plotsUD..., layout=(1,2), legend=(-0.7, -.155), bottom_margin=60Plots.px, background_color_legend = nothing, foreground_color_legend = nothing, dpi=dpi,legendfontsize=12)


# ### New Problem

# r = 0.5; c𝒯 = [-1.; 1.]; # c𝒯= [0.; 0.]
# Q𝒯 = inv(r) * diagm([1., 1.])
# max_u = 1.0; max_d = 0.5;
# J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2;
# Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2;
# 𝒯target = (J, Jˢ, (Q𝒯, c𝒯))
# t = 0.4

# Th = 0.13
# T = collect(Th:Th:t)

# ### Target Partitioning

# ntp = 5; θtp = 2π/ntp

# # minimal circle-polygon cover
# # ϵ=0.005; ra = ϵ:ϵ:r; l = ϵ:ϵ:2r
# # d² = 2 * l^2 * (1 - cos(θtp)) # ntp-polygon length^2
# # h = sqrt(l^2 - d²/2)
# # a = sqrt(d² * (4 * ra^2 - d²)) / d # dis of nhbr-circ ∩ pts
# # f(ra, l) = 0.5*(sqrt(2 * l^2 * (1 - cos(θtp)) * (4 * ra^2 - 2 * l^2 * (1 - cos(θtp)))) / sqrt(2 * l^2 * (1 - cos(θtp)))) + sqrt(l^2 - l^2 * (1 - cos(θtp))) - r

# # Gral = hcat(collect.(Iterators.product(ra, l))...)
# # fval = ones(size(Gral,2))
# # for ii=1:size(Gral,2); try; fval[ii] = f(Gral[:,ii]...); catch y; fval[ii] = 0.2; end; end
# # plotlyjs(); scatter(ra, l, reshape(fval, length(ra), length(l))')

# # l = 0.154 # regular polygon radius, defining partition centers
# # ra = 0.154 # partition radii (optimalish for avoid, using same for reach)

# l = r / 1.6 # regular polygon radius, defining partition centers
# ra = r / 1.6 # partition radii (optimalish for avoid, using same for reach)

# R(θ) = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1];
# To(x, y) = [1 0 x; 0 1 y; 0 0 1];
# Rot(pr, p, i, θ) = (To(pr...) * R(θ)^(i-1) * To(-pr...) * vcat(p,1))[1:2]
# c𝒯rri(i) = Rot(c𝒯, c𝒯 + [0, (r-ra)], i , θtp) # reach centers
# c𝒯rai(i) = Rot(c𝒯, c𝒯 + [0, l], i , θtp) # avoid centers

# θi = 0.:0.01:2π
# scatter(eachrow(c𝒯)..., color=:green)
# scatter()
# plot!([c𝒯[1] .+ r * cos.(θi)], [c𝒯[2] .+ r * sin.(θi)], lw=2, color=:green)
# for i=1:ntp; 
#     scatter!(eachrow(c𝒯rri(i))..., color=:blue)
#     plot!([c𝒯rri(i)[1] .+ ra * cos.(θi)], [c𝒯rri(i)[2] .+ ra * sin.(θi)], lw=2, color=:blue)
#     scatter!(eachrow(c𝒯rai(i))..., color=:red)
#     plot!([c𝒯rai(i)[1] .+ ra * cos.(θi)], [c𝒯rai(i)[2] .+ ra * sin.(θi)], lw=2, color=:red)
# end
# plot!(legend=false)

# δ̃ˢU_TPr, X̃_TPr, BRZu_TPr, dtU_TPr = [], [], [], []
# δ̃ˢD_TPa, X̃_TPa, BRZd_TPa, dtD_TPa = [], [], [], []

# for i=1:ntp
#     𝒯targetri = (nothing, nothing, (inv(ra) * diagm([1., 1.]), c𝒯rri(i)))    
#     δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetri, inputs, t; zono_over="U");
#     push!(δ̃ˢU_TPr, δ̃ˢU); push!(X̃_TPr, X̃); push!(BRZu_TPr, BRZu); push!(dtU_TPr, dtU)

#     𝒯targetai = (nothing, nothing, (inv(ra) * diagm([1., 1.]), c𝒯rai(i)))
#     δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯targetai, inputs, t; zono_over="D");
#     push!(δ̃ˢD_TPa, δ̃ˢD); push!(X̃_TPa, X̃); push!(BRZd_TPa, BRZd); push!(dtD_TPa, dtD)
# end

# δ̃ˢU, X̃, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U");
# δ̃ˢD, X̃, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D");

# BRZ_plot_r = plot(BRZu, vars=(1,2), alpha=0.1, lw=3, label="BRZ (𝒰, 𝒯)", legend=:bottomleft); 
# BRZ_plot_a = plot(BRZd, vars=(1,2), alpha=0.1, lw=3, label="BRZ (𝒟, 𝒯)", legend=:bottomleft);
# for i=1:ntp
#     plot!(BRZ_plot_r, BRZu_TPr[i], vars=(1,2), alpha=0.5, lw=1, label="BRZ (𝒰, 𝒯p$i)")
#     plot!(BRZ_plot_a, BRZd_TPa[i], vars=(1,2), alpha=0.5, lw=1, label="BRZ (𝒟, 𝒯p$i)")
# end
# plot(BRZ_plot_r, BRZ_plot_a)

# ### Solve w/ Various x̃ for one target (Linear Ensemble)

# nle = 5; nu = 1; nd = 1;
# # X̃0ŨD̃ = ([c𝒯, c𝒯 + [r,r], c𝒯 + [-r,r], c𝒯 + [r,-r], c𝒯 - [r,r]], # box
# # X̃0ŨD̃ = ([c𝒯, c𝒯 + [r,0], c𝒯 - [r,0], c𝒯 + [0,r], c𝒯 - [0,r]], # circle
# # X̃0 = [c𝒯, c𝒯 + [r,r]/sqrt(2), c𝒯 + [-r,r]/sqrt(2), c𝒯 + [r,-r]/sqrt(2), c𝒯 - [r,r]/sqrt(2)]
# X̃0 = [c𝒯rri(i) for i =1:nle]
# X̃0ŨD̃_LE = (X̃0, # circle ×
#         [(y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu)], 
#         [(y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd)])
# δ̃ˢU_LE, X̃_LE, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U", X̃0ŨD̃=X̃0ŨD̃_LE); # trajectories are the same, hence X̃_LE same for both
# δ̃ˢD_LE, X̃_LE, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D", X̃0ŨD̃=X̃0ŨD̃_LE);
# nle = length(X̃0ŨD̃_LE[1])

# BRZ_plot = plot(BRZu, vars=(1,2), alpha=0.3, lw=3, label="BRZ (𝒰)", legend=:bottomleft)
# plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.3, lw=3, label="BRZ (𝒟)")
# for i=1:nle
#     scatter!(BRZ_plot, eachrow(hcat(X̃_LE[i].(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃$i", alpha=0.3)
# end
# plot!()

# plot(); for i=1:nle; plot!(dtU, δ̃ˢU_LE[i][2].(dtU)); end; plot!()

# ## Solve with DP (only needed if true problem changed)

# res = 300
# Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(ones(nx)), r; shape="ball", lb=(-2, -2), ub=(1, 2), res=res)
# ϕXgT_DP_dynamics_reach = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_reach, T; BRS=true, one_shot=true);
# ϕXgT_DP_dynamics_avoid = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_avoid, T; BRS=true, one_shot=true);

# ## Solve with Hopf

# EδU_LE(i, s) = δ̃ˢU_LE[i][2](-s) * diagm([0, 1])
# EδD_LE(i, s) = δ̃ˢD_LE[i][2](-s) * diagm([0, 1])
# EδU_TP(i, s) = δ̃ˢU_TPr[i][2](-s) * diagm([0, 1])
# EδD_TP(i, s) = δ̃ˢD_TPa[i][2](-s) * diagm([0, 1])

# system_errU_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδU_LE(i,s));
# system_errD_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδD_LE(i,s));
# system_errU_TP(i) = (s -> A(X̃_TPr[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPr[i](-s)), s -> EδU_TP(i,s));
# system_errD_TP(i) = (s -> A(X̃_TPa[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPa[i](-s)), s -> EδD_TP(i,s));

# rr, ra = ra, ra
# Jr(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * rr^2;
# Jˢr(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * rr^2;
# Ja(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * ra^2;
# Jˢa(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * ra^2;

# targetri(i) = (Jr, Jˢr, (diagm(ones(nx)), c𝒯rri(i)));
# targetai(i) = (Ja, Jˢa, (diagm(ones(nx)), c𝒯rai(i)));

# # lb = (1.1 * -ρ(-[1,0,0,0], BRZu), -ρ(-[0,1,0,0], BRZu))
# # ub = (1.5 * ρ([1,0,0,0], BRZu), ρ([0,1,0,0], BRZu))
# lb = (-2.5, -1); ub = (-0.25, 3)

# # res3 = res2
# # res4 = res2
# # xig3 = xig2
# # xig4 = xig2

# res3 = 30
# Xg, _, _, xig3 = hjr_init(c𝒯, Q𝒯, r; shape="box", lb=lb, ub=ub, res=res3);

# ϕXgT_Hopf_LE_reach = []; ϕXgT_Hopf_LE_avoid = [];
# for i=1:nle
#     (ϕXgT, ϕXgT_Hopf_errU_reachi), _ = Hopf_BRS(system_errU_LE(i), 𝒯target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=false,  check_all=true, printing=true);
#     (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_LE(i), 𝒯target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=true, check_all=true,  printing=true);
#     push!(ϕXgT_Hopf_LE_reach, ϕXgT_Hopf_errU_reachi); push!(ϕXgT_Hopf_LE_avoid, ϕXgT_Hopf_errD_avoidi);
# end

# ϕXgT_Hopf_TP_reach = []; ϕXgT_Hopf_TP_avoid = [];
# for i=1:ntp
#     (ϕXgT, ϕXgT_Hopf_errU_reachi), _ = Hopf_BRS(system_errU_TP(i), targetri(i), T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true,  check_all=true, printing=true);
#     (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_TP(i), targetai(i), T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=true, check_all=true,  printing=true);
#     push!(ϕXgT_Hopf_TP_reach, ϕXgT_Hopf_errU_reachi); push!(ϕXgT_Hopf_TP_avoid, ϕXgT_Hopf_errD_avoidi);
# end

# # plotlyjs()
# # ϕXgT, _, _ = Hopf_BRS(system_errU_LE(1), 𝒯target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true,  check_all=true, printing=true);
# # plot_BRS(T, fill(Xg, length(T)+1), ϕXgT_Hopf_LE_reach[1]; interpolate=false, value_fn=true)

# ## Plot Linear Ensemble Results

# gr()
# LE_pal = palette(:oslo10)[2:7] #palette(:magma)[25:50:end], palette(:oslo10)[2:7], palette(:navia10)[2:7], palette(:lipari10)[1:5]
# colors = [:black, pal[1], pal[2], LE_pal, pal[7]]
# alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;

# tix = 1; ti = T[tix];
# dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
# reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_LE_reach) # true v lin + const err
# avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_LE_avoid)

# LE_plots = []
# for game in ["reach", "avoid"]

#     ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
#     labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-") : fill("", 6)
#     title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"

#     LE_plot = plot(title=title, dpi=dpi);
#     contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
   
#     contour!(xig1..., reshape(ϕ_DP[tix+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#     for i=1:nle
#         plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(LE_plot), ylims=ylims(LE_plot), label=labels[2], alpha=0.3, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     for i=1:nle

#         # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(LE_plot), ylims=ylims(LE_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#         # labels = i == 1 ? labels : fill("", length(labels))

#         contour!(xig3..., reshape(ϕ_Hopf[i][tix+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
#         # contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#         # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     plot!(xlims = (-2.5, -0.25), ylims = (-1, 3),legendfontsize=12,legend_columns=-1)
#     # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
#     # # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(-.25:.5:1.25, (L"-.25", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # # plot!(xlims=(-0.8, .3), ylims=(-.25, 1.45))
#     # lo, _ = collect(zip(xlims(LE_plot), ylims(LE_plot)))
#     # locxl = lo .+ ((xlims(LE_plot)[2] - xlims(LE_plot)[1])/2, -0.1)
#     # locyl = lo .+ (-0.1, (ylims(LE_plot)[2] - ylims(LE_plot)[1])/2)
#     # annotate!(locxl..., L"x_1", fontsize=16)
#     # annotate!(locyl..., L"x_2", fontsize=16)

#     push!(LE_plots, LE_plot)
# end

# LE_plots_final = plot(LE_plots..., layout=(1,2), legend=(-0.925, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

# ## Plot Target Partition Results

# gr()
# TP_pal = palette(:acton10)[[2,3,4,5,6][end:-1:1]] #palette(:seaborn_deep6), palette(:magma)[25:50:end], palette(:oslo10)[2:7], palette(:navia10)[2:7], palette(:lipari10)[1:5], palette(:acton10)[2:6]
# pal = 
# colors = [:black, pal[1], pal[2], TP_pal, pal[7]]
# alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;

# tix = 3; ti = T[tix];
# dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
# reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_TP_reach) # true v lin + const err
# avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_TP_avoid)

# TP_plots = []
# for game in ["reach", "avoid"]

#     ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
#     labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-") : fill("", 6)
#     title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"

#     TP_plot = plot(title=title, dpi=dpi);
#     contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
   
#     contour!(xig1..., reshape(ϕ_DP[tix+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#     for i=1:ntp
#         # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(TP_plot), ylims=ylims(TP_plot), label=labels[2], alpha=0.3, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#         # contour!(xig3..., reshape(ϕ_Hopf[i][1], res3, res3)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false)
#         contour!(xig3..., reshape(ϕ_Hopf[i][1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=0.25*alpha, colorbar=false)
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     for i=1:ntp

#         # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(TP_plot), ylims=ylims(TP_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#         # labels = i == 1 ? labels : fill("", length(labels))

#         contour!(xig3..., reshape(ϕ_Hopf[i][tix+1], res3, res3)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
#         # contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#         # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(TP_plot), ylims=ylims(TP_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     plot!(xlims = (-2.5, -0.25), ylims = (-1, 3),legendfontsize=12,legend_columns=-1)
#     # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
#     # # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(-.25:.5:1.25, (L"-.25", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # # plot!(xlims=(-0.8, .3), ylims=(-.25, 1.45))
#     # lo, _ = collect(zip(xlims(TP_plot), ylims(TP_plot)))
#     # locxl = lo .+ ((xlims(TP_plot)[2] - xlims(TP_plot)[1])/2, -0.1)
#     # locyl = lo .+ (-0.1, (ylims(TP_plot)[2] - ylims(TP_plot)[1])/2)
#     # annotate!(locxl..., L"x_1", fontsize=16)
#     # annotate!(locyl..., L"x_2", fontsize=16)

#     push!(TP_plots, TP_plot)
# end

# TP_plots_final = plot(TP_plots..., layout=(1,2), legend=(-0.925, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)

# ### New Problem

# r = 0.25; c𝒯 = [0.; 0.]; # c𝒯= [0.; 0.]
# Q𝒯 = inv(r) * diagm([1., 1.])
# max_u = 1.0; max_d = 0.5;
# J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2;
# Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2;
# 𝒯target = (J, Jˢ, (Q𝒯, c𝒯))
# t = 0.4

# Th = 0.13
# T = collect(Th:Th:t)

# ## Solve Linear Ensemble

# # X̃0ŨD̃ = ([c𝒯, c𝒯 + [r,r], c𝒯 + [-r,r], c𝒯 + [r,-r], c𝒯 - [r,r]], # box
# # X̃0ŨD̃ = ([c𝒯, c𝒯 + [r,0], c𝒯 - [r,0], c𝒯 + [0,r], c𝒯 - [0,r]], # circle
# # X̃0 = [c𝒯, c𝒯 + [r,r]/sqrt(2), c𝒯 + [-r,r]/sqrt(2), c𝒯 + [r,-r]/sqrt(2), c𝒯 - [r,r]/sqrt(2)]
# X̃0 = [4*c𝒯rri(i) for i=1:nle]
# X̃0ŨD̃_LE = (X̃0, # circle ×
#         [(y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu), (y,s) -> zeros(nu)], 
#         [(y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd), (y,s) -> zeros(nd)])
# δ̃ˢU_LE, X̃_LE, BRZu, dtU, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="U", X̃0ŨD̃=X̃0ŨD̃_LE); # trajectories are the same, hence X̃_LE same for both
# δ̃ˢD_LE, X̃_LE, BRZd, dtD, (lin_mat_fs, Gs) = apri_δˢ(vanderpol!, 𝒯target, inputs, t; zono_over="D", X̃0ŨD̃=X̃0ŨD̃_LE);
# nle = length(X̃0ŨD̃_LE[1])

# BRZ_plot = plot(BRZu, vars=(1,2), alpha=0.3, lw=3, label="BRZ (𝒰)", legend=:bottomleft)
# plot!(BRZ_plot, BRZd, vars=(1,2), alpha=0.3, lw=3, label="BRZ (𝒟)")
# for i=1:nle
#     scatter!(BRZ_plot, eachrow(hcat(X̃_LE[i].(dt)...)[1:2,:])..., xlims=xlims(BRZ_plot), ylims=ylims(BRZ_plot), label="x̃$i", alpha=0.3)
# end
# plot!()

# plot(); for i=1:nle; plot!(dtU, δ̃ˢU_LE[i][2].(dtU)); end; plot!()
# plot(); for i=1:nle; plot!(dtD, δ̃ˢD_LE[i][2].(dtD)); end; plot!()

# ## Solve with DP (only needed if true problem changed)

# res = 300
# Xg, Xg_DP, ϕ0Xg_DP, xig1 = hjr_init(c𝒯, diagm(ones(nx)), r; shape="ball", lb=(-1, -1), ub=(1, 1), res=res)
# ϕXgT_DP_dynamics_reach = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_reach, T; BRS=true, one_shot=true);
# ϕXgT_DP_dynamics_avoid = hjr_solve(Xg_DP, ϕ0Xg_DP, dynamics_avoid, T; BRS=true, one_shot=true);

# ## Solve with Hopf

# EδU_LE(i, s) = δ̃ˢU_LE[i][2](-s) * diagm([0, 1])
# EδD_LE(i, s) = δ̃ˢD_LE[i][2](-s) * diagm([0, 1])
# # EδU_TP(i, s) = δ̃ˢU_TPr[i][2](-s) * diagm([0, 1])
# # EδD_TP(i, s) = δ̃ˢD_TPa[i][2](-s) * diagm([0, 1])

# system_errU_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδU_LE(i,s));
# system_errD_LE(i) = (s -> A(X̃_LE[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_LE[i](-s)), s -> EδD_LE(i,s));
# # system_errU_TP(i) = (s -> A(X̃_TPr[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPr[i](-s)), s -> EδU_TP(i,s));
# # system_errD_TP(i) = (s -> A(X̃_TPa[i](-s)), max_u * B1, max_d * B2, Q₁, c₁, Q₁, c₂, s -> c(X̃_TPa[i](-s)), s -> EδD_TP(i,s));

# rr, ra = ra, ra
# Jr(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * rr^2;
# Jˢr(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * rr^2;
# Ja(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * ra^2;
# Jˢa(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * ra^2;

# targetri(i) = (Jr, Jˢr, (diagm(ones(nx)), c𝒯rri(i)));
# targetai(i) = (Ja, Jˢa, (diagm(ones(nx)), c𝒯rai(i)));

# # lb = (1.1 * -ρ(-[1,0,0,0], BRZu), -ρ(-[0,1,0,0], BRZu))
# # ub = (1.5 * ρ([1,0,0,0], BRZu), ρ([0,1,0,0], BRZu))
# lb = (-1, -1); ub = (1, 1)

# res4 = 30
# Xg, _, _, xig4 = hjr_init(c𝒯, Q𝒯, r; shape="box", lb=lb, ub=ub, res=res4);

# ϕXgT_Hopf_LE_reach = []; ϕXgT_Hopf_LE_avoid = [];
# for i=1:nle
#     (ϕXgT, ϕXgT_Hopf_errU_reachi), _ = Hopf_BRS(system_errU_LE(i), 𝒯target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
#     (_,    ϕXgT_Hopf_errD_avoidi), _ = Hopf_BRS(system_errD_LE(i), 𝒯target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
#     push!(ϕXgT_Hopf_LE_reach, ϕXgT_Hopf_errU_reachi); push!(ϕXgT_Hopf_LE_avoid, ϕXgT_Hopf_errD_avoidi);
# end

# plotlyjs()
# (ϕXgT, _ ), _ = Hopf_BRS(system_errD_LE(5), 𝒯target, T; th, Xg=Xg, error=true, game="avoid", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# plot_BRS(T, ϕXgT, ϕXgT_Hopf_LE_avoid[5]; interpolate=true, value_fn=false)

# (ϕXgT, _ ), _ = Hopf_BRS(system_errU_LE(5), 𝒯target, T; th, Xg=Xg, error=true, game="reach", opt_method=Hopf_cd, warm=true, check_all=true, printing=true);
# plot_BRS(T, ϕXgT, ϕXgT_Hopf_LE_reach[5]; interpolate=true, value_fn=false)

# ## Plot Linear Ensemble Results

# gr()
# LE_pal = palette(:oslo10)[2:7] #palette(:magma)[25:50:end], palette(:oslo10)[2:7], palette(:navia10)[2:7], palette(:lipari10)[1:5]
# colors = [:black, pal[1], pal[2], LE_pal, pal[7]]
# alpha = 0.85; fillalpha = 0.65; lw=2.5; legend_hfactor=0.5; dpi=300;

# tix = 3; ti = T[tix];
# dtix = findfirst(x -> x .+ ti > 0, dt) - 1;
# reach_comp = (ϕXgT_DP_dynamics_reach[1], ϕXgT_Hopf_LE_reach) # true v lin + const err
# avoid_comp = (ϕXgT_DP_dynamics_avoid[1], ϕXgT_Hopf_LE_avoid)

# LE_plots = []
# for game in ["reach", "avoid"]

#     ϕ_DP, ϕ_Hopf = game == "reach" ? reach_comp : avoid_comp
#     labels = game == "avoid" ? (L"𝒯", latexstring("x_{auto}"), L"\hat{𝒮}", L"ℛ / ℛ^-", L"ℛ_{δ^*} / ℛ_{δ^*}^-" , L"ℛ_{δ^*_{(τ)}} / ℛ_{δ^*_{(τ)}}^-") : fill("", 6)
#     title = game == "reach" ? L"\textrm{Reach}" : L"\textrm{Avoid}"
#     BRZ = game == "reach" ? BRZu : BRZd

#     LE_plot = plot(title=title, dpi=dpi);
#     contour!(xig1..., reshape(ϕ_DP[1], res, res)', levels=[0], color=colors[1], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[1], linecolor=colors[1], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#     plot!(BRZ[dtix], vars=(1,2), alpha=fillalpha, lw=lw, label=labels[3], color=:white, linecolor=colors[2], legend_hfactor=legend_hfactor, extra_kwargs=:subplot, linealpha=alpha);

#     contour!(xig1..., reshape(ϕ_DP[tix+1], res, res)', levels=[0], color=colors[3], lw=lw, alpha=alpha, colorbar=false);
#     plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[4], linecolor=colors[3], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
    
#     for i=1:nle
#         plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(LE_plot), ylims=ylims(LE_plot), label=labels[2], alpha=0.3, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     for i=1:nle

#         # plot!(eachrow(hcat(X̃_LE[i].(dt[end:-1:dtix])...)[1:2,:])..., xlims=xlims(LE_plot), ylims=ylims(LE_plot), label=labels[2], alpha=0.6, lw=lw, color=colors[4][i], linestyle=:dash, legend_hfactor=legend_hfactor, extra_kwargs=:subplot)

#         # labels = i == 1 ? labels : fill("", length(labels))

#         contour!(xig4..., reshape(ϕ_Hopf[i][tix+1], res4, res4)', levels=[0], color=colors[4][i], lw=lw, alpha=alpha, colorbar=false)
#         plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[5], linecolor=colors[4][i], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
  
#         # contour!(xig2..., reshape(ϕ_Hopf2[i+1], res2, res2)', levels=[0], color=colors[4], lw=lw, alpha=alpha, colorbar=false)
#         # plot!(Ball2(zeros(2), 10.), vars=(1,2), label=labels[6], linecolor=colors[4], alpha=0., lw=lw, xlims=xlims(LE_plot), ylims=ylims(LE_plot), legend_hfactor=legend_hfactor, extra_kwargs=:subplot)
#     end

#     plot!(xlims = (-1, 1), ylims = (-1, 1), legendfontsize=12, legend_columns=-1)
#     # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(0.:.25:1.25, (L"0", "", "", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # plot!(xlims=(-0.8, .3), ylims=(-0.15, 1.4))
#     # # plot!(xticks=(-0.75:0.5:0.25, (L"-.75", "", L".25")), yticks=(-.25:.5:1.25, (L"-.25", "", "", L"1.25")), xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,legendfontsize=12,legend_columns=-1)
#     # # plot!(xlims=(-0.8, .3), ylims=(-.25, 1.45))
#     # lo, _ = collect(zip(xlims(LE_plot), ylims(LE_plot)))
#     # locxl = lo .+ ((xlims(LE_plot)[2] - xlims(LE_plot)[1])/2, -0.1)
#     # locyl = lo .+ (-0.1, (ylims(LE_plot)[2] - ylims(LE_plot)[1])/2)
#     # annotate!(locxl..., L"x_1", fontsize=16)
#     # annotate!(locyl..., L"x_2", fontsize=16)

#     push!(LE_plots, LE_plot)
# end

# LE_plots_final = plot(LE_plots..., layout=(1,2), legend=(-0.925, -.175), bottom_margin=45Plots.px, foreground_color_legend = nothing, dpi=dpi)
