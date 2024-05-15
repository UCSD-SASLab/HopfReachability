
using LinearAlgebra, PyCall

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

@pydef mutable struct Linear <: hj.ControlAndDisturbanceAffineDynamics
    function __init__(self, A, B, C; input_shapes="mixed", Ushape="ball", Dshape="ball", max_u=1.0, max_d=0.5, game="reach", control_space=nothing, disturbance_space=nothing)
        """
        Linear (Potentially Time-Varying) Dynamics Class
        A, B, C: linear parameters, can be constant Matrix{Float64}, or fn of t (Float64) -> Mt (Matrix{Float64})
        """
        self.n_x = typeof(A) <: Function ? size(A(0.0))[2] : size(A)[1]
        self.n_u = typeof(B) <: Function ? size(B(0.0))[2] : size(B)[2]
        self.n_d = typeof(C) <: Function ? size(C(0.0))[2] : size(C)[2]
        self.control_mode, self.disturbance_mode = game == "reach" ? ("min", "max") : ("max", "min")
        Ushape, Dshape = input_shapes == "ball" ? ("ball", "ball") : (input_shapes == "box" ? ("box", "box") : (Ushape, Dshape))

        if isnothing(control_space)
            if Ushape == "box"
                self.control_space = hj.sets.Box(lo=-max_u*jnp.ones(self.n_u), hi=max_u*jnp.ones(self.n_u))
            elseif Ushape == "ball"
                self.control_space = hj.sets.Ball(jnp.zeros(self.n_u), max_u)
            end
        end

        if isnothing(disturbance_space)
            if Dshape == "box"
                self.disturbance_space = hj.sets.Box(lo=-max_d*jnp.ones(self.n_d), hi=max_d*jnp.ones(self.n_d))
            elseif Dshape == "ball"
                self.disturbance_space = hj.sets.Ball(jnp.zeros(self.n_d), max_d)
            end
        end

        pybuiltin(:super)(Linear, self).__init__(self.control_mode, self.disturbance_mode, self.control_space, self.disturbance_space) #(Linear, self)
        
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
        self.control_mode, self.disturbance_mode = game == "reach" ? ("min", "max") : ("max", "min")

        if isnothing(control_space)
            if Ushape == "box"
                ub = max_u*ones(self.n_u);
                self.control_space = hj.sets.Box(lo=-ub, hi=ub)
            elseif Ushape == "ball"
                self.control_space = hj.sets.Ball(jnp.zeros(self.n_u), max_u)
            end
        end

        if isnothing(disturbance_space)
            deb = vcat(max_d*ones(self.n_d), max_e*ones(self.n_e))
            self.disturbance_space = hj.sets.Box(lo=-deb, hi=deb)
        end

        pybuiltin(:super)(LinearError, self).__init__(self.control_mode, self.disturbance_mode, self.control_space, self.disturbance_space) #(Linear, self)
        
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
