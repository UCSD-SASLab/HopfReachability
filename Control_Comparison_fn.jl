#   Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   wsharpless@ucsd.edu
#   ---------------------

using LinearAlgebra, Plots
plotlyjs()

## Comparison
using iLQGames, JuMP, Gurobi

#   Utility Fn
#   ==========

function roll_out(system, target, ctrls, x0, steps; random_d=false, random_same=true, worst_d=false, surface=false, printing=false, plotting=false, title="Controlled Trajectories")
    """
    Function for rolling out a dictionary of controllers from a given initial condition.
        random_d    : apply random disturbance to all
        random_same : apply same random disturbance (if random)
        surface     : generates random disturbance on surface of constraint
        worst_d     : apply the disturbance from the "Hopf" controller to all
    
        e.g.

        ctrls = Dict("Hopf"=> Hopf_minT(intH=intH_ytc17, HJoc=HJoc_ytc17, preH=preH_ytc17, time_p = (0.05, 0.4, 2.0)),
                     "MPCs"=> MPC_stochastic(H=10, N_T=20),
                     "MPCg"=> MPC_game(H=1, its=1))
    """

    ## Initialize
    M, C, C2, Q, Q2, a, a2 = system
    Md, Cd, C2d = diagm(ones(size(M)[1])) + th * M, th * C, th * C2;
    Tˢ, dϕdz = 0, 0 #for plot_game

    ## Store Trajectories
    Xs = Dict(key => hcat(x0, zeros(nx, steps)) for key in keys(ctrls));
    Us = Dict(key => zeros(nx, steps) for key in keys(ctrls));
    Ds = Dict(key => zeros(nx, steps) for key in keys(ctrls)); 
    
    ## Iterate through time
    for s in 1:steps
        if printing; println(" on step $s ..."); end

        ## Compute Feedback
        for c in keys(ctrls)
            outputs = ctrls[c](Xs[c][:, s])
            Us[c][:, s], Ds[c][:, s] = outputs[1], outputs[2]
            if plotting && c == "Hopf"; Tˢ, dϕdz = outputs[3], outputs[5]; end
        end

        ## Choose Disturbance
        if worst_d
            if printing; print(" with worst disturbance"); end
            for c in keys(ctrls); Ds[c][:, s] = Ds["Hopf"][:, s]; end
        elseif random_d 
            if printing; print(" with random disturbance"); end
            d = ellipsoid_sample(1, Q2, 1.; surface)
            for c in keys(ctrls); Ds[c][:, s] = d; end
        else
            if printing; print(" with their estimated worst disturbance"); end
        end

        ## Evolve States
        for c in keys(ctrls)
            if random_d && !random_same; Ds[c][:, s] = ellipsoid_sample(1, Q2, 1., surface); if printing; print(" (different)"); end; end
            Xs[c][:, s + 1] = Md * Xs[c][:, s] + Cd * Us[c][:, s] + C2d * Ds[c][:, s]
        end

        ## Plot Game at First Step
        if s == 1 && plotting
            pair_strats = [(Us[c][:, s], Ds[c][:, s]) for c in keys(ctrls)]; pair_labels = collect(keys(ctrls))
            game_plot = plot_game(system, pair_strats, pair_labels; p=dϕdz, t=Tˢ)
        end
    end

    if plotting; sim_plot = plot_sim(system, target, Xs; title); display(sim_plot); end

    return Xs, Us, Ds
end

function plot_sim(system, target, Xs; grid_p=((-3, 3), 10.), title="Controlled Trajectories", plot_drift=true, plot_target=true, scale=0.6)
    """
    Plots the Controlled Trajectories, Target and Autonomous Drift from a dictionary of paths.
    """

    M, C, C2, Q, Q2, a, a2 = system
    J, _, Jp = target
    nx = size(M)[1]

    ## Make Grid
    bd, N = grid_p
    lb, ub = typeof(bd) <: Tuple ? bd : (-bd, bd)
    xig = collect(lb : 3/N : ub) .+ ϵ; lg = length(xig)
    Xg = hcat(collect.(Iterators.product([xig for i in 1:nx]...))...)[end:-1:1,:]
    
    ## Target and Field
    Jxg = reshape(J(Xg, Jp...), length(xig), length(xig));
    Fxg = M * Xg; Fn = Fxg ./ map(norm, eachcol(Fxg))'

    sim_plot = plot(title=title)

    if plot_drift; quiver!(sim_plot, Xg[1,:], Xg[2,:], quiver = 0.75 * scale .* (Fn[1,:],  Fn[2,:]), label="f", color="blue", alpha=0.1); end
    if plot_target; contour!(sim_plot, xig, xig, Jxg, levels=-1e-3:1e-3:1e-3, color="black", colorbar=false); end

    for c in sort(collect(keys(Xs))); plot!(sim_plot, Xs[c][1,:], Xs[c][2,:], label=c); end

    scatter!(sim_plot, [Xs[first(keys(Xs))][1,1]], [Xs[first(keys(Xs))][2,1]], label="x_0", legend=:bottomleft)

    return sim_plot
end


function MPC_game(system, target, x0; th=5e-2, H=1, its=1, input_constraint="coupled", solver=Gurobi.Optimizer)
    """
    Computes the optimal control over horizon H given a naive optimal disturbance over horizon H. No proof of convergence. 
    For simplicity, use H = 1 & its = 1 to just compute optimal control for 1-step worst disturbance.
    """

    M, C, C2, Q, Q2, a, a2 = system
    Md, Cd, C2d = diagm(ones(size(M)[1])) + th * M, th * C, th * C2;
    J, _, Jp = target

    nx, nu, nd = size(M)[1], size(C)[2], size(C2)[2]
    uˢH, dˢH = zeros(nu, H), zeros(nu, H)

    for it in 1:its
        for player in ["d", "u"]

            model = Model(solver); set_silent(model);
            set_optimizer_attribute(model, "NonConvex", 2);

            @variable(model, x[1:nx, 1:(H+1)]);
            @variable(model, u[1:nu, 1:H])
            @variable(model, d[1:nd, 1:H])

            ## Fix Initial Condition
            for i in 1:nx; fix(x[i, 1], x0[i]); end

            ## Fix Other Strategy
            if it != 1
                if player == "d"
                    for i in 1:nd, j in 1:H; fix(u[i, j], uˢH[i, j]); end
                elseif player == "u"
                    for i in 1:nd, j in 1:H; fix(d[i, j], dˢH[i, j]); end
                end
            else
                if player == "d"
                    for i in 1:nd, j in 1:H; fix(u[i, j], 0.); end
                elseif player == "u"
                    for i in 1:nd, j in 1:H; fix(d[i, j], dˢH[i, j]); end
                end
            end

            ## Constraints
            for h in 1:H

                ## Dynamics
                @constraint(model, x[:, h + 1] .== Md * x[:, h] + Cd * u[:, h] + C2d * d[:, h])

                ## Control and Disturbance
                if input_constraint == "coupled"
                    @constraint(model, u[:, h]' * inv(Q)  * u[:, h] ≤ 1)
                    @constraint(model, d[:, h]' * inv(Q2) * d[:, h] ≤ 1)
                elseif input_constraint == "box"
                    @constraint(model, [t; u] in MOI.NormInfinityCone(1 + nu)) #maximum(abs.(Q  * u[:, h])) ≤ 1)
                    @constraint(model, [t; d] in MOI.NormInfinityCone(1 + nd)) #maximum(abs.(Q2 * d[:, h])) ≤ 1)
                    @constraint(model, t == 1)
                end
            end

            if player == "d"
                @objective(model, Max, 0.5 * (x[:, H+1]' * inv(Ap) * x[:, H+1] - 1))
            elseif player == "u"
                @objective(model, Min, 0.5 * (x[:, H+1]' * inv(Ap) * x[:, H+1] - 1))
            end
            optimize!(model)

            uˢH, dˢH = value.(u), value.(d)
            
        end
    end

    return uˢH[:,1], dˢH[:,1]
end

function MPC_stochastic(system, target, x0; th=5e-2, H=3, N_T=10, input_constraint="coupled", solver=Gurobi.Optimizer)
    """
    Computes the optimal control over horizon H given N_T sample trajectories of disturbance.
    """

    M, C, C2, Q, Q2, a, a2 = system
    Md, Cd, C2d = diagm(ones(size(M)[1])) + th * M, th * C, th * C2;
    J, _, Jp = target

    nx, nu, nd = size(M)[2], size(C)[2], size(C2)[2]
        
    model = Model(solver); set_silent(model);
    set_optimizer_attribute(model, "NonConvex", 2);
    
    @variable(model, x[1:nx, 1:(H+1), 1:N_T]);
    @variable(model, u[1:nu, 1:H])
    
    ## Generate Random Disturbance
    if input_constraint == "coupled"
        Ds = reshape(ellipsoid_sample(H * N_T, inv(Q2), 1.), nd, H, N_T) 
    elseif input_constraint == "box"
        Ds = reshape(rand(nd, H * N_T) .* inv(diag(Q2)), nd, H, N_T)
    end

    ## Fix Initial Condition
    for i in 1:nx, k in 1:N_T; fix(x[i, 1, k], x0[i]); end

    ## Constraints
    for h in 1:H 
        
        ## Dynamics
        for k in 1:N_T
            @constraint(model, x[:, h + 1, k] .== Md * x[:, h, k] + Cd * u[:, h] + C2d * Ds[:, h, k])
        end

        ## Control
        if input_constraint == "coupled"
            @constraint(model, u[:, h]' * inv(Q) * u[:, h] ≤ 1)
        elseif input_constraint == "box"
            @constraint(model, [t; u] in MOI.NormInfinityCone(1 + nu)) #maximum(abs.(Q  * u[:, h])) ≤ 1)
            @constraint(model, t == 1)
        end
    end

    @objective(model, Min, sum(0.5 * (x[:, H+1, k]' * inv(Ap) * x[:, H+1, k] - 1) for k in 1:N_T)/k);
    optimize!(model);

    uˢH = value.(u)

    return uˢH[:,1], Ds[:, 1, 1]
end

function ellipsoid_sample(N_s, A, r; n=size(A)[1], surface=false)
    """
    Sample the ellipsoid defined by x'inv(A)x ≤ r^2, by sampling on/in spheroid and mapping to ellipsoid w inv(cholesky(inv(A))).
    # http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf (interior)
    # https://www.maxturgeon.ca/f19-stat4690/slides/quadratic_forms_ellipses.pdf (surface)
    """

    B = !surface ? randn((n+1, N_s)) : randn((n, N_s))
    Bn = B ./ map(norm, eachcol(B))'
    return inv(cholesky(inv(A)).U) * r * Bn[1:n, :] 
end

# function Hamiltonian_counter_optimal(uˢ; p, t, solver=Gurobi.Optimizer) #sub-optimal
#     """
#     Computes the optimal disturbance given an optimal control for a linear-tv, differential game Hamiltonian with coupled controls.
#     """

#     model = Model(solver); set_silent(model);
#     set_optimizer_attribute(model, "NonConvex", 2);
#     @variable(model, d[1:nd])
#     @constraint(model, d' * inv(Q2) * d ≤ 1)

#     @objective(model, Min, p' * exp(-t * M) * (C * uˢ + C2 * d))
#     optimize!(model);
#     dˢ = value.(d)

#     return dˢ
# end

function plot_game(system, pair_strats, pair_labels; p, t, p1_strats=[], p1_labels=[], title=nothing)
    """
    Plots the value of the 2D, linear TV differential game with coupled controls
    """

    M, C, C2, Q, Q2, a, a2 = system    
    
    ## Parameterize Input space with θ (w conjecture that optimal soln always lies on bound of constraint)
    θu, θd = collect(-pi:0.05:pi), collect(-pi:0.05:pi);
    Θg = hcat(collect.(Iterators.product(θd, θu))...)[end:-1:1,:]

    # Ue, De = inv(cholesky(inv(Q)).U)  * vcat(cos.(θu)', sin.(θu)'), inv(cholesky(inv(Q2)).U) * vcat(cos.(θd)', sin.(θd)')
    Hz(θ; pp, tt) = p' * exp(-t * M) * (C * inv(cholesky(inv(Q)).U) * vcat(cos.(θ[1, :])', sin.(θ[1, :])') + C2 * inv(cholesky(inv(Q2)).U) * vcat(cos.(θ[2, :])', sin.(θ[2, :])'))
    HΘg = reshape(Hz(Θg; pp=p, tt=t), length(θu), length(θd))

    ## Plot Hamiltonian Surface
    game_plot = surface(θu, θd, HΘg, xlabel="θu", ylabel="θd", alpha=0.5, title = isnothing(title) ? "H(p≈$(round.(p, digits=2)), t≈$(round.(Tˢ, digits=3)), θu, θd)" : title)

    ## Map Computed Optimal Inputs to Theta Space
    Input2θ(V, A) = map(Base.splat(atan), eachcol((cholesky(inv(A)).U * V)[end:-1:1, :])) # θu, θd = Input2θ(uˢ, Q), Input2θ(dˢ, Q2)

    for (psi, (u, d)) in enumerate(pair_strats)
        θud = reshape(vcat(Input2θ(u, Q), Input2θ(d, Q2)), 2, 1)
        scatter!(θud[1, :], θud[2, :], [Hz(θud; pp=p, tt=t)], label=pair_labels[psi])
    end

    for (psi, (u)) in enumerate(p1_strats)
        θud = vcat(Input2θ(u, Q) * (zero(θu) .+ 1)', θd')
        plot!(θud[1, :], θud[2, :], Hz(θud; pp=p, tt=t)', label=p1_labels[psi])
    end

    return game_plot
end

function plot_inputs(system, Is, itype; border=0.7)
    """
    Function for plotting a trajectory of inputs and their constraints.
    """
    M, C, C2, Q, Q2, a1, a2 = system

    if itype == "u"
        P = Q; title = "U := {u' Q⁻¹ u ≤ 1}"
    elseif itype == "d"
        P = Q2; title = "D := {d' Q₂⁻¹ d ≤ 1}"
    end

    u1g = collect(-border : 0.1 : border) .+ 1e-7; u2g = collect(-border : 0.1 : border) .+ 1e-7;
    Ug = hcat(collect.(Iterators.product(u2g, u1g))...)[end:-1:1,:]
    Jg = reshape(J(Ug, inv(P), cp), length(u1g), length(u2g))

    input_plot = contour(u1g, u2g, Jg, color="black", colorbar=false, levels=-1e-5:1e-5:1e-5, title=title);

    for key in sort(collect(keys(Is))); 
        scatter!(input_plot, Is[key][1,:], Is[key][2,:], label=key, alpha=0.5); 
    end

    return input_plot
end
