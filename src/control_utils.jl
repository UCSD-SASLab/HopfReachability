#   Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   wsharpless@ucsd.edu
#   ---------------------

using LinearAlgebra, Plots, Suppressor, LaTeXStrings, TickTock
# plotlyjs()

## Comparison
using iLQGames, JuMP, Gurobi

#   Utility Fn
#   ==========

function LinearEuler(x, u, d; p=(I, I, I), th=5e-2)
    return (I + th * p[1]) * x + (th * p[2]) * u + (th * p[3]) * d
end

function roll_out(system, target, ctrls, x0, steps; F = LinearEuler, ψ = x -> x,
                    nx = size(system[1])[2], nu = size(system[2])[2], nd = size(system[3])[2], th = 5e-2,
                    random_d=false, random_same=true, same_d=nothing, surface=false, printing=false, 
                    plotting=false, title="Controlled Trajectories")

    """
    Function for rolling out a dictionary of controllers from a given initial condition.
        nx, nu, nd  : state, control and disturbance dimensions
        th          : time interval for evolution (implicitly, control and disturbance constant during interval)
        F           : function for evolving the state (default linear Euler)
        ψ           : function for lifting the state (default id)
        random_d    : apply random disturbance to all
        random_same : apply same random disturbance (if random)
        surface     : generates random disturbance on surface of constraint
        same_d      : apply the same disturbance from the specified controller
    
        e.g.

        ctrls = Dict("Hopf" => x -> Hopf_minT(system, target, x; time_p = (0.05, 0.4, 2.0)),
                     "MPCs" => x -> MPC_stochastic(system, target, x; H=10, N_T=20),
                     "MPCg" => x -> MPC_game(system, target, x; H=1, its=1));
        
        Xs, Us, Ds = roll_out(system, target, ctrls, x0, steps)
    """

    ## System
    Xs = Dict(key => hcat(x0, zeros(nx, steps)) for key in keys(ctrls));
    Us = Dict(key => zeros(nu, steps) for key in keys(ctrls));
    Ds = Dict(key => zeros(nd, steps) for key in keys(ctrls)); 
    Ts = Dict(key => zeros(steps) for key in keys(ctrls)); 
    Tˢ, dϕdz = 0, 0 #for plot_game
    
    ## Iterate through time
    for s in 1:steps
        if printing; println(" on step $s ..."); end

        ## Compute Feedback
        for c in keys(ctrls)

            @suppress begin; tick(); end # timing

            if startswith(c, L"Hopf")
                outputs = ctrls[c](ψ(Xs[c][:, s]))
                us, ds, Tˢ, ϕ, dϕdz = outputs;
                if printing; println("$c will reach in $Tˢ s (ϕ = $ϕ)"); end
            else
                try
                    @suppress outputs = ctrls[c](ψ(Xs[c][:, s]))
                catch;
                    outputs = (Us[c][:, s-1], Ds[c][:, s-1]) # rare occurence when primal oob
                end
            end

            Ts[c][s] = tok()

            Us[c][:, s], Ds[c][:, s] = outputs[1], outputs[2]
        end

        ## Choose Disturbance
        if !isnothing(same_d)
            if printing; print(" perturbed by the $same_d disturbance"); end
            for c in keys(ctrls); Ds[c][:, s] = Ds[same_d][:, s]; end
        elseif random_d 
            if printing; print(" perturbed by random disturbance"); end
            d = ellipsoid_sample(1, system[6], 1.; surface)
            for c in keys(ctrls); Ds[c][:, s] = d; end
        else
            if printing; print(" perturbed by their own estimated worst disturbance"); end
        end

        ## Evolve States
        for c in keys(ctrls)
            if random_d && !random_same; Ds[c][:, s] = ellipsoid_sample(1, system[6], 1.; surface); if printing; print(" (different)"); end; end
            Xs[c][:, s+1] = F(Xs[c][:, s], Us[c][:, s], Ds[c][:, s]; p=system, th)
        end

        ## Plot Game at First Step
        if s == 1 && plotting
            _, _, Tˢ, _, dϕdz = Hopf_minT(system, target, x0; time_p = (0.05, 0.4, 2.0))
            pair_strats = [(Us[c][:, s], Ds[c][:, s]) for c in keys(ctrls)]; pair_labels = collect(keys(ctrls))
            game_plot = plot_game(system, pair_strats, pair_labels; p=dϕdz, t=Tˢ); display(game_plot);
        end
    end

    if plotting; sim_plot = plot_sim(system, target, Xs; title); display(sim_plot); end

    return Xs, Us, Ds, Ts
end


# function roll_out(system, target, ctrls, x0, steps; random_d=false, random_same=true, worst_d=false, surface=false, printing=false, plotting=false, title="Controlled Trajectories")
#     """
#     Function for rolling out a dictionary of controllers from a given initial condition.
#         random_d    : apply random disturbance to all
#         random_same : apply same random disturbance (if random)
#         surface     : generates random disturbance on surface of constraint
#         worst_d     : apply the disturbance from the "Hopf" controller to all
    
#         e.g.

#         ctrls = Dict("Hopf" => Hopf_minT(time_p = (0.05, 0.4, 2.0)),
#                      "MPCs" => MPC_stochastic(H=10, N_T=20),
#                      "MPCg" => MPC_game(H=1, its=1))
#     """

#     ## System
#     M, C, C2, Q, a, Q2, a2 = system
#     Md, Cd, C2d = diagm(ones(size(M)[1])) + th * M, th * C, th * C2;
#     Tˢ, dϕdz = 0, 0 #for plot_game

#     ## Store Trajectories
#     Xs = Dict(key => hcat(x0, zeros(nx, steps)) for key in keys(ctrls));
#     Us = Dict(key => zeros(nx, steps) for key in keys(ctrls));
#     Ds = Dict(key => zeros(nx, steps) for key in keys(ctrls)); 
    
#     ## Iterate through time
#     for s in 1:steps
#         if printing; println(" on step $s ..."); end

#         ## Compute Feedback
#         for c in keys(ctrls)
#             outputs = ctrls[c](Xs[c][:, s])
#             Us[c][:, s], Ds[c][:, s] = outputs[1], outputs[2]
#             if plotting && c == "Hopf"; Tˢ, dϕdz = outputs[3], outputs[5]; end
#         end

#         ## Choose Disturbance
#         if worst_d
#             if printing; print(" with worst disturbance"); end
#             for c in keys(ctrls); Ds[c][:, s] = Ds["Hopf"][:, s]; end
#         elseif random_d 
#             if printing; print(" with random disturbance"); end
#             d = ellipsoid_sample(1, Q2, 1.; surface)
#             for c in keys(ctrls); Ds[c][:, s] = d; end
#         else
#             if printing; print(" with their estimated worst disturbance"); end
#         end

#         ## Evolve States
#         for c in keys(ctrls)
#             if random_d && !random_same; Ds[c][:, s] = ellipsoid_sample(1, Q2, 1.; surface); if printing; print(" (different)"); end; end
#             Xs[c][:, s + 1] = Md * Xs[c][:, s] + Cd * Us[c][:, s] + C2d * Ds[c][:, s]
#         end

#         ## Plot Game at First Step
#         if s == 1 && plotting
#             pair_strats = [(Us[c][:, s], Ds[c][:, s]) for c in keys(ctrls)]; pair_labels = collect(keys(ctrls))
#             game_plot = plot_game(system, pair_strats, pair_labels; p=dϕdz, t=Tˢ)
#         end
#     end

#     if plotting; sim_plot = plot_sim(system, target, Xs; title); display(sim_plot); end

#     return Xs, Us, Ds
# end

function plot_sim(system, target, Xs; grid_p=((-3, 3), 0.1), title="Controlled Trajectories", plot_drift=true, plot_target=true, scale=0.6, lw_traj=2)
    """
    Plots the Controlled Trajectories, Target and Autonomous Drift from a dictionary of paths. (for 2D)
    """

    M, C, C2, Q, a, Q2, a2 = system
    J, _, Jp = target
    nx = size(M)[1]

    ## Make Grid
    Xg, xigs, _ = make_grid(grid_p..., size(M,1); return_all=true)
    
    ## Target and Field
    Jxg = reshape(J(Xg), length(xigs[1]), length(xigs[1]))';
    Fxg = M * Xg; Fn = Fxg ./ map(norm, eachcol(Fxg))'

    sim_plot = plot(title=title)

    if plot_drift; quiver!(sim_plot, Xg[1,:], Xg[2,:], quiver = 0.75 * scale .* (Fn[1,:],  Fn[2,:]), label="f", color="blue", alpha=0.1); end
    if plot_target; contour!(sim_plot, xigs[1], xigs[1], Jxg, levels=[0], color="black", colorbar=false); end

    for c in sort(collect(keys(Xs))); plot!(sim_plot, Xs[c][1,:], Xs[c][2,:], label=c, lw=lw_traj); end

    scatter!(sim_plot, [Xs[first(keys(Xs))][1,1]], [Xs[first(keys(Xs))][2,1]], label="x_0", legend=:bottomleft)

    return sim_plot
end

function plot_sim_sbs(Xs; th, steps=size(first(Xs)[2])[2]-1, nx=size(first(Xs)[2])[1], 
    labels=[L"\text{Glucose}" L"\text{Phosphate Pool}" L"1,3-\text{BPG}" L"\text{Pyruvate} \:\: \& \:\: \text{Acetate}" L"\text{NADH}" L"\text{ATP}" L"\text{Glucose}\:\: \text{Ext} \:\: (\leftarrow u_1)" L"\text{Pyruvate} \:\: \& \:\: \text{Acetate} \:\: \text{Ext}" L"\text{NAD}^+ / \text{NADH}\:\:\text{Pool}  \:\: (\leftarrow u_2) " L"\text{ATP}/\text{ADP}\:\:\text{Pool}  \:\: (\leftarrow u_3)"], 
    dim_splits=[[1,2,3,4,5,6,8], [7,9,10]], pal=:seaborn_colorblind, semilog=false,)

    sim_plots = []
    for (ci, c) in enumerate(sort(collect(keys(ctrls))))
        labels = ci == 1 ? labels : reshape(["" for _=1:nx], 1, nx)

        split_plots = []
        for (ixsi, ixs) in enumerate(dim_splits)
            split_plot = plot(th:th:th*(steps+1), Xs[c]'[:,ixs], 
                            label=labels[:, ixs], linewidth=3, alpha=0.8, yscale = semilog ? :log10 : :identity,
                            palette = palette(pal)[[end:-1:1, 1:end][ixsi]])
            
            if ixsi == 1;
                hline!([2.5], linestyle=:dash, color=:black, alpha=0.4, linewidth=2, label= ci == 1 ? L"\text{Target}\:\:\text{ATP}" : "")
            end

            if ci == 1;
                if ixsi == 1; 
                    plot!(ylims=[-0.01, 3.01], yticks = ([0,1,2,3], [L"0", L"1", L"2", L"3"])); 
                else
                    plot!(ylims=[-0.01, 5.02], yticks = ([0,1,2,3,4,5], [L"0", L"1", L"2", L"3", L"4", L"5"]));
                end
            else                
                if ixsi == 1; 
                    plot!(ylims=[-0.01, 3.01], yticks = ([0,1,2,3], ["", "", "", ""])); 
                else
                    plot!(ylims=[-0.01, 5.02], yticks = ([0,1,2,3,4,5], ["", "", "", "", "", ""]));
                end
            end

            if ixsi == 2; 
                plot!(xlims=[-0.01, 3.01], xticks = ([1,2,3], [L"1", L"2", L"3"]));
                # if ci == 1; plot!(ylabel = L"x_i", xlabel = L"t"); end 
            else
                plot!(xlims=[-0.01, 3.01], xticks = ([0,1,2,3],["", "", "", ""]));
            end

            push!(split_plots, split_plot)
        end

        sim_plot_c = plot(split_plots..., layout=(length(dim_splits),1), legend=:outertopright, plot_title=LaTeXString(c))

        push!(sim_plots, sim_plot_c)
    end
    
    sim_plot = plot(sim_plots..., layout=(1, length(keys(Xs))), legend=:outertopright, size=(900, 500))

    YLABEL = plot(title = L"\: \\ \: \\ x_i \\ \: \\ \: \\ \: \\ \: \\ \: \\ \: \\  \: \\ x_i", fontsize=10, grid = false, showaxis = false, right_margin=-50Plots.px) #, left_margin = 10Plots.px)    
    
    # empty = plot(title = L"", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    XLABEL = plot(title = L"\: \\ \text{Time (s)}", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    
    # XLABEL = plot(empty, XLABEL, layout= @layout[a{0.01h}; b])

    sim_plot = plot(sim_plot, XLABEL, layout= @layout[a; b{0.01h}], size=(900, 500), legend=:outertopright)
    sim_plot = plot(YLABEL, sim_plot, layout=@layout[a{0.01w}  b], size=(900, 500), legend=:outertopright)

    plot!(sim_plot, extra_plot_kwargs = KW(:include_mathjax => "cdn"), legend=false)

    return sim_plot
end


function MPC_game(system, target, x0; th=5e-2, H=1, its=1, input_constraint="coupled", solver=Gurobi.Optimizer)
    """
    Computes the optimal control over horizon H given a naive optimal disturbance over horizon H. No proof of convergence. 
    For simplicity, use H = 1 & its = 1 to just compute optimal control for 1-step worst disturbance.
    """

    M, C, C2, Q, a, Q2, a2 = system
    Md, Cd, C2d = diagm(ones(size(M)[1])) + th * M, th * C, th * C2;
    J, _, Jp = target
    Ap, cp = Jp

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
                @objective(model, Max, 0.5 * ((x[:, H+1] .- cp)' * inv(Ap) * (x[:, H+1] .- cp) - 1))
            elseif player == "u"
                @objective(model, Min, 0.5 * ((x[:, H+1] .- cp)' * inv(Ap) * (x[:, H+1] .- cp) - 1))
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

    M, C, C2, Q, a, Q2, a2 = system
    Md, Cd, C2d = diagm(ones(size(M)[1])) + th * M, th * C, th * C2;
    J, _, Jp = target
    Ap, cp = Jp

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

    @objective(model, Min, sum(0.5 * ((x[:, H+1, k] .- cp)' * inv(Ap) * (x[:, H+1, k] .- cp) - 1) for k in 1:N_T)/N_T);
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

    M, C, C2, Q, a, Q2, a2 = system    
    
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

function plot_inputs(input_params, Is; title="Input Set", type="ball", res=0.01)
    """
    Function for plotting a trajectory of inputs and their constraints.
    """
    Q, c, r = input_params

    if type ∈ ["ball", "Ball", "L2"]
        Ji = X -> diag((X .- c')' * inv(Q) * (X .- c'))/2 .- 0.5 * 1^2 # gives used r
        # Ji = X -> diag((X .- c')' * inv(2r*Q) * (X .- c'))/2 .- 0.5 * 1^2 # gives intended r
    elseif type ∈ ["box", "Box", "Linf"]
        Ji = X -> norm.(eachcol(inv(Q) * (X .- c')), Inf) .- 2
    else
        error("$type not supported")
    end

    Ug, ugis, _ = make_grid(1.25r, res, size(Q,1); return_all=true, shift=c)
    Jg = reshape(Ji(Ug), length(ugis[1]), length(ugis[2]))'

    input_plot = contour(ugis[1], ugis[2], Jg, color="black", colorbar=false, levels=[0], lw=2, title=title);

    for key in sort(collect(keys(Is))); 
        scatter!(input_plot, Is[key][1,:], Is[key][2,:], label=key, alpha=0.5); 
    end

    return input_plot
end
