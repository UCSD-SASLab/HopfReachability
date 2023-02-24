#   Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   wsharpless@ucsd.edu
#   ---------------------

using LinearAlgebra, Plots
plotlyjs()

## Comparison
using iLQGames, JuMP, Gurobi

## Our HopfReachability Pkg
include("/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, HJoc_ytc17, Hopf_minT


#   Initialize
#   ==========

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

nx, nu, nd = 2, 2, 2
M = [0. 1; -2 -3]
C = 0.5 * [1 0; 0 1]
C2 = 0.5 * [2 0; 0 1]
Q = 0.1 * diagm(0 => 3*ones(nu), -1 => 1*ones(nu-1), 1 => 1*ones(nu-1))
Q2 = 0.2 * diagm(0 => 2*ones(nd), -1 => 1*ones(nd-1), 1 => 1*ones(nd-1))
a1 = [0. 0.]
a2 = [0. 0.]
system = (M, C, C2, Q, Q2, a1, a2);

## Target
# J(x) = 0 is the boundary of the target

Ap = diagm(ones(nx))
cp = zeros(nx)

J(x::Vector, A, c) = ((x - c)' * A * (x - c))/2 - 0.5
Js(v::Vector, A, c) = (v' * inv(A) * v)/2 + c'v + 0.5 # Convex Conjugate

J(x::Matrix, A, c) = diag((x .- c)' * A * (x .- c))/2 .- 0.5
Js(v::Matrix, A, c) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5

target = (J, Js, (Ap, cp));

## Lookback Time(s), 
# each time solved independently! solving multiple for comparison to Dynamic Programming

th = 0.05
Th = 0.4
Tf = 2.0
T = collect(Th : Th : Tf);

## Grid Parameters
bd = (-3, 3)
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N)


#  Solve the BRS (Coordinate Descent)
#   ============

# solution, run_stats = Hopf_BRS(system, target, intH_ytc17, T; opt_method=Hopf_cd,
#                                 preH=preH_ytc17, # Hamiltonian precomputation, output to Hamiltonian integrator intH
#                                 th,
#                                 grid_p,
#                                 warm=false, # warm starts the optimizer => 10x speed, risky if H non-convex
#                                 check_all=true, # checks all points in grid (once); if false, just convolution of boundary
#                                 printing=true);
# B⁺T, ϕB⁺T = solution;

# plot = plot_BRS(T, B⁺T, ϕB⁺T; M, cres=0.1, interpolate=true);

# x0 = [-2.3, 2.6]; # test point for control
# scatter!([x0[1]], [x0[2]], color="purple", label="x₀")


#   Discretizations
#   ===============

Md, Cd, C2d = diagm(ones(nx)) + th * M, th * C, th * C2;


#   Grid for plotting Target and Field
#   ==================================

scale = 0.6
xig = collect(-3 : 0.35 : 3.) .+ 1e-7; lg = length(xig)
Xg = hcat(collect.(Iterators.product([xig for i in 1:2]...))...)[end:-1:1,:]
Fxg = M * Xg; Fn = Fxg ./ map(norm, eachcol(Fxg))'
Jxg = reshape(J(Xg, Ap, cp), length(xig), length(xig));


#   Compare Optimal Solution of 1-step
#   ==================================

x0 = [-2.3, 2.6]; steps = 1; 

## Controller Parameters
time_p = (0.05, 0.4, 2.0)
Hs, N_T = 10, 20;
Hg, its = 3, 3; 

## Compute Feedback

s = 1
uˢ_Hopf, dˢ_Hopf, Tˢ, ϕ, dϕdz = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, x0; preH=preH_ytc17, time_p, refines=3);
uˢH_MPCg, dˢH_MPCg = MPC_2P(x0; H=1, its=1);
uˢH_MPCs, dˢH_MPCs = MPC_stochastic(x0; H=3, N_T=10);

dˢ_MPCg = Hamiltonian_counter_optimal(uˢH_MPCg[:, 1]; p=dϕdz, t=Tˢ) ## true optimal p2 soln
dˢ_MPCs = Hamiltonian_counter_optimal(uˢH_MPCs[:, 1]; p=dϕdz, t=Tˢ)

x0_pair_strats = [(uˢ_Hopf, dˢ_Hopf), (uˢH_MPCg[:, 1], dˢH_MPCg[:, 1]), (uˢH_MPCg[:, 1], dˢ_MPCg), (uˢH_MPCs[:, 1], dˢ_MPCs)]
x0_pair_labels = ["Hopf", "MPCg", "dˢ | MPCg", "dˢ | MPCs"]

x0_p1_strats = [uˢH_MPCs[:, 1]]
x0_p1_labels = ["MPCs"]

plot_game(x0_pair_strats, x0_pair_labels; p=dϕdz, t=Tˢ, p1_strategies=x0_p1_strats, p1_labels=x0_p1_labels)


#   Compare Hopf Against Random Disturbance
#   =======================================

## Check Disturbance ∈ Constraint
steps = 35; N_T = 10
samples = steps * N_T; 

border = 0.7
u1g = collect(-border : 0.1 : border) .+ 1e-7; u2g = collect(-border : 0.1 : border) .+ 1e-7;
Ug = hcat(collect.(Iterators.product(u2g, u1g))...)[end:-1:1,:]
Jdg = reshape(J(Ug, inv(Q2), cp), length(u1g), length(u2g))
Dplot = contour(u1g, u2g, Jdg, color="black", colorbar=false, levels=-1e-5:1e-5:1e-5, title="D := {d' Q₂⁻¹ d ≤ 1}")

D_Rand = ellipsoid_sample(samples, Q2, 1.)
scatter!(Dplot, D_Rand[1,:], D_Rand[2,:], label="D_Rand Interior", color="blue", alpha=0.5)

D_Rand_surf = ellipsoid_sample(samples, Q2, 1., surface=true)
scatter!(Dplot, D_Rand_surf[1,:], D_Rand_surf[2,:], label="D_Rand Surface", color="red", alpha=0.5)

D_Rand = reshape(D_Rand, nd, steps, N_T);
D_Rand_surf = reshape(D_Rand_surf, nd, steps, N_T);

display(Dplot)

## Controller Parameters
time_p = (0.05, 0.4, 2.0); refines=3

## Store Trajectories
X_Hopf, X_HopR = zeros(nx, steps+1), zeros(nx, steps+1, N_T)
U_Hopf, U_HopR = zeros(nx, steps), zeros(nx, steps, N_T)
D_Hopf, D_HopR = zeros(nx, steps), zeros(nx, steps, N_T)
X_Hopf[:, 1] = x0; X_HopR[:, 1, :] .= x0;

for s in 1:steps

    println(" on step $s ...")

    ## Compute Feedback
    U_Hopf[:, s], D_Hopf[:, s], Tˢ, ϕ, dϕdz = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, X_Hopf[:, s]; preH=preH_ytc17, time_p, refines);
    X_Hopf[:, s + 1] = Md * X_Hopf[:, s] + Cd * U_Hopf[:, s] + C2d * D_Hopf[:, s]

    dϕdzR1, TˢR1 = 0, 0;
    for nt = 1:N_T
        U_HopR[:, s, nt], D_HopR[:, s, nt], TˢR, ϕR, dϕdzR = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, X_HopR[:, s, nt]; preH=preH_ytc17, time_p, refines);
        X_HopR[:, s + 1, nt] = Md * X_HopR[:, s, nt] + Cd * U_HopR[:, s, nt] + C2d * D_Rand[:, s, nt]
        if nt == 1; dϕdzR1, TˢR1 = dϕdzR, TˢR; end
    end
    
    ## Plot Game at First Step
    if s == 1
        pair_strats = cat([(U_HopR[:, s, 1], D_HopR[:, s, 1])], [(U_HopR[:, s, 1], D_Rand[:, s, nt]) for nt=1:N_T], dims=1); 
        pair_labels = cat(["Hopf (uˢ, dˢ)", "Hopf (uˢ, d_Rand)"], ["Hopf (uˢ, d_Rand)" for _=1:N_T-1], dims=1);
        p1_strategies = [U_HopR[:, s, 1]]; p1_labels = ["Hopf uˢ"];
        p = plot_game(pair_strats, pair_labels; p=dϕdzR1, t=TˢR1, p1_strategies, p1_labels, title="Hopf (uˢ,dˢ) vs. random, x=$x0")
        display(p)
    end

end

## Plot Sim
plot_sim_1 = quiver(Xg[1,:], Xg[2,:], quiver = 0.75 * scale .* (Fn[1,:],  Fn[2,:]), label="f", color="blue", alpha=0.1) # plot field
contour!(plot_sim_1, xig, xig, Jxg, levels=-1e-3:1e-3:1e-3, color="black", colorbar=false) # plot target

plot!(plot_sim_1, X_Hopf[1,:], X_Hopf[2,:], label="Hopf", color="green")
for nt=1:N_T; lab = nt == 1 ? "Hopf w/ Random" : ""; plot!(plot_sim_1, X_HopR[1,:,nt], X_HopR[2,:,nt], label=lab, color="blue"); end

scatter!(plot_sim_1, [x0[1]], [x0[2]], color="purple", label="x_0", title="Hopf: Safe v Rand", legend=:bottomleft)


#   Compare Controllers (with a random disturbance)
#   ===============================================

x0 = [-2.3, 2.6]; steps = 30; 

## Controller Parameters
time_p = (0.05, 0.4, 2.0)
Hs, N_T = 10, 20;
Hg, its = 3, 3; 

## Store Trajectories
X_Hopf, X_MPCs, X_MPCg = zeros(nx, steps+1), zeros(nx, steps+1), zeros(nx, steps+1)
U_Hopf, U_MPCs, U_MPCg = zeros(nx, steps), zeros(nx, steps), zeros(nx, steps)
D_Hopf, D_MPCs, D_MPCg = zeros(nx, steps), zeros(nx, steps), zeros(nx, steps) # keep same random disturbances from before
X_Hopf[:, 1], X_MPCs[:, 1], X_MPCg[:, 1] = x0, x0, x0

for s in 1:steps

    println(" on step $s ...")

    ## Compute Feedback
    U_Hopf[:, s], D_Hopf[:, s], Tˢ, ϕ, dϕdz = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, X_Hopf[:, s]; preH=preH_ytc17, time_p);
    uˢH_MPCs, dˢH_MPCs = MPC_stochastic(X_MPCs[:, s]; H=3, N_T=10);
    uˢH_MPCg, dˢH_MPCg = MPC_2P(X_MPCg[:, s]; H=1, its=1);

    U_MPCs[:, s], D_MPCs[:, s] = uˢH_MPCs[:,1], D_Rand[:, s, 1]
    U_MPCg[:, s], D_MPCg[:, s] = uˢH_MPCg[:,1], dˢH_MPCg[:,1]
    
    ## Evolve State (each playing against random disturbance)
    X_Hopf[:, s + 1] = Md * X_Hopf[:, s] + Cd * U_Hopf[:, s] + C2d * D_Rand[:, s, 1]
    X_MPCs[:, s + 1] = Md * X_MPCs[:, s] + Cd * U_MPCs[:, s] + C2d * D_Rand[:, s, 1]
    X_MPCg[:, s + 1] = Md * X_MPCg[:, s] + Cd * U_MPCg[:, s] + C2d * D_Rand[:, s, 1]
    
end

## Plot Sim
plot_sim_2 = quiver(Xg[1,:], Xg[2,:], quiver = 0.75 * scale .* (Fn[1,:],  Fn[2,:]), label="f", color="blue", alpha=0.1)
contour!(plot_sim_2, xig, xig, Jxg, levels=-1e-3:1e-3:1e-3, color="black", colorbar=false)

plot!(plot_sim_2, X_Hopf[1,:], X_Hopf[2,:], label="Hopf", color="green")
plot!(plot_sim_2, X_MPCs[1,:], X_MPCs[2,:], label="MPCs", color="blue")
plot!(plot_sim_2, X_MPCg[1,:], X_MPCg[2,:], label="MPCg", color="red")

scatter!(plot_sim_2, [x0[1]], [x0[2]], color="purple", label="x_0", title="Compare, random d", legend=:bottomleft)


#   Compare Controllers (against worst disturbance)
#   ===============================================

x0 = [-2.3, 2.6]; steps = 35; 

## Controller Parameters
time_p = (0.05, 0.4, 2.0)
Hs, N_T = 10, 20;
Hg, its = 3, 3; 

## Store Trajectories
X_Hopf, X_MPCs, X_MPCg = zeros(nx, steps+1), zeros(nx, steps+1), zeros(nx, steps+1)
U_Hopf, U_MPCs, U_MPCg = zeros(nx, steps), zeros(nx, steps), zeros(nx, steps)
D_Hopf, D_MPCs, D_MPCg = zeros(nx, steps), zeros(nx, steps), zeros(nx, steps)
X_Hopf[:, 1], X_MPCs[:, 1], X_MPCg[:, 1] = x0, x0, x0;

for s in 1:steps

    println(" on step $s ...")

    ## Compute Feedback
    U_Hopf[:, s], D_Hopf[:, s], Tˢ, ϕ, dϕdz = Hopf_minT(system, target, intH_ytc17, HJoc_ytc17, X_Hopf[:, s]; preH=preH_ytc17, time_p);
    uˢH_MPCs, _ = MPC_stochastic(X_MPCs[:, s]; H=3, N_T=10);
    uˢH_MPCg, _ = MPC_2P(X_MPCg[:, s]; H=1, its=1);

    U_MPCs[:, s], U_MPCg[:, s] = uˢH_MPCs[:,1], uˢH_MPCg[:,1]
    # D_MPCs[:, s], D_MPCg[:,s] = Hamiltonian_counter_optimal(U_MPCs[:, s]; p=dϕdz, t=Tˢ), Hamiltonian_counter_optimal(U_MPCg[:, s]; p=dϕdz, t=Tˢ)
    
    ## Evolve State (each playing against their own)
    X_Hopf[:, s + 1] = Md * X_Hopf[:, s] + Cd * U_Hopf[:, s] + C2d * D_Hopf[:, s]
    X_MPCs[:, s + 1] = Md * X_MPCs[:, s] + Cd * U_MPCs[:, s] + C2d * D_Hopf[:, s]
    X_MPCg[:, s + 1] = Md * X_MPCg[:, s] + Cd * U_MPCg[:, s] + C2d * D_Hopf[:, s]
    
end

## Plot Sim
plot_sim_3 = quiver(Xg[1,:], Xg[2,:], quiver = 0.75 * scale .* (Fn[1,:],  Fn[2,:]), label="f", color="blue", alpha=0.1)
contour!(plot_sim_3, xig, xig, Jxg, levels=-1e-3:1e-3:1e-3, color="black", colorbar=false)

plot!(plot_sim_3, X_Hopf[1,:], X_Hopf[2,:], label="Hopf", color="green")
plot!(plot_sim_3, X_MPCs[1,:], X_MPCs[2,:], label="MPCs", color="blue")
plot!(plot_sim_3, X_MPCg[1,:], X_MPCg[2,:], label="MPCg", color="red")

scatter!(plot_sim_3, [x0[1]], [x0[2]], color="purple", label="x_0", title="Compare, worst d", legend=:bottomleft)


#   Utility Fn
#   ==========

function MPC_2P(x0; H=1, its=1, input_constraint="coupled", solver=Gurobi.Optimizer)
    """
    Computes the optimal control over horizon H given a naive optimal disturbance over horizon H. No proof of convergence. 
    For simplicity, use H = 1 & its = 1 to just compute optimal control for 1-step worst disturbance.
    """

    nx, nu, nd = size(M)[2], size(C)[2], size(C2)[2]
    uˢ, dˢ = zeros(nu, H), zeros(nu, H)

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
                    for i in 1:nd, j in 1:H; fix(u[i, j], uˢ[i, j]); end
                elseif player == "u"
                    for i in 1:nd, j in 1:H; fix(d[i, j], dˢ[i, j]); end
                end
            else
                if player == "d"
                    for i in 1:nd, j in 1:H; fix(u[i, j], 0.); end
                elseif player == "u"
                    for i in 1:nd, j in 1:H; fix(d[i, j], dˢ[i, j]); end
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

            uˢ, dˢ = value.(u), value.(d)
            
        end
    end

    return uˢ, dˢ
end

function MPC_stochastic(x0; H=3, N_T=10, input_constraint="coupled", solver=Gurobi.Optimizer)
    """
    Computes the optimal control over horizon H given N_T sample trajectories of disturbance.
    """

    nx, nu, nd = size(M)[2], size(C)[2], size(C2)[2]
    uˢ, dˢ = zeros(nu, H), zeros(nu, H)
        
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

    uˢ, dˢ = value.(u), Ds[:, :, 1]

    return uˢ, dˢ
end

function ellipsoid_sample(N_s, A, r; n=size(A)[1], surface=false)
    """
    Sample the ellipsoid defined by x'inv(A)x ≤ r^2, by sampling the spheroid and mapping to the ellipsoid w inv(cholesky(inv(A))).
    # http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf (interior)
    # https://www.maxturgeon.ca/f19-stat4690/slides/quadratic_forms_ellipses.pdf (surface)
    """
    B = !surface ? randn((n+1, N_s)) : randn((n, N_s))
    Bn = B ./ map(norm, eachcol(B))'
    return inv(cholesky(inv(A)).U) * r * Bn[1:n, :] 
end

# function Hamiltonian_counter_optimal(uˢ; p, t, solver=Gurobi.Optimizer)
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

function plot_game(pair_strategies, pair_labels; p, t, p1_strategies=[], p1_labels=[], title=nothing)
    """
    Plots the value of the 2D, linear TV differential game with coupled controls
    """
    
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

    for (psi, (u, d)) in enumerate(pair_strategies)
        θud = reshape(vcat(Input2θ(u, Q), Input2θ(d, Q2)), 2, 1)
        scatter!(θud[1, :], θud[2, :], [Hz(θud; pp=p, tt=t)], label=pair_labels[psi])
    end

    for (psi, (u)) in enumerate(p1_strategies)
        θud = vcat(Input2θ(u, Q) * (zero(θu) .+ 1)', θd')
        plot!(θud[1, :], θud[2, :], Hz(θud; pp=p, tt=t)', label=p1_labels[psi])
    end

    return game_plot
end