
using LinearAlgebra, JLD2
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_minT, Hopf_admm_cd
using TickTock, Suppressor
using ReachabilityAnalysis, Polyhedra
using DifferentialEquations
using Plots, LaTeXStrings
using JuMP, Gurobi
import MathOptInterface as MOI

## Problem Function

# Convenience function for entire algorithm (Feasibility, Error Comp, Linearization, Hopf Solving)
function solve_MDavoid_hopfconlin(; n_ag = 3,                       # number of agents
                                    r = 0.5, r_follow = 5.,         # max distance of ego & aux agents to evader for capture
                                    θ_max = 4π,                     # theta of capture (should be inf but that blows up)
                                    v_b = 3.,                       # pursuer velocity
                                    v_a = 3.,                       # evader velocity
                                    max_u = 1., max_d = 1.,         # ctrl of pursuer & evader
                                    x_init = 3.,                    # l2 distance of pursuers from evader initially
                                    θ = 0π/(2*3),                   # evader initial heading wrt pursuers
                                    opt_p_admm_cd = (               # hopf optimization params
                                        (1e-0, 1e-0, 1e-5, 3), 
                                        (0.01, 1e2, 1e-4, 500, 3, 9, 1000), 
                                        1, 1, 3),
                                    th = 0.025,                     # hopf small time step
                                    Th = 0.05,                      # hopf mid time step
                                    Tf = nothing,                   # final time for horizon/simulation, will be set unless given
                                    run_name = "Run1_10b10b10",     # output data directory
                                    name = "test_$(rand(Int))",     # output data name
                                    saving = true,
                                    printing = false,
                                    feasibility_only = false,
                                    control_simulation = false,
                                    warm = nothing, time_p = nothing, refines = nothing, one_shot = true,
                                    pursuer_ctrl = "short",
                                    H=20, N_T=20, # for pursuer_ctrl = "mpc"
                                    plotting = false, plot_size=(375,375),
                                    )

    # solve_MDavoid_hopfconlin(saving=false, printing=true) # for testing
    # solve_MDavoid_hopfconlin(saving=false, printing=true, feasability_only=true) # for testing feasibility only
    # solve_MDavoid_hopfconlin(saving=false, printing=true, control_simulation=true) # for testing control simulation
    
    @suppress begin; tick(); end
    time_segments = Float64[];

    if printing; println("\n Solving the Hopf-Error, Multi-Dubins Problem with $n_ag Agents"); end
    if printing && feasibility_only; println("  (Feasibility Check only)\n"); end
    if printing && control_simulation; println("  (Running Control Simulation)\n"); end

    ## Static Parameters

    dim_x = 3 # agent dimension
    dim_xh = n_ag * dim_x # system dimension
    Tf = isnothing(Tf) ? (x_init / max(v_a, v_b)) + 0.01 : Tf;
    T = collect(Th : Th : Tf) # hopf time array
    p_static = (n_ag, dim_x, dim_xh, v_a, v_b, max_u, max_d, r, r_follow, θ_max)

    ## Export Prep
    if saving
        full_name = pwd() * "/Zonotoping/MultiDubins_Data/$run_name/$name.jld2";
        parameters = Dict(
            "n_ag"=>n_ag, "r"=>r, "r_follow"=>r_follow, "θ_max"=>θ_max, "v_b"=>v_b, "v_a"=>v_a, 
            "max_u"=>max_u, "max_d"=>max_d, "x_init"=>x_init, "θ"=>θ, "Tf"=>Tf,
        );
    end
    hopf_parameters = (th, Th, opt_p_admm_cd);
    pl = nothing; val_pl = nothing;
    # pal_pursuer, pal_evader = n_ag == 3 ? palette(:Reds, n_ag+5)[3:2:end] : palette(:Reds, n_ag+8)[4:2:end], :blue
    pal_pursuer, pal_evader = palette(:seaborn_colorblind), :white

    ## Initial Point

    rot(θ, p) = [cos(θ) -sin(θ); sin(θ) cos(θ)] * p
    polygon = [rot((2π/n_ag)*(i-1) + θ, [0., x_init]) for i=1:n_ag]
    xh_init = vcat([vcat(x..., -π + (i-1)*(2π/n_ag) + π/2 + θ) for (i, x) in enumerate(polygon)]...)
    xh_init = round.(xh_init, digits=12)    

    dim_xp = 2 # pos dim
    Bi = [0; 0; 1] # pursuer i (player bi)
    Qc = [max_u] # pursuer control lim
    Qd = [max_d] # evader control lim

    Wi = diagm(vcat(fill(r^2, dim_xp), fill(θ_max^2, 1))) # per-agent target set params (ego : follower)
    Wi_f = diagm(vcat(fill(r_follow^2, dim_xp), fill(θ_max^2, 1)))
    Wi_mpc = diagm(vcat(fill(r^2, dim_xp), fill(20, 1)))

    ## Combined System 

    Bh = zeros(dim_xh, n_ag);
    Qch = Qc .* I(n_ag) # 1d ctrl per agent
    Qdh = Qd .* I(1) # 1d ctrl per agent
    qc, qd = zeros(1, n_ag), [0.] #center of input constraints

    Whs = [zeros(dim_xh, dim_xh) for _ in 1:n_ag];
    Whs_mpc = [zeros(dim_xh, dim_xh) for _ in 1:n_ag]; 
    qh = zeros(dim_xh);

    for i = 1:n_ag
        Bh[dim_x * (i - 1) + 1 : dim_x * i, i] = Bi
        qh[dim_x * (i - 1) + 1 : dim_x * i] = [0; 0; xh_init[(i-1)*dim_x + 3]]

        for j=1:n_ag
            Whs[i][dim_x * (j - 1) + 1 : dim_x * j, dim_x * (j - 1) + 1 : dim_x * j] = i == j ? Wi : Wi_f
            Whs_mpc[i][dim_x * (j - 1) + 1 : dim_x * j, dim_x * (j - 1) + 1 : dim_x * j] = i == j ? Wi_mpc : 10*Wi_f
        end
    end

    ## Elliptical Target: J(x) = 0 is the boundary of the target

    Js(v::Vector, Q, q) = (v' * Q * v)/2 + q'v + 0.5
    J(x::Matrix, Q, q) = diag((x .- q)' * inv(Q) * (x .- q))/2 .- 0.5
    targets = [(J, Js, (Whs[i], qh)) for i=1:n_ag]
    targets_mpc = [(J, Js, (Whs_mpc[i], qh)) for i=1:n_ag]

    ## Solve Multi-Agent Dubins in Relative State Space 
    
    δt = 1e-2; δts = 0.:δt:(Tf+Th);

    @suppress begin; tick(); end 
    sol_de = DifferentialEquations.solve(ODEProblem(MultiDubins!, xh_init, (0., (Tf+Th)-1e-3)), p=(x,s)->(p_static, zeros(n_ag), zeros(1)), saveat=δts)
    push!(time_segments, tok())
    x̃ = s -> vcat(sol_de((Tf+Th) + s), 0.0)

    ## Define Feasibility Problem and Solve

    xi = copy(xh_init)
    X0is = [Singleton(xi[ci+1:ci+dim_x]) for ci=0:dim_x:dim_x*(n_ag-1)]
    U = Hyperrectangle(; low=-Qc, high=Qc)
    sys_is_nob = [InitialValueProblem(BlackBoxContinuousSystem((dx,x,p,t) -> SingleDubins_RA_nob!(dx,x,p_static,t), dim_x+1), X0is[ai] × U) for ai=1:n_ag];
    
    @suppress begin; tick(); end
    solz_is_nob = Dict()
   # Threads.@threads for ai=1:n_ag;
    for ai=1:n_ag;
        solz_is_nob[ai] = overapproximate(ReachabilityAnalysis.solve(sys_is_nob[ai]; tspan=(0.0, (Tf+Th)), alg=TMJets21a(; orderQ=6, abstol=1e-15)), Zonotope) # per agent parallel computations
    end
    push!(time_segments, tok())

    if feasibility_only

        ## Solve Feasible Sets with Evader Action
        D = Hyperrectangle(; low=-Qd, high=Qd)
        sys_is = [InitialValueProblem(BlackBoxContinuousSystem((dx,x,p,t) -> SingleDubins_RA!(dx,x,p_static,t), dim_x+2), X0is[ai] × U × D) for ai=1:n_ag];
        solz_is = Dict(); for ai=1:n_ag; solz_is[ai] = overapproximate(ReachabilityAnalysis.solve(sys_is[ai]; tspan=(0.0, (Tf+Th)), alg=TMJets21a(; orderQ=6, abstol=1e-15)), Zonotope); end

        ## Check if Feasible via Projection
        θi = 0.:2π/10:2π; target_disc = hcat(vcat(r*cos.(θi)', r*sin.(θi)'), zeros(2))
        dt_is, feas_dt_is = [], []

        @suppress begin; tick(); end 
        for ai=1:n_ag
            proj = ReachabilityAnalysis.project(solz_is[ai], vars=1:2)
            dt = high.(tspan.(solz_is[ai]))
            feas_dt = [!isnothing(findfirst(x -> ReachabilityAnalysis.in(x, proj[j]), eachcol(target_disc))) for j=1:length(proj)] # check discritized circle in agent feasible sets

            push!(dt_is, dt)
            push!(feas_dt_is, feas_dt)

            if printing
                print_feas = isnothing(findfirst(x->x>0, feas_dt)) ? "infeasible" : "feasible, first at T=$(dt[findfirst(x->x>0, feas_dt)])"
                println("  Capture for Agent $ai is $print_feas")
            end
        end
        push!(time_segments, tok())

        if plotting
            pl = plot_feasible(solz_is, xh_init; p_static, pal_pursuer, pal_evader, plot_size)
            display(pl)
        end
            
        total_time = tok();
        if saving; jldsave(full_name; parameters, dt_is, feas_dt_is, total_time, time_segments, pl); end
        if printing; 
            print_feas = sum(vcat(feas_dt_is...)) > 0 ? "feasible" : "infeasible"
            println("\n Team capture is $print_feas \n")
        end

        return total_time
    end

    ### Compute Error over time

    x̃i(i, s) = vcat(x̃(s)[(i-1)*dim_x+1:i*dim_x], 0.0)

    @suppress begin; tick(); end 
    δˢ_nob_fasters = [zeros(dim_x, length(solz_is_nob[ai])) for ai=1:n_ag]; δˢ_nob_tighters = [zeros(dim_x, length(solz_is_nob[ai])) for ai=1:n_ag];
    for ai=1:n_ag
        for ti=1:length(solz_is_nob[ai])
            δˢ_nob_fasters[ai][:, ti] = TSerror_Inf_SingleDubins(x̃i(ai, -(Tf+Th)+low(tspan(solz_is_nob[ai][ti]))), set(solz_is_nob[ai][ti]); p_static); 
        end
        for ti=1:length(solz_is_nob[ai])
            δˢ_nob_tighters[ai][:, ti] = TSerror_SingleDubins_Tight(x̃i(ai, -(Tf+Th)+low(tspan(solz_is_nob[ai][ti]))), set(solz_is_nob[ai][ti]); p_static); 
        end
    end
    δˢ_nob_mins = copy(δˢ_nob_tighters); for i=1:n_ag; δˢ_nob_mins[i] = min.(δˢ_nob_tighters[i], δˢ_nob_fasters[i]); end
    push!(time_segments, tok()/3)

    ## Error Functions for Hopf

    δˢt_nob_mins(s) = δˢ_combine(s, δˢ_nob_mins, solz_is_nob; p_static);
    Eδ_nob_mins(s) = diagm(δˢt_nob_mins(s))

    ## Linearize System

    system_x̃_err_min = (s -> Ãh(sol_de(s); p_static), s -> C̃h(sol_de(s); p_static), Bh, Qdh, qd, Qch, qc, s -> c̃h(sol_de(s), s; p_static), Eδ_nob_mins);
     
    ## Solve

    if !(control_simulation)

        solns = Dict(); t_sols = [];
        # Threads.@threads for i=1:n_ag
        for i=1:n_ag
            @suppress begin; tick(); end 
            solns[i], rstats = Hopf_BRS(system_x̃_err_min, targets[i], T; Xg=[xh_init;;], th, game="avoid", opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, error=true, printing, warm=false, opt_tracking=false);
            push!(t_sols, tok())
        end

        push!(time_segments, maximum(t_sols))
    end

    ## Optimal Control Simulation

    if control_simulation
        # include(joinpath(pwd(), "Comparison/Control_Comparison_fn.jl"))

        @suppress begin; tick(); end 
        warm = isnothing(warm) ? true : warm
        time_p = isnothing(time_p) ? (th, Th, Tf) : time_p
        refines = isnothing(refines) ? 2 : refines
        sim_targets = pursuer_ctrl == "mpc" ? targets_mpc : targets
        Xr, Xnr, oc_times_e, oc_times_p, sim_avoided, final_ix = simulate(system_x̃_err_min, sim_targets, xh_init, Tf; p_static, 
                                                            Δt=Th, time_p, warm, opt_p=opt_p_admm_cd, pursuer_ctrl,
                                                            refines, printing=printing, F=F_Dubins, one_shot)
        push!(time_segments, tok())

        val_pl = plotting ? plot_value(Xr, targets, T; plot_size, pal_pursuer) : val_pl
        if plotting; display(val_pl); end

        pl = plotting ? plot_reldubins_tp(Xnr, final_ix; p_static, Δt=Th, plot_size,
                                        pal_pursuer, pal_evader,
                                        lw_tail=2, tail_l=size(Xnr,2), relative=false) : pl

        if plotting; display(pl); end

        total_time = tok();
        if saving; jldsave(full_name; parameters, hopf_parameters, Xr, Xnr, oc_times_e, oc_times_p, sim_avoided, pl, val_pl, total_time, time_segments); end

        return total_time
    end

    ## SAVE 
    error = δˢt_nob_mins.(T .- 3e-3);
    values = [vcat(solns[i][2]...) for i=1:n_ag];
    total_time = tok();

    if saving; jldsave(full_name; parameters, hopf_parameters, T, error, values, total_time, time_segments); end
    if printing
        println("Name            : $name")
        println("Max Error       : $(maximum(error))")
        println("Min Value       : $(minimum(minimum.(values)))")
        println("Time Segments   : $time_segments")
        println("Total Time      : $total_time \n")
    end

    return total_time
end

### Auxiliary Functions 

eix(i; dim=dim_xh+1) = vcat(zeros(i-1), 1, zeros(dim-i)) # one hot for supports

function MultiDubins!(dx, x, p, s)
    # Frame of Evader
    p_static, us, ds = p(x, s)
    n_ag, dim_x, _, v_a, v_b = p_static
    
    for (ci, c) in enumerate(0:dim_x:dim_x*(n_ag-1))
        dx[c+1] = -v_a + v_b * cos(x[c+3]) + ds[1] * x[c+2] # x_i_Δ,1
        dx[c+2] = v_b * sin(x[c+3]) - ds[1] * x[c+1] # x_i_Δ,2
        dx[c+3] = us[ci] - ds[1] # θ_i_Δ
    end
    return dx
end

function SingleDubins_RA_nob!(dx, x, p, s) # assumes trivial evader action
    _, _, _, v_a, v_b = p
    dx[1] = -v_a + v_b * cos(x[3]) # x_i_{Δ,1}
    dx[2] = v_b * sin(x[3]) # x_i_{Δ,2}
    dx[3] = x[4] # θ_i_Δ
    dx[4] = zero(x[1]) # action of pursuer
    return dx
end

function SingleDubins_RA!(dx, x, p, s)
    # Frame of Evader
    _, _, _, v_a, v_b = p
    dx[1] = -v_a + v_b * cos(x[3]) + x[5] * x[2] # x_i_Δ,1
    dx[2] = v_b * sin(x[3]) - x[5] * x[1] # x_i_Δ,2
    dx[3] = x[4] - x[5] # θ_i_Δ
    dx[4] = zero(x[1]) # action of pursuer (ui)
    dx[5] = zero(x[1]) # action of evader (d) 
    return dx
end

function TSerror_Inf_SingleDubins(xl, shape; p_static)

    _, dim_x, _, _, v_b = p_static
    dim = dim_x+1; δˢ = zeros(dim-1)

    xi = 0
    max_norm_agent = 0
    for i in [2,3], pm in [1, -1]; max_norm_agent = max(max_norm_agent, abs(pm * ρ(pm * eix(xi + i; dim=dim), shape) - xl[xi + i])); end
    for pm in [1, -1]; max_norm_agent = max(max_norm_agent, abs(pm * ρ(pm * eix(dim; dim=dim), shape) - xl[dim])); end

    θi_hi, θi_lo = ρ(eix(xi + 3; dim=dim), shape), -ρ(-eix(xi + 3; dim=dim), shape)
    if abs(θi_hi - θi_lo) ≥ π/2
        max_sinξθi, max_cosξθi = 1, 1
    elseif sign(cos(θi_hi)) != sign(cos(θi_lo)) || sign(sin(θi_hi)) != sign(sin(θi_lo))
        max_sinξθi = sign(cos(θi_hi)) == sign(cos(θi_lo)) ? max(abs(sin(θi_hi)), abs(sin(θi_lo))) : 1.
        max_cosξθi = sign(sin(θi_hi)) == sign(sin(θi_lo)) ? max(abs(cos(θi_hi)), abs(cos(θi_lo))) : 1.
    else
        max_cosξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
        max_sinξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
    end

    δˢ[xi + 1] = 0.5 * max(abs(v_b * max_cosξθi), 1) * max_norm_agent^2
    δˢ[xi + 2] = 0.5 * max(abs(v_b * max_sinξθi), 1) * max_norm_agent^2
    return δˢ
end

function TSerror_SingleDubins_Tight(xl, shape; p_static)

    _, dim_x, _, _, v_b, _, max_d = p_static
    dim = dim_x+1; δˢ = zeros(dim-1)
    xi = 0;

    θi_hi, θi_lo = ρ(eix(xi + 3; dim=dim), shape), -ρ(-eix(xi + 3; dim=dim), shape)
    if abs(θi_hi - θi_lo) ≥ π/2
        max_sinξθi, max_cosξθi = 1, 1
    elseif sign(cos(θi_hi)) != sign(cos(θi_lo)) || sign(sin(θi_hi)) != sign(sin(θi_lo))
        max_sinξθi = sign(cos(θi_hi)) == sign(cos(θi_lo)) ? max(abs(sin(θi_hi)), abs(sin(θi_lo))) : 1.
        max_cosξθi = sign(sin(θi_hi)) == sign(sin(θi_lo)) ? max(abs(cos(θi_hi)), abs(cos(θi_lo))) : 1.
    else
        max_cosξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
        max_sinξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
    end

    max_d_diff = max(max_d-xl[dim], xl[dim]+max_d); max_d_diff2 = max_d_diff^2;
    max_ε1v, max_ε2v = 0., 0.
    for v in vertices_list(shape)
        ε1v = 0.5 * (2*abs(v[xi+2]-xl[xi+2])*max_d_diff2 + v_b*max_cosξθi*(v[xi+3]-xl[xi+3])^2)
        ε2v = 0.5 * (2*abs(v[xi+2]-xl[xi+2])*max_d_diff2 + v_b*max_sinξθi*(v[xi+3]-xl[xi+3])^2)
        max_ε1v, max_ε2v = max(max_ε1v,ε1v), max(max_ε2v,ε2v)
    end

    δˢ[xi + 1] = max_ε1v; δˢ[xi + 2] = max_ε2v
    return δˢ
end

function δˢ_combine(s, δˢ_set, solz_set; p_static)
    n_ag, dim_x, dim_xh = p_static
    δˢ = zeros(dim_xh)
    for (ai, i) in enumerate(0:dim_x:dim_x*(n_ag-1))
        try
            δˢ[i+1:i+dim_x] = δˢ_set[ai][:, findfirst(x->x<0, s .- vcat(0., high.(tspan.(solz_set[ai]))))-1] 
        catch y
            δˢ[i+1:i+dim_x] = δˢ_set[ai][:, findfirst(x->x<0, s .- vcat(0., high.(tspan.(solz_set[ai]))))]
        end
    end
    return δˢ
end

function Ãh(x̃; p_static)
    n_ag, dim_x, dim_xh, _, v_b = p_static
    Ai = [0 x̃[4] -v_b*sin(x̃[3]); -x̃[4] 0. v_b*cos(x̃[3]); 0 0 0]; Ah = zeros(dim_xh, dim_xh);
    for i = 1:n_ag; Ah[dim_x * (i - 1) + 1 : dim_x * i, dim_x * (i - 1) + 1 : dim_x * i] = Ai; end
    return Ah
end

function C̃h(x̃; p_static)
    n_ag, dim_x, dim_xh = p_static
    Ci = [x̃[2]; -x̃[1]; -1]; Ch = zeros(dim_xh, 1);
    for i = 1:n_ag; Ch[dim_x * (i - 1) + 1 : dim_x * i, 1] = Ci; end
    return Ch
end

function c̃h(x̃, s; p_static)
    n_ag, _ = p_static
    return MultiDubins!(zero(x̃), x̃, (x,s)->(p_static, zeros(n_ag), [0.]), s) - Ãh(x̃; p_static) * x̃;
end

## Control Functions

function MPC_stochastic(system, target, x0; p_static, th=5e-2, H=3, N_T=10, input_constraint="box", solver=Gurobi.Optimizer)

    n_ag, dim_x, dim_xh = p_static
    M, C, C2, Q, a, Q2, a2, c, E = system
    Md, Cd, C2d, cd, Ed  = s -> (diagm(ones(dim_xh)) + th * M(s)), th * C, s -> th * C2(s), s-> th * c(s), s -> th * E(s);
    J, _, Jp = target
    Ap, cp = Jp

    nx, nu, nd = dim_xh, n_ag, 1 # translate from other code
    
    model = nothing
    @suppress begin
        model = Model(solver); set_silent(model);
    end 
    # set_optimizer_attribute(model, "NonConvex", 2);
    
    @variable(model, x[1:nx, 1:(H+1), 1:N_T]);
    @variable(model, u[1:nu, 1:H])
    
    ## Generate Random Disturbance
    if input_constraint == "coupled"
        # Ds = reshape(ellipsoid_sample(H * N_T, inv(Q2), 1.), nd, H, N_T) 
    elseif input_constraint == "box"
        Ds = reshape((rand(nd, H * N_T) .- 0.5) .* Q2, nd, H, N_T)
        Es = reshape((rand(nx, H * N_T) .- 0.5), nx, H, N_T)
    end

    ## Fix Initial Condition
    for i in 1:nx, k in 1:N_T; fix(x[i, 1, k], x0[i]); end

    ## Constraints
    for h in 1:H 
        
        ## Dynamics
        for k in 1:N_T
            # println("Md", size(Md(h * th)))
            # println("Cd", size(Cd))
            # println("C2d", size(C2d(h * th)))
            # println("cd", size(cd(h * th)))
            @constraint(model, x[:, h + 1, k] .== Md(h * th) * x[:, h, k] + Cd * u[:, h] + C2d(h * th) * Ds[:, h, k] + cd(h * th) + Ed(h * th) * Es[:, h, k])
        end

        ## Control
        if input_constraint == "coupled"
            # @constraint(model, u[:, h]' * inv(Q) * u[:, h] ≤ 1)
        elseif input_constraint == "box"
            for i=1:nu
                @constraint(model, u[i, h] ≤ diag(Q)[i])
                @constraint(model, u[i, h] ≥ -diag(Q)[i])
            end
        end
    end

    @objective(model, Min, sum(0.5 * ((x[:, H+1, k] .- cp)' * inv(Ap) * (x[:, H+1, k] .- cp) - 1) for k in 1:N_T)/N_T);
    optimize!(model);

    uˢH = value.(u)

    return uˢH[:,1], Ds[:, 1, 1], objective_value(model)
end

function SingleDubins!(dx, x, p, s)
    v, u = p
    dx[1] = v * cos(x[3]) # x
    dx[2] = v * sin(x[3]) # y
    dx[3] = u # θ
    return dx
end

function F_Dubins(xh0, us, ds, Δt; p_static, δt=0.01, relative=true)

    n_ag, dim_x, _, v_a, v_b = p_static

    if relative
        sol_deq = DifferentialEquations.solve(ODEProblem(MultiDubins!, xh0, (0., Δt-1e-3)), p=(x,s)->(p_static, us, ds), saveat=0:δt:Δt-1e-3)
        xh = Array(sol_deq)[:,end]
    else
        xh = zeros(dim_x*(n_ag+1))
        for (ai, i) in enumerate(0:dim_x:dim_x*(n_ag-1))
            sol_deqi = DifferentialEquations.solve(ODEProblem(SingleDubins!, xh0[i+1:i+dim_x], (0., Δt-1e-3)), p=(v_b, us[ai]), saveat=0:δt:Δt-1e-3) # pursuer model
            xh[i+1:i+dim_x] = Array(sol_deqi)[:,end]
        end
        sol_deqd = DifferentialEquations.solve(ODEProblem(SingleDubins!, xh0[dim_x*n_ag+1:dim_x*(n_ag+1)], (0., Δt-1e-3)), p=(v_a, ds[1]), saveat=0:δt:Δt-1e-3); # evader model
        xh[dim_x*n_ag+1:dim_x*(n_ag+1)] = Array(sol_deqd)[:,end]
    end

    return xh
end

function simulate(system, targets, x0, tf; p_static, Δt=0.05, time_p = (0.025, 0.05, 1.0), time_p_short = (0.025, 0.05, 0.15), pursuer_ctrl="hopf_short", warm=true, v_init_s=nothing, refines=2, printing=false, F=F_Dubins, opt_p=nothing, H=20, N_T=20, one_shot=false)

    n_ag, dim_x, dim_xh = p_static; r = p_static[8]

    t = 0.:Δt:tf-Δt
    Xr, Xnr = Inf*ones(dim_xh, Int(floor(tf / Δt)) + 1), Inf*ones(dim_x*(n_ag+1), Int(floor(tf / Δt)) + 1)
    Xr[:, 1], Xnr[:, 1] = x0, vcat(x0, zeros(dim_x))
    tcs_p, tcs_e, ego_p, ego_e = zero(t), zero(t), Int.(zero(t)), Int.(zero(t)); # ego_e is leading pursuer from evader pov; may differ from ego_p when pursuer_ctrl not hopf
    avoided, final_ix = true, size(Xr, 2)
    fixed_policy = [s -> zeros(n_ag), s -> zeros(1)] # for one_shot only
    system_switch = (system[1], system[3], system[2], system[6], system[7], system[4], system[5], system[8], system[9]) # mpc needs reacher/pursuers first
    v_init_s = isnothing(v_init_s) ? 0.1*ones(dim_xh, n_ag) : v_init_s

    if printing; println(" Simulating...\n"); end
    for (ti, thi) in enumerate(t)

        ## Check if Reached or Escaped Early
        vals = vcat([targets[i][1](Xr[:,ti:ti], targets[i][3]...) for i=1:n_ag]...)
        if minimum(vals) ≤ 0
            avoided, final_ix = false, ti
            if printing; println("*** Agent $(argmin(vals)) CAPTURED target at t = $thi, (ti = $ti) ***"); end
            break
        end
        # if minimum(vals) >= 10 && thi >= tf/2
        #     avoided, final_ix = true, ti
        #     if printing; println("___ Target escaped at t = $thi, (ti = $ti) ___"); end
        #     break
        # end

        if printing; println("  solving at t=$thi \n"); end

        ### Compute Optimal Intputs

        us, ds, Ts, ϕs, dϕdzs, tci_p, tci_e = Inf*ones(n_ag, n_ag), Inf*ones(n_ag), Inf*ones(n_ag), Inf*ones(n_ag), Inf*ones(dim_xh, n_ag), Inf*ones(n_ag), Inf*ones(n_ag)
        uˢ, dˢ = nothing, nothing

        ## Control for Evader

        if !one_shot || ti == 1 # re-solved or initial solve
            for i=1:n_ag
                @suppress begin; tick(); end 
                dˢi, uˢi, Tˢi, ϕi, dϕdzi = Hopf_minT(system, targets[i], Xr[:, ti]; error=true, game="avoid", opt_method=Hopf_admm_cd, opt_p, time_p, printing=false, warm=true, v_init_=v_init_s[:, i], refines, return_policy=one_shot) # u and d switched since here we call pursuer control u but we control evader
                us[:,i], ds[i] = one_shot ? (uˢi(thi), dˢi(thi)[1]) : (uˢi, dˢi[1])
                Ts[i], ϕs[i], dϕdzs[:,i], tci_e[i] = Tˢi, ϕi, dϕdzi, tok()

                if Tˢi ≤ minimum(Ts)
                    if !one_shot || ϕi ≤ minimum(ϕs) # if all avodiable, take min value
                        ego_e[ti] = i
                        fixed_policy = one_shot ? (uˢi, dˢi) : fixed_policy
                    end
                end
            end

            dˢ = [ds[ego_e[ti]]] # evader oc (vs. argmin(ϕs)?)
            tcs_e[ti] = maximum(tci_e) # (parallizeable)

        else # one-shot policy computed at ti=1
            @suppress begin; tick(); end 
            dˢ = fixed_policy[2](thi)
            tcs_e[ti], ego_e[ti] = tok(), ego_e[ti-1]
        end

        pr_avoid_e = minimum(ϕs) < 0 ? "unavoidable" : "avoidable"
        pr_T_phi_e = "Tˢ=$(minimum(Ts)), ϕˢ=$(minimum(ϕs))"

        ## Control for Pursuer Team

        if pursuer_ctrl == "hopf" # same ctrl as evader
            tcs_p[ti], ego_p[ti] = tcs_e[ti], ego_e[ti]
            uˢ = one_shot && ti > 1 ? fixed_policy[1](thi) : us[:, ego_p[ti]]
        else
            # Ts, ϕs, dϕdzs = Inf*ones(n_ag), Inf*ones(n_ag), Inf*ones(dim_xh, n_ag)
            ϕs = Inf*ones(n_ag)
            
            for i=1:n_ag
                if pursuer_ctrl == "hopf_short"
                    @suppress begin; tick(); end 
                    _, uˢi, Tˢi, ϕi, _ = Hopf_minT(system, targets[i], Xr[:, ti]; error=true, game="avoid", opt_method=Hopf_admm_cd, opt_p, time_p=time_p_short, printing=false, warm, v_init_=v_init_s[:, i], refines) # u and d switched since here we call pursuer control u but we control evader
                    us[:,i], Ts[i], ϕs[i], tci_p[i] = uˢi, Tˢi, ϕi, tok()
                elseif pursuer_ctrl == "mpc"
                    @suppress begin; tick(); end 
                    uˢi, _, ϕi = MPC_stochastic(system_switch, targets[i], Xr[:, ti]; p_static, H, N_T)
                    us[:,i], ϕs[i], tci_p[i] = uˢi, ϕi, tok()
                end

                if (pursuer_ctrl == "hopf_short" && Tˢi ≤ minimum(Ts)) || (pursuer_ctrl == "mpc" && ϕi ≤ minimum(ϕs))
                    ego_p[ti] = i
                    tcs_p[i] = tci_p[i]
                end
            end

            uˢ = us[:, ego_p[ti]]
        end

        pr_avoid_p = minimum(ϕs) < 0 ? "unavoidable" : "avoidable"
        pr_T_phi_p = pursuer_ctrl != "mpc" ? "Tˢ=$(minimum(Ts)), ϕˢ=$(minimum(ϕs))" : "Tˢ=N/A, ϕˢ=$(minimum(ϕs))"

        if printing; println("   Evader believes   -  capture $pr_avoid_e ($pr_T_phi_e, ego=$(ego_e[ti]))"); end
        if printing; println("   Pursuers believe  -  capture $pr_avoid_p ($pr_T_phi_p, ego=$(ego_p[ti]))"); end
        if printing; println("   xr(t) : $(round.(Xr[:, ti], digits=3))"); end
        if printing; println("   uˢ(t) = $uˢ, dˢ(t) = $dˢ \n"); end

        # time_p = ... # TODO: needs to be time_p = (th, thi, tf) for TV systems, namely t needs a new/another init variable for faster minT

        ## Evolve
        Xr[:, ti+1]  = F(Xr[:, ti],  uˢ, dˢ, Δt; p_static, δt=0.01, relative=true)
        Xnr[:, ti+1] = F(Xnr[:, ti], uˢ, dˢ, Δt; p_static, δt=0.01, relative=false)

        v_init_s = warm ? dϕdzs : v_init_s
    end

    if avoided && printing; println("   ___ Target ESCAPED at t = $tf, (ti = $final_ix) ___\n"); end

    if printing; println("    Mean OC Computation Time Evader  : $(round(sum(tcs_e)/length(tcs_e), sigdigits=3)) s"); end
    if printing; println("    Mean OC Computation Time Pursuer : $(round(sum(tcs_p)/length(tcs_p), sigdigits=3)) s\n"); end
    
    return Xr, Xnr, tcs_e, tcs_p, avoided, final_ix
end

## Plotting

function plot_reldubins_tp(Xs, si; p_static, Δt=0.1, ylimz=(-5, 5), xlimz=(-5, 5), legend=:bottom, plot_size=(375,375),
                            markersize=20, alpha=0.7, alpha_tail=0.3, legend_columns=3,
                            legendfontsize=10, xtickfontsize=12, ytickfontsize=12,
                            lw_tail=2, tail_l=3, relative=true,  
                            pal_pursuer=palette(:seaborn_colorblind), pal_evader=:white,
                            # pal_pursuer=palette(:Reds, n_ag+4)[3:end-2], pal_evader=:blue,
                            )

    n_ag, dim_x, dim_xh, v_a, v_b, max_u, max_d, r, r_follow, θ_max = p_static
    gr(); pl = plot(size=plot_size,title= L"\textrm{Controlled\:\:Evolution},\:" * latexstring("t=$(round((si-1)*Δt, digits=2))"), dpi=300); 
    tail_color = pal_evader == :white ? :black : pal_evader

    # triangle = [(-sqrt(3)/6 - 0.5, 0.5), (-sqrt(3)/6 - 0.5, -0.5), (sqrt(3)/2 - sqrt(3)/6 + 0.5, 0.), (-sqrt(3)/6 - 0.5, 0.5)];
    triangle = [(-0.6, 0.45), (0.2, 0.45), (0.6, 0.), (0.2, -0.45), (-0.6, -0.45), (-0.6, 0.45)];
    rotate_marker(points, θ) = [[cos(θ) -sin(θ); sin(θ) cos(θ)] * [x; y] for (x,y) in points];
    rotate_triangle(θ) = [tuple(x...) for x in rotate_marker(triangle, θ)];

    for i=0:(n_ag-1); scatter!(pl, [Xs[1 + i*dim_x, si]], [Xs[2 + i*dim_x, si]], lw=2, color=pal_pursuer[i+1], alpha=alpha, marker=Shape(rotate_triangle(Xs[3 + i*dim_x, si])), markersize=markersize, label=L"Agent\:" * latexstring("$(i+1)")); end # 
    if si > 1
        for i=0:(n_ag-1); plot!(pl, Xs[1 + i*dim_x, si-min(si-1, tail_l):si], Xs[2 + i*dim_x, si-min(si-1, tail_l):si], label="", lw=lw_tail, color=pal_pursuer[i+1], alpha=alpha_tail); end
        if !relative; plot!(pl, Xs[1 + n_ag*dim_x, si-min(si-1, tail_l):si], Xs[2 + n_ag*dim_x, si-min(si-1, tail_l):si], label="", lw=lw_tail, color=tail_color, alpha=alpha_tail); end
    end
    plot!(pl, ylims=ylimz, xlims=xlimz, aspect_ratio=:equal)
    # θi=0.:0.01:2π; plot!([r * cos.(θi) .+ Xs[1 + n_ag*dim_x, si]], [r * sin.(θi) .+ Xs[2 + n_ag*dim_x, si]], color="black", lw=1.5, label="") # , label=L"𝒯"
    # θi=0.:0.01:2π; plot!([r_follow * cos.(θi) .+ Xs[1 + n_ag*dim_x, si]], [r_follow * sin.(θi) .+ Xs[2 + n_ag*dim_x, si]], color="black", lw=1.5, label="")
    
    if relative
        scatter!(pl, [0], [0], label=L"Evader", lw=2, color=pal_evader, alpha=alpha, marker=Shape(rotate_triangle(0.)), markersize=markersize)
    else
        scatter!(pl, [Xs[1 + n_ag*dim_x, si]], [Xs[2 + n_ag*dim_x, si]], label=L"Evader", lw=2, color=pal_evader, alpha=alpha, marker=Shape(rotate_triangle(Xs[3 + n_ag*dim_x, si])), markersize=markersize)
    end

    ## Details
    plot!(pl, ylims=ylimz, xlims=xlimz, legend=legend, aspect_ratio=:equal, legend_columns=legend_columns, legendfontsize=legendfontsize)
    xticksspots = xlimz[1]:2.5:xlimz[2]
    xticklabels = vcat(latexstring(xlimz[1]), ["" for i=1:length(xticksspots)-2]..., latexstring(xlimz[2]))
    yticksspots = ylimz[1]:2.5:ylimz[2]
    yticklabels = vcat(latexstring(ylimz[1]), ["" for i=1:length(yticksspots)-2]..., latexstring(ylimz[2]))
    plot!(pl, xticks=(xticksspots, xticklabels), yticks=(yticksspots, yticklabels), xtickfontsize=xtickfontsize, ytickfontsize=ytickfontsize)

    lo, _ = collect(zip(xlims(pl), ylims(pl)))
    locxl = lo .+ ((xlims(pl)[2] - xlims(pl)[1])/2, -0.7)
    locyl = lo .+ (-0.7, (ylims(pl)[2] - ylims(pl)[1])/2)
    annotate!(locxl..., L"x\quad(100\: m)", 12)
    annotate!(locyl..., Plots.text(L"y\quad(100\: m)", 12, :black, rotation=90))

    return pl
end

function plot_feasible(solz_is, x0; p_static, tpf = 60, ylimz=(-5, 5), xlimz=(-5, 5), legend=:topright, plot_size=(375,375), 
                        alpha=0.1, alpha_tri=0.9, markersize=20, legend_columns=2, skip=1,
                        legendfontsize=10, xtickfontsize=12, ytickfontsize=12, mixed=true,
                        pal_pursuer=palette(:seaborn_colorblind), pal_evader=:white,
                        # pal_pursuer=palette(:Reds, n_ag+4)[3:end-2], 
                        # pal_pursuer = palette(:Reds, n_ag+5)[3:2:end], # 3 agents
                        # pal_pursuer = palette(:Reds, n_ag+8)[5:2:end], # 5 agents
                        # pal_evader = :blue,
                        )

    n_ag, dim_x, dim_xh, v_a, v_b, max_u, max_d, r, r_follow, θ_max = p_static
    gr(); pl = plot(size=plot_size,title=L"\textrm{Feasible\:\:Sets}", dpi=300);
    
    ## Feasible Sets
    tff = high(tspan(solz_is[1][min(tpf, minimum(length(solz_is[i]) for i=1:n_ag))]))
    print(", tff: $tff")

    if !mixed
        for i=1:n_ag; plot!(solz_is[i][tpf:-skip:1], vars=(1,2), alpha=alpha, label=latexstring("𝒮_$i"), color=pal_pursuer[i]); end
    else
        K = min(tpf, minimum(length(solz_is[i]) for i=1:n_ag))
        for k=K:-skip:1
            for i=1:n_ag; 
                plot!(solz_is[i][k], vars=(1,2), alpha=alpha, label="", color=pal_pursuer[i]); 
            end
        end
        for i=1:n_ag; plot!(solz_is[i][1], vars=(1,2), alpha=alpha, label=latexstring("𝒮_$i"), color=pal_pursuer[i]); end
    end

    ## Triangular Markers
    # triangle = [(-sqrt(3)/6 - 0.5, 0.5), (-sqrt(3)/6 - 0.5, -0.5), (sqrt(3)/2 - sqrt(3)/6 + 0.5, 0.), (-sqrt(3)/6 - 0.5, 0.5)];
    triangle = [(-0.6, 0.45), (0.2, 0.45), (0.6, 0.), (0.2, -0.45), (-0.6, -0.45), (-0.6, 0.45)];
    rotate_marker(points, θ) = [[cos(θ) -sin(θ); sin(θ) cos(θ)] * [x; y] for (x,y) in points];
    rotate_triangle(θ) = [tuple(x...) for x in rotate_marker(triangle, θ)];
    
    for i=0:(n_ag-1); scatter!(pl, [x0[1 + i*dim_x]], [x0[2 + i*dim_x]], label="", lw=2, color=pal_pursuer[i+1], alpha=alpha_tri, marker=Shape(rotate_triangle(x0[3 + i*dim_x])), markersize=markersize); end
    scatter!(pl, [0], [0], label="", lw=2, color=pal_evader, alpha=alpha_tri, marker=Shape(rotate_triangle(0.)), markersize=markersize)
    θ=0.:0.01:2π; plot!([r * cos.(θ)], [r * sin.(θ)], color="black", lw=2, label=L"𝒯")

    ## Details
    plot!(pl, ylims=ylimz, xlims=xlimz, legend=legend, aspect_ratio=:equal, legend_columns=legend_columns, legendfontsize=legendfontsize)
    xticksspots = xlimz[1]:2.5:xlimz[2]
    xticklabels = vcat(latexstring(xlimz[1]), ["" for i=1:length(xticksspots)-2]..., latexstring(xlimz[2]))
    yticksspots = ylimz[1]:2.5:ylimz[2]
    yticklabels = vcat(latexstring(ylimz[1]), ["" for i=1:length(yticksspots)-2]..., latexstring(ylimz[2]))
    plot!(pl, xticks=(xticksspots, xticklabels), yticks=(yticksspots, yticklabels), xtickfontsize=xtickfontsize, ytickfontsize=ytickfontsize)

    lo, _ = collect(zip(xlims(pl), ylims(pl)))
    locxl = lo .+ ((xlims(pl)[2] - xlims(pl)[1])/2, -0.6)
    locyl = lo .+ (-0.6, (ylims(pl)[2] - ylims(pl)[1])/2)
    annotate!(locxl..., L"x_Δ\quad(100\: m)", 12)
    annotate!(locyl..., Plots.text(L"y_Δ\quad(100\: m)", 12, :black, rotation=90))

    return pl
end

function plot_value(Xr, targets, T; pal_pursuer=palette(:tab10), plot_size=(375,375), 
                                    legendfontsize=10, xtickfontsize=12, ytickfontsize=12,
                                    xlimz=(0, 1), ylimz=(0, 15))

    val_pl = plot(size=plot_size,)
    for (tgi, target) in enumerate(targets)
        vals = target[1](Xr, target[3]...)
        plot!(0.:T[1]:T[end], vals, label=L"Agent\:"*latexstring(tgi), color=pal_pursuer[tgi], lw=2)
    end
    plot!(val_pl, title=L"\textrm{Capture\:\:Value\:\:over\:\:x(t)}")
    xticksspots = xlimz[1]:0.2:xlimz[2]
    xticklabels = vcat(latexstring(xlimz[1]), ["" for i=1:length(xticksspots)-2]..., latexstring(xlimz[2]))
    yticksspots = ylimz[1]:5:ylimz[2]
    yticklabels = vcat(latexstring(ylimz[1]), ["" for i=1:length(yticksspots)-2]..., latexstring(ylimz[2]))
    plot!(xticks=(xticksspots, xticklabels), yticks=(yticksspots, yticklabels), xtickfontsize=xtickfontsize, ytickfontsize=ytickfontsize)
    plot!(legend_columns=1, aspect=:equal, legendfontsize=legendfontsize)

    lo, _ = collect(zip(xlims(val_pl), ylims(val_pl)))
    locxl = lo .+ ((xlims(val_pl)[2] - xlims(val_pl)[1])/2, -0.9)
    locyl = lo .+ (-0.05, (ylims(val_pl)[2] - ylims(val_pl)[1])/2)
    annotate!(locxl..., L"t", 12)
    annotate!(locyl..., Plots.text(L"J_i (x(t))", 16, :black, rotation=90))
    return val_pl
end

### TESTS

testing = false

if testing
    n_ag = 5
    v_a = 2.5
    # θ = -pi/4n_ag
    θ = 0.0
    opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.01, 50, 1e-3, 500, 3, 9, 1000), 1, 1, 2); refines = 0;
    opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.01, 40, 1e-3, 50, 3, 3, 200), 1, 1, 2); refines = 0;

    solve_MDavoid_hopfconlin(; θ, n_ag, v_a, saving=false, printing=true, opt_p_admm_cd)

    solve_MDavoid_hopfconlin(; θ, n_ag, v_a, saving=false, name="feas_sim_$(n_ag)ag_θ_$(round.(θ,digits=2))_va_$(v_a)_nice", printing=true, feasibility_only=true, plotting=true)

    solve_MDavoid_hopfconlin(; θ, n_ag, v_a, saving=false, printing=true, control_simulation=true, plotting=true, opt_p_admm_cd, refines, pursuer_ctrl="hopf", one_shot=false)

    solve_MDavoid_hopfconlin(; θ, n_ag, v_a, saving=false, printing=true, control_simulation=true, plotting=true, opt_p_admm_cd, refines, pursuer_ctrl="mpc", one_shot=false)

    solve_MDavoid_hopfconlin(; θ, n_ag, v_a, saving=false, name="ctrl_sim_$(n_ag)ag_θ_$(round.(θ,digits=2))_va_$(v_a)_nice_itr", printing=true, control_simulation=true, plotting=true, opt_p_admm_cd, refines, pursuer_ctrl="mpc", one_shot=true)

    # solve_MDavoid_hopfconlin(; max_u=1., v_a=2., saving=false, printing=true, control_simulation=true, plotting=true, opt_p_admm_cd, refines=1, pursuer_ctrl="mpc", one_shot=true)
end
