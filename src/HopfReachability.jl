module HopfReachability

using LinearAlgebra 
using StatsBase 
using TickTock, Suppressor 
using Plots, ScatteredInterpolation
import Contour: contours, levels, level, lines, coordinates
import Plots: plot, scatter, contour

##################################################################################################
##################################################################################################

### General

##################################################################################################
##################################################################################################


## Evaluate Hopf Cost
function Hopf(system, target, tbH, z, v; p2=true, game="reach")

    J, Jˢ, Jp = target
    intH, Hdata = tbH

    return Jˢ(v) - z'v + intH(system, Hdata, v; p2, game);
end

## Solve Hopf Reachability over Grid for given system, target, ∫Hamiltonian and lookback time(s) T
function Hopf_BRS(system, target, T;
                  opt_method=Hopf_cd, opt_p=nothing, inputshape=nothing, preH=preH_ball, intH=intH_ball, HJoc=HJoc_ball, error=false, Φ=nothing,
                  Xg=nothing, lg=0, ϵ=0.5e-7, grid_p=(3, 10 + 0.5e-7), th=0.02, warm=false, warm_pattern="previous",
                  p2=true, game="reach", plotting=false, printing=false, sampling=false, samples=360, zplot=false, check_all=true,
                  moving_target=false, moving_grid=false, opt_tracking=false, sigdigits=3, v_init=nothing, admm=false)

    if opt_method == Hopf_cd
        # opt_p = isnothing(opt_p) ? (0.01, 2, ϵ, 100, 5, 10, 400) : opt_p # Faster
        opt_p = isnothing(opt_p) ? (0.01, 5, ϵ, 500, 20, 20, 2000) : opt_p # Safer
        admm = false
    elseif opt_method == Hopf_admm
        opt_p = isnothing(opt_p) ? (1e-1, 1e-2, 1e-5, 100) : opt_p 
        admm = true
    elseif opt_method == Hopf_admm_cd
        opt_p = isnothing(opt_p) ? ((1e-0, 1e-0, 1e-5, 3), (0.01, 2, ϵ, 50, 1, 3, 125), 1, 1, 10) : opt_p 
        admm = true
    end

    ## System
    simple_problem = !moving_grid && !moving_target
    preH, intH, HJoc = inputshape ∈ ["ball", "Ball", "BALL"] ? (preH_ball, intH_ball, HJoc_ball) : 
                      (inputshape ∈ ["box", "Box", "BOX"] ? (preH_box, intH_box, HJoc_box) : 
                      (preH, intH, HJoc))
    preH, intH, HJoc = error ? (preH_error, intH_error, HJoc_error) : (preH, intH, HJoc) # forces inputshape to be box atm
    
    if printing; 
        pr_A = typeof(system[1]) <: Function ? "A(t)" : (length(size(system[1])) == 1 ? "A[t]" : "A")
        pr_B₁ = typeof(system[2]) <: Function ? "B₁(t)" : (length(size(system[2])) == 1 ? "B₁[t]" : "B₁")
        pr_B₂ = typeof(system[3]) <: Function ? "B₂(t)" : (length(size(system[3])) == 1 ? "B₂[t]" : "B₂")
        pr_affine = length(system) >= 8 ? (typeof(system[8]) <: Function ? "+ c(t) " : (length(size(system[8][1])) == 0 ? "+ c " : "+ c[t] ")) : ""
        pr_E = length(system) == 9 ? (typeof(system[9]) <: Function ? "Eδ(t)" : (length(size(system[9])) == 1 ? "Eδ[t]" : "Eδ")) : ""
        
        pr_x_dim = typeof(system[1]) <: Function ? size(system[1](0),1) : (length(size(system[1])) == 1 ? size(system[1][1],1) : size(system[1],1))
        pr_u_dim, pr_d_dim = size(system[4])[1], size(system[6])[1]
        pr_u_bds, pr_d_bds = preH == preH_ball ? ("𝒰 := {|| (u-c₁)ᵗ inv(Q₁) (u-c₁)||₂ ≤ 1}", "𝒟 := {|| (d-c₂)ᵗ inv(Q₂) (d-c₂)||₂ ≤ 1}") : ("𝒰 := {||inv(Q₁) (u-c₁)||∞ ≤ 1}", "𝒟 := {||inv(Q₂) (d-c₂)||∞ ≤ 1}")
        pr_error, pr_error_bds, pr_error_dim = error && length(system) == 9 ? ("+ $pr_E ε", ", ℰ := {||ε||∞ ≤ 1}", ", ε ∈ ℝ^{$pr_x_dim}") : ("", "", "")
        
        println("\nGiven,")
        println("\n  ẋ = $pr_A x + $pr_B₁ u + $pr_B₂ d $pr_affine$pr_error")
        println("\n  s.t. ")
        println("\n  $pr_u_bds, $pr_d_bds$pr_error_bds")
        println("\n  for x ∈ ℝ^{$pr_x_dim}, u ∈ ℝ^{$pr_u_dim}, d ∈ ℝ^{$pr_d_dim}$pr_error_dim")
    end
    
    nx = typeof(system[1]) <: Function ? size(system[1](0.),1) : (length(size(system[1])) == 1 ? size(system[1][1],1) : size(system[1],1))
    system = length(system) == 7 ? (system..., zeros(nx)) : system
    
    J, Jˢ, Jp = target
    M, Q𝒯, c𝒯 = system[1], Jp[1], Jp[2]
    t = collect(th: th: T[end])

    ## System Data
    index, ϕX, B⁺, ϕB⁺, B⁺T, ϕB⁺T = [], [], [], [], [], []
    averagetimes, pointstocheck, N, last_bi = [], [], 0, 1
    opt_data = opt_tracking ? [] : nothing

    ## Precomputation
    if printing; println("\nPrecomputation, ..."); end
    Hmats, Φ = preH(system, target, t; admm, opt_p, Φ, printing)

    ## Grid Set Up
    if isnothing(Xg)
        Xg = make_grid(grid_p, nx; small_shift=ϵ)
    elseif moving_grid
        # N = Int.([floor(inv(norm(Xg[i][:,1] - Xg[i][:,2]))) for i in eachindex(Xg)])
        N = [10 for i in eachindex(Xg)] # TODO: arbitrary atm, fix
    end

    ## Compute Near-Boundary set of Target in X
    if simple_problem
        
        @suppress begin; tick(); end # timing
        ϕX = J(Xg)
        push!(averagetimes, tok() / length(ϕX)) # initial grid eval time

        if check_all
            B⁺, ϕB⁺ = Xg, ϕX
            index = warm && warm_pattern == "spiral" ? mat_spiral(lg, lg)[1] : collect(1:length(ϕX))
        else
            index = boundary(ϕX; lg, N, nx)
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index] 
        end

        push!(B⁺T, copy(B⁺)); push!(ϕB⁺T, copy(ϕB⁺)) # for plotting ϕ0
        push!(pointstocheck, length(ϕB⁺))
    end
    warm_old, warm_new = zero(B⁺), zero(B⁺) # warm matrices

    prgame = game == "reach" ? "Reach" : "Avoid"
    prerror = error ? " with Linear Error" : ""
    if printing; println("\nSolving Backwards $prgame Set$prerror,"); end
    @suppress begin; tick(); end # timing

    ## Loop Over Time Frames
    for Tix in eachindex(T)

        if printing; println("   for t=-", T[Tix], "..."); end
        Ti = T[Tix]
        tix = findfirst(abs.(t .-  Ti) .< th/2);

        Xgi = !moving_grid ? Xg : Xg[Tix]
        Ai, ci = !moving_target ? Jp : (Jp[1][Tix], Jp[2][Tix])
        
        ## Update Near-Boundary set for moving problems
        if moving_grid || moving_target
            Ni = moving_grid ? N[Tix] : N
            lgi = moving_grid ? lg[Tix] : lg

            J, Jˢ = moving_target ? make_levelset_fs(ci, r; Q=Ai) : J, Jˢ # FIXME: make target tv fn of t
            ϕX = J(Xgi)
            if check_all
                B⁺, ϕB⁺, index = Xgi, ϕX, collect(1:length(ϕX))
            else
                index = boundary(ϕX; lg=lgi, N=Ni, nx)
                B⁺, ϕB⁺ = Xgi[:, index], ϕX[index] 
            end

            push!(B⁺T, copy(B⁺)); push!(ϕB⁺T, copy(ϕB⁺))
        end

        if isempty(index) && printing 
            println("At T=" * string(Ti) * ", no x in the grid s.t. |J(x)| < " * string(ϵ))
        end

        ## Map X to Z
        B⁺z = Φ(Ti) * B⁺
        target = (J, Jˢ, (Ai, ci))

        ## Packaging Hdata, tbH
        Hdata = (Hmats, tix, th)
        tbH = (intH, Hdata)

        ## Pre solve details
        index_pts = sampling ? sample(1:length(index),samples) : eachindex(index) # sample for speed test
        opt_data_Ti = opt_tracking ? [] : nothing

        @suppress begin; tick(); end # timing

        ## Solve Over Grid (near ∂ID if !check_all)
        for bi in index_pts

            ## Warm Starting 
            if warm
                if Tix == 1 || warm_pattern == "previous" || warm_pattern == "spiral"
                    v_init = bi == first(index_pts) ? v_init : warm_old[:, last_bi]
                elseif warm_pattern == "temporal"
                    v_init = warm_old[:, bi]
                elseif warm_pattern == "spatiotemporal"
                    v_init = bi == first(index_pts) ? warm_old[:, bi] : warm_old[:, last_bi]
                end
                last_bi = copy(bi)
            end

            ## Solve Hopf
            z = B⁺z[:, bi]
            ϕB⁺[bi], dϕdz, v_path = opt_method(system, target, z; p2, game, tbH, opt_p, v_init)
            
            warm_new[:, bi] = copy(dϕdz)
            if bi == last(index_pts); warm_old = copy(warm_new); end # overwrite warm matrix

            ## Store Optimization Path & Value
            if opt_tracking
                vals = zeros(size(v_path, 2))
                for (vi, v) in enumerate(eachcol(v_path))
                    vals[vi] = Hopf(system, target, tbH, z, v_path[:,vi]; p2, game)
                end
                push!(opt_data_Ti, (v_path, vals)); 
            end
        end

        ## Store Data
        push!(averagetimes, tok()/length(index_pts));
        push!(pointstocheck, length(index_pts));
        push!(B⁺T, copy(B⁺))
        push!(ϕB⁺T, copy(ϕB⁺))
        if opt_tracking; push!(opt_data, opt_data_Ti); end

        ## Update Near-Boundary index to intermediate solution
        if simple_problem && Tix != length(T) && !check_all
            ϕX[index] = ϕB⁺
            index = boundary(ϕX; lg, N, nx)
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index]
        end
    end
    
    totaltime = tok()

    pr_totaltime = round(totaltime, sigdigits=sigdigits)
    pr_pointstocheck = Int.(pointstocheck)
    pr_averagetimes = round.(float.(averagetimes), sigdigits=sigdigits)
    min_ϕB⁺T = round.(minimum.(ϕB⁺T), sigdigits=sigdigits)
    max_ϕB⁺T = round.(maximum.(ϕB⁺T), sigdigits=sigdigits)

    if plotting; plot_nice(T, (B⁺T, ϕB⁺T); Φ, simple_problem, zplot); end
    if printing; println("TOTAL TIME: $pr_totaltime s"); end
    if printing; println("\nAt t = $(vcat(0., T)) over Xg,"); end
    if printing; println("  TOTAL PTS: $pr_pointstocheck"); end
    if printing; println("  MEAN TIME: $pr_averagetimes s/pt"); end
    if printing; println("  MIN VALUE: $min_ϕB⁺T"); end
    if printing; println("  MAX VALUE: $max_ϕB⁺T \n"); end
    run_stats = (totaltime, pointstocheck, averagetimes)

    ## Return arrays of B⁺ (near-boundary or all pts) over time and corresponding ϕ's, where the first is target (1 + length(T) arrays)
    # if moving problem, target inserted at each Ti (2 * length(T) arrays) #TODO could be better
    return (B⁺T, ϕB⁺T), run_stats, opt_data
end

## Solve Hopf Problem to find minimum T* so ϕ(z,T*) = 0 and the corresponding optimal strategies
function Hopf_minT(system, target, x; 
                  opt_method=Hopf_cd, inputshape=nothing, preH=preH_ball, intH=intH_ball, HJoc=HJoc_ball, error=false,
                  T_init=nothing, v_init_=nothing, Φ=nothing,
                  time_p=(0.01, 0.1, 2.), opt_p=nothing, tol=1e-5,
                  refine=0.5, refines=2, depth_counter=0,
                  p2=true, game="reach", printing=false, moving_target=false, warm=false)

    if depth_counter == 0; @suppress begin; tick(); end; end # timing

    if opt_method == Hopf_cd
        opt_p = isnothing(opt_p) ? (0.01, 5, 1e-5, 500, 20, 20, 2000) : opt_p
        admm = false
    elseif opt_method == Hopf_admm
        opt_p = isnothing(opt_p) ? (1e-1, 1e-2, 1e-5, 100) : opt_p 
        admm = true
    elseif opt_method == Hopf_admm_cd
        opt_p = isnothing(opt_p) ? ((1e-1, 1e-2, 1e-5, 100), (0.01, 5, 1e-5, 500, 1, 4, 2000), 1, 1, 10) : opt_p 
        admm = true
    end

    preH, intH, HJoc = inputshape ∈ ["ball", "Ball", "BALL"] ? (preH_ball, intH_ball, HJoc_ball) : 
                      (inputshape ∈ ["box", "Box", "BOX"] ? (preH_box, intH_box, HJoc_box) : 
                      (preH, intH, HJoc))
    preH, intH, HJoc = error ? (preH_error, intH_error, HJoc_error) : (preH, intH, HJoc) # forces inputshape to be box atm
    
    if printing && depth_counter == 0; 
        pr_A = typeof(system[1]) <: Function ? "A(t)" : (length(size(system[1])) == 1 ? "A[t]" : "A")
        pr_B₁ = typeof(system[2]) <: Function ? "B₁(t)" : (length(size(system[2])) == 1 ? "B₁[t]" : "B₁")
        pr_B₂ = typeof(system[3]) <: Function ? "B₂(t)" : (length(size(system[3])) == 1 ? "B₂[t]" : "B₂")
        pr_affine = length(system) >= 8 ? (typeof(system[8]) <: Function ? "+ c(t) " : (length(size(system[8][1])) == 0 ? "+ c " : "+ c[t] ")) : ""
        pr_E = length(system) == 9 ? (typeof(system[9]) <: Function ? "Eδ(t)" : (length(size(system[9])) == 1 ? "Eδ[t]" : "Eδ")) : ""
        
        pr_x_dim = typeof(system[1]) <: Function ? size(system[1](0),1) : (length(size(system[1])) == 1 ? size(system[1][1],1) : size(system[1],1))
        pr_u_dim, pr_d_dim = size(system[4])[1], size(system[6])[1]
        pr_u_bds, pr_d_bds = preH == preH_ball ? ("𝒰 := {|| (u-c₁)ᵗ inv(Q₁) (u-c₁)||₂ ≤ 1}", "𝒟 := {|| (d-c₂)ᵗ inv(Q₂) (d-c₂)||₂ ≤ 1}") : ("𝒰 := {||inv(Q₁) (u-c₁)||∞ ≤ 1}", "𝒟 := {||inv(Q₂) (d-c₂)||∞ ≤ 1}")
        pr_error, pr_error_bds, pr_error_dim = error && length(system) == 9 ? ("+ $pr_E ε", ", ℰ := {||ε||∞ ≤ 1}", ", ε ∈ ℝ^{$pr_x_dim}") : ("", "", "")
        
        println("\nGiven,")
        println("\n  ẋ = $pr_A x + $pr_B₁ u + $pr_B₂ d $pr_affine$pr_error")
        println("\n  s.t. ")
        println("\n  $pr_u_bds, $pr_d_bds $pr_error_bds")
        println("\n  for x ∈ ℝ^{$pr_x_dim}, u ∈ ℝ^{$pr_u_dim}, d ∈ ℝ^{$pr_d_dim} $pr_error_dim")

        println("\nSolving Optimal Control at x=$x,")
    end
    system = length(system) == 7 && depth_counter == 0 ? (system..., zeros(size(system[1],1))) : system

    M = system[1]
    J, Jˢ, Jp = target
    th, Th, maxT = time_p; 
    uˢ, dˢ, Tˢ, ϕ, dϕdz = 0, 0, 0, 0, 0
    t = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)

    ## System ϕs
    ϕ, dϕdz = Inf, 0
    v_init = isnothing(v_init_) ? nothing : copy(v_init_)

    ## Precompute entire Φ for speed
    # Φ = isnothing(Φ) ? (!(typeof(M) <: Function) ? s->exp(s*M) : solveΦ(M, collect(th:th:maxT))[2]; printing) : Φ
    Φ = isnothing(Φ) ? (typeof(M) <: Function || length(size(M)) == 1 ? solveΦ(M, collect(th:th:maxT); printing) : s->exp(s*M)) : Φ;

    ## Precomputing some of H for efficiency (#TODO check its faster than solving (0., maxT) once)
    Hmats, _ = preH(system, target, t; opt_p, admm, Φ, printing)

    ## Loop Over Time Frames until ϕ(z, T) > 0
    Thi = 0; Tˢ = t[end]
    while ϕ > tol

        if printing; println("   checking T = $Tˢ..."); end
        tix = length(t)
        Tix = Thi + 1

        ## Support Moving Targets
        Ai, ci = !moving_target ? Jp : (Jp[1][Tix], Jp[2][Tix])
        target = (J, Jˢ, (Ai, ci))

        ## Packaging Hdata, tbH
        Hdata = (Hmats, tix, th)
        tbH = (intH, Hdata)

        ## Hopf-Solving to check current Tˢ
        z = Φ(Tˢ) * x;
        ϕ, dϕdz, v_arr = opt_method(system, target, z; p2, game, tbH, opt_p, v_init)
        v_init = warm ? copy(dϕdz) : nothing

        ## Check if Tˢ yields valid ϕ
        if ϕ <= 0;
            if refines > 0
                if printing; println("At $Tˢ, ϕ=$(round(ϕ, digits=2)) < 0 with (th,Th)=($th,$Th), but refining with ($(th*refine),$(Th*refine)),"); end
                
                uˢ, dˢ, Tˢ, ϕ, dϕdz = Hopf_minT(system, target, x; 
                                    # time_p=(th*refine, Th*refine, Tˢ+Th), # refine t params maxT
                                    time_p=(th*refine, Th*refine, Tˢ), # refine t params maxT
                                    T_init = Thi == 0 ? T_init : (Tˢ-Th), # initialize 1 Th step back unless first step
                                    refines=refines-1, depth_counter=depth_counter+1,
                                    v_init_ = warm ? copy(dϕdz) : nothing, Φ,
                                    refine, printing, opt_method, opt_p, preH, intH, HJoc, tol, p2, game, moving_target, warm);

                # t = t_refined # need to do this in case refinement time goes past ϕ<0 time (i.e. refinements showed ϕ>0 actually)

            elseif printing; 
                println("\nTˢ ∈ [$(Tˢ-Th), $Tˢ], overestimating with Tˢ=$Tˢ"); 
            end

            if ϕ <= 0; break; end # need to do this in case refinement contradicts trigger
        end

        if Tˢ ≥ maxT; 
            if depth_counter == 0
                if printing && game == "reach"; println("!!! Not reachable under max time of $Tˢ !!!"); end
                if printing && game == "avoid"; println("Avoidable until max time of $Tˢ"); end
            else
                if printing; println("False trigger: refinement showed ϕ=$(round(ϕ, digits=2))>0, returning to original step size..."); end
            end
            break; 
        end    

        ## Update the t, to increase Tˢ
        t_ext = collect(Tˢ + th : th : Tˢ + Th + th/2); push!(t, t_ext...)
        Tˢ, Thi = t[end], Thi + 1

        ## Parsimoniously amending Hdata for increased final time (#TODO check its faster than solving (0., maxT) once)
        F_init = admm ? Hmats[end] : nothing
        Hmats_ext, _ = preH(system, target, t_ext; opt_p, admm, F_init, Φ, printing)
        Rt_ext, R2t_ext = vcat(Hmats[1], Hmats_ext[1]), vcat(Hmats[2], Hmats_ext[2]) # extend R1t, R2t or QR1t, QR2t
        if preH != preH_error; 
            Hmats = (Rt_ext, R2t_ext, Hmats[3:end-1]..., Hmats_ext[end])
        else
            cRt_ext, δR2t_ext = vcat(Hmats[3], Hmats_ext[3]), vcat(Hmats[4], Hmats_ext[4]); # extend cRt, δRt
            Hmats = (Rt_ext, R2t_ext, cRt_ext, δR2t_ext, Hmats_ext[end])
        end 
    end

    ## Compute Optimal Control (dH/dp(dϕdz, Tˢ) = exp(-Tˢ * M)Cuˢ + exp(-Tˢ * M)C2dˢ)
    if depth_counter == 0
        uˢ, dˢ = HJoc(system, dϕdz, t, Tˢ; p2, game, Φ, printing)

        totaltime = tok()
        if printing; println("  ϕ($Tˢ) = $ϕ \n ∇ϕ($Tˢ) = $dϕdz \n uˢ($Tˢ) = $uˢ \n dˢ($Tˢ) = $dˢ"); end
        if printing; println("TOTAL TIME: $totaltime s \n"); end
    end

    return uˢ, dˢ, Tˢ, ϕ, dϕdz
end

# ## Solve Hopf Problem to find minimum T* so ϕ(z,T*) = 0, Bisection Method (Will Only Work for BRTs)
# function Hopf_minT_BM(system, target, x; 
#     opt_method=Hopf_cd, preH=preH_ball, intH=intH_ball, HJoc=HJoc_ball,
#         T_init=nothing, v_init_=nothing,
#         time_p=(0.02, 0.1, 1.), opt_p=nothing, 
#         nx=size(system[1])[2], p2=true, game="reach", printing=false, moving_target=false, warm=false)

#     if printing; println("\nSolving Optimal Control at x=$x,"); end
#     @suppress begin; tick(); end # timing

#     if opt_method == Hopf_cd
#         opt_p = isnothing(opt_p) ? (0.01, 5, 0.5e-7, 500, 20, 20) : opt_p
#         admm = false
#     elseif opt_method == Hopf_admm
#         opt_p = isnothing(opt_p) ? (1e-4, 1e-4, 1e-5, 100) : opt_p 
#         admm = true
#     elseif opt_method == Hopf_admm_cd
#         opt_p = isnothing(opt_p) ? ((1e-1, 1e-2, 1e-5, 100), (0.01, 5, ϵ, 500, 1, 4), 1, 1, 10) : opt_p 
#         admm = true
#     end

#     J, Jˢ, Jp = target
#     th, Th, maxT = time_p
#     t = isnothing(T_init) ? collect(th: th: maxT) : collect(th: th: T_init) # Bisection

#     ## System ϕs
#     ϕ, dϕdz = Inf, 0
#     v_init = isnothing(v_init_) ? nothing : copy(v_init_) 

#     ## Precomputing some of H for efficiency
#     Hmats = preH(system, target, t; opt_p, admm, printing)

#     ## Loop over Time Frames until ϕ(z, T) > 0
#     c = 0; break_flag = false;
#     Tl, Tu = t[1], t[end]; 
#     Tˢ, tix = copy(Tu), length(t)

#     while true
#         if printing && !break_flag; println("   checking T = $Tˢ ∈ [$Tl, $Tu] ..."); end
#         Tix = findfirst(abs.(t .-  Tˢ) .< Th); #useless unless moving target

#         ## Support Moving Targets
#         Ai, ci = !moving_target ? Jp : (Jp[1][Tix], Jp[2][Tix])
#         target = (J, Jˢ, (Ai, ci))

#         ## Packaging Hdata, tbH
#         Hdata = (Hmats, tix, th)
#         tbH = (intH, Hdata)

#         ## Hopf-Solving to check current Tˢ
#         z = Φ(Tˢ * system[1]) * x
#         ϕ, dϕdz, v_arr = opt_method(system, target, z; p2, game, tbH, opt_p, v_init)
#         v_init = warm ? copy(dϕdz) : nothing

#         ## Check if Tˢ yields valid ϕ
#         if c == 0; 
#             if ϕ > 0; println("!!! Not reachable under max time of $maxT seconds !!!"); break; end
#             tix = argmin(abs.(t .- (Tl + Tu)/2)); Tˢ = t[tix];
#         elseif Tu - Tl ≈ th
#             if break_flag; if printing; println("Tˢ ∈ [$Tl, $Tu], overestimating with Tˢ=$Tu"); end; break; end
#             tix = argmin(abs.(t .- (Tl + Tu)/2)); Tˢ = t[tix]; break_flag = true # finish on over-estimate
#             # could also defeine th and refine at this point
#         elseif ϕ > 0
#             Tl = copy(Tˢ); tix = argmin(abs.(t .- (Tl + Tu)/2)); Tˢ = t[tix]
#         else # ϕ ≤ 0
#             Tu = copy(Tˢ); tix = argmin(abs.(t .- (Tl + Tu)/2)); Tˢ = t[tix]
#         end    

#         c += 1
#         if c > -log(th/maxT)/log(2) + 2 + ceil(abs(log(th)/log(10))); 
#             println(Tu - Tl, " vs ", th); 
#             println("Tu - Tl ≈ th is ", Tu - Tl ≈ th); println("Tu - Tl ≤ th*1.5 is ", Tu - Tl ≤ th*1.5)
#             error("Bisection should've converged :/"); end
#     end

#     ## Compute Optimal Control (dH/dp(dϕdz, Tˢ) = exp(-Tˢ * M)Cuˢ + exp(-Tˢ * M)C2dˢ)
#     uˢ, dˢ = HJoc(system, dϕdz, t, Tˢ; p2, game, printing)

#     totaltime = tok()
#     if printing; println("  ϕ($Tˢ) = $ϕ \n ∇ϕ($Tˢ) = $dϕdz \n uˢ($Tˢ) = $uˢ \n dˢ($Tˢ) = $dˢ"); end
#     if printing; println("TOTAL TIME: $totaltime s \n"); end

#     return uˢ, dˢ, Tˢ, ϕ, dϕdz
# end


##################################################################################################
##################################################################################################

### Optimization Methods

##################################################################################################
##################################################################################################


## Iterative Coordinate Descent of the Hopf Cost at Point z
function Hopf_cd(system, target, z; p2, game, tbH, opt_p, v_init=nothing, nx=length(z))

    vh, L, tol, lim, lll, max_runs, max_its = opt_p
    solution, v = Inf, nothing
    v_arr = zeros(nx, max_its)

    converged_runs, runs = 0, 0
    while converged_runs < lll

        v = isnothing(v_init) ? 10*(rand(nx) .- 0.5) : copy(v_init) # Hopf optimizer variable
        v_arr_temp = zeros(nx, max_its)

        kcoord = copy(nx); # coordinate counter
        stopcount = 0; # convergence flag
        happycount = 0; # step-size flag

        ## Init Hopf val: J*(v) - z'v + ∫Hdt
        fnow = Hopf(system, target, tbH, z, v; p2, game)
        
        ci = 1
        while true

            if ci < max_its + 1; v_arr_temp[:, ci] = copy(v); end

            ## Nearby Point along Coordinate
            kcoord = mod(kcoord, nx) + 1; #iterate coords
            v_coord = copy(v);

            v_coord[kcoord] = v_coord[kcoord] + vh; # nearby pt

            ## Nearby Hopf for Finite Differencing
            fnew = Hopf(system, target, tbH, z, v_coord; p2, game)

            ## Coordinate Descent Step (could be a view)
            v[kcoord] = v[kcoord] - 1/L * ((fnew - fnow)/vh);

            ## Updated Hopf
            fnownow = Hopf(system, target, tbH, z, v; p2, game)
            
            ## Convergence Criteria
            stopcount = abs(fnownow - fnow) < tol ? stopcount + 1 : 0; # in thresh for coordinate
            happycount, L = happycount > lim ? (1, 2L) : (happycount + 1, L); 

            if stopcount == nx || ci == max_its # in thresh for all coordinates
                if -Inf < fnow && fnow < Inf
                    solution = min(solution, fnow) # min with iter, to overcome potential local convergence
                    v_arr = solution == fnow ? v_arr_temp[:, 1:ci] : v_arr # save minimum descent
                    converged_runs += 1
                end

                break
            end

            fnow = fnownow;
            ci += 1
        end

        if runs > max_runs 
            if converged_runs > 0
                # println("Caution only ", 100*converged_runs/max_runs, "% of runs converged.")
                break
            else
                # error("All optimization runs diverged! :(")
                break
            end
        else
            runs += 1
        end
    end

    return -solution, v, v_arr
end

## Proximal Splitting Algorithm for optimizing the Hopf Cost at point z (vectorized)
function Hopf_admm(system, target, z; p2, game, tbH, opt_p, v_init=nothing, nx=length(z))

    ## System
    Q, Q2 = system[4], system[6]
    ρ, ρ2, tol, max_its = opt_p
    Hmats, tix, th = tbH[2]
    Rt, R2t, F = Hmats[1:2]..., Hmats[end] # TODO transfer SVD from preH
    nu, nd = size(Q)[1], size(Q2)[1]

    U, Σ, Vt    = svd(Q * ρ)  # TODO transfer this from preH
    U2, Σ2, Vt2 = svd(Q2 / ρ2)
    Σ, Σ2 = diagm(Σ), diagm(Σ2)

    ## Costate, Artificial Constraint Curves, Augmented Lagrange Multipliers
    v = isnothing(v_init) ? 0.1*ones(nx) : v_init
    γ, γ2, λ, λ2  = 0.1*ones(nu, tix), 0.1*ones(nd, tix), 0.1*ones(nu, tix), 0.1*ones(nd, tix) 
    v_arr = zeros(nx, max_its+1)
    # v_arr, γ_arr, γ2_arr, λ_arr, λ2_arr = zeros(nx, max_its), [], [], [], []

    ## Update Quadrature Mats
    Rvt, R2vt = reshape(view(Rt,1:nu*tix,:) * v, nu, tix), reshape(view(R2t,1:nd*tix,:) * v, nd, tix)

    ci = 1
    while true

        v_old = copy(v)
        v_arr[:, ci] = copy(v)

        ## Proximal Updates of Characteristic Curves
        γ  = Q  != zero(Q)  ? Rvt  + λ  - U  * Proj_Ellipse_BM(Vt *  (Rvt  + λ),  Σ)  : Rvt  + λ
        γ2 = Q2 != zero(Q2) ? R2vt + λ2 + U2 * Fur_Ellipse_BM(Vt2 * -(R2vt + λ2), Σ2) : R2vt + λ2
        # push!(γ_arr, γ); push!(γ2_arr, γ2);

        ## Proximal Update of Costate
        # v = update_v(z, γ - λ, γ2 - λ2, tbH, ρ, ρ2); 
        y = z + target[3][2];
         for s = 1:tix; y += th * ((ρ * view(Rt, nu*(s-1)+1: nu*s,:)' * view(γ - λ,:,s) + (ρ2 * view(R2t, nd*(s-1)+1: nd*s,:)' * view(γ2 - λ2,:,s)))); end
        v = F * y

        ## Stopping Criteria
        if norm(v_old - v) < tol || ci == max_its+1; break; end
        ci += 1;

        ## Update Quadrature Mats
        Rvt, R2vt = reshape(view(Rt,1:nu*tix,:) * v, nu, tix), reshape(view(R2t,1:nd*tix,:) * v, nd, tix)

        ## Linear Update of ALM
        λ  = λ  + ρ * (Rvt - γ)
        λ2 = λ2 + 1/ρ2 * (R2vt - γ2)
        # push!(λ_arr, λ); push!(λ2_arr, λ2)

    end

    return -Hopf(system, target, tbH, z, v; p2, game), v, v_arr[:, 1:ci] #, γ_arr, γ2_arr, λ_arr, λ2_arr
end

## Projection for the Shrink Operator and Characteristic Curve of Convex Player (Bisection Method for Ellipses, vectorized from GTools 2013)
function Proj_Ellipse_BM(Wi, A; tol=0.5e-9, ϵ=1e-5)

    Wo = copy(Wi); 
    mask = map(w -> ((w' * inv(A) * w) > 1), eachcol(Wo)); # select points outside
    if sum(mask) == 0; return Wo; end
    W = view(Wo, :, mask)

    W[abs.(W) .< ϵ] = W[abs.(W) .< ϵ] .+ (ϵ * sign.(W[abs.(W) .< ϵ] .+ tol)) # for zeros-robustness

    amin, imin = minimum(diag(A)), argmin(diag(A))
    μ0 = -amin .+ sqrt(amin) * W[imin,:]' 
    μ1 = -amin .+ sqrt.(sum(A * W.^2, dims = 1))

    while minimum(abs.(μ0 - μ1)) > tol

        μ = (μ0 + μ1)/2

        ix = sum((A * W.^2) ./ (diag(A) .+ μ).^2, dims=1) .- 1 .> 0 
        μ0[ix], μ1[.!ix] = μ[ix], μ[.!ix]

    end

    Wo[:, mask] = (A * W) ./ (diag(A) .+ (μ0 + μ1)/2)

    return Wo
end

## Projection for the Stretch Operator and Characteristic Curve of Concave Player (Bisection Method for Ellipses, WAS)
function Fur_Ellipse_BM(Wi, A; tol=0.5e-9, ϵ=1e-5)

    Wo = copy(Wi); 
    mask = map(w -> ((w' * inv(A) * w) < 1), eachcol(Wo)); # select points outside
    if sum(mask) == 0; return Wo; end
    W = view(Wo, :, mask)

    W[abs.(W) .< ϵ] = W[abs.(W) .< ϵ] .+ (ϵ * sign.(W[abs.(W) .< ϵ] .+ tol)) # for zeros-robustness
    
    amin, imin = maximum(diag(A)), argmax(diag(A))
    μ0 = -amin .- sqrt(amin) * W[imin,:]' 
    μ1 = -amin .- sqrt.(sum(A * W.^2, dims = 1))
    
    while minimum(abs.(μ0 - μ1)) > tol

        μ = (μ0 + μ1)/2

        ix = sum((A * W.^2) ./ (diag(A) .+ μ).^2, dims=1) .- 1 .> 0 
        μ0[ix], μ1[.!ix] = μ[ix], μ[.!ix]

    end

    Wo[:, mask] = (A * W) ./ (diag(A) .+ (μ0 + μ1)/2)

    return Wo
end

## Proximal update for v
# function update_v(z, ξ, ξ2, tbH, ρ, ρ2; nu=size(tbH[2][1][1])[2], nd=size(tbH[2][1][2])[2])

#     Hmats, tix, th = tbH[2]
#     Rt, R2t = Hmats[1:2]
#     F = Hmats[end]

#     y = copy(z)
#     for s = 1:tix
#         y += th * ((ρ * view(Rt, nu*(s-1)+1: nu*s,:)' * view(ξ,:,s) + (ρ2 * view(R2t, nd*(s-1)+1: nd*s,:)' * view(ξ2,:,s))))
#     end

#     return F * y # TODO, need to subtract c𝒯 from RHS/y
# end

## ADMM-CD Hybrid Method
function Hopf_admm_cd(system, target, z; p2, game, tbH, opt_p, v_init=nothing, nx=length(z))

    opt_p_admm, opt_p_cd, ρ_grid_pts, ρ2_grid_pts, runs = opt_p
    max_ϕz, argmax_dϕdz, argmax_dϕdz_arr, v_arr_admm, min_k = -Inf, nothing, nothing, nothing, nothing

    for r = 1:(runs-1)
        v = isnothing(v_init) ? rand(nx) .- 0.5 : r == 1 ? v_init : v_init .+ 0.5 * norm(v_init) * (rand(nx) .- 0.5)

        for i = 0:ρ_grid_pts-1, j = 0:ρ2_grid_pts-1

            opt_p_admm = (opt_p_admm[1]/(10^i), opt_p_admm[2]/(10^j), opt_p_admm[3:end]...)
            ϕz_admm, dϕdz_admm, v_arr_admm = Hopf_admm(system, target, z; p2, game, tbH, opt_p=opt_p_admm, nx, v_init=v)

            for k in [1, 2, size(v_arr_admm)[2]] # initialize at no admm, first admm step and admm converged/max-it step

                ϕz_cd, dϕdz_cd, v_arr_cd = Hopf_cd(system, target, z; p2, game, tbH, opt_p=opt_p_cd, nx, v_init=v_arr_admm[:, k])

                max_ϕz = max(max_ϕz, ϕz_cd)
                argmax_dϕdz = max_ϕz == ϕz_cd ? dϕdz_cd : argmax_dϕdz
                argmax_dϕdz_arr = max_ϕz == ϕz_cd ? hcat(v_arr_admm[:, 1:k], v_arr_cd) : argmax_dϕdz_arr
                min_k = max_ϕz == ϕz_cd ? k : min_k

            end
        end
    end

    return max_ϕz, argmax_dϕdz, argmax_dϕdz_arr, min_k
end


##################################################################################################
##################################################################################################

### Utility fn's

##################################################################################################
##################################################################################################

## Make Grid
function make_grid(bd, res, nx; return_all=false, small_shift=0.)
    lbs, ubs = typeof(bd) <: Tuple ? (typeof(bd[1]) <: Tuple ? bd : (bd[1]*ones(nx), bd[2]*ones(nx))) : (-bd*ones(nx), bd*ones(nx))
    xigs = [collect(lbs[i] : res : ubs[i]) .+ small_shift for i in 1:nx]
    Xg = hcat(collect.(Iterators.product(xigs...))...) ## TODO : do this better
    output = return_all ? (Xg, xigs, (lbs, ubs)) : Xg
    return output
end

## Make Target
function make_levelset_fs(c, r; Q=diagm(one(c)), type="ball") # TODO: tv arg instead of redifing for tv params
    if type ∈ ["ball", "Ball", "ellipse", "Ellipse", "l2", "L2"]
        J(x::Vector) = ((x - c)' * inv(Q) * (x - c))/2 - 0.5 * r^2
        Jˢ(v::Vector) = (v' * Q * v)/2 + c'v + 0.5 * r^2
        J(x::Matrix) = diag((x .- c)' * inv(Q) * (x .- c))/2 .- 0.5 * r^2
        Jˢ(v::Matrix) = diag(v' * Q * v)/2 + (c'v)' .+ 0.5 * r^2
    else
        error("$type not supported yet") # TODO: l1, linf 
    end
    return J, Jˢ
end

## Find points near boundary ∂ of f(z) = 0
function boundary(ϕ; lg, N, nx, δ = 20/N) ## MULTI-D FIX

    A = Float64.(abs.(reshape(ϕ, [lg for i=1:nx]...)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    # A = Float64.(abs.(reshape(ϕ, lg, lg)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones([Int(floor(N/2)) for i=1:nx]...); # kernel
    # B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel

    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv

    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Contour Plot
function contour(solution; value=true, xigs=nothing, grid=false, title="BRS", titleval="Value", labels=nothing, color_range=["red", "blue"], alpha=0.9, value_alpha=0.2, lw=2, ϵs=0.1, markersize=2, interp_alg=Polyharmonic(), interp_grid_bds=nothing, interp_res=0.1, camera=(50,30), BRS_on_value=true, plotting_kwargs...)
    
    @assert size(solution[1][1],1) < 3 "For 3 dimensions, use plot_nice(). Instead, you could consider projection into 2 dimensions."
    @assert !(grid && isnothing(xigs)) "If plotting a solution on a grid, insert the discretized axes xig (used to make Xg)."
    labels = isnothing(labels) ? vcat("Target", ["t$i" for i=1:length(solution[1])-1]...) : labels
    colors = vcat(:black, palette(color_range, length(solution[1])-1)...)

    vals = copy(solution[2])
    if !grid
        interp_grid_bds = isnothing(interp_grid_bds) ? [minimum(minimum.(solgi for solgi in solution[1])), maximum(maximum.(solgi for solgi in solution[1]))] : interp_grid_bds
        Xg, xigs, _ = make_grid(interp_grid_bds, interp_res, size(solution[1][1], 1); return_all=true)
        for ti=1:length(solution[1])
            vals[i] = evaluate(interpolate(interp_alg, solution[1][i], solution[2][i]), Xg)
        end
    end

    BRS_plot = plot(title=title)
    value_plot = value ? plot(title=titleval, camera=camera) : nothing
    
    for ti=1:length(solution[1]); 
        Plots.contour!(BRS_plot, xigs..., reshape(vals[ti], length(xigs[1]), length(xigs[1]))', levels=[0], color=colors[ti], lw=lw, alpha=alpha, colorbar=false, plotting_kwargs...)
        plot!(BRS_plot, [1e5, 2e5], [1e5, 2e5], color=colors[ti], lw=lw, alpha=alpha, label=labels[ti], xlims=xlims(BRS_plot), ylims=ylims(BRS_plot)); # contour label workaround
    
        if value; surface!(value_plot, xigs..., reshape(vals[ti], length(xigs[1]), length(xigs[1]))', color=colors[ti], alpha=value_alpha, colorbar=false, plotting_kwargs...); end
    
        if value && BRS_on_value
            cl = levels(contours(xigs..., reshape(solution[2][ti], length(xigs[1]), length(xigs[1]))', [0]))[1]
            for line in lines(cl)
                ys, xs = coordinates(line)
                zs = 0 * xs
                Plots.plot!(value_plot, xs, ys, zs, alpha=0.5, label="", lw=1, color=colors[ti])
            end
        end
    end
    
    output = value ? plot(BRS_plot, value_plot) : BRS_plot
    return output
end

## Scatter Plot of BRS
function scatter(solution; value=true, xigs=nothing, grid=false, title="BRS", titleval="Value", labels=nothing, color_range=["red", "blue"], alpha=0.9, value_alpha=0.2, lw=2, ϵs=0.1, markersize=2, interp_alg=Polyharmonic(), interp_grid_bds=nothing, interp_res=0.1, camera=(50,30), BRS_on_value=true, plotting_kwargs...)
    
    @assert size(solution[1][1],1) < 3 "For 3 dimensions, use plot_nice(). Instead, you could consider projection into 2 dimensions."
    labels = isnothing(labels) ? vcat("Target", ["t$i" for i=1:(length(solution[1])-1)]...) : labels
    colors = vcat(:black, palette(color_range, length(solution[1])-1)...)

    BRS_plot = plot(title=title)
    value_plot = value ? plot(title=titleval, camera=camera) : nothing
    
    for ti=1:length(solution[1]);
        b = solution[1][ti][:, abs.(solution[2][ti]) .< ϵs] 
        xlims = [minimum(minimum.(solgi[1,:] for solgi in solution[1])), maximum(maximum.(solgi[1,:] for solgi in solution[1]))]
        ylims = [minimum(minimum.(solgi[2,:] for solgi in solution[1])), maximum(maximum.(solgi[2,:] for solgi in solution[1]))]
        Plots.scatter!(BRS_plot, collect(eachrow(b))..., color=colors[ti], label=labels[ti], alpha=alpha, markersize=markersize, markerstrokewidth=0., xlims=xlims, ylims=ylims, plotting_kwargs...);
        
        if value && BRS_on_value; Plots.scatter!(value_plot, b[1,:], b[2,:], -0*ones(length(b[1,:])), color=colors[ti], label="", alpha=0.5, markersize=markersize, markerstrokewidth=0., plotting_kwargs...); end
    end
    if value; for ti=1:length(solution[1]); Plots.scatter!(value_plot, solution[1][ti][1,:], solution[1][ti][2,:], solution[2][ti], color=colors[ti], label="", alpha=value_alpha, markersize=markersize, markerstrokewidth=0.); end; end

    output = value ? plot(BRS_plot, value_plot) : BRS_plot
    return output
end

function plot(solution; seriestype=:contour, kwargs...)
    @assert seriestype ∈ [:scatter, :contour] "Only contour and scatter are currently supported."
    if seriestype == :contour
        contour(solution; kwargs...)
    else
        scatter(solution; kwargs...)
    end
end

## Contour or Scatter Plot of BRS using plotly for interactive 3d plots: surfaces & volumes (needs https://plotly.com/julia/getting-started/)
function plot_nice(T, solution; Φ=nothing, simple_problem=true, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, 
    zplot=false, interpolate=false, pal_colors=["red", "blue"], alpha=0.5, 
    title=nothing, value_fn=false, xlims=[-2, 2], ylims=[-2, 2], base_plot=nothing)

    B⁺T, ϕB⁺T = solution
    nx=size(B⁺T[1])[1]
    if interpolate; @assert (isdefined(Main, Symbol("PlotlyJS"))) "Load PlotlyJS.jl for isosurfaces (`using PlotlyJS`)."; end
    if nx > 2 && value_fn; println("4D plots are not supported yet, can't plot Value fn"); value_fn = false; end

    if isnothing(base_plot)
        Xplot = isnothing(title) ? Main.Plots.plot(title="BRS") : Main.Plots.plot(title=title)
    else
        Xplot = base_plot
    end
    if zplot; Zplot = Main.Plots.plot(title="BRS"); end

    plots = zplot ? [Xplot, Zplot] : [Xplot]
    if value_fn; vfn_plots = zplot ? [Main.Plots.plot(title="Value"), Main.Plots.plot(title="Value")] : [Main.Plots.plot(title="Value")]; end

    B⁺Tc, ϕB⁺Tc = copy(B⁺T), copy(ϕB⁺T)
    
    ϕlabels = "t=" .* string.(-T)
    Jlabels = "Target, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1]))
    labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

    Tcolors = length(T) > 1 ? Main.Plots.palette(pal_colors, length(T)) : Main.Plots.palette([pal_colors[2]], 2)
    B0colors = length(T) > 1 ? Main.Plots.palette(["black", "gray"], length(T)) : Main.Plots.palette(["black"], 2)
    plot_colors = collect(Iterators.flatten(zip(B0colors, Tcolors)))

    ## Zipping Target to Plot Variation in Z-space over Time (already done in moving problems)
    if simple_problem && (length(T) > 1)
        for i = 3 : 2 : 2*length(T)
            insert!(B⁺Tc, i, B⁺T[1])
            insert!(ϕB⁺Tc, i, ϕB⁺T[1])
        end
    end

    if nx > 2 && interpolate; plotly_pl = zplot ? [Array{Main.PlotlyJS.GenericTrace{Dict{Symbol, Any}},1}(), Array{Main.PlotlyJS.GenericTrace{Dict{Symbol, Any}},1}()] : [Array{Main.PlotlyJS.GenericTrace{Dict{Symbol, Any}},1}()]; end

    for (j, i) in enumerate(1 : 2 : 2*length(T))        
        B⁺0, B⁺, ϕB⁺0, ϕB⁺ = B⁺Tc[i], B⁺Tc[i+1], ϕB⁺Tc[i], ϕB⁺Tc[i+1]
        Bs = zplot ? [B⁺0, B⁺, Φ(-T[j]) * B⁺0, Φ(-T[j]) * B⁺] : [B⁺0, B⁺]

        for (bi, b⁺) in enumerate(Bs)
            if simple_problem && bi == 1 && i !== 1; continue; end

            ϕ = bi % 2 == 1 ? ϕB⁺0 : ϕB⁺
            label = simple_problem && i == 1 && bi == 1 ? "Target" : labels[i + (bi + 1) % 2]

            ## Plot Scatter
            if interpolate == false

                ## Find Boundary in Near-Boundary
                b = b⁺[:, abs.(ϕ) .< ϵs]

                Main.scatter!(plots[Int(bi > 2) + 1], [b[i,:] for i=1:nx]..., label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                # scatter!(plots[Int(bi > 2) + 1], b[1,:], b[2,:], label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                
                if value_fn
                    Main.scatter!(vfn_plots[Int(bi > 2) + 1], b⁺[1,:], b⁺[2,:], ϕ, label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha, xlims=xlims, ylims=ylims)
                    # scatter!(vfn_plots[Int(bi > 2) + 1], b[1,:], b[2,:], ϕ, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], label=label)
                end
            
            ## Plot Interpolation
            else 

                if nx == 2
                    Main.contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] for i=1:nx]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label, linewidth=3)

                    if value_fn

                        ## Make Grid
                        xig = [collect(minimum(b⁺[i,:]) : cres : maximum(b⁺[i,:])) for i=1:nx]
                        G = hcat(collect.(Iterators.product(xig...))...)'
                        
                        ## Construct Interpolation (Should skip this in the future and just use Plotly's built in one for contour)
                        itp = Main.ScatteredInterpolation.interpolate(Main.Polyharmonic(), b⁺, ϕ)
                        itpd = Main.evaluate(itp, G')
                        iϕG = Main.reshape(itpd, length(xig[1]), length(xig[2]))'
                        
                        Main.surface!(vfn_plots[Int(bi > 2) + 1], xig..., iϕG, colorbar=false, color=plot_colors[i + (bi + 1) % 2], label=label, alpha=alpha)
                    end
            
                else
                    # isosurface!(plots[Int(bi > 2) + 1], xig..., iϕG, isomin=-ϵc, isomax=ϵc, surface_count=2, lc=plot_colors[i + (bi + 1) % 2], alpha=0.5)
                    pl = Main.isosurface(x=b⁺[1,:], y=b⁺[2,:], z=b⁺[3,:], value=ϕ[:], opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=Main.PlotlyJS.attr(x_show=false, y_show=false, z_show=false),
                        name=label, colorscale=[[0, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]], [1, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]]])

                    push!(plotly_pl[Int(bi > 2) + 1], pl)
                end
            end
        end
    end

    if value_fn
        Xplot = Main.Plots.plot(vfn_plots[1], Xplot)
        if zplot; Zplot = Main.Plots.plot(vfn_plots[2], Zplot); end
    end

    if nx > 2 && interpolate 
        Xplot = Main.PlotlyJS.plot(plotly_pl[1], Main.PlotlyJS.Layout(title="BRS of T, in X"));
        if zplot; Zplot = PlotlyJS.plot(plotly_pl[2], Main.PlotlyJS.Layout(title="BRS of T, in X")); end
    end

    Main.display(Xplot); plots = [Xplot]
    if zplot; Main.display(Zplot); plots = [Xplot, Zplot]; end

    return plots
end

## Generate index for a matrix spiral (for better warm-starting over space)
function mat_spiral(rows, cols)

    matr_ix = reshape(collect(1:rows * cols), rows, cols)
    top, bottom, left, right = 1, copy(rows), 1, copy(cols)
    spiral_ix, unspiral_ix = [], []
    
    while length(spiral_ix) < rows * cols
        for i in left:right; push!(spiral_ix, matr_ix[top,i]); end; top += 1
        for i in top:bottom; push!(spiral_ix, matr_ix[i,right]); end; right -= 1
        if top <= bottom; for i in right:-1:left; push!(spiral_ix, matr_ix[bottom,i]); end; bottom -= 1; end  
        if left <= right; for i in bottom:-1:top; push!(spiral_ix, matr_ix[i,left]); end; left += 1; end
    end
    
    spiral_ix = spiral_ix[end:-1:1] #flip to go outward

    for i=1:rows * cols
        push!(unspiral_ix, findfirst(isequal(i), spiral_ix))
    end

    return spiral_ix, unspiral_ix
end

## Solve the Fundamental i.e. Flow-map for a Time-Varying Linear System
function solveΦ(A, t; printing=false)
    @assert (isdefined(Main, Symbol("OrdinaryDiffEq")) || isdefined(Main, Symbol("OrdinaryDiffEq"))) "Time-varying drift requires integration of the fundamental matrix. Load OrdinaryDiffEq.jl or DifferentialEquations.jl (e.g. `using OrdinaryDiffEq`)."
    if printing; print("LTV System inputted, integrating Φ(t)... \r"); flush(stdout); end
    nx = typeof(A) <: Function ? size(A(t[1]))[1] : size(A[1])[1];
    f = typeof(A) <: Function ? (U,p,s) -> U * A(t[end] - s) : (U,p,s) -> U * A[end:-1:1][findfirst(x->x<=0, (-t .+ s))]
    sol = Main.solve(Main.ODEProblem(f, diagm(ones(nx)), (0., t[length(t)])), saveat=t, alg=Main.Tsit5());
    if printing; print("LTV System inputted, integrating Φ(t)... Success!\n"); end
    return sol
end


##################################################################################################
##################################################################################################

### Example Hamiltonians and Precomputation fn's

##################################################################################################
##################################################################################################


### Ellipsoid-Constrained Inputs (Ellipse norm, 2-norm w SVD)

## Time integral of Hamiltonian for YTC et al. 2017, quadratic control constraints
function intH_ball(system, Hdata, v; p2=true, game="reach",)

    M, C, C2, Q, a, Q2, a2, c = system
    Hmats, tix, th = Hdata
    Rt, R2t, G, G2, F = Hmats
    nu, nd = size(Q)[1], size(Q2)[1]
    sgn_p1, sgn_p2 = game == "reach" ? (1, -1) : (-1, 1)

    ## Quadrature mats
    Rvt, R2vt = reshape(view(Rt,1:nu*tix,:) * v, nu, tix), reshape(view(R2t,1:nd*tix,:) * v, nd, tix)
    
    ## Quadrature sum
    H1 = th * sgn_p1 * (sum(map(norm, eachcol(G * Rvt))) + sum(a * Rvt)) # player 1 / control
    H2 = p2 ? th * sgn_p2 * (-sum(map(norm, eachcol(G2 * R2vt))) + sum(a2 * R2vt)) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_ball(system, target, t; opt_p=nothing, admm=false, F_init=nothing, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c = system
    ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : (typeof(opt_p[1]) <: Number ? opt_p[1:2] : opt_p[1][1:2])
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (typeof(M) <: Function || length(size(M)) == 1 ? solveΦ(M, t; printing) : s->exp(s*M)) : Φ; Φt = Φ.(t)
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx);

    ## Precompute ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## TODO (admm only) only works when c𝒯 (in J) == 0, need to include c𝒯 in update_v on other side of eqn

    ## Precompute Rt Mats
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(t[end] - t[si]) : (length(size(C)) == 2 ? C : C[end:-1:1][si])
        C2ti = typeof(C2) <: Function ? C2.(t[end] - t[si]) : (length(size(C2)) == 2 ? C2 : C2[end:-1:1][si])  
        R, R2 = -(Φt[si] * Cti)', -(Φt[si] * C2ti)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F
        Rt[ nu*(si-1) + 1 : nu*si, :] = R;
        R2t[nd*(si-1) + 1 : nd*si, :] = R2;
    end

    ## Precomputing SVD for matrix sqrt
    _, Σ,  VV  = svd(Q);
    _, Σ2, VV2 = svd(Q2);
    # G  = Diagonal(sqrt.(Σ)) * VV; #TODO FIX & TEST
    # G2 = Diagonal(sqrt.(Σ2)) * VV2;
    G  = Diagonal(Σ)  * VV; #TODO FIX & TEST
    G2 = Diagonal(Σ2) * VV2;

    ## Precomputing ADMM matrix
    F = admm ? inv(F) : F

    return (Rt, R2t, G, G2, F), Φ
end

## Optimal HJB Control
function HJoc_ball(system, dϕdz, t, s; p2=true, game="reach", Hdata=nothing, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c = system
    nu, nd = size(Q)[1], size(Q2)[1]
    sgn_p1, sgn_p2 = game == "reach" ? (1, -1) : (-1, 1)
    
    ## Handle Time-Varying Systems (constant or fn(time), array nonsensible)
    Φ = isnothing(Φ) ? (typeof(M) <: Function ? solveΦ(M,[s]; printing) : s->exp(s*M)) : Φ; Φs = Φ(s)
    Cs  = typeof(C)  <: Function ? C(t[end] - s)  : C
    C2s = typeof(C2) <: Function ? C2(t[end] - s) : C2
    R, R2 = -(Φs * Cs)', -(Φs * C2s)'

    _,Σ,VV = svd(Q);
    _,Σ2,VV2 = svd(Q2);
    G, G2 = Diagonal(sqrt.(Σ)) * VV, Diagonal(sqrt.(Σ2)) * VV2;

    uˢ = Q  != zero(Q)  ? sgn_p1 * inv(norm(G * R * dϕdz)) * Q * R * dϕdz + a' : zeros(nu)
    dˢ = Q2 != zero(Q2) && p2 ? sgn_p2 * (inv(norm(G2 * R2 * dϕdz)) * Q2 * R2 * dϕdz - a2') : zeros(nd)

    return uˢ, dˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)
# could be faster by using Hdata

### Box-Constrained Inputs (Inf norm)

## Time integral of Hamiltonian for MRK et al. 2018, inf-norm control constraints
function intH_box(system, Hdata, v; p2=true, game="reach")

    M, C, C2, Q, a, Q2, a2, c = system
    Hmats, tix, th = Hdata
    QRt, QR2t, F = Hmats
    nu, nd = size(Q)[1], size(Q2)[1]
    sgn_p1, sgn_p2 = game == "reach" ? (1, -1) : (-1, 1)

    ## Quadrature mats
    QRvt, QR2vt = reshape(view(QRt,1:nu*tix,:) * v, nu, tix), reshape(view(QR2t,1:nd*tix,:) * v, nd, tix)
    
    ## Quadrature sum
    H1 = th * sgn_p1 * sum(map(x->sum(abs,x), eachcol(QRvt))) # player 1 / control
    H2 = p2 ? th * sgn_p2 * sum(map(x->sum(abs,x), eachcol(QR2vt))) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_box(system, target, t; opt_p=nothing, admm=false, F_init=nothing, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c = system
    ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : (typeof(opt_p[1]) <: Number ? opt_p[1:2] : opt_p[1][1:2])
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (typeof(M) <: Function || length(size(M)) == 1 ? solveΦ(M, t; printing) : s->exp(s*M)) : Φ; Φt = Φ.(t)
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Precomputing ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## THIS ONLY WORKS WHEN cT == 0 (admm only), ... TODO

    ## Transformation Mats Q * R := Q * (exp(-(T-t)M)C)' over t
    QRt, QR2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx);
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(t[end] - t[si]) : (length(size(C)) == 2 ? C : C[end:-1:1][si])
        C2ti = typeof(C2) <: Function ? C2.(t[end] - t[si]) : (length(size(C2)) == 2 ? C2 : C2[end:-1:1][si])  
        R, R2 = -(Φt[si] * Cti)', -(Φt[si] * C2ti)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F  ## Does this need to be adapated for ADMM + Box?
        QRt[nu*(si-1) + 1 : nu*si, :], QR2t[nd*(si-1) + 1 : nd*si, :] = Q * R, Q2 * R2
    end

    F = admm ? inv(F) : F

    return (QRt, QR2t, F), Φ
end

## Optimal HJB Control # todo: check graphically
function HJoc_box(system, dϕdz, t, s; p2=true, game="reach", Hdata=nothing, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c = system
    nu, nd = size(Q)[1], size(Q2)[1]
    sgn_p1, sgn_p2 = game == "reach" ? (1, -1) : (-1, 1)

    ## Handle Time-Varying Systems (constant or fn(time), array nonsensible)
    Φ = isnothing(Φ) ? (typeof(M) <: Function ? solveΦ(M,[s]; printing) : s->exp(s*M)) : Φ; Φs = Φ(s)
    Cs  = typeof(C)  <: Function ? C(t[end] - s)  : C
    C2s = typeof(C2) <: Function ? C2(t[end] - s) : C2
    R, R2 = -(Φs * Cs)', -(Φs * C2s)'

    QR, QR2 = Q * R, Q2 * R2

    # uˢ = inv(R') * (QR)' * sign.(QR * dϕdz)
    # dˢ = p2 ? - inv(R2') * (QR2)' * sign.(QR2 * dϕdz) : zeros(nd)

    uˢ = Q  != zero(Q)  ? sgn_p1 * Q * sign.(QR * dϕdz) : zeros(nu)
    dˢ = Q2 != zero(Q2) && p2 ? sgn_p2 * Q2 * sign.(QR2 * dϕdz) : zeros(nd)

    return uˢ, dˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)

### Standard Hamiltonian precomputation fn

function preH_std(system, target, t; opt_p=nothing, admm=false, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c = system
    ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : (typeof(opt_p[1]) <: Number ? opt_p[1:2] : opt_p[1][1:2])
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (!(typeof(M) <: Function) || length(size(M)) == 2 ? s->exp(s*M) : solveΦ(M, t; printing)) : Φ; Φt = Φ.(t) 
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Precomputing ADMM matrix
    F = Jp[1]

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx);
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(t[end] - t[si]) : (length(size(C)) == 2 ? C : C[si])
        C2ti = typeof(C2) <: Function ? C2.(t[end] - t[si]) : (length(size(C2)) == 2 ? C2 : C2[si])  
        R, R2 = -(Φt[si] * Cti)', -(Φt[si] * C2ti)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F
        Rt[ nu*(si-1) + 1 : nu*si, :] = R;
        R2t[nd*(si-1) + 1 : nd*si, :] = R2;
    end

    F = admm ? inv(F) : F

    return (Rt, R2t, F), Φ
end

### Hamiltonian with Error

## Time integral of Hamiltonian for MRK et al. 2018, inf-norm control constraints
function intH_error(system, Hdata, v; p2=true, game="reach")

    M, C, C2, Q, a, Q2, a2, c, Eδ = system
    Hmats, tix, th = Hdata
    QRt, QR2t, cRt, δRt, F = Hmats
    nu, nd = size(Q)[1], size(Q2)[1]
    nx = typeof(M) <: Function ? size(M(0.),1) : (length(size(M)) == 1 ? size(M[1])[1] : size(M)[1])
    sgn_p1, sgn_p2 = game == "reach" ? (1, -1) : (-1, 1)

    ## Quadrature mats
    QRvt, QR2vt = reshape(view(QRt,1:nu*tix,:) * v, nu, tix), reshape(view(QR2t,1:nd*tix,:) * v, nd, tix)
    δRvt = reshape(view(δRt,1:nx*tix,:) * v, nx, tix)
    
    ## Quadrature sum
    H1 = th * sgn_p1 * sum(map(x->sum(abs,x), eachcol(QRvt))) # player 1 / control
    H2 = p2 ? th * sgn_p2 * sum(map(x->sum(abs,x), eachcol(QR2vt))) : 0 # player 2 / disturbance, opponent   
 
    Hc = th * sum(view(cRt, 1:tix, :) * v) # affine term  
    Hδ = th * sgn_p2 * sum(map(x->sum(abs,x), eachcol(δRvt))) # error

    return H1 + H2 + Hc + Hδ
end

## Hamiltonian Precomputation
function preH_error(system, target, t; opt_p=nothing, admm=false, F_init=nothing, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c, Eδ = system
    ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : (typeof(opt_p[1]) <: Number ? opt_p[1:2] : opt_p[1][1:2])
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (typeof(M) <: Function || length(size(M)) == 1 ? solveΦ(M, t; printing) : s->exp(s*M)) : Φ; Φt = Φ.(t)
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Precomputing ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## THIS ONLY WORKS WHEN cT == 0 (admm only), ... TODO

    ## Transformation Mats Q * R := Q * (exp(-(T-t)M)C)' over t
    QRt, QR2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx) 
    cRt = zeros(length(t), nx); δRt = zeros(nx*length(t), nx);
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(t[end] - t[si]) : (length(size(C)) == 2 ? C : C[end:-1:1][si])
        C2ti = typeof(C2) <: Function ? C2.(t[end] - t[si]) : (length(size(C2)) == 2 ? C2 : C2[end:-1:1][si])  

        cti = typeof(c) <: Function ? c(t[end] - t[si]) : (length(size(c[1])) == 0 ? c : c[end:-1:1][si]) # note difference
        Eδti = typeof(Eδ) <: Function ? Eδ(t[end] - t[si]) : (length(size(Eδ)) == 2 ? Eδ : Eδ[end:-1:1][si])

        R, R2 = -(Φt[si] * Cti)', -(Φt[si] * C2ti)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F  ## Does this need to be adapated for ADMM + Box?

        QRt[nu*(si-1) + 1 : nu*si, :], QR2t[nd*(si-1) + 1 : nd*si, :] = Q * R, Q2 * R2
        cRt[si, :] = - (Φt[si] * cti)'
        δRt[nx*(si-1) + 1 : nx*si, :] = - (Φt[si] * Eδti)'
    end

    F = admm ? inv(F) : F

    return (QRt, QR2t, cRt, δRt, F), Φ
end

## Optimal HJB Control # todo: check graphically
function HJoc_error(system, dϕdz, t, s; p2=true, game="reach", Hdata=nothing, Φ=nothing, printing=false)

    M, C, C2, Q, a, Q2, a2, c, Eδ = system
    nu, nd = size(Q)[1], size(Q2)[1]
    sgn_p1, sgn_p2 = game == "reach" ? (1, -1) : (-1, 1)

    ## Handle Time-Varying Systems (constant or fn(time), array nonsensible)
    Φ = isnothing(Φ) ? (typeof(M) <: Function ? solveΦ(M,[s]; printing) : s->exp(s*M)) : Φ; Φs = Φ(s)
    Cs  = typeof(C)  <: Function ? C(t[end] - s)  : C
    C2s = typeof(C2) <: Function ? C2(t[end] - s) : C2
    # cs = typeof(c) <: Function ? c(t[end] - s) : c # unneeded for oc
    Eδs = typeof(Eδ) <: Function ? Eδ(t[end] - s) : C2

    R, R2 = -(Φs * Cs)', -(Φs * C2s)'

    QR, QR2 = Q * R, Q2 * R2

    # uˢ = inv(R') * (QR)' * sign.(QR * dϕdz)
    # dˢ = p2 ? - inv(R2') * (QR2)' * sign.(QR2 * dϕdz) : zeros(nd)

    uˢ = Q  != zero(Q)  ? sgn_p1 * Q * sign.(QR * dϕdz) : zeros(nu)
    dˢ = Q2 != zero(Q2) && p2 ? sgn_p2 * Q2 * sign.(QR2 * dϕdz) : zeros(nd)
    δˢ = sgn_p2 * Eδs * sign.((-Φs * Eδs)' * dϕdz)

    return uˢ, dˢ, δˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)

end


### WAYS THIS PKG CAN BE IMPROVED:
# - Parallelization
# - Sparse Arrays
# - Type Declaration
# - Integration with other optimizers (& Automatic differentiation?)
#   (e.g. for Automatic Convex Conjugation)