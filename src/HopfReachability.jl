module HopfReachability

using LinearAlgebra, StatsBase, ScatteredInterpolation, DifferentialEquations
using Plots, ImageFiltering, TickTock, Suppressor, PlotlyJS, LaTeXStrings
plotlyjs()

##################################################################################################
##################################################################################################

### General

##################################################################################################
##################################################################################################


## Evaluate Hopf Cost
function Hopf(system, target, tbH, z, v; p2)

    J, Jˢ, Jp = target
    intH, Hdata = tbH

    return Jˢ(v, Jp...) - z'v + intH(system, Hdata, v; p2);
end

## Solve Hopf Reachability over Grid for given system, target, ∫Hamiltonian and lookback time(s) T
function Hopf_BRS(system, target, T;
                  opt_method=Hopf_cd, inputshape="ball", preH=preH_ball, intH=intH_ball, HJoc=HJoc_ball, Φ=nothing,
                  Xg=nothing, lg=0, ϵ=0.5e-7, grid_p=(3, 10 + 0.5e-7), th=0.02, opt_p=nothing, warm=false, warm_pattern="",
                  p2=true, plotting=false, printing=false, sampling=false, samples=360, zplot=false, check_all=true,
                  moving_target=false, moving_grid=false)

    if printing; println("\nPrecomputation, ..."); end

    if opt_method == Hopf_cd
        opt_p = isnothing(opt_p) ? (0.01, 5, ϵ, 500, 20, 20, 2000) : opt_p
        ρ, ρ2 = 1, 1
        admm = false
    elseif opt_method == Hopf_admm
        opt_p = isnothing(opt_p) ? (1e-1, 1e-2, 1e-5, 100) : opt_p 
        ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : opt_p[1:2]
        admm = true
    elseif opt_method == Hopf_admm_cd
        opt_p = isnothing(opt_p) ? ((1e-1, 1e-2, 1e-5, 10), (0.01, 5, ϵ, 500, 1, 4, 2000), 1, 1, 10) : opt_p 
        ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : opt_p[1][1:2]
        admm = true
    end

    ## Initialize
    simple_problem = !moving_grid && !moving_target
    preH, intH, HJoc = inputshape ∈ ["ball", "Ball", "BALL"] ? (preH_ball, intH_ball, HJoc_ball) : 
                      (inputshape ∈ ["box", "Box", "BOX"] ? (preH_box, intH_box, HJoc_box) : 
                      (preH, intH, HJoc))

    J, Jˢ, Jp = target
    M, A, c = system[1], Jp[1], Jp[2]
    t = collect(th: th: T[end])

    ## Initialize Data
    index, ϕX, B⁺, ϕB⁺, B⁺T, ϕB⁺T, v_init = [], [], [], [], [], [], nothing
    averagetimes, pointstocheck, N = [], [], 0

    ## Precomputation
    Hmats, Φ = preH(system, target, t; admm, ρ, ρ2, Φ)
    nx = size(Hmats[1])[2]

    ## Grid Set Up
    if isnothing(Xg)
        bd, N = grid_p
        lb, ub = typeof(bd) <: Tuple ? bd : (-bd, bd)
        xig = collect(lb : 1/N : ub) .+ ϵ; lg = length(xig)
        Xg = hcat(collect.(Iterators.product([xig for i in 1:nx]...))...)[end:-1:1,:]
        #takes a while in REPL, but not when timed... ## TODO : Do this better

    elseif moving_grid
        # N = Int.([floor(inv(norm(Xg[i][:,1] - Xg[i][:,2]))) for i in eachindex(Xg)])
        N = [10 for i in eachindex(Xg)] # fix!
    end

    ## Compute Near-Boundary set of Target in X
    if simple_problem
        
        ϕX = J(Xg, A, c)
        if check_all
            B⁺, ϕB⁺ = Xg, ϕX
            index = warm && warm_pattern == "spiral" ? mat_spiral(lg, lg)[1] : collect(1:length(ϕX))
        else
            index = boundary(ϕX; lg, N, nx)
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index] 
        end

        push!(B⁺T, copy(B⁺)); push!(ϕB⁺T, copy(ϕB⁺)) # for plotting ϕ0
    end

    if printing; println("\nSolving Backwards Reachable Set,"); end
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

            ϕX = J(Xgi, Ai, ci)
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

        @suppress begin; tick(); end # timing

        ## Pre solve details
        index_pts = sampling ? sample(1:length(index),samples) : eachindex(index) # sample for speed test

        ## Solve Over Grid (near ∂ID if !check_all)
        for bi in index_pts
            z = B⁺z[:, bi]
            ϕB⁺[bi], dϕdz, opt_arr = opt_method(system, target, z; p2, tbH, opt_p, v_init)
            v_init = warm ? copy(dϕdz) : nothing
        end

        ## Store Data
        push!(averagetimes, tok()/length(index_pts));
        push!(pointstocheck, length(index_pts));
        push!(B⁺T, copy(B⁺))
        push!(ϕB⁺T, copy(ϕB⁺))

        ## Update Near-Boundary index to intermediate solution
        if simple_problem && Tix != length(T) && !check_all
            ϕX[index] = ϕB⁺
            index = boundary(ϕX; lg, N, nx)
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index]
        end
    end
    
    totaltime = tok()

    if plotting; plot_BRS(T, B⁺T, ϕB⁺T; Φ, simple_problem, zplot); end
    if printing; println("TOTAL TIME: $totaltime s"); end
    if printing; println("MEAN TIME[s] PER TIME POINT: $averagetimes"); end
    if printing; println("TOTAL POINTS PER TIME POINT: $pointstocheck"); end
    run_stats = (totaltime, averagetimes, pointstocheck)

    ## Return arrays of B⁺ over time and corresponding ϕ's, where the first is target (1 + length(T) arrays)
    # if moving problem, target inserted at each Ti (2 * length(T) arrays)
    return (B⁺T, ϕB⁺T), run_stats
end

## Solve Hopf Problem to find minimum T* so ϕ(z,T*) = 0 and the corresponding optimal strategies
function Hopf_minT(system, target, x; 
                  opt_method=Hopf_cd, inputshape="ball", preH=preH_ball, intH=intH_ball, HJoc=HJoc_ball,
                  T_init=nothing, v_init_=nothing, Φ=nothing,
                  time_p=(0.01, 0.1, 2.), opt_p=nothing, tol=1e-5,
                  refine=0.5, refines=2, depth_counter=0,
                  p2=true, printing=false, moving_target=false, warm=false)

    if printing && depth_counter == 0; println("\nSolving Optimal Control at x=$x,"); end
    if depth_counter == 0; @suppress begin; tick(); end; end # timing

    if opt_method == Hopf_cd
        opt_p = isnothing(opt_p) ? (0.01, 5, 1e-5, 500, 20, 20, 2000) : opt_p
        ρ, ρ2 = 1, 1
        admm = false
    elseif opt_method == Hopf_admm
        opt_p = isnothing(opt_p) ? (1e-1, 1e-2, 1e-5, 100) : opt_p 
        ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : opt_p[1:2]
        admm = true
    elseif opt_method == Hopf_admm_cd
        opt_p = isnothing(opt_p) ? ((1e-1, 1e-2, 1e-5, 100), (0.01, 5, 1e-5, 500, 1, 4, 2000), 1, 1, 10) : opt_p 
        ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : opt_p[1][1:2]
        admm = true
    end

    preH, intH, HJoc = inputshape ∈ ["ball", "Ball", "BALL"] ? (preH_ball, intH_ball, HJoc_ball) : 
                      (inputshape ∈ ["box", "Box", "BOX"] ? (preH_box, intH_box, HJoc_box) : 
                      (preH, intH, HJoc))

    M = system[1]
    J, Jˢ, Jp = target
    th, Th, maxT = time_p; 
    uˢ, dˢ, Tˢ, ϕ, dϕdz = 0, 0, 0, 0, 0
    t = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)

    ## Initialize ϕs
    ϕ, dϕdz = Inf, 0
    v_init = isnothing(v_init_) ? nothing : copy(v_init_)

    ## Precompute entire Φ for speed
    Φ = isnothing(Φ) ? (!(typeof(M) <: Function) ? s->exp(s*M) : solveΦ(M, collect(th:th:maxT))[2]) : Φ

    ## Precomputing some of H for efficiency (#TODO check its faster than solving (0., maxT) once)
    Hmats, _ = preH(system, target, t; ρ, ρ2, admm, Φ)

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
        ϕ, dϕdz, v_arr = opt_method(system, target, z; p2, tbH, opt_p, v_init)
        v_init = warm ? copy(dϕdz) : nothing

        ## Check if Tˢ yields valid ϕ
        if ϕ <= 0;
            if refines > 0
                if printing; println("At $Tˢ, ϕ<0 with (th,Th)=($th,$Th), but refining with ($(th*refine),$(Th*refine)),"); end
                
                uˢ, dˢ, Tˢ, ϕ, dϕdz = Hopf_minT(system, target, x; 
                                    time_p=(th*refine, Th*refine, Tˢ+Th), # refine t params
                                    T_init = Thi == 0 ? T_init : (Tˢ-Th), # initialize 1 Th step back unless first step
                                    refines=refines-1, depth_counter=depth_counter+1,
                                    v_init_ = warm ? copy(dϕdz) : nothing, Φ,
                                    refine, printing, opt_method, opt_p, preH, intH, HJoc, tol, p2, moving_target, warm);

            elseif printing; 
                println("Tˢ ∈ [$(Tˢ-Th), $Tˢ], overestimating with Tˢ=$Tˢ"); 
            end
            break; 
        end    
        if Tˢ > maxT && depth_counter == 0; println("!!! Not reachable under max time of $Tˢ !!!"); break; end    

        ## Update the t, to increase Tˢ
        t_ext = collect(Tˢ + th : th : Tˢ + Th + th/2); push!(t, t_ext...)
        Tˢ, Thi = t[end], Thi + 1

        ## Parsimoniously amending Hdata for increased final time (#TODO check its faster than solving (0., maxT) once)
        F_init = admm ? Hmats[end] : nothing
        Hmats_ext, _ = preH(system, target, t_ext; ρ, ρ2, admm, F_init, Φ)
        Rt_ext, R2t_ext = Hmats_ext[1], Hmats_ext[2]
        Rt, R2t  = vcat(Hmats[1], Rt_ext), vcat(Hmats[2], R2t_ext)
        Hmats = (Rt, R2t, Hmats[3:end-1]..., Hmats_ext[end])

    end

    ## Compute Optimal Control (dH/dp(dϕdz, Tˢ) = exp(-Tˢ * M)Cuˢ + exp(-Tˢ * M)C2dˢ)
    if depth_counter == 0
        uˢ, dˢ = HJoc(system, dϕdz, Tˢ; p2, Φ)

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
#         nx=size(system[1])[2], p2=true, printing=false, moving_target=false, warm=false)

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
#         ρ, ρ2 = isnothing(opt_p) ? (1e-1, 1e-2) : opt_p[1][1:2]
#         admm = true
#     end

#     J, Jˢ, Jp = target
#     th, Th, maxT = time_p
#     t = isnothing(T_init) ? collect(th: th: maxT) : collect(th: th: T_init) # Bisection

#     ## Initialize ϕs
#     ϕ, dϕdz = Inf, 0
#     v_init = isnothing(v_init_) ? nothing : copy(v_init_) 

#     ## Precomputing some of H for efficiency
#     Hmats = preH(system, target, t; ρ, ρ2, admm)

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
#         ϕ, dϕdz, v_arr = opt_method(system, target, z; p2, tbH, opt_p, v_init)
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
#     uˢ, dˢ = HJoc(system, dϕdz, Tˢ; p2)

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
function Hopf_cd(system, target, z; p2, tbH, opt_p, v_init=nothing, nx=length(z))

    vh, L, tol, lim, lll, max_runs, max_its = opt_p
    solution, v = 1e9, nothing
    v_arr = zeros(nx, max_its)

    converged_runs, runs = 0, 0
    while converged_runs < lll

        v = isnothing(v_init) ? 10*(rand(nx) .- 0.5) : copy(v_init) # Hopf optimizer variable
        v_arr_temp = zeros(nx, max_its)

        kcoord = copy(nx); # coordinate counter
        stopcount = 0; # convergence flag
        happycount = 0; # step-size flag

        ## Init Hopf val: J*(v) - z'v + ∫Hdt
        fnow = Hopf(system, target, tbH, z, v; p2)
        
        ci = 1
        while true

            if ci < max_its + 1; v_arr_temp[:, ci] = copy(v); end

            ## Nearby Point along Coordinate
            kcoord = mod(kcoord, nx) + 1; #iterate coords
            v_coord = copy(v);

            v_coord[kcoord] = v_coord[kcoord] + vh; # nearby pt

            ## Nearby Hopf for Finite Differencing
            fnew = Hopf(system, target, tbH, z, v_coord; p2)

            ## Coordinate Descent Step (could be a view)
            v[kcoord] = v[kcoord] - 1/L * ((fnew - fnow)/vh);

            ## Updated Hopf
            fnownow = Hopf(system, target, tbH, z, v; p2)
            
            ## Convergence Criteria
            stopcount = abs(fnownow - fnow) < tol ? stopcount + 1 : 0; # in thresh for coordinate
            happycount, L = happycount > lim ? (1, 2L) : (happycount + 1, L); 

            if stopcount == nx || ci == max_its # in thresh for all coordinates
                if -1e9 < fnow && fnow < 1e9
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
function Hopf_admm(system, target, z; p2, tbH, opt_p, v_init=nothing, nx=length(z))

    ## Initialize
    M, B, C, Q, Q2, a, a2 = system
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

    return -Hopf(system, target, tbH, z, v; p2), v, v_arr[:, 1:ci] #, γ_arr, γ2_arr, λ_arr, λ2_arr
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

#     return F * y # TODO, need to subtract c from RHS/y
# end

## ADMM-CD Hybrid Method
function Hopf_admm_cd(system, target, z; p2, tbH, opt_p, v_init=nothing, nx=length(z))

    opt_p_admm, opt_p_cd, ρ_grid_pts, ρ2_grid_pts, runs = opt_p
    max_ϕz, argmax_dϕdz, argmax_dϕdz_arr, v_arr_admm, min_k = -Inf, nothing, nothing, nothing, nothing

    for r = 1:(runs-1)
        v = isnothing(v_init) ? rand(nx) .- 0.5 : r == 1 ? v_init : v_init .+ 0.5 * norm(v_init) * (rand(nx) .- 0.5)

        for i = 0:ρ_grid_pts-1, j = 0:ρ2_grid_pts-1

            opt_p_admm = (opt_p_admm[1]/(10^i), opt_p_admm[2]/(10^j), opt_p_admm[3:end]...)
            ϕz_admm, dϕdz_admm, v_arr_admm = Hopf_admm(system, target, z; p2, tbH, opt_p=opt_p_admm, nx, v_init=v)

            for k in [1, 2, size(v_arr_admm)[2]] # initialize at no admm, first admm step and admm converged/max-it step

                ϕz_cd, dϕdz_cd, v_arr_cd = Hopf_cd(system, target, z; p2, tbH, opt_p=opt_p_cd, nx, v_init=v_arr_admm[:, k])

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

## Find points near boundary ∂ of f(z) = 0
function boundary(ϕ; lg, N, nx, δ = 20/N) ## MULTI-D FIX

    A = Float64.(abs.(reshape(ϕ, [lg for i=1:nx]...)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    # A = Float64.(abs.(reshape(ϕ, lg, lg)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones([Int(floor(N/2)) for i=1:nx]...); # kernel
    # B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel

    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv

    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Plots BRS over T in X and Z space
function plot_BRS(T, B⁺T, ϕB⁺T; Φ=nothing, simple_problem=true, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, 
    zplot=false, interpolate=false, inter_method=Polyharmonic(), pal_colors=["red", "blue"], alpha=0.5, 
    title=nothing, value_fn=false, nx=size(B⁺T[1])[1], xlims=[-2, 2], ylims=[-2, 2], base_plot=nothing)

    if nx > 2 && value_fn; println("4D plots are not supported yet, can't plot Value fn"); value_fn = false; end

    if isnothing(base_plot)
        Xplot = isnothing(title) ? Plots.plot(title="BRS: ϕ(X, T) = 0") : Plots.plot(title=title)
    else
        Xplot = base_plot
    end
    if zplot; Zplot = Plots.plot(title="BRS: ϕ(Z, T) = 0"); end

    plots = zplot ? [Xplot, Zplot] : [Xplot]
    if value_fn; vfn_plots = zplot ? [Plots.plot(title="Value: ϕ(X, T)"), Plots.plot(title="Value: ϕ(Z, T)")] : [Plots.plot(title="Value: ϕ(X, T)")]; end

    B⁺Tc, ϕB⁺Tc = copy(B⁺T), copy(ϕB⁺T)
    
    ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
    Jlabels = "J(⋅, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
    labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

    Tcolors = length(T) > 1 ? palette(pal_colors, length(T)) : palette([pal_colors[2]], 2)
    B0colors = length(T) > 1 ? palette(["black", "gray"], length(T)) : palette(["black"], 2)
    plot_colors = collect(Iterators.flatten(zip(B0colors, Tcolors)))

    ## Zipping Target to Plot Variation in Z-space over Time (already done in moving problems)
    if simple_problem && (length(T) > 1)
        for i = 3 : 2 : 2*length(T)
            insert!(B⁺Tc, i, B⁺T[1])
            insert!(ϕB⁺Tc, i, ϕB⁺T[1])
        end
    end

    if nx > 2 && interpolate; plotly_pl = zplot ? [Array{GenericTrace{Dict{Symbol, Any}},1}(), Array{GenericTrace{Dict{Symbol, Any}},1}()] : [Array{GenericTrace{Dict{Symbol, Any}},1}()]; end

    for (j, i) in enumerate(1 : 2 : 2*length(T))        
        B⁺0, B⁺, ϕB⁺0, ϕB⁺ = B⁺Tc[i], B⁺Tc[i+1], ϕB⁺Tc[i], ϕB⁺Tc[i+1]
        Bs = zplot ? [B⁺0, B⁺, Φ(-T[j]) * B⁺0, Φ(-T[j]) * B⁺] : [B⁺0, B⁺]

        for (bi, b⁺) in enumerate(Bs)
            if simple_problem && bi == 1 && i !== 1; continue; end

            ϕ = bi % 2 == 1 ? ϕB⁺0 : ϕB⁺
            label = simple_problem && i == 1 && bi == 1 ? "J(⋅)" : labels[i + (bi + 1) % 2]

            ## Plot Scatter
            if interpolate == false

                ## Find Boundary in Near-Boundary
                b = b⁺[:, abs.(ϕ) .< ϵs]

                scatter!(plots[Int(bi > 2) + 1], [b[i,:] for i=1:nx]..., label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                # scatter!(plots[Int(bi > 2) + 1], b[1,:], b[2,:], label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                
                if value_fn
                    scatter!(vfn_plots[Int(bi > 2) + 1], b⁺[1,:], b⁺[2,:], ϕ, label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha, xlims=xlims, ylims=ylims)
                    # scatter!(vfn_plots[Int(bi > 2) + 1], b[1,:], b[2,:], ϕ, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], label=label)
                end
            
            ## Plot Interpolation
            else 

                if nx == 2
                    contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] for i=1:nx]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label, linewidth=3)

                    if value_fn

                        ## Make Grid
                        xig = [collect(minimum(b⁺[i,:]) : cres : maximum(b⁺[i,:])) for i=1:nx]
                        G = hcat(collect.(Iterators.product(xig...))...)'
                        
                        ## Construct Interpolationb (Should skip this in the future and just use Plotly's built in one for contour)
                        itp = ScatteredInterpolation.interpolate(inter_method, b⁺, ϕ)
                        itpd = evaluate(itp, G')
                        iϕG = reshape(itpd, length(xig[1]), length(xig[2]))'
                        
                        surface!(vfn_plots[Int(bi > 2) + 1], xig..., iϕG, colorbar=false, color=plot_colors[i + (bi + 1) % 2], label=label, alpha=alpha)
                    end
            
                else
                    # isosurface!(plots[Int(bi > 2) + 1], xig..., iϕG, isomin=-ϵc, isomax=ϵc, surface_count=2, lc=plot_colors[i + (bi + 1) % 2], alpha=0.5)
                    pl = isosurface(x=b⁺[1,:], y=b⁺[2,:], z=b⁺[3,:], value=ϕ[:], opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
                        name=label, colorscale=[[0, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]], [1, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]]])

                    push!(plotly_pl[Int(bi > 2) + 1], pl)
                end
            end
        end
    end

    if value_fn
        Xplot = Plots.plot(vfn_plots[1], Xplot)
        if zplot; Zplot = Plots.plot(vfn_plots[2], Zplot); end
    end

    if nx > 2 && interpolate 
        Xplot = PlotlyJS.plot(plotly_pl[1], Layout(title="BRS of T, in X"));
        if zplot; Zplot = PlotlyJS.plot(plotly_pl[2], Layout(title="BRS of T, in X")); end
    end

    display(Xplot); plots = [Xplot]
    if zplot; display(Zplot); plots = [Xplot, Zplot]; end

    return plots
end

## Generate index for a matrix spiral (for better warm-starting over space)
function mat_spiral(rows, cols)

    matr_ix = reshape(collect(1:rows * cols), rows, cols)
    top, bottom, left, right = 1, copy(rows), 1, copy(cols)
    spiral_ix, unspiral_ix = [], []
    
    while length(spiral_ix) < rows * cols
        for i in left:right; push!(spiral_ix, matr_ix[top,i]); end; top += 1AIKCN
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
function solveΦ(A, t; alg=Tsit5())
    print("LTV System inputted, integrating Φ(t)... \r"); flush(stdout); 
    nx = typeof(A) <: Function ? size(A(t[1]))[1] : size(A[1])[1];
    f = typeof(A) <: Function ? (U,p,s) -> U * A(t[end] - s) : (U,p,s) -> U * A[end:-1:1][findfirst(x->x<=0, (-t .+ s))]
    sol = solve(ODEProblem(f, diagm(ones(nx)), (0., t[length(t)])), saveat=t, alg);
    print("LTV System inputted, integrating Φ(t)... Success!\n");
    return sol
end


##################################################################################################
##################################################################################################

### Example Hamiltonians and Precomputation fn's

##################################################################################################
##################################################################################################


### Ellipsoid-Constrained Inputs (Ellipse norm, 2-norm w SVD)

## Time integral of Hamiltonian for YTC et al. 2017, quadratic control constraints
function intH_ball(system, Hdata, v; p2)

    M, C, C2, Q, Q2, a, a2 = system
    Hmats, tix, th = Hdata
    Rt, R2t, G, G2, F = Hmats
    nu, nd = size(Q)[1], size(Q2)[1]

    ## Quadrature mats
    Rvt, R2vt = reshape(view(Rt,1:nu*tix,:) * v, nu, tix), reshape(view(R2t,1:nd*tix,:) * v, nd, tix)
    
    ## Quadrature sum
    H1 = th * (sum(map(norm, eachcol(G * Rvt))) + sum(a * Rvt)) # player 1 / control
    H2 = p2 ? th * (-sum(map(norm, eachcol(G2 * R2vt))) + sum(a2 * R2vt)) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_ball(system, target, t; ρ=1, ρ2=1, admm=false, F_init=nothing, Φ=nothing)

    M, C, C2, Q, Q2, a, a2 = system
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (typeof(M) <: Function || length(size(M)) == 1 ? solveΦ(M, t) : s->exp(s*M)) : Φ; Φt = Φ.(t)
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx);

    ## Precompute ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## TODO (admm only) only works when c (in J) == 0, need to include c in update_v on other side of eqn

    ## Precompute Rt Mats
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(t[end] - si) : (length(size(C)) == 2 ? C : C[end:-1:1][si])
        C2ti = typeof(C2) <: Function ? C2.(t[end] - si) : (length(size(C2)) == 2 ? C2 : C2[end:-1:1][si])  
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
function HJoc_ball(system, dϕdz, T; p2=true, Hdata=nothing, Φ=nothing)

    M, C, C2, Q, Q2, a, a2 = system
    nu, nd = size(Q)[1], size(Q2)[1]
    
    ## Handle Time-Varying Systems (constant or fn(time), array nonsensible)
    Φ = isnothing(Φ) ? (typeof(M) <: Function ? solveΦ(M,[T]) : s->exp(s*M)) : Φ; ΦT = Φ(T)
    CT  = typeof(C)  <: Function ? C(t[end] - T)  : C
    C2T = typeof(C2) <: Function ? C2(t[end] - T) : C2
    R, R2 = -(ΦT * CT)', -(ΦT * C2T)'

    _,Σ,VV = svd(Q);
    _,Σ2,VV2 = svd(Q2);
    G, G2 = Diagonal(sqrt.(Σ)) * VV, Diagonal(sqrt.(Σ2)) * VV2;

    uˢ = Q  != zero(Q)  ? inv(norm(G * R * dϕdz)) * Q * R * dϕdz + a' : zeros(nu)
    dˢ = Q2 != zero(Q2) && p2 ? - (inv(norm(G2 * R2 * dϕdz)) * Q2 * R2 * dϕdz - a2') : zeros(nd)

    return uˢ, dˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)
# could be faster by using Hdata

### Box-Constrained Inputs (Inf norm)

## Time integral of Hamiltonian for MRK et al. 2018, inf-norm control constraints
function intH_box(system, Hdata, v; p2)

    M, C, C2, Q, Q2, a, a2 = system
    Hmats, tix, th = Hdata
    QRt, QR2t, F = Hmats
    nu, nd = size(Q)[1], size(Q2)[1]

    ## Quadrature mats
    QRvt, QR2vt = reshape(view(QRt,1:nu*tix,:) * v, nu, tix), reshape(view(QR2t,1:nd*tix,:) * v, nd, tix)
    
    ## Quadrature sum
    H1 = th * sum(map(c->sum(abs,c), eachcol(QRvt))) # player 1 / control
    H2 = p2 ? th * -sum(map(c->sum(abs,c), eachcol(QR2vt))) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_box(system, target, t; ρ=1, ρ2=1, admm=false, F_init=false, Φ=nothing)

    M, C, C2, Q, Q2, a, a2 = system
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (typeof(M) <: Function || length(size(M)) == 1 ? solveΦ(M, t) : s->exp(s*M)) : Φ; Φt = Φ.(t)
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Precomputing ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## THIS ONLY WORKS WHEN c == 0 (admm only), ... unless we move c over todo

    ## Transformation Mats Q * R := Q * (exp(-(T-t)M)C)' over t
    QRt, QR2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx);
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(t[end] - si) : (length(size(C)) == 2 ? C : C[end:-1:1][si])
        C2ti = typeof(C2) <: Function ? C2.(t[end] - si) : (length(size(C2)) == 2 ? C2 : C2[end:-1:1][si])  
        R, R2 = -(Φt[si] * Cti)', -(Φt[si] * C2ti)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F  ## Does this need to be adapated for ADMM + Box?
        QRt[nu*(si-1) + 1 : nu*si, :], QR2t[nd*(si-1) + 1 : nd*si, :] = Q * R, Q2 * R2
    end

    F = admm ? inv(F) : F

    return (QRt, QR2t, F), Φ
end

## Optimal HJB Control # todo: check graphically
function HJoc_box(system, dϕdz, T; p2=true, Hdata=nothing, Φ=nothing)

    M, C, C2, Q, Q2, a, a2 = system
    nu, nd = size(Q)[1], size(Q2)[1]

    ## Handle Time-Varying Systems (constant or fn(time), array nonsensible)
    Φ = isnothing(Φ) ? (typeof(M) <: Function ? solveΦ(M,[T]) : s->exp(s*M)) : Φ; ΦT = Φ(T)
    CT  = typeof(C)  <: Function ? C(t[end] - T)  : C
    C2T = typeof(C2) <: Function ? C2(t[end] - T) : C2
    R, R2 = -(ΦT * CT)', -(ΦT * C2T)'

    QR, QR2 = Q * R, Q2 * R2

    # uˢ = inv(R') * QR * sign.(QR * dϕdz)
    # dˢ = p2 ? - inv(R2') * QR2 * sign.(QR2 * dϕdz) : zeros(nd)

    uˢ = Q  != zero(Q)  ? Q * sign.(QR * dϕdz) : zeros(nu)
    dˢ = Q2 != zero(Q2) && p2 ? - Q2 * sign.(QR2 * dϕdz) : zeros(nd)

    return uˢ, dˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)

### Standard Hamiltonian precomputation fn

function preH_std(system, target, t; ρ=1, ρ2=1, admm=false, Φ=nothing)

    M, C, C2, Q, Q2, a, a2 = system
    J, Jˢ, Jp = target
    th = t[2]

    ## Solve Φ
    Φ = isnothing(Φ) ? (!(typeof(M) <: Function) || length(size(M)) == 2 ? s->exp(s*M) : solveΦ(M, t)) : Φ; Φt = Φ.(t) 
    nx, nu, nd = size(Φt[1])[1], size(Q)[1], size(Q2)[1]

    ## Precomputing ADMM matrix
    F = Jp[1]

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(nu*length(t), nx), zeros(nd*length(t), nx);
    for si in eachindex(t)
        Cti = typeof(C) <: Function ? C.(si) : (length(size(C)) == 2 ? C : C[si])
        C2ti = typeof(C2) <: Function ? C2.(si) : (length(size(C2)) == 2 ? C2 : C2[si])  
        R, R2 = -(Φt[si] * Cti)', -(Φt[si] * C2ti)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F
        Rt[ nu*(si-1) + 1 : nu*si, :] = R;
        R2t[nd*(si-1) + 1 : nd*si, :] = R2;
    end

    F = admm ? inv(F) : F

    return (Rt, R2t, F), Φ
end

end


### WAYS THIS PKG CAN BE IMPROVED:
# - Parallelization
# - Sparse Arrays
# - Type Declaration
# - Automatic Convex Conjugate
# - Integration with other optimizers (& Automatic differentiation?)