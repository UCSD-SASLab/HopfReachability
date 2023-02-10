module HopfReachability

using LinearAlgebra, StatsBase, ScatteredInterpolation
using Plots, ImageFiltering, TickTock, Suppressor, PlotlyJS
plotlyjs()

##################################################################################################
##################################################################################################

### General

##################################################################################################
##################################################################################################


## Evaluate Hopf Cost
function Hopf(system, target, tbH, z, v; p2, dim=size(system[1])[1])

    J, Jˢ, Jp = target
    intH, Hdata = tbH

    return Jˢ(v, Jp...) - z'v + intH(system, Hdata, v; p2); ### MULTI-D FIX
    # return Jˢ(v, Jp...) - z'view(v,1:2) + intH(system, Hdata, v; p2); ### MULTI-D FIX
end

## Solve Hopf Reachability over Grid for given system, target, ∫Hamiltonian and lookback time(s) T
function Hopf_BRS(system, target, intH, T;
                  opt_method = Hopf_cd,
                  preH=preH_std, Xg=nothing, ϵ=0.5e-7, dim=size(system[1])[1], 
                  grid_p=(3, 10 + 0.5e-7), th=0.02, opt_p=nothing, warm=false, 
                  p2=true, plotting=false, printing=false, sampling=false, samples=360, zplot=false, check_all=true,
                  moving_target=false, moving_grid=false)

    if printing; println("\nSolving Backwards Reachable Set,"); end

    if opt_method == Hopf_cd
        opt_p = isnothing(opt_p) ? (0.01, 5, ϵ, 500, 20, 20) : opt_p
        admm = false
    elseif opt_method == Hopf_admm
        opt_p = isnothing(opt_p) ? (1e-4, 1e-4, 1e-5, 100) : opt_p 
        admm = true
    end

    ## Initialize
    # moving_target = typeof(target[3][1]) <: Matrix || typeof(target[3][1]) <: Vector || typeof(target[3][1]) <: Number ? false : true
    # moving_grid = typeof(Xg) <: Matrix || isnothing(Xg) ? false : true
    simple_problem = !moving_grid && !moving_target

    J, Jˢ, Jp = target
    M, A, c = system[1], Jp[1], Jp[2]
    t = collect(th: th: T[end])

    ## Initialize Data
    index, ϕX, B⁺, ϕB⁺, B⁺T, ϕB⁺T, v_init = [], [], [], [], [], [], nothing
    averagetimes, pointstocheck, N, lg = [], [], 0, 0

    ## Grid Set Up
    if isnothing(Xg)
        bd, N = grid_p
        lb, ub = typeof(bd) <: Tuple ? bd : (-bd, bd)
        xig = collect(lb : 1/N : ub) .+ ϵ; lg = length(xig)
        Xg = hcat(collect.(Iterators.product([xig for i in 1:dim]...))...)[end:-1:1,:] ## MULTI-D FIX
        # Xg = hcat(collect.(Iterators.product(xig .- ϵ, xig .+ ϵ))...)[end:-1:1,:] ## MULTI-D FIX
        
        #takes a while in REPL, but not when timed...
        
    elseif moving_grid
        # N = Int.([floor(inv(norm(Xg[i][:,1] - Xg[i][:,2]))) for i in eachindex(Xg)])
        N = [10 for i in eachindex(Xg)] # fix!
        lg = Int.([sqrt(size(Xg[i])[2])  for i in eachindex(Xg)])
    end

    ## Compute Near-Boundary set of Target in X
    if simple_problem
        
        ϕX = J(Xg, A, c) ## MULTI-D FIX
        # ϕX = J(vcat(Xg, zeros(dim-2, lg^2)), A, c) ## MULTI-D FIX
        if check_all
            B⁺, ϕB⁺, index = Xg, ϕX, collect(1:length(ϕX))
        else
            index = boundary(ϕX; lg, N, dim=dim) ## MULTI-D FIX
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index] 
        end

        push!(B⁺T, copy(B⁺)); push!(ϕB⁺T, copy(ϕB⁺)) # for plotting ϕ0
    end

    ## Precomputation
    Hmats = preH(system, target, t; opt_p, admm)

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

            ϕX = J(Xgi, Ai, ci) ## MULTI-D FIX
            # ϕX = J(vcat(Xgi, zeros(dim-2, size(Xgi)[2])), Ai, ci) ## MULTI-D FIX
            if check_all
                B⁺, ϕB⁺, index = Xgi, ϕX, collect(1:length(ϕX))
            else
                index = boundary(ϕX; lg=lgi, N=Ni, dim=dim) ## MULTI-D FIX
                B⁺, ϕB⁺ = Xgi[:, index], ϕX[index] 
            end

            push!(B⁺T, copy(B⁺)); push!(ϕB⁺T, copy(ϕB⁺))
        end

        if isempty(index)
            # error("At T=" * string(Ti) * ", no x in the grid s.t. |J(x)| < " * string(ϵ))
            if printing; println("At T=" * string(Ti) * ", no x in the grid s.t. |J(x)| < " * string(ϵ)); end
        end

        ## Map X to Z
        B⁺z = exp(Ti * M) * B⁺ ## MULTI-D FIX
        # B⁺z = exp(Ti * M) * vcat(B⁺, zeros(dim-2, size(B⁺)[2])) ## MULTI-D FIX
        target = (J, Jˢ, (Ai, ci))

        ## Packaging Hdata, tbH
        Hdata = (Hmats, tix, th)
        tbH = (intH, Hdata)

        @suppress begin; tick(); end # timing

        ## Pre solve details
        index_pts = sampling ? sample(1:length(index),samples) : eachindex(index) # sample for speed test

        ## Solve Over Grid (near ∂ID if !check_all)
        for bi in index_pts
            z = B⁺z[:, bi] ## MULTI-D FIX
            # z = B⁺z[1:2, bi] ## MULTI-D FIX
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
            index = boundary(ϕX; lg, N, dim=dim)
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index]
        end
    end
    
    totaltime = tok()

    if plotting; plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem, zplot); end
    if printing; println("TOTAL TIME: ", totaltime); end
    if printing; println("MEAN TIME[s] PER TIME POINT: ", averagetimes); end
    if printing; println("TOTAL POINTS PER TIME POINT: ", pointstocheck); end
    run_stats = (totaltime, averagetimes, pointstocheck)

    ## Return arrays of B⁺ over time and corresponding ϕ's, where the first is target (1 + length(T) arrays)
    # if moving problem, target inserted at each Ti (2 * length(T) arrays)
    return (B⁺T, ϕB⁺T), run_stats
end

## Solve Hopf Problem to find minimum T* so ϕ(z,T*) = 0, for given system & target
function Hopf_minT(system, target, intH, HJoc, x; 
                  opt_method=Hopf_cd, preH=preH_std,
                  T_init=nothing, v_init_=nothing,
                  time_p=(0.02, 0.1, 1.), opt_p=nothing, 
                  dim=size(system[1])[2], p2=true, printing=false, moving_target=false, warm=false)

    if opt_method == Hopf_cd
        opt_p = isnothing(opt_p) ? (0.01, 5, 0.5e-7, 500, 20, 20) : opt_p
        admm = false
    elseif opt_method == Hopf_admm
        opt_p = isnothing(opt_p) ? (1e-4, 1e-4, 1e-5, 100) : opt_p 
        admm = true
    end

    J, Jˢ, Jp = target
    th, Th, maxT = time_p
    t = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)

    ## Initialize ϕs
    ϕ, dϕdz = Inf, 0
    v_init = isnothing(v_init_) ? nothing : copy(v_init_) 

    ## Precomputing some of H for efficiency
    Hmats = preH(system, target, t; opt_p, admm)

    ## Loop Over Time Frames until ϕ(z, T) > 0
    Thi = 0; Tˢ = t[end]
    while ϕ > 0
        if printing; println("   checking T =-", Tˢ, "..."); end
        tix = length(t)
        Tix = Thi + 1

        ## Support Moving Targets
        Ai, ci = !moving_target ? Jp : (Jp[1][Tix], Jp[2][Tix])
        target = (J, Jˢ, (Ai, ci))

        ## Packaging Hdata, tbH
        Hdata = (Hmats, tix, th)
        tbH = (intH, Hdata)

        ## Hopf-Solving to check current Tˢ
        z = exp(Tˢ * system[1]) * x
        ϕ, dϕdz, v_arr = opt_method(system, target, z; p2, tbH, opt_p, v_init)
        v_init = warm ? copy(dϕdz) : nothing

        ## Check if Tˢ yields valid ϕ
        if ϕ <= 0; break; end    
        if Tˢ > maxT; println("!!! Not reachable under max time !!!"); break; end    

        ## Update the t, to increase Tˢ
        t_ext = collect(Tˢ + th : th : Tˢ + Th + th/2); push!(t, t_ext...)
        Tˢ, Thi = t[end], Thi + 1

        ## Extending previous precomputed Hdata for increased final time
        F_init = admm ? Hmats[end] : nothing
        Hmats_ext = preH(system, target, t_ext; opt_p, admm, F_init)
        Rt_ext, R2t_ext = Hmats_ext[1], Hmats_ext[2]
        Rt, R2t  = vcat(Hmats[1], Rt_ext), vcat(Hmats[2], R2t_ext)
        Hmats = (Rt, R2t, Hmats[3:end-1]..., Hmats_ext[end])

    end

    ## Compute Optimal Control (dH/dp(dϕdz, Tˢ) = exp(-Tˢ * M)Cuˢ + exp(-Tˢ * M)C2dˢ)
    uˢ, dˢ = HJoc(system, dϕdz, Tˢ; p2)

    return uˢ, dˢ, Tˢ, ϕ
end


##################################################################################################
##################################################################################################

### Optimization Methods

##################################################################################################
##################################################################################################


## Iterative Coordinate Descent of the Hopf Cost at Point z
function Hopf_cd(system, target, z; p2, tbH, opt_p, v_init=nothing, dim=size(system[1])[1], max_its=100)

    vh, L, tol, lim, lll, max_runs = opt_p
    solution, v = 1e9, nothing
    v_arr = zeros(dim, max_its)

    converged_runs, runs = 0, 0
    while converged_runs < lll

        v = isnothing(v_init) ? 10*(rand(dim) .- 0.5) : copy(v_init) # Hopf optimizer variable
        v_arr_temp = zeros(dim, max_its)

        kcoord = copy(dim); # coordinate counter
        stopcount = 0; # convergence flag
        happycount = 0; # step-size flag

        ## Init Hopf val: J*(v) - z'v + ∫Hdt
        fnow = Hopf(system, target, tbH, z, v; p2)
        
        ci = 1
        while true

            if ci < max_its + 1; v_arr_temp[:, ci] = copy(v); end

            ## Nearby Point along Coordinate
            kcoord = mod(kcoord, dim) + 1; #iterate coords
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

            if stopcount == dim || ci == max_its # in thresh for all coordinates
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
                println("Caution only ", 100*converged_runs/max_runs, "% of runs converged.")
                break
            else
                error("All optimization runs diverged! :(")
            end
        else
            runs += 1
        end
    end

    return -solution, v, v_arr
end

## Proximal Splitting Algorithm for optimizing the Hopf Cost at point z (vectorized)
function Hopf_admm(system, target, z; p2, tbH, opt_p, v_init=nothing, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    ## Initialize
    M, B, C, Q, Q2, a, a2 = system
    ρ, ρ2, tol, max_its = opt_p
    tix = tbH[2][2]
    Rt, R2t = tbH[2][1][1:2]

    ## Costate, Artificial Constraint Curves, Augmented Lagrange Multipliers
    v = isnothing(v_init) ? 0*ones(dim) : v_init
    γ, γ2, λ, λ2  = 0*ones(dim_u, tix), 0*ones(dim_d, tix), 0*ones(dim_u, tix), 0*ones(dim_d, tix) 
    v_arr = zeros(dim, max_its)

    ## Update Quadrature Mats
    Rvt, R2vt = reshape(view(Rt,1:dim_u*tix,:) * v, dim_u, tix), reshape(view(R2t,1:dim_d*tix,:) * v, dim_d, tix)

    ci = 1
    while true

        v_old = copy(v)
        v_arr[:, ci] = copy(v)

        ## Proximal Updates of Characteristic Curves
        γ  = Q  != zero(Q)  ? Rvt  + λ  -  shrink_BM(Rvt  + λ,  Q)   : Rvt  + λ
        γ2 = Q2 != zero(Q2) ? R2vt + λ2 + stretch_BM(R2vt + λ2, Q2) : R2vt + λ2
        ξ, ξ2 = γ - λ, γ2 - λ2

        ## Proximal Update of Costate
        v = update_v(z, ξ, ξ2, tbH, ρ, ρ2)

        ## Linear Update of ALM
        λ  = Rvt  - ξ
        λ2 = R2vt - ξ2

        # println("tol: ", norm(v_old - v))
        if norm(v_old - v) < tol || ci >= max_its; break; end
        ci += 1

        ## Update Quadrature Mats
        Rvt, R2vt = reshape(view(Rt,1:dim_u*tix,:) * v, dim_u, tix), reshape(view(R2t,1:dim_d*tix,:) * v, dim_d, tix)
    end

    return -Hopf(system, target, tbH, z, v; p2), v, v_arr
end

## Shrink Operator for Characteristic Curve of Convex Player (Bisection Method specifically for Ellipses, from GTools 2013)
function shrink_BM(Wi, A; tol=0.5e-9, ϵ=1e-3)

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

## Stretch Operator for Characteristic Curve of Concave Player (Bisection Method specifically for Ellipses, WAS)
function stretch_BM(Wi, A; tol=0.5e-9, ϵ=1e-3)

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
function update_v(z, ξ, ξ2, tbH, ρ, ρ2; dim_u=size(tbH[2][1][1])[2], dim_d=size(tbH[2][1][2])[2])

    Hmats, tix, th = tbH[2]
    Rt, R2t = Hmats[1:2]
    F = Hmats[end]

    y = z
    for s = 1:tix
        y += th * ((ρ * view(Rt, dim_u*(s-1)+1: dim_u*s,:)' * view(ξ,:,s) + (ρ2 * view(R2t, dim_d*(s-1)+1: dim_d*s,:)' * view(ξ2,:,s))))
    end

    return F * y # need to subtract c from RHS/y todo
end


##################################################################################################
##################################################################################################

### Utility fn's

##################################################################################################
##################################################################################################

## Find points near boundary ∂ of f(z) = 0
function boundary(ϕ; lg, N, δ = 20/N, dim=size(system[1])[1]) ## MULTI-D FIX

    A = Float64.(abs.(reshape(ϕ, [lg for i=1:dim]...)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    # A = Float64.(abs.(reshape(ϕ, lg, lg)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones([Int(floor(N/2)) for i=1:dim]...); # kernel
    # B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel

    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv

    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Plots BRS over T in X and Z space
function plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem=true, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, zplot=false, interpolate=false, inter_method=Polyharmonic(), pal_colors=[:red, :blue], alpha=0.5, title=nothing, value_fn=false, dim=size(B⁺T[1])[1])

    if dim > 2 && value_fn; println("4D plots are not supported yet, can't plot Value fn"); value_fn = false; end

    Xplot = isnothing(title) ? Plots.plot(title="BRS: Φ(X, T) = 0") : Plots.plot(title=title)
    if zplot; Zplot = Plots.plot(title="BRS: Φ(Z, T) = 0"); end

    plots = zplot ? [Xplot, Zplot] : [Xplot]
    if value_fn; vfn_plots = zplot ? [Plots.plot(title="Value: Φ(X, T)"), Plots.plot(title="Value: Φ(Z, T)")] : [Plots.plot(title="Value: Φ(X, T)")]; end

    B⁺Tc, ϕB⁺Tc = copy(B⁺T), copy(ϕB⁺T)
    
    ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
    Jlabels = "J(⋅, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
    labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

    Tcolors = length(T) > 1 ? palette(pal_colors, length(T)) : [pal_colors[2]]
    B0colors = length(T) > 1 ? palette([:black, :gray], length(T)) : [:black]
    plot_colors = collect(Iterators.flatten(zip(B0colors, Tcolors)))

    ## Zipping Target to Plot Variation in Z-space over Time (already done in moving problems)
    if simple_problem && (length(T) > 1)
        for i = 3 : 2 : 2*length(T)
            insert!(B⁺Tc, i, B⁺T[1])
            insert!(ϕB⁺Tc, i, ϕB⁺T[1])
        end
    end

    if dim > 2 && interpolate; plotly_pl = zplot ? [Array{GenericTrace{Dict{Symbol, Any}},1}(), Array{GenericTrace{Dict{Symbol, Any}},1}()] : [Array{GenericTrace{Dict{Symbol, Any}},1}()]; end

    for (j, i) in enumerate(1 : 2 : 2*length(T))        
        B⁺0, B⁺, ϕB⁺0, ϕB⁺ = B⁺Tc[i], B⁺Tc[i+1], ϕB⁺Tc[i], ϕB⁺Tc[i+1]
        Bs = zplot ? [B⁺0, B⁺, exp(-T[j] * M) * B⁺0, exp(-T[j] * M) * B⁺] : [B⁺0, B⁺]

        for (bi, b⁺) in enumerate(Bs)
            if simple_problem && bi == 1 && i !== 1; continue; end

            ϕ = bi % 2 == 1 ? ϕB⁺0 : ϕB⁺
            label = simple_problem && i == 1 && bi == 1 ? "J(⋅)" : labels[i + (bi + 1) % 2]

            ## Plot Scatter
            if interpolate == false

                ## Find Boundary in Near-Boundary
                b = b⁺[:, abs.(ϕ) .< ϵs]

                scatter!(plots[Int(bi > 2) + 1], [b[i,:] for i=1:dim]..., label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                # scatter!(plots[Int(bi > 2) + 1], b[1,:], b[2,:], label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                
                if value_fn
                    scatter!(vfn_plots[Int(bi > 2) + 1], b⁺[1,:], b⁺[2,:], ϕ, label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha)
                    # scatter!(vfn_plots[Int(bi > 2) + 1], b[1,:], b[2,:], ϕ, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], label=label)
                end
            
            ## Plot Interpolation
            else 

                if dim == 2
                    contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] for i=1:dim]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label)

                    if value_fn

                        ## Make Grid
                        xig = [collect(minimum(b⁺[i,:]) : cres : maximum(b⁺[i,:])) for i=1:dim]
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

    if dim > 2 && interpolate 
        Xplot = PlotlyJS.plot(plotly_pl[1], Layout(title="BRS of T, in X"));
        if zplot; Zplot = PlotlyJS.plot(plotly_pl[2], Layout(title="BRS of T, in X")); end
    end

    display(Xplot); plots = [Xplot]
    if zplot; display(Zplot); plots = [Xplot, Zplot]; end

    return plots
end


##################################################################################################
##################################################################################################

### Example Hamiltonians and Precomputation fn's

##################################################################################################
##################################################################################################


### Example 1 

## Time integral of Hamiltonian for YTC et al. 2017, quadratic control constraints
function intH_ytc17(system, Hdata, v; p2, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2, a, a2 = system
    Hmats, tix, th = Hdata
    Rt, R2t, G, G2 = Hmats

    ## Quadrature mats
    Rvt, R2vt = reshape(view(Rt,1:dim_u*tix,:) * v, dim_u, tix), reshape(view(R2t,1:dim_d*tix,:) * v, dim_d, tix)
    
    ## Quadrature sum
    H1 = th * (sum(map(norm, eachcol(G * Rvt))) + sum(a * Rvt)) # player 1 / control
    H2 = p2 ? th * (-sum(map(norm, eachcol(G2 * R2vt))) + sum(a2 * R2vt)) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_ytc17(system, target, t; opt_p, admm=false, F_init=nothing, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2 = system
    J, Jˢ, Jp = target
    th = t[2]
    ρ, ρ2 = opt_p[1:2]

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(dim_u*length(t), dim), zeros(dim_d*length(t), dim);

    ## Precomputing ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## todo only works when c (in J) == 0, need to include c in update_v

    for sstep in eachindex(t)
        R, R2 = -(exp(t[sstep] * M) * C)', -(exp(t[sstep] * M) * C2)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F
        Rt[ dim_u*(sstep-1) + 1 : dim_u*sstep, :] = R;
        R2t[dim_d*(sstep-1) + 1 : dim_d*sstep, :] = R2;
    end

    ## Precomputing SVD for matrix sqrt
    _, Σ,  VV  = svd(Q);
    _, Σ2, VV2 = svd(Q2);
    G  = Diagonal(sqrt.(Σ)) * VV;
    G2 = Diagonal(sqrt.(Σ2)) * VV2;

    ## Precomputing ADMM matrix
    F = admm ? inv(F) : F

    return Rt, R2t, G, G2, F
end

## Optimal HJB Control
function HJoc_ytc17(system, dϕdz, T; p2=true, Hdata=nothing, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2, a, a2 = system
    R, R2 = -(exp(T * M) * C)', -(exp(T * M) * C2)'

    _,Σ,VV = svd(Q);
    _,Σ2,VV2 = svd(Q2);
    G, G2 = Diagonal(sqrt.(Σ)) * VV, Diagonal(sqrt.(Σ2)) * VV2;

    uˢ = inv(norm(G * R * dϕdz)) * Q * R * dϕdz + a' 
    dˢ = p2 ? - (inv(norm(G2 * R2 * dϕdz)) * Q2 * R2 * dϕdz - a2') : zeros(dim_d)

    return uˢ, dˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)
# could be faster by using Hdata

### Example 2

## Time integral of Hamiltonian for MRK et al. 2018, inf-norm control constraints
function intH_mrk18(system, Hdata, v; p2, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2 = system
    QRt, QR2t, tix, th = Hdata

    ## Quadrature mats
    QRvt, QR2vt = reshape(view(QRt,1:dim_u*tix,:) * v, dim_u, t), reshape(view(QR2t,1:dim_d*tix,:) * v, dim_d, t)
    
    ## Quadrature sum
    H1 = th * sum(map(c->sum(abs,c), eachcol(QRvt))) # player 1 / control
    H2 = p2 ? th * -sum(map(c->sum(abs,c), eachcol(QR2vt))) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_mrk18(system, target, t; opt_p, admm=false, F_init=false, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2 = system
    J, Jˢ, Jp = target
    th = t[2]
    ρ, ρ2 = opt_p[1:2]
    
    ## Precomputing ADMM matrix
    F = isnothing(F_init) ? Jp[1] : inv(F_init) ## THIS ONLY WORKS WHEN c == 0 (admm only), ... unless we move c over todo

    ## Transformation Mats Q * R := Q * (exp(-(T-t)M)C)' over t
    QRt, QR2t = zeros(dim_u*length(t), dim), zeros(dim_d*length(t), dim);
    for sstep in eachindex(t)
        R, R2 = - (exp(t[sstep] * M) * C)', - (exp(t[sstep] * M) * C2)';
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F
        QRt[dim_u*(sstep-1) + 1 : dim_u*sstep, :], QR2t[dim_d*(sstep-1) + 1 : dim_d*sstep, :] = Q * R, Q2 * R2
    end

    F = admm ? inv(F) : F

    return QRt, QR2t, F
end

## Optimal HJB Control # todo: check graphically
function HJoc_mrk18(system, dϕdz, T; Hdata=nothing, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2 = system
    R, R2 = -(exp(T * M) * C)', -(exp(T * M) * C2)'
    QR, QR2 = Q * R, Q * R2

    uˢ = inv(R') * QR * sign.(QR * dϕdz)
    dˢ = p2 ? - inv(R2') * QR2 * sign.(QR2 * dϕdz) : zeros(dim_d)

    return uˢ, dˢ
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)

### Standard Hamiltonian precomputation fn

function preH_std(system, target, t; opt_p, admm=false, dim=size(system[1])[1], dim_u=size(system[2])[2], dim_d=size(system[3])[2])

    M, C, C2, Q, Q2 = system
    J, Jˢ, Jp = target
    th = t[2]
    ρ, ρ2 = opt_p[1:2]

    ## Precomputing ADMM matrix
    F = Jp[1]

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(dim_u*length(t), dim), zeros(dim_d*length(t), dim);
    for sstep in eachindex(t)
        R, R2 = -(exp(t[sstep] * M) * C)', -(exp(t[sstep] * M) * C2)'
        F = admm ? F + th * ((ρ * R' * R) + (ρ2 * R2' * R2)) : F
        Rt[ dim_u*(sstep-1) + 1 : dim_u*sstep, :] = R;
        R2t[dim_d*(sstep-1) + 1 : dim_d*sstep, :] = R2;
    end

    F = admm ? inv(F) : F

    return Rt, R2t, F
end

end