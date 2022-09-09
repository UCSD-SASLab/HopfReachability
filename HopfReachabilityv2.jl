module HopfReachabilityv2

using LinearAlgebra, StatsBase, ScatteredInterpolation
using Plots, ImageFiltering, TickTock, Suppressor

## Evaluate Hopf Cost
function Hopf(system, target, tbH, z, v; p2, kkk=size(system[1])[1])

    J, Jˢ, Az = target
    intH, Hdata = tbH

    return Jˢ(v, Az) - z'view(v,1:2) + intH(system, Hdata, v; p2);
end

## Iterative Coordinate Descent of the Hopf Cost at Point z
function Hopf_cd(system, target, z; p2, tbH, opt_p, v_init=nothing, kkk=size(system[1])[1])

    vh, L, tol, lim, lll = opt_p
    solni, v = -Inf, nothing

    # Loop over Hopf optimizations (coordinate descents)
    for ll = 1:lll

        v = isnothing(v_init) ? 10*(rand(kkk) .- 0.5) : v_init # Hopf optimizer variable
        kcoord = kkk; # coordinate counter
        stopcount = 0; # convergence flag
        happycount = 0; # step-size flag

        ## Init Hopf val: J*(v) - z'v + ∫Hdt
        fnow = Hopf(system, target, tbH, z, v; p2)
        
        while true

            ## Nearby Point along Coordinate
            kcoord = mod(kcoord,kkk) + 1; #iterate coords
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
            happycount, L = happycount > lim ? (1, L*2) : (happycount+1, L); 

            if stopcount == kkk # in thresh for all coordinates
                # why are these all negative????
                solni = -min(-solni, fnow) # min with iter, to overcome potential local convergence
                break
            end

            fnow = fnownow;
        end
    end

    if isnothing(v)
        error("dϕdz has not overwritten")
    end

    if solni == -Inf
        error("Solution is -Inf ?")
    end

    return solni, v
end

## Solve Hopf Problem for given system, target and lookback time(s) T
function Hopf_BRS(system, target, intH, T; preH=preH_std, Xg=nothing, ϵ=0.5e-7, grid_p=(3, 10 + 0.5e-7), th=0.02, opt_p=(0.01, 5, 0.5e-7, 500, 20), kkk=size(system[1])[1], p2=true, plotting=false, printing=false, sampling=false, samples=360, zplot=false)

    ## Initialize
    moving_target = typeof(target[3]) <: Matrix ? false : true
    moving_grid = typeof(Xg) <: Matrix ? false : true
    simple_problem = !moving_grid && !moving_target

    J, Jˢ, A = target
    M = system[1]
    t = collect(th: th: T[end])
    index, ϕX, B⁺, ϕB⁺, B⁺T, ϕB⁺T = [], [], [], [], [], [] 
    averagetime, N, lg = 0, 0, 0

    ## Grid Set Up
    if isnothing(Xg)
        bd, N = grid_p
        xig = collect(-bd : 1/N : bd); lg = length(xig)
        # Xg = [[j,i] for (i,j) in collect(Iterators.product(xig .- ϵ, xig .+ ϵ))[:]]
        Xg = hcat(collect.(Iterators.product(xig .- ϵ, xig .+ ϵ))...)[end:-1:1,:] #takes a while in REPL, but not when timed...
    elseif moving_grid
        # N = Int.([floor(inv(norm(Xg[i][:,1] - Xg[i][:,2]))) for i in eachindex(Xg)])
        N = [10 for i in eachindex(Xg)] # fix!
        lg = Int.([sqrt(size(Xg[i])[2])  for i in eachindex(Xg)])
    end

    ## Compute Near-Boundary set of Target in X
    if simple_problem
        ϕX = J(vcat(Xg, zeros(kkk-2, lg^2)), A)#plotting in z bcuz ytc paper in z
        index = boundary(ϕX; lg, N)
        B⁺, ϕB⁺ = Xg[:, index], ϕX[index] 

        push!(B⁺T, copy(B⁺)); push!(ϕB⁺T, copy(ϕB⁺)) # for plotting ϕ0
    end

    ## Precomputing some of H for efficiency
    Hmats = preH(system, t)

    ## Loop Over Time Frames
    for Tix in eachindex(T)

        Ti = T[Tix]
        Xgi = !moving_grid ? Xg : Xg[Tix]
        Ai = !moving_target ? A : A[Tix]
        tix = findfirst(abs.(t .-  Ti) .< th/2);
        if printing; println("   for t=-", Ti, "..."); end
        
        ## Update Near-Boundary set for moving problems
        if moving_grid || moving_target
            Ni = moving_grid ? N[Tix] : N
            lgi = moving_grid ? lg[Tix] : lg

            ϕX = J(vcat(Xgi, zeros(kkk-2, size(Xgi)[2])), Ai)
            index = boundary(ϕX; lg=lgi, N=Ni)
            B⁺, ϕB⁺ = Xgi[:, index], ϕX[index]

            push!(B⁺T, copy(B⁺))
            push!(ϕB⁺T, copy(ϕB⁺))
        end

        ## Map X to Z
        B⁺z = exp(-Ti * M) * B⁺
        Az = exp(Ti * M)' * Ai * exp(Ti * M) #A → Az so J, Jˢ → Jz, Jˢz
        target = (J, Jˢ, Az)

        if isempty(index)
            error("At T=" * string(Ti) * ", no x in the grid s.t. |J(x)| < " * string(ϵ))
        end

        ## Packaging Hdata, tbH
        Hdata = (Hmats..., tix, th)
        tbH = (intH, Hdata)

        @suppress begin; tick(); end # timing

        ## Loop Over Grid (near ∂ID)
        ϕB⁺before = copy(ϕB⁺)
        
        index_pts = sampling ? sample(1:length(index),samples) : eachindex(index) # sample for speed test
        for bi in index_pts
            z = B⁺z[:, bi]
            ϕB⁺[bi], dϕdz = Hopf_cd(system, target, z; p2, tbH, opt_p)
        end

        averagetime = tok()/length(index_pts); #includes lll reopts/pt

        ## Store Near-Boundary solution
        push!(B⁺T, copy(B⁺))
        push!(ϕB⁺T, copy(ϕB⁺))

        ## Update Near-Boundary index to intermediate solution
        if simple_problem && Tix != length(T)
            ϕX[index] = ϕB⁺
            index = boundary(ϕX; lg=lgi, N=Ni)
            B⁺, ϕB⁺ = Xg[:, index], ϕX[index]
        end
    end

    if plotting; plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem, zplot); end
    if printing; println("MEAN TIME: ", averagetime); end

    return (B⁺T, ϕB⁺T), averagetime
end

## Solve Hopf Problem to find minimum T* so ϕ(z,T*) = 0, for given system & target
function Hopf_minT(system, target, intH, HJoc, x; T_init=nothing, preH=preH_std, time_p=(0.02, 0.1, 1), opt_p=(0.01, 5, 0.5e-7, 500, 20), kkk=size(system[1])[1], p2=true, printing=false)

    J, Jˢ, A = target
    th, Th, maxT = time_p
    t = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)
    moving_target = typeof(A) <: Matrix ? false : true

    ## Initialize ϕs
    ϕ, dϕdz = Inf, 0
    v_init = nothing 

    ## Precomputing some of H for efficiency
    Hmats = preH(system, t)

    ## Loop Over Time Frames until ϕ(z, T) > 0
    Thi = 0; Tˢ = t[end]
    while ϕ > 0
        if printing; println("   checking T =-", Tˢ, "..."); end
        # tix = findfirst(abs.(t .-  Ti) .< th/2);
        tix = length(t)
        Tix = Thi + 1 #assuming Thi == stepping in A

        ## A → Az so J, Jˢ → Jz, Jˢz
        M = system[1]
        if !(moving_target)
            Az = exp(Tˢ * M)' * A * exp(Tˢ * M)
        else
            Az = exp(Tˢ * M)' * A[Tix] * exp(Tˢ * M)
        end
        z = exp(Tˢ * M) * x
        target = (J, Jˢ, Az)

        ## Packaging Hdata, tbH
        Hdata = (Hmats..., tix, th)
        tbH = (intH, Hdata)

        ## Hopf-Solving to check current Tˢ
        ϕ, dϕdz = Hopf_cd(system, target, z; p2, tbH, opt_p, v_init)

        ## Check if Tˢ yields valid ϕ
        if ϕ <= 0
            break
        end        

        ## Update the t, to increase Tˢ
        Thi += 1; 
        # t_ext = isnothing(T_init) ? collect((Thi-1)*Th + th: th: Thi*Th) : collect(T_init + (Thi-1)*Th + th: th: Thi*Th)
        t_ext = collect(Tˢ + th : th : Tˢ + Th)
        push!(t, t_ext...)
        Tˢ = t[end]

        ## Extending previous precomputed Hdata for increased final time
        exts = preH(system, t_ext)
        Rt_ext = cat(Hmats[1], exts[1], dims=2) # Rt and R2t need to be first ***
        R2t_ext = cat(Hmats[2], exts[2], dims=2)
        Hmats = length(Hmats) == 2 ? (Rt_ext, R2t_ext) : (Rt_ext, R2t_ext, Hmats[3:end]...)

        # v_init = dϕdz ???

        if Tˢ > maxT
            println("!!! Couldn't find a ϕ under max time")
            break
        end
    end

    ## Compute Optimal Control (dH/dp(dϕdz, Tˢ) = exp(-Tˢ * M)Cuˢ + exp(-Tˢ * M)C2dˢ)
    uˢ = HJoc(system, dϕdz, Tˢ)

    return ϕ, uˢ, Tˢ
end

##################################################################################################
##################################################################################################

### Utility fn's

##################################################################################################


## Find points near boundary ∂ of f(z) = 0
function boundary(ϕ; lg, N, δ = 5/N)

    A = Float64.(abs.(reshape(ϕ, lg, lg)) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel

    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv

    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Plots BRS over T in X and Z space (scatter)
function plot_BRS_scatter(T, B⁺T, ϕB⁺T; M, simple_problem, ϵ = 0.1, zplot=true)

    Xplot = plot(title="BRS of T, in X") 
    if zplot; Zplot = plot(title="BRS of T, in Z"); end
    ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
    Tcolors = palette([:red, :blue], length(T))
    
    if simple_problem
        labels = vcat("J(⋅)", ϕlabels) # length(T) + 1
        colors = [:black, Tcolors...]

        for i = 1 : length(T) + 1 # first is Target
            B⁺, ϕB⁺ = B⁺T[i], ϕB⁺T[i]
            B = B⁺[:, abs.(ϕB⁺) .< ϵ]
            Bz = exp(-[Ti] * M) * B

            ## Plot
            scatter!(Xplot, B[1,:], B[2,:], label=labels[i], markersize=1.5, markercolor=colors[i], markerstrokewidth=0)
            if zplot; scatter!(Zplot, Bz[1,:], Bz[2,:], label=labels[i], markersize=1.5, markercolor=colors[i], markerstrokewidth=0); end
        end

    else # moving target and/or grid
        Jlabels = "J(⋅," .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
        labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)
        B0colors = palette([:black, :gray], length(T))

        for i = 1 : 2 : 2*length(T)
            B⁺, ϕB⁺0, ϕB⁺ = B⁺T[i], ϕB⁺T[i], ϕB⁺T[i+1] # B⁺0 = B⁺ for any Ti
            B0, B = B⁺[:, abs.(ϕB⁺0) .< ϵ], B⁺[:, abs.(ϕB⁺) .< ϵ]
            Bz0, Bz = exp(-T[Int((i+1)/2)] * M) * B0, exp(-T[Int((i+1)/2)] * M) * B

            ## Plot
            scatter!(Xplot, B0[1,:], B0[2,:], label=labels[i], markersize=2, markercolor=B0colors[Int((i+1)/2)], markerstrokewidth=0, markershape=:square)
            if zplot; scatter!(Zplot, Bz0[1,:], Bz0[2,:], label=labels[i], markersize=2, markercolor=B0colors[Int((i+1)/2)], markerstrokewidth=0, markershape=:square); end

            scatter!(Xplot, B[1,:], B[2,:], label=labels[i+1], markersize=1.5, markercolor=Tcolors[Int((i+1)/2)], markerstrokewidth=0)
            if zplot; scatter!(Zplot, Bz[1,:], Bz[2,:], label=labels[i+1], markersize=1.5, markercolor=Tcolors[Int((i+1)/2)], markerstrokewidth=0); end
        end
    end

    display(Xplot)
    if zplot; display(Zplot); end
end

## Plots BRS over T in X and Z space (contour)
function plot_BRS(T, B⁺T, ϕB⁺T; M, simple_problem, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, zplot=true, contour=false, inter_method=Polyharmonic())

    Xplot = plot(title="BRS of T, in X") 
    if zplot; Zplot = plot(title="BRS of T, in Z"); end
    plots = zplot ? [Xplot, Zplot] : [Xplot]

    ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
    simple_labels = vcat("J(⋅)", ϕlabels) # length(T) + 1
    Jlabels = "J(⋅, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
    moving_labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

    Tcolors = palette([:red, :blue], length(T))
    B0colors = palette([:black, :gray], length(T))

    I = simple_problem ? (1 : length(T)+1) : (1 : 2 : 2*length(T))
    j = 1

    for i in I
        B⁺, ϕB⁺0, ϕB⁺ = simple_problem ? (B⁺T[i], 0, ϕB⁺T[i]) : (B⁺T[i], ϕB⁺T[i], ϕB⁺T[i+1]) # B⁺0 = B⁺ for any Ti
        ϕs = simple_problem ? [ϕB⁺] : [ϕB⁺0, ϕB⁺]
        B⁺z = exp(-T[j] * M) * B⁺

        Bs = zplot ? [B⁺, B⁺z] : [B⁺]

        for (bi, b⁺) in enumerate(Bs)
            for (ϕi, ϕ) in enumerate(ϕs)

                if contour == false
                    ## Find Boundary in Near-Boundary
                    b = b⁺[:, abs.(ϕ) .< ϵs]

                    ## Plot Scatter
                    color = simple_problem ? [:black, Tcolors...][i] : (ϕi == 1 ? B0colors[j] : Tcolors[j])
                    label = simple_problem ? simple_labels[i] : moving_labels[i]
                    scatter!(plots[bi], b[1,:], b[2,:], label=label, markersize=2, markercolor=color, markerstrokewidth=0)
                
                else
                    ## Make Grid
                    xg = collect(minimum(b⁺[1,:]) : cres : maximum(b⁺[1,:]))
                    yg = collect(minimum(b⁺[2,:]) : cres : maximum(b⁺[2,:]))
                    G = hcat(collect.(Iterators.product(xg, yg))...)'
                    
                    ## Construct Interpolation
                    itp = interpolate(inter_method, b⁺, ϕ)
                    itpd = evaluate(itp, G')
                    iϕG = reshape(itpd, length(xg), length(yg))'

                    ## Plot Contour
                    color = simple_problem ? [:black, Tcolors...][i] : (ϕi == 1 ? B0colors[j] : Tcolors[j])
                    contour!(plots[bi], xg, yg, iϕG, levels=[-ϵc, ϵc], colorbar = false, lc=color, lw=2)
                end
            end
        end

        j+=1
    end

    display(Xplot)
    if zplot; display(Zplot); end

    return plots
end

##################################################################################################
##################################################################################################

### Example Hamiltonians and Precomputation fn's

##################################################################################################

### Example 1 

## Time integral of Hamiltonian for YTC et al. 2017, quadratic control constraints
function intH_ytc17(system, Hdata, v; p2, kkk=size(system[1])[1])

    M, C, C2, Q, Q2, a1, a2 = system
    Rt, R2t, G, G2, tix, th = Hdata

    ## Quadrature mats
    Temp1 = zeros(kkk, tix); Temp2 = copy(Temp1);
    for i in axes(Temp1,2)
        Temp1[:,i] = view(view(Rt,:,1 : kkk*tix), :, (i-1)*kkk + 1 : i*kkk) * v;
        Temp2[:,i] = view(view(R2t,:,1 : kkk*tix), :, (i-1)*kkk + 1 : i*kkk) * v;
    end
    
    ## Quadrature sum
    H1 = th * (sum(map(norm, eachcol(G * Temp1))) + sum(a1 * Temp1)) # player 1 / control
    H2 = p2 ? th * (-sum(map(norm, eachcol(G2 * Temp2))) + sum(a2 * Temp2)) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_ytc17(system, t; kkk=size(system[1])[1])

    M, C, C2, Q, Q2 = system

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(kkk, kkk*length(t)), zeros(kkk, kkk*length(t));
    for sstep in eachindex(t)
        Rt[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(t[end] - t[sstep]) * M) * C)';
        R2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(t[end] - t[sstep]) * M) * C2)';
    end

    ## Precomputing SVD for matrix sqrt
    _,Σ,VV = svd(Q);
    _,Σ2,VV2 = svd(Q2);
    G = Diagonal(sqrt.(Σ)) * VV;
    G2 = Diagonal(sqrt.(Σ2)) * VV2;

    return Rt, R2t, G, G2
end

## Optimal HJB Control
function HJoc_ytc17(system, dϕdz, T; Hdata=nothing, kkk=size(system[1])[1])

    M, C, C2, Q, Q2, a1, a2 = system

    _,Σ,VV = svd(Q);
    G = Diagonal(sqrt.(Σ)) * VV;
    R = (exp(-T * M) * C)'

    # println(size(G))
    # println(size(R))
    # println(size(Q))
    # println(size(a1))
    # println(size(dϕdz))

    return inv(norm(G * R * dϕdz)) * Q * R * dϕdz + a1'
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)
# could be faster by using Hdata

##################################################################################################

### Example 2

## Time integral of Hamiltonian for MRK et al. 2018, inf-norm control constraints
function intH_mrk18(system, Hdata, v; p2, kkk=size(system[1])[1])

    M, C, C2, Q, Q2 = system
    QRt, QR2t, tix, th = Hdata

    ## Quadrature mats (does this speed anything up vs. repmat(v))
    Temp1 = zeros(kkk, tix); Temp2 = copy(Temp1);
    for i in axes(Temp1,2)
        Temp1[:,i] = view(view(QRt,:,1 : kkk*tix), :, (i-1)*kkk + 1 : i*kkk) * v;
        Temp2[:,i] = view(view(QR2t,:,1 : kkk*tix), :, (i-1)*kkk + 1 : i*kkk) * v;
    end
    
    ## Quadrature sum
    H1 = th * sum(map(c->sum(abs,c), eachcol(Temp1))) # player 1 / control
    H2 = p2 ? th * -sum(map(c->sum(abs,c), eachcol(Temp2))) : 0 # player 2 / disturbance, opponent   

    return H1 + H2
end

## Hamiltonian Precomputation
function preH_mrk18(system, t; kkk=size(system[1])[1])

    M, C, C2, Q, Q2 = system

    ## Transformation Mats Q * R := Q * (exp(-(T-t)M)C)' over t
    QRt, QR2t = zeros(kkk, kkk*length(t)), zeros(kkk, kkk*length(t));
    for sstep in eachindex(t)
        QRt[:,kkk*(sstep-1) + 1 : kkk*sstep] = Q * (exp(-(t[end] - t[sstep]) * M) * C)';
        QR2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = Q2 * (exp(-(t[end] - t[sstep]) * M) * C2)';
    end

    return QRt, QR2t
end

## Optimal HJB Control 
function HJoc_mrk18(system, dϕdz, T; Hdata=nothing, kkk=size(system[1])[1])

    M, C, C2, Q, Q2 = system
    RT = exp(-T * M) * C
    QR = Q * RT'

    return inv(RT) * QR * sign.(QR * dϕdz)
end
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)

##################################################################################################

### Standard Hamiltonian precomputation fn

function preH_std(system, t)

    M, C, C2, Q, Q2 = system

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(kkk, kkk*length(t)), zeros(kkk, kkk*length(t));
    for sstep in eachindex(t)
        Rt[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(t[end] - t[sstep]) * M) * C)';
        R2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(t[end] - t[sstep]) * M) * C2)';
    end

    return Rt, R2t
end

end