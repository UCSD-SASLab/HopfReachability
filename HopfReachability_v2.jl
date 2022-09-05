module HopfReachability

using LinearAlgebra, StatsBase
using Plots, ImageFiltering, TickTock, Suppressor

export Hopf_BRS

## Find points near boundary ∂ of f(z) = 0
function boundary(solution; zg, N, δ = 5/N)

    A = Float64.(abs.(reshape(solution, length(zg), length(zg))) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel

    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv

    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Evaluate Hopf Cost
function Hopf(system, target, tbH, z, v; p2, kkk=size(system[1])[1])

    J, Jˢ, Az = target
    intH, Hdata = tbH

    return Jˢ(v; Az) - z'view(v,1:2) + intH(system, Hdata; p2);
end

## Iterative Coordinate Descent of the Hopf Cost at Point z
function Hopf_cd(system, target, z; p2, tbH, opt_p, v_init=nothing, kkk=size(system[1])[1])

    vh, L, tol, lim, lll = opt_p
    solni = -Inf

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

    if solni == -Inf
        error("!!! Coordinate Descent value did not change from -Inf")
    end

    return solni, v
end

## Solve Hopf Problem for given system, target and lookback time(s) T
function Hopf_BRS(system, target, intH, T; preH=preH_std, grid_p=(3, 0.5e-7, 10 + 0.5e-7, 0.02), opt_p=(0.01, 5, 0.5e-7, 500, 20), kkk=size(system[1])[1], p2=true, plotting=false, sampling=false, samples=360)

    J, Jˢ, A = target
    bd, ϵ, N, th = grid_p
    tvec = collect(th: th: T[end])
    checkpoints = length(T) > 1
    averagetime = 0

    ## Generate grid (rewrite after debug)
    zg = collect(-bd : 1/N : bd)
    XYvec = [[j,i] for (i,j) in collect(Iterators.product(zg .- ϵ, zg .+ ϵ))[:]]
    solution = [J(cat(i,j,zeros(kkk-2), dims=1); Az) for (i,j) in XYvec] #plotting in z bcuz ytc paper in z
    index = boundary(solution; zg, N)
    # to make this work backwards, need xg -> zg ie. grid_p_x -> grid_p

    ## Plot Initial Data Boundary
    if plotting
        solution_sq = reshape(solution, length(zg), length(zg)) 
        contour(zg .+ ϵ, zg .- ϵ, solution_sq, levels=[-ϵ, ϵ], colorbar = false)
        plot!()
    end

    ## Precomputing some of H for efficiency
    Hmats = preH(system, tvec)

    ## Loop Over Time Frames
    for Ti in eachindex(T)
        println("   for t=-", T[Ti], "...")
        tix = findfirst(abs.(tvec .-  T[Ti]) .< th/2);

        ## A → Az so J, Jˢ → Jz, Jˢz
        M = system[1]
        Az = exp(T * M)' * A * exp(T * M)
        target = J, Jˢ, Az
        # to make this work backwards, need example to make Az = A' so A = (τ^t)^-1 A' τ^-1 

        # Packaging Hdata, tbH
        Hdata = (Hmats..., tix, th)
        tbH = (intH, Hdata)

        ## Time Hopf-Solving
        @suppress begin; tick(); end

        ## Loop Over Grid (near ∂ID)
        index_pts = sampling ? sample(1:length(index),samples) : index # sample for speed test
        for kk in eachindex(index_pts)

            z = XYvec[index[kk]]
            solution[index[kk]], dϕdz = Hopf_cd(system, target, z; p2, tbH, opt_p)

        end
        averagetime = tok()/length(index_pts); #includes lll reopts/pt

        ## Plot Safe-Set at Checkpoint  (unneccesary in speed test)
        if plotting
            solution_sq = reshape(solution, length(zg), length(zg));
            tcolor = RGB(1- Ti/size(T,1), (Ti/size(T,1)-0.5).^2,  Ti/size(T,1));
            contour!(zg .+ ϵ, zg .- ϵ, solution_sq, levels=[-ϵ, ϵ], colorbar = false, lc = tcolor);
        end

        ## Update Boundary Index to new Solution
        if checkpoints
            index = boundary(solution; zg, N)
        end
    end

    println("MEAN TIME: ", averagetime)
    return solution, averagetime
end

## Solve Hopf Problem to find minimum T* so ϕ(z,T*) = 0, for given system & target
function Hopf_BRS_minT(system, target, intH, x; T_init=nothing, preH=preH_std, time_p=(0.02, 0.1, 5), opt_p=(0.01, 5, 0.5e-7, 500, 20), kkk=size(system[1])[1], p2=true, plotting=false)

    J, Jˢ, A = target
    th, Th, maxT = time_p
    tvec = isnothing(T_init) ? collect(th: th: Th) : collect(th: th: T_init)

    ## Initialize solutions
    solution, dϕdz = J(x; A), 0
    v_init = nothing 

    ## Precomputing some of H for efficiency
    Hmats = preH(system, tvec)

    ## Loop Over Time Frames until ϕ(z, T) > 0
    Thi = 0; Tˢ = tvec[end]
    while solution > 0
        println("   checking T =-", Tˢ, "...")
        # tix = findfirst(abs.(tvec .-  T[Ti]) .< th/2);
        tix = length(tvec)

        ## A → Az so J, Jˢ → Jz, Jˢz
        M = system[1]
        Az = exp(Tˢ * M)' * A * exp(Tˢ * M)
        z = exp(Tˢ * M) * x
        target = J, Jˢ, Az

        ## Packaging Hdata, tbH
        Hdata = (Hmats..., tix, th)
        tbH = (intH, Hdata)

        ## Hopf-Solving to check current Tˢ
        solution, dϕdz = Hopf_cd(system, target, z; p2, tbH, opt_p, v_init)

        ## Check if Tˢ yields valid solution
        if solution <= 0
            break
        end        

        ## Update the tvec, to increase Tˢ
        Thi += 1; 
        tvec_ext = isnothing(T_init) ? collect((Thi-1)*Th + th: th: Thi*Th) : collect(T_init + (Thi-1)*Th + th: th: Thi*Th)
        push!(tvec, tvec_ext...)
        Tˢ = tvec[end]

        ## Extending previous precomputed Hdata for increased final time
        exts = preH(system, tvec_ext)
        Rt_ext = cat(Hmats[1], exts[1], dims=2) # Rt and R2t need to be first ***
        R2t_ext = cat(Hmats[2], exts[2], dims=2)
        Hmats = length(Hmats) == 2 ? (Rt_ext, R2t_ext) : (Rt_ext, R2t_ext, Hmats[3:end]...)

        # v_init = dϕdz ???

        if Tˢ > maxT
            println("!!! Couldn't find a solution under max Time")
            break
        end
    end

    ## Compute Optimal Control (dH/dp(dϕdz, Tˢ) = exp(-Tˢ * M)Cuˢ + exp(-Tˢ * M)C2dˢ)
    uˢ = HJoc(system, dϕdz, Tˢ; Hdata)

    return solution, uˢ, Tˢ
end


##################################################################################################
##################################################################################################

### Example Hamiltonians and Precomputation fn's

##################################################################################################

### Example 1 

## Time integral of Hamiltonian for YTC et al. 2017, quadratic control constraints
function intH_ytc17(system, Hdata; p2, kkk=size(system[1])[1])

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
function preH_ytc17(system, tvec)

    M, C, C2, Q, Q2 = system

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(kkk, kkk*length(tvec)), zeros(kkk, kkk*length(tvec));
    for sstep in eachindex(tvec)
        Rt[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(tvec[end] - tvec[sstep]) * M) * C)';
        R2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(tvec[end] - tvec[sstep]) * M) * C2)';
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

return inv(norm(G * R * dϕdz)) * Q * R * dϕdz + a1
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)
# could be faster by using Hdata

##################################################################################################

### Example 2

## Time integral of Hamiltonian for MRK et al. 2018, inf-norm control constraints
function intH_mrk18(system, Hdata; p2, kkk=size(system[1])[1])

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
function preH_mrk18(system, tvec; kkk=size(system[1])[1])

    M, C, C2, Q, Q2 = system

    ## Transformation Mats Q * R := Q * (exp(-(T-t)M)C)' over t
    QRt, QR2t = zeros(kkk, kkk*length(tvec)), zeros(kkk, kkk*length(tvec));
    for sstep in eachindex(tvec)
        QRt[:,kkk*(sstep-1) + 1 : kkk*sstep] = Q * (exp(-(tvec[end] - tvec[sstep]) * M) * C)';
        QR2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = Q2 * (exp(-(tvec[end] - tvec[sstep]) * M) * C2)';
    end

    return QRt, QR2t
end

## Optimal HJB Control 
function HJoc_mrk18(system, dϕdz, T; Hdata=nothing, kkk=size(system[1])[1])

    M, C, C2, Q, Q2 = system
    RT = exp(-T * M) * C
    QR = Q * RT'

return inv(RT) * QR * sign.(QR * dϕdz) 
# ∇pH(dϕdz, Tˢ) = R_c' uˢ + R_d' dˢ => uˢ = R_c'^-1 * control_portion(∇pH(dϕdz, Tˢ)) [ytc 17] (?)

##################################################################################################

### Standard Hamiltonian precomputation fn

function preH_std(system, tvec)

    M, C, C2, Q, Q2 = system

    ## Transformation Mats R := (exp(-(T-t)M)C)' over t
    Rt, R2t = zeros(kkk, kkk*length(tvec)), zeros(kkk, kkk*length(tvec));
    for sstep in eachindex(tvec)
        Rt[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(tvec[end] - tvec[sstep]) * M) * C)';
        R2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(tvec[end] - tvec[sstep]) * M) * C2)';
    end

    return Rt, R2t
end

end