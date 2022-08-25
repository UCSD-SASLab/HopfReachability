module HopfReachability

using LinearAlgebra, StatsBase
using Plots, ImageFiltering, TickTock, Suppressor

export Hopf_BRS

## Find points near boundary ∂ of f(x) = 0
function boundary(solution; xg, N, δ = 5/N)

    A = Float64.(abs.(reshape(solution, length(xg), length(xg))) .< δ); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel

    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv

    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Evaluate Hopf Cost
function Hopf(system, target, quadrature, x, v; p2, kkk=size(system[1])[1])

    M, C, C2, Q, Q2, a1, a2 = system
    J, Js = target
    St, S2t, G, G2, th = quadrature

    ## Quadrature mats
    Temp1 = zeros(kkk, Int(size(St)[2]/kkk)); Temp2 = copy(Temp1);
    for i in axes(Temp1,2)
        Temp1[:,i] = view(St,:,(i-1)*kkk + 1 : i*kkk) * v;
        Temp2[:,i] = view(S2t,:,(i-1)*kkk + 1 : i*kkk) * v;
    end
    
    ## Quadrature sum
    H1 = th * (sum(map(norm, eachcol(G * Temp1))) + sum(a1 * Temp1)) # player 1 / control
    H2 = p2 ? th * (-sum(map(norm, eachcol(G2 * Temp2))) + sum(a2 * Temp2)) : 0 # player 2 / disturbance, opponent   

    ## Hopf val: J*(v) - x'v + ∫Hdt
    return Js(v) - x'view(v,1:2) + H1 + H2;
end

## Iterative Coordinate Descent of the Hopf Cost at Point x
function Hopf_cd(system, target, quadrature, x; p2, opt_p, kkk=size(system[1])[1])

    vh, L, tol, lim, lll = opt_p
    solni = -Inf

    # Loop over Hopf optimizations (coordinate descents)
    for ll = 1:lll

        v = 10*(rand(kkk) .- 0.5); # Hopf min val, how picked??
        kcoord = kkk; # coordinate counter
        stopcount = 0; # convergence flag
        happycount = 0; # step-size flag

        ## Init Hopf val: J*(v) - z'v + ∫Hdt
        fnow = Hopf(system, target, quadrature, x, v; p2)
        
        while true

            ## Nearby Point along Coordinate
            kcoord = mod(kcoord,kkk) + 1; #iterate coords
            v_coord = copy(v);
            v_coord[kcoord] = v_coord[kcoord] + vh; # nearby pt

            ## Nearby Hopf for Finite Differencing
            fnew = Hopf(system, target, quadrature, x, v_coord; p2)

            ## Coordinate Descent Step (could be a view)
            v[kcoord] = v[kcoord] - 1/L * ((fnew - fnow)/vh);

            ## Updated Hopf
            fnownow = Hopf(system, target, quadrature, x, v; p2)
            
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
        error("Coordinate Descent value did not change from -Inf")
    end
    return solni
end

## Solve Hopf Problem for given system, target and lookback time(s) T
function Hopf_BRS(system, target, T;  grid_p=(3, 0.5e-7, 10 + 0.5e-7, 0.02), opt_p=(0.01, 5, 0.5e-7, 500, 20), kkk=size(system[1])[1], p2=true, plotting=false, sampling=false, samples=360)

    M, C, C2, Q, Q2, a1, a2 = system
    J, Js = target
    bd, ϵ, N, th = grid_p
    tvec = collect(th: th: T[end])
    checkpoints = length(T) > 1
    averagetime = 0

    ## generate grid (rewrite after debug)
    xg = collect(-bd : 1/N : bd)
    XYvec = [[j,i] for (i,j) in collect(Iterators.product(xg .- ϵ, xg .+ ϵ))[:]]
    solution = [J(cat(i,j,zeros(kkk-2), dims=1)) for (i,j) in XYvec]
    index = boundary(solution; xg, N)

    ## Plot Initial Data Boundary
    if plotting
        solution_sq = reshape(solution, length(xg), length(xg)) 
        contour(xg .+ ϵ, xg .- ϵ, solution_sq, levels=[-ϵ, ϵ], colorbar = false)
        plot!()
    end

    ## Precomputing SVD for matrix sqrt
    _,Σ,VV = svd(Q);
    _,Σ2,VV2 = svd(Q2);
    G = Diagonal(sqrt.(Σ)) * VV;
    G2 = Diagonal(sqrt.(Σ2)) * VV2;

    ## Precomputing Quadrature Mats
    Rt, R2t = zeros(kkk, kkk*length(tvec)), zeros(kkk, kkk*length(tvec));
    for sstep in eachindex(tvec)
        Rt[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(T[end] - tvec[sstep]) * M) * C)';
        R2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(T[end] - tvec[sstep]) * M) * C2)';
    end

    ## Loop Over Time Frames
    for Ti in eachindex(T)
        println("   for t=-", T[Ti], "...")
        Tix = findfirst(abs.(tvec .-  T[Ti]) .< th/2);
        
        ## Quadrature mats (could be views)
        St = view(Rt,:,1 : kkk*Tix)
        S2t = view(R2t,:,1 : kkk*Tix)
        quadrature = (St, S2t, G, G2, th)

        ## Time Hopf-Solving
        @suppress begin
            tick();
        end

        ## Loop Over Grid (near ∂ID)
        index_pts = sampling ? sample(1:length(index),samples) : index # sample for speed test
        for kk in eachindex(index_pts)

            x = XYvec[index[kk]]
            solution[index[kk]] = Hopf_cd(system, target, quadrature, x; p2, opt_p)

            fnow = -solution[index[kk]]; # for printing?
        end
        
        averagetime = tok()/length(index_pts); #note this includes lll iterations of guesses per point

        ## Plot Safe-Set at Checkpoint  (unneccesary in speed test)
        if plotting
            solution_sq = reshape(solution, length(xg), length(xg));
            tcolor = RGB(1- Ti/size(T,1), (Ti/size(T,1)-0.5).^2,  Ti/size(T,1));
            contour!(xg .+ ϵ, xg .- ϵ, solution_sq, levels=[-ϵ, ϵ], colorbar = false, lc = tcolor);
        end

        ## Update Boundary Index to new Solution
        if checkpoints
            index = boundary(solution; xg, N)
        end
    end

    println("MEAN TIME: ", averagetime)
    return solution, averagetime
end

end

