### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, DifferentialEquations, StatsBase
using Plots, ImageFiltering, JLD
using TickTock, Suppressor

## Find points near boundary ∂ of f(x) = 0
function boundary(solution)
    A = Float64.(abs.(reshape(solution, length(x), length(y))) .< 5/N); # near ∂ID |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # kernel
    ind = imfilter(A, centered(B), Fill(0)); # cushion boundary w conv
    return findall(ind[:] .> 0); # index of XY near ∂ID
end

## Evaluate Hopf Cost
function Hopf(v; tstep, St, S2t, Hello, Hello2, basept, basept2, xyveci, a, kkk)

    ## Quadrature mats
    Temp1 = zeros(kkk, tstep); Temp2 = copy(Temp1);
    # Temp12 = copy(Temp1);  Temp22 = copy(Temp1);
    for i in axes(Temp1,2)
        Temp1[:,i] = view(St,:,(i-1)*kkk + 1 : i*kkk) * v;
        # Temp12[:,i] = view(HSt,:,(i-1)*kkk + 1 : i*kkk) * v; #this works but slow
        Temp2[:,i] = view(S2t,:,(i-1)*kkk + 1 : i*kkk) * v;
        # Temp22[:,i] = view(HS2t,:,(i-1)*kkk + 1 : i*kkk) * v;
    end
    # replaces: rr = Hello * reshape(S*v,kkk,tstep) # or rr = Hello * (S * repmat(v));
    
    ## Quadrature sum, control
    # probably should just roll this all out with the temps and fnow eval
    hahahaha = th * (sum(map(norm, eachcol(Hello * Temp1))) + sum(basept * Temp1)) 
    
    ## Quadrature sum, disturbance
    # this be condensed with a julia ?
    if shutdown == 0
        # rr2 = Hello2 * (S2 .* v);
        hahahaha2 = th * (-sum(map(norm, eachcol(Hello2 * Temp2))) + sum(basept2 * Temp2));                
    end

    ## Hopf val: J*(v) - z'v + ∫Hdt
    return 0.5 + (sum(abs2, v .* a'))/2 - xyveci'*view(v,1:2) + cc * (hahahaha + hahahaha2);
end

## Iterative Coordinate Descent of the Hopf Cost
function Hopf_cd!(solni; tstep, St, S2t, Hello, Hello2, basept, basept2, xyveci, a, kkk)

    # Loop over Hopf optimizations (coordinate descent)
    for lllll = 1:lll

        fnow = 10000; #current hopf val
        fnownow = 10000; #next hopf val
        fnew = 10000; #nearby hopf val for fd coord descent
        hahahaha = 0; 
        hahahaha2 = 0; 
        L = 5; #lipschitz constant ie. 1/(descent step)
        v = 10*(rand(kkk) .- 0.5); # Hopf min val, how picked??
        kcoord = kkk; # coordinate counter
        stopcount = 0; # convergence flag
        happycount = 0; # step-size flag

        ## Init Hopf val: J*(v) - z'v + ∫Hdt
        fnow = Hopf(v; tstep, St, S2t, Hello, Hello2, basept, basept2, xyveci, a, kkk)
        
        while true

            ## Nearby Point along Coordinate
            kcoord = mod(kcoord,kkk) + 1; #iterate coords
            v_coord = copy(v);
            v_coord[kcoord] = v_coord[kcoord] + xh; # nearby pt

            ## Nearby Hopf for Finite Differencing
            fnew = Hopf(v_coord; tstep, St, S2t, Hello, Hello2, basept, basept2, xyveci, a, kkk)

            ## Coordinate Descent Step (could be a view)
            v[kcoord] = v[kcoord] - 1/L * ((fnew - fnow)/xh);

            ## Updated Hopf
            fnownow = Hopf(v; tstep, St, S2t, Hello, Hello2, basept, basept2, xyveci, a, kkk)
            
            ## Convergence Criteria
            stopcount = abs(fnownow - fnow) < ϵ ? stopcount + 1 : 0; # in thresh for coordinate
            happycount, L = happycount > 500 ? (1, L*2) : (happycount, L);

            if stopcount == kkk # in thresh for all coordinates
                solni = -min(-solni, fnow)
                # solution[index[kk]] = -min(-solni, fnow); # take min with previous opt run, in case of local convergence
                break
            end

            fnow = fnownow;
            happycount += 1;
        end
    end

    return solni
end

## Solve Hopf Problem for given initial data and system
# in future, should package grid params, time and space steps, and ID = (J, J*) and feed to Hopf_solve() so its exportable
function Hopf_solve!(solution, index; system, a, kkk)

    averagetime = 0;
    count = 0
    (M, C, C2, Q, Q2, basept, basept2) = system

    # Moving Q to the principal space ???
    UU,Sigma,VV = svd(Q);
    Hello = Diagonal(sqrt.(Sigma))*VV;
    UU2,Sigma2,VV2 = svd(Q2);
    Hello2 = Diagonal(sqrt.(Sigma2))*VV2;

    # Rt, Rt2 = zeros(kkk*length(tvec), kkk), zeros(kkk*length(tvec), kkk);
    Rt, R2t = zeros(kkk, kkk*length(tvec)), zeros(kkk, kkk*length(tvec));
    for sstep in eachindex(tvec)
        Rt[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(tend - tvec[sstep]) * M) * C)';
        R2t[:,kkk*(sstep-1) + 1 : kkk*sstep] = (exp(-(tend - tvec[sstep]) * M) * C2)';
    end

    ## Loop Over Time Frames
    for tstep_large in eachindex(tvec_large)
        println("   for t=-", tvec_large[tstep_large], "...")
        tstep = findfirst( abs.(tvec .-  tvec_large[tstep_large] ) .< th/2 );
        
        # quadrature mats (could be views)
        St = Rt[:,1 : kkk*tstep];
        S2t = R2t[:,1 : kkk*tstep];
        # HSt = Hello * St
        # HS2t = Hello2 * S2t

        averagetime = 0;
        count = 0

        ## Loop Over Grid (near ∂ID)
        for kk in eachindex(index)
        # for kkr = 1:360 # Sample 10% of the of the index
            # kk = sample(1:length(index))
            xyveci = XYvec[index[kk]]
            # solni = solution[index[kk]]

            @suppress begin
                tick();
            end
            
            solution[index[kk]] = Hopf_cd!(-Inf; tstep, St, S2t, Hello, Hello2, basept, basept2, xyveci, a, kkk)

            fnow = -solution[index[kk]]; # for printing?

            count += 1;
            averagetime = (averagetime * (count - 1) + tok())/count;

        end
        
        # println("For T=-"*string(tvec_large[tstep_large])*", mean cpu_t/pt = "*string(averagetime)*" s")

        ## Plot Safe-Set at Checkpoint  (unneccesary in speed test)
        if plotting
            solution_sq = reshape(solution, length(x), length(y));
            tcolor = RGB(1- tstep_large/size(tvec_large,1), (tstep_large/size(tvec_large,1)-0.5).^2,  tstep_large/size(tvec_large,1));
            contour!(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false, lc = tcolor);
        end

        ## Update Boundary Index to new Solution (only used if checkpoints)
        if checkpoints
            index = boundary(solution)
        end
    end

    println("MEAN TIME: ", averagetime/lll)
    return solution, averagetime/lll
end

# functions J, J*?

## Alg param
const cc = 1; # scales all quadrature sums, but why?
const shutdown = 0; # turn-off disturbance??
const plotting = false;
const saving = true;

const ϵ = 0.5e-7;
const N = 10 + ϵ; # grid resolution, inter-integer spacing in 1D
kkk_ini = 2; # dimensions
const lll = 20; # number of re-opts in case of local convergence??

a = [[2.5 1] 0.5*ones(1, kkk_ini - 2)]; # radii of initial ellipse
rho_ori = 2; # unused

const xh = 0.01; # finite differencing step
const th = 0.2; # time step (within integration)
const th_large = 1.; # time step, large (for BRS checkpoints)
const tend = 1.; # final time

const tvec = collect(th : th : tend); 
const tvec_large = collect(th_large : th_large : tend); 
const checkpoints = length(tvec_large) > 1

## Make Grid
const x = collect(-3 : 1/N : 3) .+ ϵ;
const y = collect(-3 : 1/N : 3) .- ϵ;
const xyg = collect(Iterators.product(y, x)); #flipped to match matlab order
const XYvec = [[j,i] for (i,j) in xyg[:]]; #flipped to match matlab order

## Find Points Near Initial Data Boundary (J(XY)=0 iff ∂ID)
solution = -0.5 .+ [(i/a[1]).^2 + (j/a[2]).^2 for (i,j) in XYvec]/2; # J(XY), ignores all other dim meaning ID is hyper-cylinder
index = boundary(solution);

## Plot Initial Data Boundary
if plotting
    solution_sq = reshape(solution, length(x), length(y)) 
    contour(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false)
    plot!()
end

const ds = 2:5
# const ds = 2:2:20
# ds = 2
solutions_d = zeros(length(solution), length(ds));
avg_times = zeros(length(ds));

## Iterate thru dimension size
for kkk in ds
    global solution, index, solutions_d, avg_times

    println()
    println("Running d=", kkk, "...")

    ## Define Problem
    a = [[2.5 1] 0.5*ones(1, kkk - 2)]; # radii of target

    ## System (speed)
    M = diagm(0 => ones(kkk), -1 => ones(kkk-1), 1 => ones(kkk-1));
    C = diagm(0 => ones(kkk), -1 => 0.5*ones(kkk-1), 1 => 0.5*ones(kkk-1));
    C2 = 0.1 * diagm(ones(kkk));
    Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1));
    Q2 = 0.1 * diagm(ones(kkk));
    basept = zeros(kkk)'; #10*[1 1 1 1 1 1 1 1 1 1]; #a_c
    basept2 = zeros(kkk)'; #-2*[1 1 1 1 1 1 1 1 1 1]; #a_d

    ## System (og, for validity)
    # M = [0. 1; -2 -3];
    # C = 0.5 * I;
    # C2 = 0.5 * [2 0; 0 1];
    # Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
    # Q2 = 0.2 * diagm(0 => 2*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
    # basept = [0.5 0.75]; #10*[1 1 1 1 1 1 1 1 1 1]; #[10 5 0 0 0 0];# 
    # basept2 = -[0.5 0]; #-2*[1 1 1 1 1 1 1 1 1 1]; #[1 1 1 1 1 1];#0.001*[10 5 0 0 0 0 0 0 0 0]; #[0.5 0.2 0 ]; #[0.5 0.75]; #0* [0 0];   #; #0.001*[10 5 0 0 0 0 0 0 0 0 ];
    
    system = (M, C, C2, Q, Q2, basept, basept2);
    solution, averagetime = Hopf_solve!(solution, index; system, a, kkk);

    # avg_times[Int(kkk/2)] = averagetime;
    # solutions_d[:, Int(kkk/2)] = solution;
    avg_times[Int(kkk-1)] = averagetime;
    solutions_d[:, Int(kkk-1)] = solution;

    if saving
        # save("Hopf_test_081222.jld", "avg_times")
        file = jldopen("Hopf_test_081722_T-1.jld", "w");
        file["avg_times"] = avg_times;
        file["solutions_d"] = solutions_d;
        close(file);
        println("** Wrote results for d="*string(kkk)*" **");
    end
end

if plotting
    ## Finish off the BRS evolution plot
    # title!("Multiple Dimensions, T = -"*string(tvec_large[end]))

    ## Computation Speed Plot
    plot(ds, avg_times, label="Julia (naive, i5/2.3 GHz)")
    plot!(ds, cpp_times, label="C++ (i7/1.7 GHz)")
    plot!(xticks=2:2:20, xlabel="d", ylabel="t (s)", title="Mean CPU time/pt vs. Dimension", legend=:topleft)
end

# computed remotely
# avg_times_v3 = [0.0004823995618739448, 
#             0.0035204544789251544,
#             0.008375455881879585,
#             0.015562773529375337,
#             0.02811180597371967,
#             0.04000357769946536,
#             0.05458015838849176,
#             0.06464738353342735,
#             0.08317038722723705,
#             0.08731980386662924]

# avg_times_v3 = [0.0005552788891389986, 
# 0.003639561169667983, 
# 0.008221533367135641, 
# 0.014750507249887437, 
# 0.027909042340658403,
# ]

avg_times_naive = [0.012, 0.022, 0.038, 0.072, 0.126, 0.172, 0.21, 0.235, 0.275, 0.293]
cpp_times = [0.004, 0.007, 0.015, 0.028, 0.042, 0.065, 0.081, 0.11, 0.139, 0.179]

plot(ds, avg_times_naive, label="Julia (naive, i5/2.3 GHz)")
plot!(ds, cpp_times, label="C++ (YTC's, i7/1.7 GHz)")
plot!(ds, avg_times, label="Julia (v2, i5/2.3 GHz)")
plot!(xticks=2:2:20, xlabel="d", ylabel="t (s)", title="Mean CPU time/pt vs. Dimension", legend=:topleft)


solution = -0.5 .+ [(i/a[1]).^2 + (j/a[2]).^2 for (i,j) in XYvec]/2;
solution_sq = reshape(solution, length(x), length(y)) 
contour(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false)
plot!(ylims = (-1.5,1.5))

for soli in axes(solutions_d,2)
    solution_sq = reshape(view(solutions_d,:,soli), length(x), length(y)) ;
    tcolor = RGB(1.0 - soli/length(soli), 0.1, soli/length(soli));
    contour!(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false, lc = tcolor, label = soli)
end
plot!()