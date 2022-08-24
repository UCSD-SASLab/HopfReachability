### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, DifferentialEquations
using Plots
using ImageFiltering
using TickTock

## Alg param
cc = 1; # scales all quadrature sums, but why?
shutdown = 0; # turn-off disturbance??

ϵ = 0.5e-7;
N = 12 + ϵ; # grid resolution between integers in 1D
kkk = 2; # dimensions
lll = 1; # unused

a = [[2.5 1] 0.5*ones(1, kkk - 2)]; # radii of initial ellipse
rho_ori = 2; # unused

xh = 0.01; # finite differencing step
th = 0.02; # time step (within integration)
th_large = 0.1; # time step, large (for BRS checkpoints)
tend=0.7; # final time

tvec = collect(th : th : tend ); 
tvec_large = collect(th_large : th_large : tend); 

## Make Grid
x = collect(-3 : 1/N : 3) .+ ϵ;
y = collect(-3 : 1/N : 3) .- ϵ;
xyg = collect(Iterators.product(y, x)); #flipped to match matlab order
XYvec = [[j,i] for (i,j) in xyg[:]]; #flipped to match matlab order

## Find Points Near Initial Data Boundary
solution = -0.5 .+ [(i/a[1]).^2 + (j/a[2]).^2 for (i,j) in XYvec]/2; # J*(XY)
A = Float64.(abs.(reshape(solution, length(x), length(y))) .< 5/N); # |J*(XY)| < 5/N, signal
B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # convolution kernel
ind = imfilter(A, centered(B), Fill(0));
index = findall(ind[:] .> 0); # index of XY near ∂ID

## Plot Initial Data Boundary
# plot(a[1] * cos.(0:0.01:2*pi),  a[2] * sin.(0:0.01:2*pi))
# plot!(xlim=(-3, 3), ylim=(-3, 3))
solution_sq = reshape(solution, length(x), length(y)) 
contour(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false)
plot!()
# tcolor = RGB(1- tstep_large/size(tvec_large,2), (tstep_large/size(tvec_large,2)-0.5).^2 , tstep_large/size(tvec_large,2))
# tcolor_next = RGB(1- (tstep_large+1)/size((tvec_large),2), ((tstep_large+1)/size(tvec_large,2)-0.5).^2 , (tstep_large+1)/size(tvec_large,2))
# contour!(x, y, solution_sq.+0.01, levels=[-ϵ, ϵ], colorbar = false, seriescolor=cgrad([tcolor, tcolor]))
# title!("t = 0")

averagetime = 0;
count = 0;

## Define System
M = [0. 1; -2 -3];
#[0 0 0 1 0 0 0 0 0 0; 0 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0;...
#     0 0 0 0 0 0 0 0 0 0; 0 -32.2 0 0 0 0 0 0 0 0; 32.2 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 1 0];
#
#0.1*[-20 0 0 0 0 0; 0 -25 0 0 0 0; 0 0 0 0 0 0; -0.744 -0.032 0 -0.154 -0.0052 1.54; 0.337 -1.12 0 0.249 -0.1 -5.2; -0.02 0 0.0386 -0.996 -0.000295 -0.117];
#0.001*[0 1 0 0 0 0 0 0 0 0;-62500 -250 0 0 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0 0;0 -2450000 0 -130 0 0 0 0 0 0;0 0 -10000 0 -2 0 0 0 0 0;...
    #0 0 900 0 0 0 1 0 0 0;0 0 0 -7260000 0 -62500 -1 0 0 0;0 0 0 0 0 0 0 0 1 0;0 0 0 0 40000 40000 0 -40000 -200 0;0 0 0 0 0 0 0 1 0 0]; 
#0.1*[-20 0 0 0 0 0; 0 -25 0 0 0 0; 0 0 0 0 0 0; -0.744 -0.032 0 -0.154 -0.0052 1.54; 0.337 -1.12 0 0.249 -0.1 -5.2; -0.02 0 0.0386 -0.996 -0.000295 -0.117];
#[ 0 1.0000 0;  0 0  -0.0136; 0 0 -2.0000]; # [0 1; -2 -3]; #eye(2); #
#0.1*[-20 0 0 0 0 0; 0 -25 0 0 0 0; 0 0 0 0 0 0; -0.744 -0.032 0 -0.154 -0.0052 1.54; 0.337 -1.12 0 0.249 -0.1 -5.2; -0.02 0 0.0386 -0.996 -0.000295 -0.117];
#[ 0 1.0000 0;  0 0  -0.0136; 0 0 -2.0000];  #

C = 0.5 * I;
#0.5*[0 0; 1 0];
#[0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0.06039 0 -0.06039 0 0 0 0 0 0; -0.05985 0 0.05985 0 0 0 0 0 0 0;...
#     -0.003292 0.003292 -0.003292 0.003292 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0; -0.006387 -0.006387 -0.006387 -0.006387 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0];
#0.5*eye(2);
#0.1*[20 0 0 0 0 0; 0 25 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]; # 0.5*[0 0; 1 0]; #0.5*eye(2);

C2 = 0.5 * [2 0; 0 1];
#0.0005*eye(kkk); #
    #0.1*eye(kkk);#
    #
    #0.001*[0 0 0 0 0 0 0 0 0 0;1 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0;10000 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0;...
    #0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0 0 0];

#0.1*[20 0 0 0 0 0; 0 25 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0];  #[0 0 0 ; 0 0 0; 2 0 0];   #0.5*[0 0; 1 0]; #0.5*eye(2);#
#[0 0 0 ; 0 0 0; 2 0 0]; #0.5*[0 0; 1 0] #0.5*eye(2); #eye(2); # 
#0.1*[20 0 0 0 0 0; 0 25 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0];  #[0 0 0 ; 0 0 0; 2 0 0];   #0.5*[0 0; 1 0]; #0.5*eye(2);#

Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
Q2 = 0.2 * diagm(0 => 2*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))

# a_c, a_d
basept = [0.5 0.75]; #10*[1 1 1 1 1 1 1 1 1 1]; #[10 5 0 0 0 0];# 
basept2 = -[0.5 0]; #-2*[1 1 1 1 1 1 1 1 1 1]; #[1 1 1 1 1 1];#0.001*[10 5 0 0 0 0 0 0 0 0]; #[0.5 0.2 0 ]; #[0.5 0.75]; #0* [0 0];   #; #0.001*[10 5 0 0 0 0 0 0 0 0 ];

# Moving Q to the principal space ???
UU,Sigma,VV = svd(Q);
Hello = Diagonal(sqrt.(Sigma))*VV;
UU2,Sigma2,VV2 = svd(Q2);
Hello2 = Diagonal(sqrt.(Sigma2))*VV2;

R, R2 = zeros(kkk*length(tvec), kkk), zeros(kkk*length(tvec), kkk);
for sstep = 1:length(tvec)
    R[kkk*(sstep-1) + 1 : kkk*sstep,:] = (exp(-(tend - tvec[sstep]) * M) * C)';
    R2[kkk*(sstep-1) + 1 : kkk*sstep,:] = (exp(-(tend - tvec[sstep]) * M) * C2)';
end

## Loop Over Time Frames
for tstep_large = 1:length(tvec_large)
    println("Running ", tvec_large[tstep_large])
    tstep = findfirst( abs.(tvec .-  tvec_large[tstep_large] ) .< th/2 );
    
    # quadrature mats
    S = R[1 : kkk*tstep,:];
    S2 = R2[1 : kkk*tstep,:];

    ## Loop Over Grid (near ∂ID)
    for kk = 1:size(index, 1)

        # tick();

        # first erase original J^*(XY)
        solution[index[kk]] = -Inf

        # Loop over Hopf optimizations (coordinate descent), currently running 20 regardless
        for lllll = 1:20

            fnow = 10000; #current hopf val
            fnownow = 10000; #next hopf val
            fnew = 10000; #nearby hopf val for fd coord descent
            hahahaha = 0; #quadrature sum, control
            hahahaha2 = 0; #quadrature sum, disturbance
            L = 5; #lipschitz constant ie. 1/(descent step)
            v = 10*(rand(kkk) .- 0.5); # Hopf min val, how picked??
            kcoord = kkk; # coordinate counter
            stopcount = 0; # convergence flag
            happycount = 0; # step-size flag

            rr = Hello * reshape(S*v,kkk,tstep); # !! we could preshape S so that reshape is unneccesary
            
            # !! map(norm, eachcol(rr)) is a little faster than sqrt.(sum(abs2, rr, dims=1))
            hahahaha = th * (sum(sqrt.(sum(abs2, rr, dims=1))) + sum(basept * reshape(S*v,kkk,tstep)))
            
            # this could all be condensed with a julia ?
            if shutdown == 0
                rr2 = Hello2 * reshape(S2*v,kkk,tstep);
                hahahaha2 = th * (-sum(sqrt.(sum(abs2, rr2, dims=1))) + sum(basept2 * reshape(S2*v,kkk,tstep)));                
            end

            ## Hopf val: J*(v) - z'v + ∫Hdt
            fnow = 0.5 + (sum(abs2, v .* a'))/2 - XYvec[index[kk]]'*v[1:2] + cc * (hahahaha + hahahaha2);
            
            while true

                ## Nearby Point along Coordinate
                kcoord = mod(kcoord,kkk) + 1; #iterate coords
                v_coord = copy(v);
                v_coord[kcoord] = v_coord[kcoord] + xh; # nearby pt

                ## Nearby Hopf for Finite Differencing
                rr = Hello * reshape(S*v_coord,kkk,tstep);
                hahahaha = th * (sum(sqrt.(sum(abs2, rr, dims=1))) + sum(basept * reshape(S*v_coord,kkk,tstep)))
                if shutdown == 0
                    rr2 = Hello2 * reshape(S2*v_coord,kkk,tstep);
                    hahahaha2 = th * (-sum(sqrt.(sum(abs2, rr2, dims=1))) + sum(basept2 * reshape(S2*v_coord,kkk,tstep)));                
                end
                fnew = 0.5 + (sum(abs2, v_coord .* a'))/2 - XYvec[index[kk]]'*v_coord[1:2] + cc * (hahahaha + hahahaha2); #nearby hopf
    
                ## Coordinate Descent Step
                v[kcoord] = v[kcoord] - 1/L * ((fnew - fnow)/xh);

                ## Updated Hopf
                rr = Hello * reshape(S*v,kkk,tstep);
                hahahaha = th * (sum(sqrt.(sum(abs2, rr, dims=1))) + sum(basept * reshape(S*v,kkk,tstep)))
                if shutdown == 0
                    rr2 = Hello2 * reshape(S2*v,kkk,tstep);
                    hahahaha2 = th * (-sum(sqrt.(sum(abs2, rr2, dims=1))) + sum(basept2 * reshape(S2*v,kkk,tstep)));                
                end
                fnownow = 0.5 + (sum(abs2, v .* a'))/2 - XYvec[index[kk]]'*v[1:2] + cc * (hahahaha + hahahaha2); #nearby hopf
                
                ## Convergence Criteria
                stopcount = abs(fnownow - fnow) < ϵ ? stopcount + 1 : 0; # in thresh for coordinate
                happycount, L = happycount > 500 ? (1, L*2) : (happycount, L);

                if stopcount == kkk # in thresh for all coordinates
                    solution[index[kk]] = -min(-solution[index[kk]], fnow); # take min with previous opt run
                    break
                end

                fnow = fnownow;
                happycount += 1;

            end

        end

        fnow = -solution[index[kk]]; #why? this will be overwritten in hopf loop

        count += 1;
        # averagetime = (averagetime * (count - 1) + tok())/count
    end

    # println("For T=-"*string(tvec_large[tstep_large])*", mean cpu_t/pt = "*string(averagetime)*" s")

    solution_sq = reshape(solution, length(x), length(y))
    tcolor = RGB(1- tstep_large/size(tvec_large,1), (tstep_large/size(tvec_large,1)-0.5).^2,  tstep_large/size(tvec_large,1)) 
    contour!(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false, lc = tcolor)

    # Update Boundary Index to new Solution
    A = Float64.(abs.(reshape(solution, length(x), length(y))) .< 5/N); # |J*(XY)| < 5/N, signal
    B = 1/(N/2)^2 * ones(Int(floor(N/2)),Int(floor(N/2))); # convolution kernel
    ind = imfilter(A, centered(B), Fill(0));
    index = findall(ind[:] .> 0);

end

title!("T = -"*string(tvec_large[end]))
plot!()