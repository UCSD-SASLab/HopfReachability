
### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra
push!(LOAD_PATH,"/Users/willsharpless/Documents/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachability

## System
# ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
const kkk = 2

# speed test
# const M = diagm(0 => ones(kkk), -1 => ones(kkk-1), 1 => ones(kkk-1));
# const C = diagm(0 => ones(kkk), -1 => 0.5*ones(kkk-1), 1 => 0.5*ones(kkk-1));
# const C2 = 0.1 * diagm(ones(kkk));
# const Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1));
# const Q2 = 0.1 * diagm(ones(kkk));
# const a1 = zeros(kkk)';
# const a2 = zeros(kkk)';

# original, validity
const M = [0. 1; -2 -3]
const C = 0.5 * I
const C2 = 0.5 * [2 0; 0 1]
const Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
const Q2 = 0.2 * diagm(0 => 2*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
const a1 = [0.5 0.75]
const a2 = -[0.5 0]

system = (M, C, C2, Q, Q2, a1, a2)

## Target
# J(x) = 0 is the boundary of the target
const aJ = cat([2.5, 1], 0.5*ones(kkk - 2), dims=1)
J  = x -> sum(abs2, x/aJ)/2 - 0.5
Js = v -> sum(abs2, v.*aJ')/2 + 0.5
target = (J, Js)

## Lookback Time(s)
const Th = 0.1
const Tf = 0.7
T = collect(Th : Th : Tf)

## Grid Parameters (optional, deafult here)
const bd = 3
const ϵ = 0.5e-7
const N = 10 + ϵ
const th = 0.02
grid_p = (bd, ϵ, N, th)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
const vh = 0.01
const L = 5
const tol = ϵ
const lim = 500
const lll = 20
opt_p = (vh, L, tol, lim, lll)


solution, averagetime = HopfReachability.Hopf_BRS(system, target, T; plotting=true)




# const ds = 2:2:20
# ds = 2
# solutions_d = zeros(length(solution), length(ds));
# avg_times = zeros(length(ds));

# ## Iterate thru dimension size
# for kkk in ds
#     global solution, index, solutions_d, avg_times

#     println()
#     println("Running d=", kkk, "...")

#     ## Define Problem
#     a = [[2.5 1] 0.5*ones(1, kkk - 2)]; # radii of target

#     ## System (speed)
#     M = diagm(0 => ones(kkk), -1 => ones(kkk-1), 1 => ones(kkk-1));
#     C = diagm(0 => ones(kkk), -1 => 0.5*ones(kkk-1), 1 => 0.5*ones(kkk-1));
#     C2 = 0.1 * diagm(ones(kkk));
#     Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1));
#     Q2 = 0.1 * diagm(ones(kkk));
#     basept = zeros(kkk)'; #10*[1 1 1 1 1 1 1 1 1 1]; #a_c
#     basept2 = zeros(kkk)'; #-2*[1 1 1 1 1 1 1 1 1 1]; #a_d

#     ## System (og, for validity)
#     # M = [0. 1; -2 -3];
#     # C = 0.5 * I;
#     # C2 = 0.5 * [2 0; 0 1];
#     # Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
#     # Q2 = 0.2 * diagm(0 => 2*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1))
#     # basept = [0.5 0.75]; #10*[1 1 1 1 1 1 1 1 1 1]; #[10 5 0 0 0 0];# 
#     # basept2 = -[0.5 0]; #-2*[1 1 1 1 1 1 1 1 1 1]; #[1 1 1 1 1 1];#0.001*[10 5 0 0 0 0 0 0 0 0]; #[0.5 0.2 0 ]; #[0.5 0.75]; #0* [0 0];   #; #0.001*[10 5 0 0 0 0 0 0 0 0 ];
    
#     system = (M, C, C2, Q, Q2, basept, basept2);
#     solution, averagetime = Hopf_solve!(solution, index; system, a, kkk);

#     # avg_times[Int(kkk/2)] = averagetime;
#     # solutions_d[:, Int(kkk/2)] = solution;
#     avg_times[Int(kkk-1)] = averagetime;
#     solutions_d[:, Int(kkk-1)] = solution;

#     if saving
#         # save("Hopf_test_081222.jld", "avg_times")
#         file = jldopen("Hopf_test_081722_T-1.jld", "w");
#         file["avg_times"] = avg_times;
#         file["solutions_d"] = solutions_d;
#         close(file);
#         println("** Wrote results for d="*string(kkk)*" **");
#     end
# end

# if plotting
#     ## Finish off the BRS evolution plot
#     # title!("Multiple Dimensions, T = -"*string(tvec_large[end]))

#     ## Computation Speed Plot
#     plot(ds, avg_times, label="Julia (naive, i5/2.3 GHz)")
#     plot!(ds, cpp_times, label="C++ (i7/1.7 GHz)")
#     plot!(xticks=2:2:20, xlabel="d", ylabel="t (s)", title="Mean CPU time/pt vs. Dimension", legend=:topleft)
# end

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

# avg_times_naive = [0.012, 0.022, 0.038, 0.072, 0.126, 0.172, 0.21, 0.235, 0.275, 0.293]
# cpp_times = [0.004, 0.007, 0.015, 0.028, 0.042, 0.065, 0.081, 0.11, 0.139, 0.179]

# plot(ds, avg_times_naive, label="Julia (naive, i5/2.3 GHz)")
# plot!(ds, cpp_times, label="C++ (YTC's, i7/1.7 GHz)")
# plot!(ds, avg_times, label="Julia (v2, i5/2.3 GHz)")
# plot!(xticks=2:2:20, xlabel="d", ylabel="t (s)", title="Mean CPU time/pt vs. Dimension", legend=:topleft)


# solution = -0.5 .+ [(i/a[1]).^2 + (j/a[2]).^2 for (i,j) in XYvec]/2;
# solution_sq = reshape(solution, length(x), length(y)) 
# contour(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false)
# plot!(ylims = (-1.5,1.5))

# for soli in axes(solutions_d,2)
#     solution_sq = reshape(view(solutions_d,:,soli), length(x), length(y)) ;
#     tcolor = RGB(1.0 - soli/length(soli), 0.1, soli/length(soli));
#     contour!(x, y, solution_sq, levels=[-ϵ, ϵ], colorbar = false, lc = tcolor, label = soli)
# end
# plot!()