
### Hopf-Lax Minimization for Linear Hamilton Jacobi Reachability
# wsharpless@ucsd.edu

using LinearAlgebra, Plots, JLD
push!(LOAD_PATH,"/Users/willsharpless/Documents/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachability

## Target
# J(x) = 0 is the boundary of the target
const J  = x -> sum(abs2, x./aJ)/2 - 0.5
const Js = v -> sum(abs2, v.*aJ)/2 + 0.5
const target = (J, Js)

## Lookback Time(s)
const T = 0.1

## Grid Parameters (optional, deafult here)
const bd = 3
const ϵ = 0.5e-7
const N = 10 + ϵ
const th = 0.02
const grid_p = (bd, ϵ, N, th)

## Hopf Coordinate-Descent Parameters (optional, deafult here)
const vh = 0.01
const L = 5
const tol = ϵ
const lim = 500
const lll = 20
const opt_p = (vh, L, tol, lim, lll)

## Iterate Through Size to Test Speed
const ds = 2:2:20
solutions_d = zeros(Int(floor(2*3*N)+1)^2, length(ds));
avg_times = zeros(length(ds));
saving = true

for kkk in ds
    global solution, index, solutions_d, avg_times, saving, aJ

    ## System
    # ẋ = Mx + Cu + C2d subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d
    M = diagm(0 => ones(kkk), -1 => ones(kkk-1), 1 => ones(kkk-1));
    C = diagm(0 => ones(kkk), -1 => 0.5*ones(kkk-1), 1 => 0.5*ones(kkk-1));
    C2 = 0.1 * diagm(ones(kkk));
    Q = 0.1 * diagm(0 => 3*ones(kkk), -1 => 1*ones(kkk-1), 1 => 1*ones(kkk-1));
    Q2 = 0.1 * diagm(ones(kkk));
    a1 = zeros(kkk)';
    a2 = zeros(kkk)';
    system = (M, C, C2, Q, Q2, a1, a2)

    ## Target
    aJ = cat([2.5, 1], 0.5*ones(kkk - 2), dims=1)

    println()
    println("Running d=", kkk, "...")

    ## Run the solver
    solution, averagetime = HopfReachability.Hopf_BRS(system, target, T, sampling=true);
    avg_times[Int(kkk/2)] = averagetime;
    solutions_d[:, Int(kkk/2)] = solution;

    if saving
        file = jldopen("data/HopfReachability_speedtest_082522.jld", "w");
        file["avg_times"] = avg_times;
        file["solutions_d"] = solutions_d;
        close(file);
        println("** Wrote results for d="*string(kkk)*" **");
    end
end

## Compare Speed with Older Versions
avg_times_naive = [0.012, 0.022, 0.038, 0.072, 0.126, 0.172, 0.21, 0.235, 0.275, 0.293]
cpp_times = [0.004, 0.007, 0.015, 0.028, 0.042, 0.065, 0.081, 0.11, 0.139, 0.179]
avg_times_script = [0.0005, 0.0035, 0.0084, 0.0156, 0.0281, 0.0400, 0.0546, 0.0646, 0.0832, 0.0873]

plot(ds, avg_times_naive, label="Julia (matlab translation, i5/2.3 GHz)")
plot!(ds, cpp_times, label="C++ (YTC's, i7/1.7 GHz)")
plot!(ds, avg_times_script, label="Julia (script w fn, i5/2.3 GHz)")
plot!(ds, avg_times/20, label="Julia (module, i5/2.3 GHz)")
plot!(xticks=2:2:20, xlabel="d", ylabel="t (s)", title="Mean CPU time/pt vs. Dimension", legend=:topleft)
