
using LinearAlgebra, Plots
plotlyjs()
# push!(LOAD_PATH,"/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR");
# using HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, plot_BRS, Hopf

include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, plot_BRS, Hopf, Hopf_minT, HJoc_ytc17

## System (2D)
M = [0. 1; -2 -3]
C = 0.5 * [1 0; 0 1]
C2 = 0.5 * [2 0; 0 1]
Q = 0.1 * 3 * [1 0; 0 1]
Q2 = 0.2 * 2 * [1 0; 0 1]
a1 = zero([0.5 0.75])
a2 = zero([0.5 0])
system = (M, C, C2, Q, Q2, a1, a2)

## System (3D)
# M = [0. 1 0.; -2 -3 0.; 0. 0. -1.]
# B = 0.5 * [1 0; 0 1; 0. 0.]
# C = 0.5 * [2 0; 0 1; 0. 0.]
# Q = 0.1 * 3 * [1 0; 0 1]
# Q2 = 0.2 * 2 * [1 0; 0 1]
# a1 = 0*[0.5 0.75]
# a2 = -0*[0.5 0]
# system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.02
Th = 0.2
Tf = 0.6
T = collect(Th : Th : Tf)

## Elliptical Target: J(x) = 0 is the boundary of the target
Ap = diagm([1., 1.]) # dim x
cp = [0.; 0.] # dim x
r = 1.0
J(x::Vector, A, c) = ((x - c)' * inv(A) * (x - c))/2 - 0.5 * r^2 #don't need yet
Js(v::Vector, A, c) = (v' * A * v)/2 + c'v + 0.5 * r^2
J(x::Matrix, A, c) = diag((x .- c)' * inv(A) * (x .- c))/2 .- 0.5 * r^2
Js(v::Matrix, A, c) = diag(v' * A * v)/2 + (c'v)' .+ 0.5 * r^2 #don't need yet
target = (J, Js, (Ap, cp))

## Point Target (Indicator Set)
# ap = -1.
# cp = [0.; 0.]
# J(x::Vector, a, c) = x == c ? a : Inf
# Js(v::Vector, a, c) = v'c - a
# J(x::Matrix, a, c) = map(x -> x == c ? a : Inf, eachcol(x))
# Js(v::Matrix, a, c) = v'c .- a
# target = (J, Js, (ap, cp))

## Affine Target
# ap = [1.; 1.]
# cp = 0.
# J(x::Vector, a, c) = a'x - c #don't need yet
# Js(v::Vector, a, c) = v == a ? c : 10
# J(x::Matrix, a, c) = (a'x)' .- c
# Js(v::Matrix, a, c) = map(v -> v == a ? c : 10, eachcol(v))
# target = (J, Js, (ap, cp))

## Points to Solve
bd = 3. # (-3, 3) for ellipse
ϵ = 0.5e-7
N = 10 + ϵ
grid_p = (bd, N)

## Hopf ADMM Parameters (default)
ρ, ρ2 = 1e-4, 1e-4
tol = 1e-5
max_its = 10
opt_p_admm = (ρ, ρ2, tol, max_its)

## Hopf CD Parameters (default)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 20
max_runs = 40
opt_p_cd = (vh, L, tol, lim, lll, max_runs)

### Solve the BRS to Validate our Controller
solution, run_stats = Hopf_BRS(system, target, T;
                                opt_method=Hopf_cd,
                                th,
                                grid_p,
                                opt_p=opt_p_cd,
                                warm=false,
                                check_all=true,
                                printing=true);
B⁺T, ϕB⁺T = solution;

plot_scatter = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵs=1e-2, interpolate=false, alpha=0.1)
plot_contour = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵc=1e-3, interpolate=true, value_fn=true, alpha=0.5)

## Find Boundary Pts of one BRS for Time of Interest
Toi = 0.2
Tix = findfirst(abs.(T .- Toi) .< Th/2);
B = B⁺T[Tix + 1][:, abs.(ϕB⁺T[Tix + 1]) .< 1e-2] # Tix + 1 because target set is first

## Test a Single Call
x = [-0.8, 2.4];
uˢ, dˢ, Tˢ, ϕ = Hopf_minT(system, target, x; time_p=(th, Th, Tf + ϵ), printing=true)

## Compute Minimum Time Tˢ until reachable and the Optimal Control uˢ and Disturbance dˢ at that time
Fu, Fd = zero(B), zero(B)
for (bi, x) in enumerate(eachcol(B))
    uˢ, dˢ, Tˢ, ϕ = Hopf_minT(system, target, x; time_p=(th, Th, Tf + ϵ))
    Fu[:, bi], Fd[:, bi] = C * uˢ, C2 * dˢ; 
end

## Compute Autonomous Field
xig = collect(-3 : 0.35 : 3.) .+ ϵ; lg = length(xig)
Xg = hcat(collect.(Iterators.product([xig for i in 1:2]...))...)[end:-1:1,:]
F = M * Xg

## Normalize
F, Fu, Fd = F ./ map(norm, eachcol(F))', Fu ./ map(norm, eachcol(Fu))', Fd ./ map(norm, eachcol(Fd))'

## Plot
scale = 0.6
quiver(Xg[1,:], Xg[2,:], quiver = 0.75 * scale .* (F[1,:],  F[2,:]),  label="f",  color="blue", alpha=0.1)
scatter!(B[1,:], B[2,:], xlims=[-3, 3], ylims=[-3, 3], label="BRS pts", color="pink", alpha=1.)
quiver!(B[1,:], B[2,:], quiver = scale .* (Fu[1,:], Fu[2,:]), label="fu", color="green")
quiver!(B[1,:], B[2,:], quiver = scale .* (Fd[1,:], Fd[2,:]), label="fd", color="red")
plot!([10, 11], [10, 11],  alpha=0.1, color=:blue, label="Ax")
plot!([10, 11], [10, 11],  alpha=1., color=:green, label="Buˢ")
plot!([10, 11], [10, 11],  alpha=1., color=:red,   label="Cdˢ")
