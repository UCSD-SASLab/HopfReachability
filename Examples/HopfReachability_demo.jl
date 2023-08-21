
using LinearAlgebra, Plots
plotlyjs()
include("/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, Hopf

## Initialize (2D Example)
A = [0. 1; -2 -3]
B₁ = 0.5 * [1 0; 0 1]
B₂ = 0.5 * [2 0; 0 1]
Q₁ = 0.1 * 3 * [1 0; 0 1]
Q₂ = 0.2 * 2 * [1 0; 0 1]
c₁ = 0*[0.5 0.75]
c₂ = -0*[0.5 0]
system = (A, B₁, B₂, Q₁, Q₂, c₁, c₂)

## Initialize (uncommet for 3D Example)
# A = [0. 1 0.; -2 -3 0.; 0. 0. -1.]
# B₁ = 0.5 * [1 0; 0 1; 0. 0.]
# B₂ = 0.5 * [2 0; 0 1; 0. 0.]
# Q₁ = 0.1 * 3 * [1 0; 0 1]
# Q₂ = 0.2 * 2 * [1 0; 0 1]
# c₁ = 0*[0.5 0.75]
# c₂ = -0*[0.5 0]
# system = (A, B₁, B₂, Q₁, Q₂, c₁, c₂)

## Time
th = 0.02
Th = 0.2
Tf = 0.8
T = collect(Th : Th : Tf)

## Target: J(x) = 0 is the boundary of the target
Qₓ = diagm(vcat([1.5], ones(size(A)[1]-1)))
cₓ = zero(A[:,1])
r = 1.0
J(x::Vector, Qₓ, cₓ) = ((x - cₓ)' * inv(Qₓ) * (x - cₓ))/2 - 0.5 * r^2 #don't need yet
Jˢ(v::Vector, Qₓ, cₓ) = (v' * Qₓ * v)/2 + cₓ'v + 0.5 * r^2
J(x::Matrix, Qₓ, cₓ) = diag((x .- cₓ)' * inv(Qₓ) * (x .- cₓ))/2 .- 0.5 * r^2
Jˢ(v::Matrix, Qₓ, cₓ) = diag(v' * Qₓ * v)/2 + (cₓ'v)' .+ 0.5 * r^2 #don't need yet
target = (J, Jˢ, (Qₓ, cₓ))

## Automatic Grid Parameters (can also define matrix of points Xg)
bd = 4
ϵ = 0.5e-7
N = 3 + ϵ
grid_p = (bd, N)

## Hopf Coordinate-Descent Parameters (optional)
vh = 0.01
L = 5
tol = ϵ
lim = 500
lll = 5
max_runs = 3
max_its = 500
opt_p_cd = (vh, L, tol, lim, lll, max_runs, max_its)

# Hopf ADMM Parameters (optional)
ρ, ρ2 = 1e-1, 1e-1
tol = 1e-5
max_its = 3
opt_p_admm = (ρ, ρ2, tol, max_its)

# ADMM-CD Hybrid Parameters (optional)
ρ_grid_vals = 1 
hybrid_runs = 3
opt_p_admm_cd = ((1e-1, 1e-1, 1e-5, 3), (0.005, 5, 1e-4, 500, 1, 3, 500), ρ_grid_vals, ρ_grid_vals, hybrid_runs)

solution, run_stats = Hopf_BRS(system, target, T; th, grid_p, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot_scatter = plot_BRS(T, solution...; A, ϵs=2e-1, interpolate=false, value_fn=true, alpha=0.1)
plot_contour = plot_BRS(T, solution...; A, ϵc=1e-3, interpolate=true, value_fn=true, alpha=0.5)