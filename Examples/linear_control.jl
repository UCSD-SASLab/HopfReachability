
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, plot_nice, Hopf, Hopf_minT, make_set_params, make_target, make_grid
using LinearAlgebra, Plots

## System & Game
A, B₁, B₂ = [0. 1; -2 -3], [2 0; 0 1], [1 0; 0 1] # system
max_u, max_d, input_center, input_shapes = 0.4, 0.3, zeros(2), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Times to Solve
th, Th, Tf = 0.02, 0.2, 0.6
T = collect(Th : Th : Tf);

## Elliptical Target: J(x) = 0 is the boundary of the target
center, radius = zero(A[:,1]), 1.
target = make_target(center, radius; type="ball")

## Points to Solve
bd, res, ϵ = 4, 0.1, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ);

## Hopf Coordinate-Descent Parameters
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

### Solve the BRS to Validate our Controller
solution, run_stats = Hopf_BRS(system, target, T; Xg, opt_method=Hopf_cd, th, opt_p=opt_p_cd, warm=false, check_all=true, printing=true);
XgT, ϕXgT = solution;

plot(solution; xigs=xigs, interpolate=true)
plot(solution; interpolate=false)

## Find Boundary Pts of one BRS for Time of Interest
Toi = 0.4
Tix = findfirst(abs.(T .- Toi) .< Th/2);
B = XgT[Tix + 1][:, abs.(ϕXgT[Tix + 1]) .< 1e-2] # Tix + 1 because target set is first

## Test a Single Call
x = [-0.8, 2.4];
uˢ, dˢ, Tˢ, ϕ = Hopf_minT(system, target, x; time_p=(th, Th, Tf + ϵ), printing=true)

## Compute Minimum Time Tˢ until reachable and the Optimal Control uˢ and Disturbance dˢ at that time
Fu, Fd = zero(B), zero(B)
for (bi, x) in enumerate(eachcol(B))
    uˢ, dˢ, Tˢ, ϕ = Hopf_minT(system, target, x; time_p=(th, Th, Tf + ϵ))
    Fu[:, bi], Fd[:, bi] = B₁ * uˢ, B₂ * dˢ; 
end

## Compute Autonomous Field
bd, res, ϵ = 4.5, 0.35, .5e-7
Xg2, xigs2, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ);
F = A * Xg2

## Normalize
F, Fu, Fd = F ./ map(norm, eachcol(F))', Fu ./ map(norm, eachcol(Fu))', Fd ./ map(norm, eachcol(Fd))'

## Plot
scale = 0.35
ocd_plot = quiver(Xg2[1,:], Xg2[2,:], quiver = 0.75 * scale .* (F[1,:],  F[2,:]), label="", color="blue", alpha=0.05, lw=2, dpi=300)
contour!(xigs..., reshape(ϕXgT[1], length(xigs[1]), length(xigs[1]))', levels=[0], label="", lw=2, color="black", colorbar=false)
plot!([100, 100], [200, 200], lw=2, color="black", label="Target", xlims=(-bd,bd), ylims=(-bd,bd))
contour!(xigs..., reshape(ϕXgT[Tix+1], length(xigs[1]), length(xigs[1]))', levels=[0], label="", lw=2, color="purple", colorbar=false)
plot!([100, 100], [200, 200], lw=2, color="purple", label="BRS, t=-$(T[Tix])", xlims=(-bd,bd), ylims=(-bd,bd))
quiver!(B[1,:], B[2,:], quiver = scale .* (Fu[1,:], Fu[2,:]), label="", color="green", lw=1.5, alpha=0.9)
quiver!(B[1,:], B[2,:], quiver = scale .* (Fd[1,:], Fd[2,:]), label="", color="red", lw=1.5, alpha=0.9)
scatter!(B[1,:], B[2,:], label="BRS pts", color="orange", alpha=1, markersize=2, markerstrokewidth=0)
plot!([10, 11], [10, 11], alpha=0.5, color="blue", label="Ax", lw=1.5)
plot!([10, 11], [10, 11], alpha=0.9, color="green", label="Buˢ", lw=1.5)
plot!([10, 11], [10, 11], alpha=0.9, color="red",   label="Cdˢ", lw=1.5)
plot!(title="Optimal Control and Disturbance in a Reach Game")