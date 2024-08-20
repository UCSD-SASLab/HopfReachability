
using LinearAlgebra, Plots
using Metal
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_BRS_metal, Hopf_cd, make_grid, make_levelset_fs, make_set_params, make_sample

## System & Game
A, B₁, B₂ = [0. 1; -2 -3], [2 0; 0 1], [1 0; 0 1] # system
max_u, max_d, input_center, input_shapes = 0.4, 0.3, zeros(2), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
Q, center, radius = diagm(vcat([1.25^2], ones(size(A)[1]-1))), zero(A[:,1]), 1.
J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.2, 0.8
times = collect(Th : Th : Tf);

## Point(s) to Solve (any set works!)
bd, res, ϵ = 4, 0.25, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ); # solve over grid
Xg_rand = 2bd*rand(2, 500) .- bd .+ ϵ; # solve over random samples

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 2, 1e-3, 50, 2, 5, 200
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

# solution, run_stats = Hopf_BRS(system, target, times; Xg, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);
# solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

# plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=true, camera=(30, 15))
# plot(solution_sampled; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=false, value=true, camera=(30, 15))

solution, run_stats = @device_code_warntype Hopf_BRS_metal(system, target, times; Xg, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);


# solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

# plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=true, camera=(30, 15))
# plot(solution_sampled; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=false, value=true, camera=(30, 15))



function vadd(a, b, c)
    i = thread_position_in_grid_1d()
    c[i] = a[i] + b[i]
    return
end

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = MtlArray(a)
d_b = MtlArray(b)
d_c = MtlArray(c)

len = prod(dims)
@device_code_warntype @metal threads=len vadd(d_a, d_b, d_c)

## testing 

Qt, ct, rt = target[3]
iQt = inv(Qt)
iQd, rd, cd = convert(Main.MtlArray{Float32}, iQt), convert(Float32, rt), convert(Main.MtlArray{Float32}, ct)

num_samples = 20000
nx = 2
lbs, ubs = typeof(bd) <: Tuple || typeof(bd) <: Array ? (typeof(bd[1]) <: Tuple || typeof(bd[1]) <: Array ? bd : (bd[1]*ones(nx), bd[2]*ones(nx))) : (-bd*ones(nx), bd*ones(nx))
X = make_sample(bd, nx, num_samples)
X = convert(Main.MtlArray{Float32}, X)

ϕX = Main.Metal.zeros(num_samples)

function target_kernel(Xd::MtlDeviceMatrix{Float32, 1}, ϕXd::MtlDeviceVector{Float32, 1})
    i = thread_position_in_grid_1d()
    # ϕXd[i] = J(Xd[:,i])
    # ϕXd[i] = ((Xd[:,i] - cd)' * iQd * (Xd[:,i] - cd))/2 - 0.5 * rd^2
    
    x = @inbounds @view Xd[:,i]
    # xmc = x
    ϕXd[i] = (sum(Vector(x .* x)) - 0.5f0^2)/2
    # ϕXd[i] = (sum(x) - 0.5f0^2)/2
    # v = @inbounds @view Xd[:, i]
    # ϕXd[i] = sum(v)
    return 
end

@device_code_warntype @metal threads=100 target_kernel(X, ϕX)