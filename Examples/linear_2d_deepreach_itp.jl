
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_levelset_fs, make_set_params
using LinearAlgebra, Plots

## System & Game
A, B₁, B₂ = [0. 0.5; -1 -1], [0.4 0; 0 0.1], [0. 0; 0 0.1] # system
max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(2), "box"
Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
system, game = (A, B₁, B₂, Q₁, c₁, Q₂, c₂), "reach"

## Target
Q, center, radius = diagm(ones(size(A)[1])), zero(A[:,1]), 0.25
J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
target = (J, Jˢ, (Q, center, radius));

## Times to Solve
Th, Tf = 0.25, 1.0
times = collect(Th : Th : Tf);

## Point(s) to Solve (any set works!)
bd, res, ϵ = 1, 0.025, .5e-7
Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ); # solve over grid
Xg_rand = 2bd*rand(2, 500) .- bd .+ ϵ; # solve over random samples

## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

solution, run_stats = Hopf_BRS(system, target, times; Xg, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=false, check_all=true, printing=true);
# solution_sampled, run_stats = Hopf_BRS(system, target, times; Xg=Xg_rand, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=true, check_all=true, printing=true);

plot(solution; interpolate=false, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=true, camera=(30, 15), ϵs=0.001)
# plot(solution_sampled; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=false, value=true, camera=(30, 15))

function make_interpolation(solution, alltimes, xigs, method="grid", itp_alg_grid=Gridded(Linear()), itp_alg_scatter=Polyharmonic())
    # only supports 2d interp atm

    ## Grid-based (much faster evaluation, but requires grid solution)
    if method == "grid"
        
        VXgt = zeros(length(xigs[1]), length(xigs[2]), length(alltimes))
        for ti=1:length(times)+1
            VXgt[:,:,ti] = reshape(solution[2][ti], length(xigs[1]), length(xigs[2]))'
        end
        V_itp = Interpolations.interpolate((xigs..., alltimes), VXgt, itp_alg_grid);
    
    ## Scatter-based (about 1e3x slower, eg ~3000 pts/s)
    elseif method == "scatter"
        V_itp = ScatteredInterpolation.interpolate(itp_alg_scatter, solution[1][ti], solution[2][ti])
    end

    return V_itp, fast_interp
end

function fast_interp(_V_itp, tXg, method="grid")
    if method == "grid"
        Vg = zeros(size(tXg,2))
        for i=1:length(Vg)
            Vg[i] = _V_itp(tXg[:,i][end:-1:1]...)
        end
    else
        Vg = ScatteredInterpolation.evaluate(_V_itp, tXg)
    end
    return Vg
end

## Test
alltimes = vcat(0., times...)
tXg = zeros(1+size(Xg,1), 0)
for ti=1:length(times)+1
    tXg = hcat(tXg, vcat(zero(solution[1][ti])[[1],:] .+ alltimes[ti], solution[1][ti]))
end
const tXg2 = copy(tXg)

## grid
V_itp, fast_interp = make_interpolation(solution, alltimes, xigs)
Vg = @time fast_interp(V_itp, tXg2)

# ## scatter (works but slow)
# V_itp, fast_interp = make_interpolation(solution, alltimes, xigs, method="scatter")
# Vg = @time fast_interp(V_itp, tXg2)

using JLD2, Interpolations
# save("lin2d_hopf_interp_linear.jld", "V_itp", V_itp, "solution", solution, "alltimes", alltimes, "xigs", xigs)

V_itp_loaded = load("lin2d_hopf_interp_linear.jld")["V_itp"]
Vg = @time fast_interp(V_itp_loaded, zeros(3, 60000))

## Plot
Xg_near = tXg[:, abs.(Vg) .≤ 0.005]
plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false, camera=(30, 15))
scatter!(eachrow(Xg_near[2:end,:])..., alpha=0.5)

x = y = range(-0.95, stop = 0.95, length = 133)
surface(x, y, (x, y) -> V_itp_loaded(y, x, 1.))