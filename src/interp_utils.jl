
using Interpolations, ScatteredInterpolation, LinearAlgebra

function make_interpolation(solution, alltimes; xigs=nothing, method="grid", itp_alg_grid=Interpolations.Gridded(Interpolations.Linear()), itp_alg_scatter=ScatteredInterpolation.Polyharmonic())
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

        tXs = zeros(1+size(solution[1][1],1), 0)
        for ti=1:length(times)+1
            tXs = hcat(tXs, vcat(zero(solution[1][1])[[1],:] .+ alltimes[ti], solution[1][ti]))
        end
        
        V_itp = ScatteredInterpolation.interpolate(itp_alg_scatter, tXs, vcat(solution[2]...))
    end

    return V_itp
end

function fast_interp(_V_itp, tXg; method="grid")
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

# ### TEST 2D

# include(pwd() * "/src/HopfReachability.jl");
# using .HopfReachability: Hopf_BRS, Hopf_cd, make_grid, make_levelset_fs, make_set_params
# using JLD2, Plots

# ## System & Game
# A, B₁, B₂ = [0. 0.5; -1 -1], [0.4; 0.], [0.1; 0.] # system
# max_u, max_d, input_center, input_shapes = 0.5, 0.3, zeros(1), "box"
# Q₁, c₁ = make_set_params(input_center, max_u; type=input_shapes) 
# Q₂, c₂ = make_set_params(input_center, max_d; type=input_shapes) # 𝒰 & 𝒟
# system, game = (A, reshape(B₁, 2, 1), reshape(B₂, 2, 1), Q₁, c₁, Q₂, c₂), "reach"

# ## Target
# Q, center, radius = diagm(ones(size(A)[1])), zero(A[:,1]), 0.25
# J, Jˢ = make_levelset_fs(center, radius; Q, type="ellipse")
# target = (J, Jˢ, (Q, center, radius));

# ## Times to Solve
# Th, Tf = 0.25, 1.0
# times = collect(Th : Th : Tf);

# ## Point(s) to Solve (any set works!)
# bd, res, ϵ = 1.0001, 0.025, .5e-7
# Xg, xigs, (lb, ub) = make_grid(bd, res, size(A)[1]; return_all=true, shift=ϵ); # solve over grid
# Xg_rand = 2bd*rand(2, 500) .- bd .+ ϵ; # solve over random samples

# ## Hopf Coordinate-Descent Parameters (optional, note the default are conserative/slower)
# vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its = 0.01, 5, 1e-3, 50, 4, 5, 500
# opt_p_cd = (vh, stepsz, tol, conv_runs_rqd, stepszstep_its, max_runs, max_its)

# solution, run_stats = Hopf_BRS(system, target, times; Xg, th=0.05, input_shapes, game, opt_method=Hopf_cd, opt_p=opt_p_cd, warm=false, check_all=true, printing=true);

# ## Interpolate Hopf Solution

# ## Test
# alltimes = vcat(0., times...)
# tXg = zeros(1+size(Xg,1), 0)
# for ti=1:length(times)+1
#     tXg = hcat(tXg, vcat(zero(solution[1][ti])[[1],:] .+ alltimes[ti], solution[1][ti]))
# end
# const tXg2 = copy(tXg)

# ## grid
# V_itp_hopf = make_interpolation(solution, alltimes; xigs, method="grid")
# Vg = @time fast_interp(V_itp_hopf, tXg2)

# ## scatter (works but slow)
# V_itp_hopf, fast_interp = make_interpolation(solution, alltimes; method="scatter")
# Vg = @time fast_interp(V_itp, tXg2)

# tXs = zeros(1+size(solution[1][1],1), 0)
# for ti=1:length(times)+1
#     tXs = hcat(tXs, vcat(zero(solution[1][1])[[1],:] .+ alltimes[ti], solution[1][ti]))
# end

# test_samp = vcat(hcat(tXs[[1],:],tXs[[1],:]), rand(2, 2*size(tXs,2)))
# # scat_itp_methods = [Multiquadratic, InverseMultiquadratic, Gaussian, InverseQuadratic, Polyharmonic, Shepard, NearestNeighbor]
# for method in scat_itp_methods
#     V_itp = ScatteredInterpolation.interpolate(method(), tXs, vcat(solution[2]...));
#     println("\nMethod $method takes (3x):")
#     for _=1:3
#         @time ScatteredInterpolation.evaluate(V_itp, test_samp)
#     end 
# end

# samples = [0.0; 0.5; 0.5; 0.5; 1.0];
# points = [0.0 0.0; 0.0 1.0; 1.0 0.0; 0.5 0.5; 1.0 1.0]';
# itp = ScatteredInterpolation.interpolate(Multiquadratic(), points, samples);
# evaluate(itp, [0.6; 0.6])


# # save("lin2d_hopf_interp_linear.jld", "V_itp", V_itp_hopf, "solution", solution, "alltimes", alltimes, "xigs", xigs)
# # V_itp_hopf_loaded = load("hopf_interp_linear.jld")["V_itp"]
# # Vg = @time fast_interp(V_itp_hopf_loaded, tXg2)

# ## Plot
# Xg_near = tXg[:, abs.(Vg) .≤ 0.005]
# plot(solution; interpolate=true, labels=vcat("Target", ["t=-$ti" for ti in times]...), color_range=["red", "blue"], grid=true, xigs=xigs, value=false)
# scatter!(eachrow(Xg_near[2:end,:])..., alpha=0.5)

# x = y = range(-0.95, stop = 0.95, length = 133)
# surface(x, y, (x, y) -> V_itp_hopf(y, x, 1.))

# ### TEST 3D


