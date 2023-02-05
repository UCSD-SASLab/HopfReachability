
using LinearAlgebra, Plots
plotlyjs()
# push!(LOAD_PATH,"/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR");
# using HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, Hopf

include("/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR/HopfReachability_dev.jl");
using .HopfReachability_dev: Hopf_BRS, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, Hopf

## Initialize
M = [0. 1; -2 -3]
B = 0.5 * [1 0; 0 1]
C = 0.5 * [2 0; 0 1]
Q = 0.1 * 3 * [1 0; 0 1]
Q2 = 0.2 * 2 * [1 0; 0 1]
a1 = 0*[0.5 0.75]
a2 = -0*[0.5 0]
system = (M, B, C, Q, Q2, a1, a2)

# M = [0. 1 0.; -2 -3 0.; 0. 0. -1.]
# B = 0.5 * [1 0; 0 1; 0. 0.]
# C = 0.5 * [2 0; 0 1; 0. 0.]
# Q = 0.1 * 3 * [1 0; 0 1]
# Q2 = 0.2 * 2 * [1 0; 0 1]
# a1 = 0*[0.5 0.75]
# a2 = -0*[0.5 0]
# system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.05
Th = 0.2
Tf = 0.8
T = collect(Th : Th : Tf)

## Target: J(x) = 0 is the boundary of the target
Ap = diagm([1.5, 1])
cp = [0.; 0.]
# Ap = diagm([1.5, 1, 1.])
# cp = [0.; 0.; 0.]
r = 2.0
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

## Grid Parameters
bd = 3. # (-3, 3) for ellipse
ϵ = 0.5e-7
N = 3 + ϵ
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

solution, run_stats = Hopf_BRS(system, target, intH_ytc17, T;
                                                    opt_method=Hopf_cd,
                                                    preH=preH_ytc17,
                                                    th,
                                                    grid_p,
                                                    opt_p=opt_p_cd,
                                                    warm=false,
                                                    check_all=true,
                                                    printing=true);
B⁺T, ϕB⁺T = solution;

plot_scatter = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵs=0.1, interpolate=false, value_fn=true, alpha=0.5)
plot_contour = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵc=0.01, interpolate=true, value_fn=true, alpha=0.5)

# using ScatteredInterpolation, Plots, PlotlyJS
# plotlyjs()

# dim = 3
# xig = [collect(-2. : 0.1 : 2.) for i=1:dim]
# b⁺ = hcat(collect.(Iterators.product(xig...))...)[end:-1:1,:]
# ϕ = reshape(sum(b⁺.^2, dims=1) .- 0.5, size(b⁺)[2])
# # # # itp = interpolate(Polyharmonic(), b⁺, reshape(ϕ, size(b⁺)[2]))
# # # # itpd = evaluate(itp, G')
# # # # iϕG = reshape(itpd, [length(xigi) for xigi in xig]...)
# ϵc, ϵs = 1e-5, 0.1

# # pl = Plots.plot(title="BRS of T, in X")

# b = b⁺[:, abs.(ϕ) .< ϵs]
# # scatter!(pl, [b[i,:] for i=1:dim]..., label="label", markersize=2, markercolor=:blue, markerstrokewidth=0)
# # contour!(pl, [b⁺[i,:] for i=1:dim]..., ϕ, levels=-ϵc:ϵc:ϵc, lc=:blue, lw=2, colorbar=false)

# # surface!(pl, xig..., reshape(ϕ, length.(xig)...)', color=:blue, lw=2, colorbar=false, alpha=0.8)
# # surface!(pl, b⁺[1,:], b⁺[2,:], ϕ[:], color=:blue, lw=2, colorbar=false, alpha=0.8)

# one = isosurface(
#                 x=b⁺[1,:],
#                 y=b⁺[2,:],
#                 z=b⁺[3,:],
#                 value=(sum(b⁺.^2, dims=1) .- 1)[:],
#                 opacity=0.5,
#                 isomin=-1,
#                 isomax=1,
#                 surface_count=1,
#                 showscale=false,
#                 caps=attr(x_show=false, y_show=false),
#                 colorscale=colors.RdBu_3,
#                 name="one",
#                 showlegend=true,
#                 title="BRS"
#                 )

# two = isosurface(
#                 x=b⁺[1,:],
#                 y=b⁺[2,:],
#                 z=b⁺[3,:],
#                 value=(sum(cos.(b⁺).^3, dims=1) .- 1.2)[:],
#                 opacity=0.9,
#                 isomin=-ϵc,
#                 isomax=ϵc,
#                 surface_count=1,
#                 showscale=true,
#                 caps=attr(x_show=false, y_show=false, z_show=false),
#                 colorscale=[[0, "rgb(1.0,0.0,0.0)"], [1, "rgb(1.0,0.0,0.0)"]], # "rgb(255,0,0)"
#                 name="two",
#                 showlegend=true,
#                 title="BRS"
#                 )
# # pli = PlotlyJS.plot(two)

# pls = Array{GenericTrace{Dict{Symbol, Any}},1}()
# push!(pls, one)
# push!(pls, two)
# # PlotlyJS.plot([one, two], Layout(title="Wohoo"))

# PlotlyJS.plot(pls, Layout(title="Wohoo"))


# ## Plots BRS over T in X and Z space
# function plot_BRS_dev(T, B⁺T, ϕB⁺T; M, simple_problem=true, ϵs = 0.1, ϵc = 1e-5, cres = 0.1, zplot=false, interpolate=false, inter_method=Polyharmonic(), pal_colors=[:red, :blue], alpha=0.5, title=nothing, value_fn=false, dim=size(B⁺T[1])[1])

#     if dim > 2 && value_fn; println("4D plots are not supported yet, can't plot Value fn"); value_fn = false; end

#     Xplot = isnothing(title) ? Plots.plot(title="BRS: Φ(X, T) = 0") : Plots.plot(title=title)
#     if zplot; Zplot = Plots.plot(title="BRS: Φ(Z, T) = 0"); end

#     plots = zplot ? [Xplot, Zplot] : [Xplot]
#     if value_fn; vfn_plots = zplot ? [Plots.plot(title="Value: Φ(X, T)"), Plots.plot(title="Value: Φ(Z, T)")] : [Plots.plot(title="Value: Φ(X, T)")]; end

#     B⁺Tc, ϕB⁺Tc = copy(B⁺T), copy(ϕB⁺T)
    
#     ϕlabels = "ϕ(⋅,-" .* string.(T) .* ")"
#     Jlabels = "J(⋅, t=" .* string.(-T) .* " -> ".* string.(vcat(0.0, -T[1:end-1])) .* ")"
#     labels = collect(Iterators.flatten(zip(Jlabels, ϕlabels))) # 2 * length(T)

#     Tcolors = length(T) > 1 ? palette(pal_colors, length(T)) : [pal_colors[2]]
#     B0colors = length(T) > 1 ? palette([:black, :gray], length(T)) : [:black]
#     plot_colors = collect(Iterators.flatten(zip(B0colors, Tcolors)))

#     ## Zipping Target to Plot Variation in Z-space over Time (already done in moving problems)
#     if simple_problem && (length(T) > 1)
#         for i = 3 : 2 : 2*length(T)
#             insert!(B⁺Tc, i, B⁺T[1])
#             insert!(ϕB⁺Tc, i, ϕB⁺T[1])
#         end
#     end

#     if dim > 2 && interpolate; plotly_pl = zplot ? [Array{GenericTrace{Dict{Symbol, Any}},1}(), Array{GenericTrace{Dict{Symbol, Any}},1}()] : [Array{GenericTrace{Dict{Symbol, Any}},1}()]; end

#     for (j, i) in enumerate(1 : 2 : 2*length(T))        
#         B⁺0, B⁺, ϕB⁺0, ϕB⁺ = B⁺Tc[i], B⁺Tc[i+1], ϕB⁺Tc[i], ϕB⁺Tc[i+1]
#         Bs = zplot ? [B⁺0, B⁺, exp(-T[j] * M) * B⁺0, exp(-T[j] * M) * B⁺] : [B⁺0, B⁺]

#         for (bi, b⁺) in enumerate(Bs)
#             if simple_problem && bi == 1 && i !== 1; continue; end

#             ϕ = bi % 2 == 1 ? ϕB⁺0 : ϕB⁺
#             label = simple_problem && i == 1 && bi == 1 ? "J(⋅)" : labels[i + (bi + 1) % 2]

#             ## Plot Scatter
#             if interpolate == false

#                 ## Find Boundary in Near-Boundary
#                 b = b⁺[:, abs.(ϕ) .< ϵs]

#                 scatter!(plots[Int(bi > 2) + 1], [b[i,:] for i=1:dim]..., label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
#                 # scatter!(plots[Int(bi > 2) + 1], b[1,:], b[2,:], label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0)
                
#                 if value_fn
#                     scatter!(vfn_plots[Int(bi > 2) + 1], b⁺[1,:], b⁺[2,:], ϕ, label=label, markersize=2, markercolor=plot_colors[i + (bi + 1) % 2], markerstrokewidth=0, alpha=alpha)
#                     # scatter!(vfn_plots[Int(bi > 2) + 1], b[1,:], b[2,:], ϕ, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], label=label)
#                 end
            
#             ## Plot Interpolation
#             else 

#                 if dim == 2
#                     contour!(plots[Int(bi > 2) + 1], [b⁺[i,:] for i=1:dim]..., ϕ, levels=-ϵc:ϵc:ϵc, colorbar=false, lc=plot_colors[i + (bi + 1) % 2], lw=2, label=label)

#                     if value_fn

#                         ## Make Grid
#                         xig = [collect(minimum(b⁺[i,:]) : cres : maximum(b⁺[i,:])) for i=1:dim]
#                         G = hcat(collect.(Iterators.product(xig...))...)'
                        
#                         ## Construct Interpolationb (Should skip this in the future and just use Plotly's built in one for contour)
#                         itp = ScatteredInterpolation.interpolate(inter_method, b⁺, ϕ)
#                         itpd = evaluate(itp, G')
#                         iϕG = reshape(itpd, length(xig[1]), length(xig[2]))'
                        
#                         surface!(vfn_plots[Int(bi > 2) + 1], xig..., iϕG, colorbar=false, color=plot_colors[i + (bi + 1) % 2], label=label, alpha=alpha)
#                     end
            
#                 else
#                     # isosurface!(plots[Int(bi > 2) + 1], xig..., iϕG, isomin=-ϵc, isomax=ϵc, surface_count=2, lc=plot_colors[i + (bi + 1) % 2], alpha=0.5)
#                     pl = isosurface(x=b⁺[1,:], y=b⁺[2,:], z=b⁺[3,:], value=ϕ[:], opacity=alpha, isomin=-ϵc, isomax=ϵc, surface_count=1, showlegend=true, showscale=false, caps=attr(x_show=false, y_show=false, z_show=false),
#                         name=label, colorscale=[[0, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]], [1, "rgb" * string(plot_colors[i + (bi + 1) % 2])[13:end]]])

#                     println(length(plotly_pl[Int(bi > 2) + 1]))
#                     println(typeof(pl))
#                     push!(plotly_pl[Int(bi > 2) + 1], pl)
#                     println(length(plotly_pl[Int(bi > 2) + 1]))
#                 end
#             end
#         end
#     end

#     if value_fn
#         Xplot = Plots.plot(vfn_plots[1], Xplot)
#         if zplot; Zplot = Plots.plot(vfn_plots[2], Zplot); end
#     end

#     if dim > 2 && interpolate 
#         Xplot = PlotlyJS.plot(plotly_pl[1], Layout(title="BRS of T, in X", gridcolor="black", plot_bgcolor="rgb(0,0,0)", showbackground=false));
#         if zplot; Zplot = PlotlyJS.plot(plotly_pl[2], Layout(title="BRS of T, in X")); end
#     end

#     display(Xplot)
#     if zplot; display(Zplot); end

#     return plots, plotly_pl
# end