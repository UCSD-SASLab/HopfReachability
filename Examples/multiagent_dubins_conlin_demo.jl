
using LinearAlgebra, Plots, LaTeXStrings, JLD2
plotlyjs()
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_minT, Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, intH_box, preH_box, plot_nice, HJoc_box
using TickTock, Suppressor
using ReachabilityAnalysis, Polyhedra
using DifferentialEquations

## Variable Parameters
n_ag = 5; # have pushed up to 25 (100 dim, with success)
r, r_follow = 0.5, 5; # radius of ego & aux agents for capture
θ_max = 4π; # theta of capture (should be inf but thats bad for numerics)
v_b = 3.; # evader velocity
v_a = 1.; # pursuer velocity
max_u, max_d = 0.36, 1.; # ctrl of pursuer, evader
x_init = 3.; # l2 distance away of pursuers initially
θ = pi/(2 * n_ag); # evader initial heading wrt pursuers, base pi/6
Tf = x_init / max(v_a, v_b); # time to analyze, usually set to time it takes for swarm to collide at speed

## Static Parameters
dim_x = 3 # agent dimension
n_d = 1
dim_xh = n_ag * dim_x # system dimension
dim_xh_d = dim_xh + n_d

## Initial Point

rot(θ, p) = [cos(θ) -sin(θ); sin(θ) cos(θ)] * p
polygon = [rot((2π/n_ag)*(i-1) + θ, [0., x_init]) for i=1:n_ag]
xh_init = vcat([vcat(x..., -π + (i-1)*(2π/n_ag) + π/2 + θ) for (i, x) in enumerate(polygon)]...)
xh_init = round.(xh_init, digits=12)

# plot_reldubins_tp([vcat(xh_init, zeros(dim_x));;], 1; relative=false)

# equi_triangle = [(0., 1), (-sqrt(3)/2, -1/2), (sqrt(3)/2, -1/2)];
# equi_triangle_scaled = map(x -> x_init .* x, equi_triangle);
# equi_triangle_scaled_rot = [[cos(θ) -sin(θ); sin(θ) cos(θ)] * [x; y] for (x,y) in equi_triangle_scaled]
# xh_init = vcat([vcat(x..., -π + (i-1)*(2π/n_ag) + π/2 + θ) for (i, x) in enumerate(equi_triangle_scaled_rot)]...)
# xh_init = round.(xh_init, digits=12)

dim_xp = 2 # pos dim
Bi = [0; 0; 1] # pursuer i (player bi)
Qc = [max_u] # pursuer control lim
Qd = [max_d] # evader control lim

Wi = diagm(vcat(fill(r^2, dim_xp), fill(θ_max^2, 1))) # per-agent target set params (ego : follower)
Wi_true = diagm(vcat(fill(r^2, dim_xp), fill(Inf, 1))) # unsolveable but true evaluation metric
qi_high = [0; 0; π]; qi_low = [0; 0; -π] # target center

ρW = 0.5; # partition scale
W_split_i = diagm(vcat(fill((ρW * r)^2, dim_xp), fill((ρW * θ_max)^2, 1))) # per-agent target set params (ego : follower)

## Combined System 

Bh = zeros(dim_xh, n_ag);

Qch = Qc .* I(n_ag) # 1d ctrl per agent
Qdh = Qd .* I(n_d) # 1d ctrl per agent
qc, qd = zeros(1, n_ag), [0.] #center of input constraints

Whs = [zeros(dim_xh, dim_xh) for _ in 1:n_ag]; Whs_true = [zeros(dim_xh, dim_xh) for _ in 1:n_ag]; Whs_split = [zeros(dim_xh, dim_xh) for _ in 1:n_ag]
qh = zeros(dim_xh)

for i = 1:n_ag
    Bh[dim_x * (i - 1) + 1 : dim_x * i, i] = Bi
    qh[dim_x * (i - 1) + 1 : dim_x * i] = [0; 0; xh_init[(i-1)*dim_x + 3]] # OLD: i ≤ ceil(n_ag/2) ? qi_high : qi_low

    for j=1:n_ag
        Whs[i][dim_x * (j - 1) + 1 : dim_x * j, dim_x * (j - 1) + 1 : dim_x * j] = i == j ? Wi : diagm(vcat(fill(r_follow^2, dim_xp), fill(θ_max^2, 1)))
        Whs_true[i][dim_x * (j - 1) + 1 : dim_x * j, dim_x * (j - 1) + 1 : dim_x * j] = i == j ? Wi_true : diagm(Inf*ones(dim_x))
        Whs_split[i][dim_x * (j - 1) + 1 : dim_x * j, dim_x * (j - 1) + 1 : dim_x * j] = i == j ? W_split_i : diagm(vcat(fill(r_follow^2, dim_xp), fill(θ_max^2, 1)))
    end
end

## Elliptical Target: J(x) = 0 is the boundary of the target

Js(v::Vector, Q, q) = (v' * Q * v)/2 + q'v + 0.5
J(x::Matrix, Q, q) = diag((x .- q)' * inv(Q) * (x .- q))/2 .- 0.5
targets = [(J, Js, (Whs[i], qh)) for i=1:n_ag]
targets_true = [(J, Js, (Whs_true[i], qh)) for i=1:n_ag]

qps_split(agi) = [qh + vcat(zeros((agi-1)*dim_x + i - 1), dr, zeros(dim_xh_d-1-((agi-1)*dim_x+i))) for (dr, i) in [(-0.5, 1), (-0.5, 2), (0.5, 2)]]
n_part = length(qps_split(1));
targets_partitions = [[(J, Js, (Whs[i], qps_split(i)[j])) for j=1:n_part] for i=1:n_ag];

## Observe trajectory solution

function SingleDubins!(dx, x, p, s)
    v, u, ai = p
    dx[1] = v * cos(x[3]) # x
    dx[2] = v * sin(x[3]) # y
    dx[3] = u(x,s)[ai] # θ
    return dx
end

function MultiDubins!(dx, x, p, s)
    # Frame of Evader
    us, ds = p(x, s)
    for (ci, c) in enumerate(0:dim_x:dim_x*(n_ag-1))
        dx[c+1] = -v_a + v_b * cos(x[c+3]) + ds[1] * x[c+2] # x_i_Δ,1
        dx[c+2] = v_b * sin(x[c+3]) - ds[1] * x[c+1] # x_i_Δ,2
        dx[c+3] = us[ci] - ds[1] # θ_i_Δ
    end
    return dx
end

function plot_reldubins_tp(Xs, si; Δt=0.1, ylimz=(-5, 5), xlimz=(-5, 5), legend=:topright, 
                            markersize=20, alpha=0.7, alpha_tail=0.3, lw_tail=2, tail_l=3, relative=true,
                            pal_Hopf=palette(:Blues, n_ag+4)[3:end-2], pal_evader=:red)

    gr(); pl = plot(title="t=$(round(si*Δt, digits=2))", xlabel="Horizontal (100m)", ylabel="Vertical (100m)", dpi=300); 

    triangle = [(-sqrt(3)/6, 0.5), (-sqrt(3)/6, -0.5), (sqrt(3)/2 - sqrt(3)/6 + 0.5, 0.), (-sqrt(3)/6, 0.5)];
    rotate_marker(points, θ) = [[cos(θ) -sin(θ); sin(θ) cos(θ)] * [x; y] for (x,y) in points];
    rotate_triangle(θ) = [tuple(x...) for x in rotate_marker(triangle, θ)];

    if relative
        scatter!(pl, [0], [0], label="Evader", lw=2, color=pal_evader, alpha=alpha, marker=Shape(rotate_triangle(0.)), markersize=markersize)
    else
        scatter!(pl, [Xs[1 + n_ag*dim_x, si]], [Xs[2 + n_ag*dim_x, si]], label="Evader", lw=2, color=pal_evader, alpha=alpha, marker=Shape(rotate_triangle(Xs[3 + n_ag*dim_x, si])), markersize=markersize)
    end

    for i=0:(n_ag-1); scatter!(pl, [Xs[1 + i*dim_x, si]], [Xs[2 + i*dim_x, si]], label="Pursuer $(i+1)", lw=2, color=pal_Hopf[i+1], alpha=alpha, marker=Shape(rotate_triangle(Xs[3 + i*dim_x, si])), markersize=markersize); end
    if si > 1
        for i=0:(n_ag-1); plot!(pl, Xs[1 + i*dim_x, si-min(si-1, tail_l):si], Xs[2 + i*dim_x, si-min(si-1, tail_l):si], label="", lw=lw_tail, color=pal_Hopf[i+1], alpha=alpha_tail); end
        if !relative; plot!(pl, Xs[1 + n_ag*dim_x, si-min(si-1, tail_l):si], Xs[2 + n_ag*dim_x, si-min(si-1, tail_l):si], label="", lw=lw_tail, color=pal_evader, alpha=alpha_tail); end
    end
    plot!(pl, ylims=ylimz, xlims=xlimz, legend=legend, aspect_ratio=:equal)
    return pl
end

## Solve Multi-Agent Dubins in Relative State Space 

δt = 1e-2; δts = 0.:δt:Tf;

sol_deq = DifferentialEquations.solve(ODEProblem(MultiDubins!, xh_init, (0., Tf-1e-3)), p=(x,s)->(zeros(n_ag), zeros(1)), saveat=δts)
Xs = hcat(sol_deq.u...)

xlimz, ylimz = (-(x_init + 2), (x_init + 2)), (-(x_init + 2) + x_init/4, (x_init + 2) + x_init/4)

gr()
plot(title="Relative Distance", xlabel="Relative Horizontal (m)", ylabel="Relative Vertical (m)"); 
for i=0:(n_ag-1); plot!(Xs[1 + i*dim_x,:], Xs[2 + i*dim_x,:], label="Agent $(i+1)", lw=2); end
plot!(legend=:topright, aspect_ratio=:equal, ylims=ylimz, xlims=xlimz)

## Solve each in State Space

Xs_nonr = zeros(dim_x*(n_ag+1), length(δts)-1)
for (ai, i) in enumerate(0:dim_x:dim_x*(n_ag-1))
    sol_deqi = DifferentialEquations.solve(ODEProblem(SingleDubins!, xh_init[i+1:i+dim_x], (0., Tf-1e-3)), p=[v_a, (x,s)->zeros(n_ag), ai], saveat=δts)
    Xs_nonr[i+1:i+dim_x, :] = Array(sol_deqi)
end
sol_deqd = DifferentialEquations.solve(ODEProblem(SingleDubins!, zeros(3), (0., Tf-1e-3)), p=[v_b, (x,s)->zero(1), 1], saveat=δts);
Xs_nonr[dim_x*n_ag+1:dim_x*(n_ag+1), :] = Array(sol_deqd)

plot_reldubins_tp(Xs_nonr, 1; Δt=δt, ylimz, xlimz, legend=true, tail_l=size(Xs_nonr,2), relative=false)
plot_reldubins_tp(Xs_nonr, size(Xs_nonr, 2); Δt=δt, ylimz, xlimz, legend=true, tail_l=size(Xs_nonr,2), relative=false)

### Error bounding with feasibility methods (DI)

eix(i; dim=dim_xh+1) = vcat(zeros(i-1), 1, zeros(dim-i)) # onehot for supports

function SingleDubins_RA!(dx, x, p, s)
    dx[1] = -v_a + v_b * cos(x[3]) + x[dim_x + 2] * x[2] # x_i_{Δ,1}
    dx[2] = v_b * sin(x[3]) - x[dim_x + 2] * x[1] # x_i_{Δ,2}
    dx[3] = x[dim_x + 1] - x[dim_x + 2] # θ_i_Δ
    dx[4] = zero(x[1]) # a_i
    dx[5] = zero(x[1]) # b_i
    return dx
end

function SingleDubins_RA_nob!(dx, x, p, s)
    dx[1] = -v_a + v_b * cos(x[3]) # x_i_{Δ,1}
    dx[2] = v_b * sin(x[3]) # x_i_{Δ,2}
    dx[3] = x[dim_x + 1] # θ_i_Δ
    dx[4] = zero(x[1]) # a_i
    return dx
end

## Define Forwards Problem and Solve

# xi = copy(Xs["Hopf"][:, intr_step[3]]) 
xi = copy(xh_init)

X0 = Singleton(xi)
X0is = [Singleton(xi[ci+1:ci+dim_x]) for ci=0:dim_x:dim_x*(n_ag-1)]

U = Hyperrectangle(; low=-Qc, high=Qc)
D = Hyperrectangle(; low=-Qd, high=Qd)
D0 = Hyperrectangle(; low=zeros(n_d), high=zeros(n_d))

sys_is = [InitialValueProblem(BlackBoxContinuousSystem((dx,x,p,t) -> SingleDubins_RA!(dx,x,p,t), dim_x+2), X0is[ai] × U × D) for ai=1:n_ag];
sys_is_nob = [InitialValueProblem(BlackBoxContinuousSystem((dx,x,p,t) -> SingleDubins_RA_nob!(dx,x,p,t), dim_x+1), X0is[ai] × U) for ai=1:n_ag];

@time solz_is = [overapproximate(ReachabilityAnalysis.solve(sys_is[ai]; tspan=(0.0, Tf), alg=TMJets21a(; orderQ=6, abstol=1e-12)), Zonotope) for ai=1:n_ag]; # per agent parallel computations
@time solz_is_nob = [overapproximate(ReachabilityAnalysis.solve(sys_is_nob[ai]; tspan=(0.0, Tf), alg=TMJets21a(; orderQ=6, abstol=1e-15)), Zonotope) for ai=1:n_ag]; # per agent parallel computations

## Per-Agent Feasible Set

Tff = Tf;
# Tff = high(tspan(solz_is[tpf]))
sol_de = solve(ODEProblem(MultiDubins!, xh_init, (0., Tff-1e-3)), p=(x,s)->(zeros(n_ag), zero(1))); Xs = hcat(sol_deq.u...); 
x̃ = s -> vcat(sol_de(Tff + s), 0.0)

gr()
alpha = 0.1;
# solz = solz_is_nob;
feas_plot = plot(title=L"\textrm{Feasible\:\:Sets}")
# for i=1:n_ag; plot!(solz_is[i][end:-1:1], vars=(1,2), alpha=alpha, label=latexstring("𝒮_$i"), color=palette(:seaborn_colorblind)[i]); end
for i=1:n_ag; plot!(solz_is_nob[i][end:-1:1], vars=(1,2), alpha=alpha, label=latexstring("𝒮_$i"), color=palette(:seaborn_colorblind)[i]); end
scatter!([0], [0], label="", xlims=(-0.5, 10.5), ylims=(-6,6), color="black")
for i=0:(n_ag-1); plot!(Xs[1 + i*dim_x,:], Xs[2 + i*dim_x,:], label=latexstring(" ̃x_{$(i+1)}"), lw=2, color=palette(:seaborn_colorblind)[i+1]); end
for i=0:(n_ag-1); scatter!([Xs[1 + i*dim_x,1]], [Xs[2 + i*dim_x,1]], label="", lw=2, color=palette(:seaborn_colorblind)[i+1]); end
θi=0.:0.01:2π; plot!([r * cos.(θi)], [r * sin.(θi)], color="black", lw=2, label=L"𝒯", legend=:topright, xlims = xlimz, ylims = ylimz, aspect_ratio=:equal)
# plot!(xlims=[-6, 12], ylims=[-10, 10], legend=true)
# plot!(legend=:bottom, legend_columns=3, legendfontsize=8)
# plot!(xticks=(-2.5:2.5:10, (L"-2.5", "", "", "", "", L"10")), yticks=(-10:5:5, (L"-10", "", "", L"5")), xtickfontsize=10, ytickfontsize=10)
# lo, _ = collect(zip(xlims(feas_plot), ylims(feas_plot)))
# locxl = lo .+ ((xlims(feas_plot)[2] - xlims(feas_plot)[1])/2, -1.0)
# locyl = lo .+ (-1.0, (ylims(feas_plot)[2] - ylims(feas_plot)[1])/2)
# annotate!(locxl..., L"x_Δ\:(100\: m)", 12)
# annotate!(locyl..., Plots.text(L"y_Δ\:(100\: m)", 12, :black, rotation=90))

## Check Tightness with Random Ensemble 

# Tf = 3.
@time solz_is_ens = [ReachabilityAnalysis.solve(sys_is[ai]; tspan=(0.0, Tf), alg=TMJets21a(; orderQ=4, abstol=1e-12), ensemble=true, trajectories=1000) for ai=1:n_ag]; # per agent parallel computations
@time solz_is_nob_ens = [ReachabilityAnalysis.solve(sys_is_nob[ai]; tspan=(0.0, Tf), alg=TMJets21a(; orderQ=4, abstol=1e-15), ensemble=true, trajectories=1000) for ai=1:n_ag]; # per agent parallel computations

tr_sample = plot(title="MultiDubins System Feasible Sets")
for i=1:n_ag; plot!(tr_sample, solz_is_nob[i][end:-1:1], vars=(1,2), alpha=0.05, label=latexstring("𝒮_$i"), color=palette(:seaborn_colorblind)[i]); end
# for i=1:n_ag; plot!(tr_sample, ensemble(solz_is_ens[i]), vars=(1,2), alpha=0.15, label="", color=palette(:seaborn_colorblind)[i]); end
for i=1:n_ag; plot!(tr_sample, ensemble(solz_is_nob_ens[i]), vars=(1,2), alpha=0.15, label="", color=palette(:seaborn_colorblind)[i]); end
scatter!(tr_sample, [0], [0], label="", xlims= xlimz, ylims= ylimz, color="black")
for i=0:(n_ag-1); plot!(tr_sample, Xs[1 + i*dim_x,:], Xs[2 + i*dim_x,:], label="Agent $(i+1) Nominal", lw=2, color=palette(:seaborn_colorblind)[i+1]); end
θi=0.:0.01:2π; plot!(tr_sample, [r * cos.(θi)], [r * sin.(θi)], color="black", lw=2, label="")

## Check Feasibility

θi = 0.:2π/10:2π; target_disc = hcat(vcat(r*cos.(θi)', r*sin.(θi)'), zeros(2))
dt_is, feas_dt_is = [], []

for ai=1:n_ag
    proj = ReachabilityAnalysis.project(solz_is_nob[ai], vars=1:2)
    push!(dt_is, high.(tspan.(solz_is_nob[ai])))
    push!(feas_dt_is, [!isnothing(findfirst(x -> ReachabilityAnalysis.in(x, proj[j]), eachcol(target_disc))) for j=1:length(proj)]) # check discritized circle in agent feasible sets
end

### Compute Error over time

function TSerror_Inf_SingleDubins(xl, shape)
    dim = dim_x+1; δˢ = zeros(dim-1)

    xi = 0
    max_norm_agent = 0
    for i in [2,3], pm in [1, -1]; max_norm_agent = max(max_norm_agent, abs(pm * ρ(pm * eix(xi + i; dim=dim), shape) - xl[xi + i])); end
    for pm in [1, -1]; max_norm_agent = max(max_norm_agent, abs(pm * ρ(pm * eix(dim; dim=dim), shape) - xl[dim])); end

    θi_hi, θi_lo = ρ(eix(xi + 3; dim=dim), shape), -ρ(-eix(xi + 3; dim=dim), shape)
    if abs(θi_hi - θi_lo) ≥ π/2
        max_sinξθi, max_cosξθi = 1, 1
    elseif sign(cos(θi_hi)) != sign(cos(θi_lo)) || sign(sin(θi_hi)) != sign(sin(θi_lo))
        max_sinξθi = sign(cos(θi_hi)) == sign(cos(θi_lo)) ? max(abs(sin(θi_hi)), abs(sin(θi_lo))) : 1.
        max_cosξθi = sign(sin(θi_hi)) == sign(sin(θi_lo)) ? max(abs(cos(θi_hi)), abs(cos(θi_lo))) : 1.
    else
        max_cosξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
        max_sinξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
    end

    δˢ[xi + 1] = 0.5 * max(abs(v_b * max_cosξθi), 1) * max_norm_agent^2
    δˢ[xi + 2] = 0.5 * max(abs(v_b * max_sinξθi), 1) * max_norm_agent^2
    return δˢ
end

function TSerror_SingleDubins_Tight(xl, shape)
    dim = dim_x+1; δˢ = zeros(dim-1)
    xi = 0;

    θi_hi, θi_lo = ρ(eix(xi + 3; dim=dim), shape), -ρ(-eix(xi + 3; dim=dim), shape)
    if abs(θi_hi - θi_lo) ≥ π/2
        max_sinξθi, max_cosξθi = 1, 1
    elseif sign(cos(θi_hi)) != sign(cos(θi_lo)) || sign(sin(θi_hi)) != sign(sin(θi_lo))
        max_sinξθi = sign(cos(θi_hi)) == sign(cos(θi_lo)) ? max(abs(sin(θi_hi)), abs(sin(θi_lo))) : 1.
        max_cosξθi = sign(sin(θi_hi)) == sign(sin(θi_lo)) ? max(abs(cos(θi_hi)), abs(cos(θi_lo))) : 1.
    else
        max_cosξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
        max_sinξθi = max(abs(cos(θi_hi)), abs(cos(θi_lo)))
    end

    max_d_diff = max(Qdh[1]-xl[dim], xl[dim]+Qdh[1]); max_d_diff2 = max_d_diff^2;
    max_ε1v, max_ε2v = 0., 0.
    for v in vertices_list(shape)
        ε1v = 0.5 * (2*abs(v[xi+2]-xl[xi+2])*max_d_diff2 + v_b*max_cosξθi*(v[xi+3]-xl[xi+3])^2)
        ε2v = 0.5 * (2*abs(v[xi+2]-xl[xi+2])*max_d_diff2 + v_b*max_sinξθi*(v[xi+3]-xl[xi+3])^2)
        max_ε1v, max_ε2v = max(max_ε1v,ε1v), max(max_ε2v,ε2v)
    end

    δˢ[xi + 1] = max_ε1v; δˢ[xi + 2] = max_ε2v
    return δˢ
end

## Solve SingleDubins Errors

# tix = tpf; 
x̃i(i, s) = vcat(x̃(s)[(i-1)*dim_x+1:i*dim_x], 0.0)
@time TSerror_Inf_SingleDubins(x̃i(1, 0.), set(solz_is_nob[1][1]))
@time TSerror_SingleDubins_Tight(x̃i(1, 0.), set(solz_is_nob[1][1]))

δˢ_nob_fasters = [zeros(dim_x, length(solz_is_nob[ai])) for ai=1:n_ag]; δˢ_nob_tighters = [zeros(dim_x, length(solz_is_nob[ai])) for ai=1:n_ag];
@time for ai=1:n_ag
    for ti=1:length(solz_is_nob[ai])
        δˢ_nob_fasters[ai][:, ti] = TSerror_Inf_SingleDubins(x̃i(ai, -Tff+low(tspan(solz_is_nob[ai][ti]))), set(solz_is_nob[ai][ti])); 
    end
end
@time for ai=1:n_ag
    for ti=1:length(solz_is_nob[ai])
        δˢ_nob_tighters[ai][:, ti] = TSerror_SingleDubins_Tight(x̃i(ai, -Tff+low(tspan(solz_is_nob[ai][ti]))), set(solz_is_nob[ai][ti])); 
    end
end

δˢ_nob_mins = copy(δˢ_nob_tighters); for i=1:n_ag; δˢ_nob_mins[i] = min.(δˢ_nob_tighters[i], δˢ_nob_fasters[i]); end

gr()
error_plot = plot(xlabel=L"t")
for ai=1:n_ag
    i=0;
    plot!(high.(tspan.(solz_is_nob[ai])), δˢ_nob_fasters[ai][i+1,:], label=latexstring("δ^*_{$(ai)_1} Fast"), color=palette(:seaborn_colorblind)[ai], lw=2); 
    plot!(high.(tspan.(solz_is_nob[ai])), δˢ_nob_fasters[ai][i+2,:], label=latexstring("δ^*_{$(ai)_2} Fast"), color=palette(:seaborn_colorblind)[ai], lw=2); 
    plot!(high.(tspan.(solz_is_nob[ai])), δˢ_nob_tighters[ai][i+1,:], label=latexstring("δ^*_{$(ai)_1} Tight"), color=palette(:seaborn_colorblind)[ai], lw=2, alpha=0.3); 
    plot!(high.(tspan.(solz_is_nob[ai])), δˢ_nob_tighters[ai][i+2,:], label=latexstring("δ^*_{$(ai)_2} Tight"), color=palette(:seaborn_colorblind)[ai], lw=2, alpha=0.3); 
    plot!(high.(tspan.(solz_is_nob[ai])), δˢ_nob_mins[ai][i+1,:], label=latexstring("δ^*_{$(ai)_2} min"), color=:black, lw=2, alpha=0.6, linestyle=:dash); 
    plot!(high.(tspan.(solz_is_nob[ai])), δˢ_nob_mins[ai][i+2,:], label=latexstring("δ^*_{$(ai)_2} min"), color=:black, lw=2, alpha=0.6, linestyle=:dash); 
end
plot!(legend=false, title=L"\textrm{Forward\:Feasible\:Errors}", xlims=[0, 1.], ylims=[0, 4.])
# plot!(xlims=(0, 2), ylims=(-0.01, 5))

## Error Functions for Hopf

E0(s) = 0 * diagm(ones(dim_xh))

function δˢ_combine(s, δˢ_set, solz_set)
    δˢ = zeros(dim_xh)
    for (ai, i) in enumerate(0:dim_x:dim_x*(n_ag-1))
        δˢ[i+1:i+dim_x] = δˢ_set[ai][:, findfirst(x->x<0, s .- vcat(0., high.(tspan.(solz_set[ai]))))]
    end
    return δˢ
end

# δˢt_nob_faster(s) = δˢ_combine(s, δˢ_nob_fasters, solz_is_nob);
# δˢt_nob_tighter(s) = δˢ_combine(s, δˢ_nob_tighters, solz_is_nob);
δˢt_nob_mins(s) = δˢ_combine(s, δˢ_nob_mins, solz_is_nob);

# plot(.25:.25:1.5, [δˢt_nob_tighter(s)[2] for s=.25:.25:1.5])

# Eδ_nob_fast(s) = diagm(δˢt_nob_faster(s))
# Eδ_nob_tight(s) = diagm(δˢt_nob_tighter(s))
Eδ_nob_mins(s) = diagm(δˢt_nob_mins(s))

## Verification with Hopf

function Ãh(x̃)
    Ai = [0 x̃[4] -v_b*sin(x̃[3]); -x̃[4] 0. v_b*cos(x̃[3]); 0 0 0]; Ah = zeros(dim_xh, dim_xh);
    for i = 1:n_ag; Ah[dim_x * (i - 1) + 1 : dim_x * i, dim_x * (i - 1) + 1 : dim_x * i] = Ai; end
    return Ah
end

function C̃h(x̃)
    Ci = [x̃[2]; -x̃[1]; -1]; Ch = zeros(dim_xh, n_d);
    for i = 1:n_ag; Ch[dim_x * (i - 1) + 1 : dim_x * i, 1] = Ci; end
    return Ch
end

c̃h(x̃,s) = MultiDubins!(zero(x̃), x̃, (x,s)->(zeros(n_ag), [0.]), s) - Ãh(x̃) * x̃;  # linearizing around auto so only drift needed here

system_x̃_auto      = (s -> Ãh(sol_de(s)), s -> C̃h(sol_de(s)), Bh, zero(Qdh), qd, zero(Qch), qc, s -> c̃h(sol_de(s),s), Eδ_nob_mins);
system_x̃_autod     = (s -> Ãh(sol_de(s)), s -> C̃h(sol_de(s)), Bh, zero(Qdh), qd, Qch, qc, s -> c̃h(sol_de(s),s), Eδ_nob_mins);

system_x̃_noerr     = (s -> Ãh(sol_de(s)), s -> C̃h(sol_de(s)), Bh, Qdh, qd, Qch, qc, s -> c̃h(sol_de(s),s), E0);
# system_x̃_err_fast  = (s -> Ãh(sol_de(s)), Bh, s -> C̃h(sol_de(s)), Qch, qc, Qdh, qd, s -> c̃h(sol_de(s),s), Eδ_nob_fast);
# system_x̃_err_tight = (s -> Ãh(sol_de(s)), Bh, s -> C̃h(sol_de(s)), Qch, qc, Qdh, qd, s -> c̃h(sol_de(s),s), Eδ_nob_tight);
system_x̃_err_min   = (s -> Ãh(sol_de(s)), s -> C̃h(sol_de(s)), Bh, Qdh, qd, Qch, qc, s -> c̃h(sol_de(s),s), Eδ_nob_mins);

## Hopf CD Parameters

vh = 0.01
L = 100
tol = 1e-5
step_lim = 2*2000
re_inits = 20
max_runs = 100
max_its = 2*10000
opt_p_cd = (vh, L, tol, step_lim, re_inits, max_runs, max_its)
opt_p_admm_cd = ((1e-0, 1e-0, 1e-5, 3), (0.01, 1e2, 1e-4, 500, 3, 9, 1000), 1, 1, 3)

## Time
th = 0.025; Th = 0.05
T = collect(Th : Th : Tf)

## Solve
# soln, rstats, opt_data = Hopf_BRS(system_x̃_auto,  targets[2], T; Xg=[xh_init;;], th=th, game="avoid", opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, error=true, printing=true, warm=true, opt_tracking=true);
# soln, rstats, opt_data = Hopf_BRS(system_x̃_autod, targets[2], T; Xg=[xh_init;;], th=th, game="avoid", opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, error=true, printing=true, warm=true, opt_tracking=true);
# # soln, rstats = Hopf_BRS(system_x̃_autod, targets[2], T; Xg=[xh_init;;], th=th, game="avoid", opt_method=Hopf_cd, opt_p=opt_p_cd, error=true, printing=true, warm=true);
# # soln, rstats = Hopf_BRS(system_x̃_noerr, targets[2], T; Xg=[xh_init;;], th=th, game="avoid", opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, error=true, printing=true, warm=true);
soln, rstats, opt_data = Hopf_BRS(system_x̃_err_min, targets[2], T; Xg=[xh_init;;], th=th, game="avoid", opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, error=true, printing=true, warm=false, opt_tracking=true);

## Solve Optimal Control
# soln, rstats = Hopf_BRS(system_x̃_err_min, targets[2], T; Xg=[xh_init;;], th=th, game="avoid", opt_method=Hopf_admm_cd, opt_p=opt_p_admm_cd, error=true, printing=true, warm=true);

# NEED TO SAVE 
# run_name = "Run1_10b10b10"
# name = "Zonotoping/MultiDubins_Data/$run_name/"
# values =
# min_value = 
# feasible = 
# min_random_value? # well thats incomprable unless we also solve nod
