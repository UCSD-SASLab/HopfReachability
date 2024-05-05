
using LinearAlgebra, Plots
plotlyjs()
include(pwd() * "/src/HopfReachability.jl");
using .HopfReachability: Hopf_minT, Hopf_BRS, Hopf_admm_cd, Hopf_admm, Hopf_cd, plot_BRS

## System

dim_x = 4 # agent dimension
n_ag = 2
dim_xh = n_ag * dim_x # system dimension
tail_engag = true
r, v_max = 1, 2 # ??

# Agent System: ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

A = vcat(hcat(zeros(2,2), I(2)), zeros(2, dim_x))
B = tail_engag ? vcat(zeros(dim_x - 1), 1) : vcat(zeros(dim_x - 1), -1)
C = vcat(zeros(dim_x - 1), -1)

Qc = [1] # agent control lim
Qd = [0.4] # evader control lim

W = diagm([r^2, r^2, v_max^2, v_max^2]) # agent-target set param (ellipse mat)

## Combined System 

Ah = zeros(dim_xh, dim_xh)
Bh = zeros(dim_xh, n_ag)
Ch = zeros(dim_xh, 1)

Qch = Qc .* I(n_ag) # 1d ctrl per agent
Qdh = Qd .* I(1) # 1d ctrl per agent
qc, qd = [0. 0.], [0.] #center of input constraints

Whs = [zeros(dim_xh, dim_xh) for _ in 1:n_ag]

for i = 1:n_ag
    Ah[dim_x * (i - 1) + 1 : dim_x * i, dim_x * (i - 1) + 1 : dim_x * i] = A
    Bh[dim_x * (i - 1) + 1 : dim_x * i, i] = B
    Ch[dim_x * (i - 1) + 1 : dim_x * i, 1] = C
    for j=1:n_ag
        Whs[i][dim_x * (j - 1) + 1 : dim_x * j, dim_x * (j - 1) + 1 : dim_x * j] = i == j ? W : v_max^2 * I(dim_x)
    end
end

system = (Ah, Bh, Ch, Qch, Qdh, qc, qd)

## Time
th = 0.5
Th = 2.0
Tf = 40.0
T = collect(Th : Th : Tf)
time_p = (th, Th, Tf + 0.01)

## Elliptical Target: J(x) = 0 is the boundary of the target
qp = zeros(dim_xh) # target center
J(x::Vector, Q, q) = ((x - q)' * inv(Q) * (x - q))/2 - 0.5 #don't need yet
Js(v::Vector, Q, q) = (v' * Q * v)/2 + q'v + 0.5
J(x::Matrix, Q, q) = diag((x .- q)' * inv(Q) * (x .- q))/2 .- 0.5
Js(v::Matrix, Q, q) = diag(v' * Q * v)/2 + (q'v)' .+ 0.5 #don't need yet
targets = [(J, Js, (Whs[i], qp)) for i=1:n_ag]

## Grid Definition (too sparse to be interpretable, 8D has too many pts)
# N = 5
# xg = collect(-8 : (2 - (-8)) / (N-1) : 2)
# yg = collect(-5 : (5 - (-5)) / (N-1) : 5)
# Vyg = collect(-2 : (2 - (-2)) / (N-1) : 2)
# # Xg_noVx = hcat(collect.(Iterators.product([xg, yg, Vyg] for _ =1:n_ag]...))...)
# Xg_noVx = hcat(collect.(Iterators.product(xg, yg, Vyg, xg, yg, Vyg))...)
# Vxg = 2*ones(size(Xg_noVx)[2]) # fixed horizontal vel
# Xg = vcat(Xg_noVx[1:2, :], Vxg', Xg_noVx[3:5, :], Vxg', Xg_noVx[6, :]') 

## Hopf ADMM Parameters (default)
ρ, ρ2 = 1e-4, 1e-4
tol = 1e-5
max_its = 10
opt_p_admm = (ρ, ρ2, tol, max_its)

## Hopf CD Parameters
vh = 0.01
L = 2
tol = 1e-5
step_lim = 2000
re_inits = 10
max_runs = 100
max_its = 10000
opt_p_cd = (vh, L, tol, step_lim, re_inits, max_runs, max_its)

# ### Solve the BRS to Validate our Controller
# solution, run_stats = Hopf_BRS(system, target, T;
#                                 opt_method=Hopf_cd,
#                                 th,
#                                 grid_p,
#                                 opt_p=opt_p_cd,
#                                 warm=false,
#                                 check_all=true,
#                                 printing=true);
# B⁺T, ϕB⁺T = solution;

# plot_scatter = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵs=1e-2, interpolate=false, alpha=0.1)
# plot_contour = plot_BRS(T, B⁺T, ϕB⁺T; M, ϵc=1e-3, interpolate=true, value_fn=true, alpha=0.5)

## Find Boundary Pts of one BRS for Time of Interest
# Toi = 0.2
# Tix = findfirst(abs.(T .- Toi) .< Th/2);
# B = B⁺T[Tix + 1][:, abs.(ϕB⁺T[Tix + 1]) .< 1e-2] # Tix + 1 because target set is first

## Test a Single Call
# agents at (x_init, ±y_init_offset) both flying horizontal
x_init = -20
y_init_offset = 0.5
x_vel = 0.5 # constant (no ctrl)
y_vel_init = 0.
x = [x_init, y_init_offset, x_vel, y_vel_init, x_init + 1e-1, -y_init_offset, x_vel, y_vel_init]

uˢ, dˢ, Tˢ, ϕ, dϕdz = Hopf_minT(system, targets[1], x; opt_method=Hopf_cd, opt_p=opt_p_cd, input_shape="box", time_p, printing=true, warm=true)
uˢ, dˢ, Tˢ, ϕ, dϕdz2 = Hopf_minT(system, targets[2], x; opt_method=Hopf_cd, opt_p=opt_p_cd, input_shape="box", time_p, printing=true, warm=true)

## Evolution with Optimal Inputs

function Flinear(xh, uh, dh, Δt)
    xhd = Ah * xh + Bh * uh + Ch * dh
    return xh + Δt * xhd
end

Δt, tf = 0.5, 40.0
X = Inf*ones(dim_xh, Int(tf / Δt) + 1)
X[:, 1] = x
catcher = Inf*ones(Int(tf / Δt))

warm = true
# v_init_ = copy(dϕdz)
v_init_1, v_init_2 = copy(dϕdz), copy(dϕdz2)

# Fast parameters don't go well yet
# vh = 0.01 # L = 2 # tol = 1e-5 # step_lim = 2000 # re_inits = 10 # max_runs = 100 # max_its = 10000
# opt_p_cd_quick = (vh, L, tol, Int(step_lim/4), Int(floor(re_inits/3)), max_runs, Int(max_its/4))

for (ti, th) in enumerate(0.:Δt:tf-Δt)

    ## Check if Reached
    ag_dists = norm.(X[ai:ai+1, ti] for ai = 1:dim_x:dim_x*n_ag)
    if minimum(ag_dists) <= r 
        reacher = argmin(ag_dists)
        println("*** Agent $reacher reached target at t = $th, (ti = $ti) ***")
        break
    end

    println("Solving at t=$th")

    ## Compute Optimal Intputs
    # uˢ1, dˢ1, Tˢ1, ϕ1, dϕdz1 = Hopf_minT(system, targets[1], X[:, ti]; opt_method=Hopf_cd, opt_p=opt_p_cd_quick, input_shape="box", time_p, warm, v_init_=v_init_1)
    # uˢ2, dˢ2, Tˢ2, ϕ2, dϕdz2 = Hopf_minT(system, targets[2], X[:, ti]; opt_method=Hopf_cd, opt_p=opt_p_cd_quick, input_shape="box", time_p, warm, v_init_=v_init_2)

    uˢ1, dˢ1, Tˢ1, ϕ1, dϕdz1 = Hopf_minT(system, targets[1], X[:, ti]; opt_method=Hopf_cd, opt_p=opt_p_cd, input_shape="box", time_p, warm, v_init_=v_init_1)
    uˢ2, dˢ2, Tˢ2, ϕ2, dϕdz2 = Hopf_minT(system, targets[2], X[:, ti]; opt_method=Hopf_cd, opt_p=opt_p_cd, input_shape="box", time_p, warm, v_init_=v_init_2)

    uˢ, dˢ, Tˢ, ϕ, dϕdz, catcher[ti] = Tˢ1 < Tˢ2 ? (uˢ1, dˢ1, Tˢ1, ϕ1, dϕdz1, 1) : (uˢ2, dˢ2, Tˢ2, ϕ2, dϕdz2, 2)

    # time_p = ... #could be changed 
    v_init_1, v_init_2 = warm ? (dϕdz1, dϕdz2) : (zero(dϕdz), zero(dϕdz))

    ## Evolve
    X[:, ti+1] = Flinear(X[:, ti], uˢ, dˢ, Δt)
    
end

plot(title="Relative Distance", xlabel="Relative Horizontal (m)", ylabel="Relative Vertical (m)"); 
plot!(X[1,:], X[2,:], label="agent 1", color=:blue); 
plot!(X[5,:], X[6,:], label="agent 2", color=:orange)

plot(title="Relative Vertical Velocity", xlabel="Time (s)", ylabel="Relative Vertical (m/s)"); 
plot!(collect(0.:Δt:tf), X[4,:], label="agent 1", color=:blue, alpha=0.7); 
plot!(collect(0.:Δt:tf), X[8,:], label="agent 2", color=:orange, alpha=0.7)


# using JLD
# save("MultiAgent_X.jld", "data", X, "params", opt_p_cd)



    
