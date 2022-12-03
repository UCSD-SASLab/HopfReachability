
using LinearAlgebra, Plots, JLD, TickTock
push!(LOAD_PATH,"/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachability: Hopf_BRS, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, Hopf

## Initialize
M = [0. 1; -2 -3]
B = 0.5 * I
C = 0.5 * [2 0; 0 1]
Q = 0.1 * 3 * [1 0; 0 1]
Q2 = 0.2 * 2 * [1 0; 0 1]
a1 = 0*[0.5 0.75]
a2 = -0*[0.5 0]
system = (M, B, C, Q, Q2, a1, a2)

## Time
th = 0.05
Th = 0.1
Tf = 0.7
T = collect(Th : Th : Tf)

## Target: J(x) = 0 is the boundary of the target
Ap = diagm([1, 1])
cp = [0.; 0.]
J(x::Vector, A, c) = ((x - c)' * A * (x - c))/2 - 0.5 #don't need yet
Js(v::Vector, A, c) = (v' * inv(A) * v)/2 + c'v + 0.5
J(x::Matrix, A, c) = diag((x .- c)' * A * (x .- c))/2 .- 0.5
Js(v::Matrix, A, c) = diag(v' * inv(A) * v)/2 + (c'v)' .+ 0.5 #don't need yet
target = (J, Js, (Ap, cp))

## Grid Parameters
bd = (-3, 3)
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
max_runs = 20
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

plot = plot_BRS(T, B⁺T, ϕB⁺T; M, cres=0.1, contour=true)



## BRT (can only be solved with coordinate descent, and errors are observable)

function intH_ytc17_BRT(system, Hdata, v; p2, kkk=size(system[1])[1])

    M, C, C2, Q, Q2, a1, a2 = system
    Hmats, tix, th = Hdata
    Rt, R2t, G, G2, F = Hmats

    ## Quadrature mats
    Rvt, R2vt = reshape(view(Rt,1:kkk*tix,:) * v, kkk, tix), reshape(view(R2t,1:kkk*tix,:) * v, kkk, tix)
    
    ## Quadrature sum
    H = map(norm, eachcol(G * Rvt)) + reshape(a1 * Rvt, tix) - map(norm, eachcol(G2 * R2vt)) + reshape(a2 * R2vt, tix)
    intH = sum(H[H .> 0]) # important BRT step

    return intH
end

solution, run_stats = Hopf_BRS(system, target, intH_ytc17_BRT, T; 
                                                    preH=preH_ytc17,
                                                    th,
                                                    grid_p,
                                                    opt_p,
                                                    check_all=true,
                                                    warm=true, #slow if false
                                                    printing=true);
B⁺T, ϕB⁺T = solution;

plot = plot_BRS(T, B⁺T, ϕB⁺T; M, cres=0.01, contour=false, title="BRT of T, in X");