
using LinearAlgebra, Plots, JLD, TickTock
push!(LOAD_PATH,"/Users/willsharpless/Library/Mobile Documents/com~apple~CloudDocs/Herbert/Koop_HJR/HL_fastHJR");
using HopfReachabilityv2: Hopf_BRS, Hopf_admm, Hopf_cd, intH_ytc17, preH_ytc17, plot_BRS, Hopf

## Initialize

dim_ag = 4
n_ag = 2
dim = n_ag * dim_ag
tail_engag = true
r, V_max = 0.5, 100 # ??

# Agent System: ẋ = Ax + Bu + Cd subject to y ∈ {(y-a)'Q(y-a) ≤ 1} for y=u,d

A = vcat(hcat(zeros(2,2), I(2)), zeros(2, dim_ag))
B = tail_engag ? vcat(zeros(dim_ag - 1), 1) : vcat(zeros(dim_ag - 1), -1)
C = vcat(zeros(dim_ag - 1), -1)

Qc = 
Qd = 

W = diagm(r^2, r^2, V_max^2, V_max^2)

## Combined System 

Ah = zeros(n_ag * dim_ag, n_ag * dim_ag)
Bh = zeros(n_ag * dim_ag, n_ag)
Ch = zeros(n_ag * dim_ag)

for i = 1:n_ag
    Ah[dim_ag * (i - 1) + 1 : dim_ag * i, dim_ag * (i - 1) + 1 : dim_ag * i] = A
    Bh[dim_ag * (i - 1) + 1 : dim_ag * i, i] = B
    Ch[dim_ag * (i - 1) + 1 : dim_ag * i] = C
end





