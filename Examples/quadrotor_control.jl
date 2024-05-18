
include(pwd() * "/src/lin_utils.jl");
include(pwd() * "/src/control_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations

### i/c, inputs, model params

### make/load controller(s) (LQR, MPC, load LQR matrix from cfclean)

### simulate 
## - go to waypoint
## - track figure 8 
## - invert and stabilize (reach upright at initial point/ any point?)
## Score with quantitative metrics (oc time, problem solved)

### NEW SCRIPT TIME ####

### try solcing oc w/ hopf (Hopf_minT)
## timing & parameter refinement pt 1

# using .HopfReachability: Hopf_BRS, Hopf_cd, make_levelset_fs
# include(pwd() * "/src/HopfReachability.jl");

### simulate hopf controller
## timing & parameter optimization pt 2
## score

### MAYBE compute lin error w/ RA.jl (historgram?) for LQR-fig8-sim segment

# include(pwd() * "/src/cons_lin_utils.jl");

### simulate hopf controller w error (levels) + MPCg
## score

### load koopman models, trajectory data, compute lifted error on trajectory set

### MAYBE compute max error on lifted feasible set 

### simulate koopman-hopf controller w error (levels) + MPCg
## timing & parameter optimization pt 3
## score 

## now go call from ros/rospy and see timing

## more complicated experiments (P-E, Dodge incoming projectile)? 
## active learning of KO in sim (ProximalAlgorithms.jl or Flux.jl)? this probably calls for comparison w/ Ian's learning
# https://github.com/FluxML/model-zoo/blob/da4156b4a9fb0d5907dcb6e21d0e78c72b6122e0/other/diffeq/neural_ode.jl