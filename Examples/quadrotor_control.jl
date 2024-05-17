
include(pwd() * "/src/lin_utils.jl");
include(pwd() * "/src/control_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations

### i/c, inputs, model params

### model for simulation (12D?)

### model for controller (7D? 12D?)

### linearize model for controller

### make controller(s) (LQR, MPC, load LQR matrix from cfclean)

### simulate 
## - go to waypoint
## - track figure 8 
## - invert and stabilize (reach upright at initial point/ any point?)
## - dodge obstacle(s) (TV cost)
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

## active learning of KO in sim? more complicated experiments (P-E, ) ?

