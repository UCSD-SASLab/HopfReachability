
include(pwd() * "/Examples/quadrotor/quadrotor_utils.jl");
using LinearAlgebra, Plots, DifferentialEquations

## Initialize

nx = 12
x0 = zeros(nx)
test_params = 1.0, 9.81, 0.1, 0.1, 0.2
cf_params = 0.034, 9.81, 16.571710e-6, 16.655602e-6, 29.261652e-6 # kg, m s^-2, kg m^2, kg m^2, kg m^2
params = cf_params

ut, ur = t -> 9.81*params[1] + 5e-3, t -> 1e-7 * sin(t)
zero_f = t -> 0.
sine_thrust = function (t); t < 2π ? 9.81*params[1] + 0.01 * sin(t) : 9.81*params[1] - 0.01 * sin(t); end

ctrls_thrust = (x,t) -> [sine_thrust(t), zero_f(t), zero_f(t), zero_f(t)]
ctrl_pitch = (x,t) -> [ut(t), zero_f(t), ur(t), zero_f(t)]
ctrl_yaw = (x,t) -> [ut(t), zero_f(t), zero_f(t), 10*ur(t)]
ctrl_roll = (x,t) -> [ut(t), ur(t), zero_f(t), zero_f(t)]

# Thrust Test

tspan = (0, 4π)
p = [params, ctrls_thrust]

prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_thrust = DifferentialEquations.solve(prob)

flight_plot(sol_thrust, ctrls_thrust; plot_title=L"\textrm{Thrust}")
# flight_gif(sol_thrust, ctrls_thrust; plot_title=L"\textrm{Thrust}", fname="test.gif")

## Positive Roll

tspan = (0, 2π)
p = [params, ctrl_roll]

prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_roll = DifferentialEquations.solve(prob)

flight_plot(sol_roll, ctrl_roll; plot_title=L"\textrm{Roll}", backend=plotly)

## Positive Pitch

p = [params, ctrl_pitch]
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_pitch = DifferentialEquations.solve(prob)

flight_plot(sol_pitch, ctrl_pitch; plot_title=L"\textrm{Pitch}")

## Positive Yaw

p = [params, ctrl_yaw]
prob = ODEProblem(quadrotor_12D!, x0, tspan, p)
sol_yaw = DifferentialEquations.solve(prob)

flight_plot(sol_yaw, ctrl_yaw; plot_title=L"\textrm{Yaw}")
