#!/usr/bin/env python
# coding: utf-8

# In[2]:


import jax
import jax.numpy as jnp
import numpy as np

from IPython.display import HTML
import matplotlib.animation as anim
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import hj_reachability as hj

import hj_reachability.dynamics as dynamics
from hj_reachability import time_integration

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(formatter={'float': '{: >9.10f}'.format})
jnp.set_printoptions(formatter={'float': '{: >9.10f}'.format})

# %%

class two_way_crash_x(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 flipped=False,
                 u_bd=1.,
                 d_bd=1.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.flipped = flipped
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-u_bd]), jnp.array([u_bd]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-d_bd]), jnp.array([d_bd]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        # _, _, psi = state
        # v_a, v_b = self.evader_speed, self.pursuer_speed
        # return jnp.array([-v_a + v_b * jnp.cos(psi), v_b * jnp.sin(psi)])
        x, y = state
        if not self.flipped:
            return (x > 0.) * jnp.array([-1, 0]) + \
                   (x < 0.) * jnp.array([1, 0])
        else:
            return (x > 0.) * jnp.array([1, 0]) + \
                   (x < 0.) * jnp.array([-1, 0])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])
    
# %%

class linear_rotation(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 flipped=False,
                 u_bd=1.,
                 d_bd=1.,
                #  v_xx = -0.2, v_xy = 0.1, v_yx = -0.1, v_yy = -0.2,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        # self.v_xx = v_xx,
        # self.v_xy = v_xy,
        # self.v_yx = v_yx,
        # self.v_yy = v_yy,
        self.flipped = flipped
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-u_bd]), jnp.array([u_bd]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-d_bd]), jnp.array([d_bd]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y = state
        if self.flipped:
            return jnp.array([-(-0.2 * x + 0.1 * y), -(-0.1 * x - 0.2 * y)])
        else:
            return jnp.array([-0.2 * x + 0.1 * y, -0.1 * x - 0.2 * y])
            ## tuple/batch tracer err when using stored params?

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])
    
# %%

## PRINT EVERYTHING

b_lo, b_hi = 0, 5; btitle = "LOWERL"
# c_lo, c_hi = 498, 503; ctitle = "CENTER" # center
# c_lo, c_hi = 498 - 250, 503 - 250; ctitle = "LWLCTR" # lower left quarter-center
c_lo, c_hi = 996, 1001; ctitle = "UPPERR" # upper right

def state_print():

    print(f"\nSTATES - {btitle} - X \n")
    print(grid.states[:b_hi, b_hi-1::-1, 0].T)
    print(f"\nSTATES - {ctitle} - X \n")
    print(grid.states[c_lo:c_hi, c_hi-1:c_lo-1:-1, 0].T)
    print(f"\nSTATES - {btitle} - Y \n")
    print(grid.states[:b_hi, b_hi-1::-1, 1].T)
    print(f"\nSTATES - {ctitle} - Y \n")
    print(grid.states[c_lo:c_hi, c_hi-1:c_lo-1:-1, 1].T)

    return

def val_print(iv, values_sbs):

    print(f"\nINIT VALUES - {btitle}\n")
    print(iv[:b_hi, b_hi-1::-1].T)
    print(f"\nINIT VALUES - {ctitle}\n")
    print(iv[c_lo:c_hi, c_hi-1:c_lo-1:-1].T)

    print(f"\nSTEP VALUES - {btitle}\n")
    print(values_sbs[:b_hi, b_hi-1::-1].T)
    print(f"\nSTEP VALUES - {ctitle}\n")
    print(values_sbs[c_lo:c_hi, c_hi-1:c_lo-1:-1].T)

    return

def grad_print():

    print(f"\nLEFT GRADS - {btitle} - X \n")
    print(left_grad_values[:b_hi, b_hi-1::-1, 0].T)
    print(f"\nLEFT GRADS - {ctitle} - X \n")
    print(left_grad_values[c_lo:c_hi, c_hi-1:c_lo-1:-1, 0].T)
    print(f"\nRIGHT GRADS - {btitle} - X \n")
    print(right_grad_values[:b_hi, b_hi-1::-1, 0].T)
    print(f"\nRIGHT GRADS - {ctitle} - X \n")
    print(right_grad_values[c_lo:c_hi, c_hi-1:c_lo-1:-1, 0].T)

    # print(f"\nLEFT GRADS - {btitle} - Y \n")
    # print(left_grad_values[:b_hi, b_hi-1::-1, 1].T)
    # print(f"\nRIGHT GRADS - {btitle} - Y \n")
    # print(right_grad_values[:b_hi, b_hi-1::-1, 1].T)

    # print(f"\nLEFT GRADS - {ctitle} - Y \n")
    # print(left_grad_values[c_lo:c_hi, c_hi-1:c_lo-1:-1, 1].T)
    # print(f"\nRIGHT GRADS - {ctitle} - Y \n")
    # print(right_grad_values[c_lo:c_hi, c_hi-1:c_lo-1:-1, 1].T)

    lravg_grad_values = 0.5 * (right_grad_values + left_grad_values)
    
    print(f"\nLR AVG GRAD - {btitle} - X\n")
    print(lravg_grad_values[:b_hi, b_hi-1::-1, 0].T)
    print(f"\nLR AVG GRAD - {ctitle} - X\n")
    print(lravg_grad_values[c_lo:c_hi, c_hi-1:c_lo-1:-1, 0].T)

    return

def diss_print():

    print(f"\nDISS COEF - {btitle} - X \n")
    print(dissipation_coefficients[:b_hi, b_hi-1::-1, 0].T)

    print(f"\nDISS COEF - {ctitle} - X \n")
    print(dissipation_coefficients[c_lo:c_hi, c_hi-1:c_lo-1:-1, 0].T)

    print(f"\nDISS COEF - {btitle} - Y \n")
    print(dissipation_coefficients[:b_hi, b_hi-1::-1, 1].T)

    print(f"\nDISS COEF - {ctitle} - Y \n")
    print(dissipation_coefficients[c_lo:c_hi, c_hi-1:c_lo-1:-1, 1].T)

    rml_grad_values = (right_grad_values - left_grad_values) / 2
    dissipation_value = (dissipation_coefficients * rml_grad_values).sum(2)

    print(f"\nDISS VAL - {btitle} \n")
    print(dissipation_value[:b_hi, b_hi-1::-1].T)

    print(f"\nDISS VAL - {ctitle} \n")
    print(dissipation_value[c_lo:c_hi, c_hi-1:c_lo-1:-1].T)

    return

def diss_ham():

    print(f"\nDISS HAM - {btitle} \n")
    print(dvalues_dt[:b_hi, b_hi-1::-1].T)
    print(f"\nDISS HAM - {ctitle} \n")
    print(dvalues_dt[c_lo:c_hi, c_hi-1:c_lo-1:-1].T)

    return

def ham():
    ## has to be back-computed due to nature of multi map fn

    rml_grad_values = (right_grad_values - left_grad_values) / 2
    dissipation_value = (dissipation_coefficients * rml_grad_values).sum(2)
    ham = dvalues_dt + dissipation_value

    print(f"\nHAM - {btitle}\n")
    print(ham[:b_hi, b_hi-1::-1].T)
    print(f"\nHAM - {ctitle}\n")
    print(ham[c_lo:c_hi, c_hi-1:c_lo-1:-1].T)

    return
    
# In[27]:

c = np.array([0., 0.])

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-2, -2]),
                                                                           np.array([2, 2])),
                                                               (1001, 1001))
                                                            #    (1000, 1000))


backwards_reachable_set = lambda x : x
solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=backwards_reachable_set)
# solver_settings = hj.SolverSettings.with_accuracy("low", hamiltonian_postprocessor=backwards_reachable_set)
# solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

# init_values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 5
r = 1.0
# init_values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(np.subtract(grid.states[..., :2], c))), axis=-1)) - r ** 2) * 0.5 # UNIT CIRCLE

r = 0.5
init_values = (jnp.array(np.max(np.abs(np.multiply([1., 1.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r) # UNIT SQUARE
# init_values = (jnp.array(np.max(np.abs(np.multiply([1., 0.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r) # UNIT CYCLINDER
# init_values = jnp.minimum((jnp.array(np.max(np.abs(np.multiply([1., 0.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r), 
#                           (jnp.array(np.max(np.abs(np.multiply([0.5, 0.5], np.subtract(grid.states[..., :2], c))), axis=-1)) - r)) # UNIT CYLINDER & TWO SQUARE

print("Max BC value:", init_values.max())
print("Min BC value:", init_values.min())

# In[24]:

tf = 1.0

# dynamics_ud = two_way_crash_x(flipped=True)
dynamics_ud = linear_rotation(flipped=True)

ntimes = 5
times = np.linspace(0., tf, ntimes)
# time_step = -tf/(ntimes-1)
time_step = -0.3

# THIS TIME STEP NEEDS TO BE FLIPPED FOR SOME REASON
time_values_tsteps = hj.step(solver_settings, dynamics_ud, grid, 0., init_values, time_step)
val_print(init_values, time_values_tsteps)

# %%

## STEP 1 - CHECK

iv = init_values.copy()

left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, iv)

time = 0.
dissipation_coefficients = solver_settings.artificial_dissipation_scheme(dynamics_ud.partial_max_magnitudes,
                                                                             grid.states, time, iv,
                                                                             left_grad_values, right_grad_values)

max_time_step=time_step - time
time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics_ud.hamiltonian(*args, **kwargs)

def lax_friedrichs_numerical_hamiltonian(hamiltonian, state, time, value, left_grad_value, right_grad_value,
                                         dissipation_coefficients):
    hamiltonian_value = hamiltonian(state, time, value, (left_grad_value + right_grad_value) / 2)
    dissipation_value = dissipation_coefficients @ (right_grad_value - left_grad_value) / 2
    return hamiltonian_value - dissipation_value

from hj_reachability import utils

dvalues_dt = -solver_settings.hamiltonian_postprocessor(time_direction * utils.multivmap(
        lambda state, value, left_grad_value, right_grad_value, dissipation_coefficients:
        (lax_friedrichs_numerical_hamiltonian(signed_hamiltonian, state, time, value,
                                              left_grad_value, right_grad_value, dissipation_coefficients)),
        np.arange(grid.ndim))(grid.states, iv, left_grad_values, right_grad_values, dissipation_coefficients))

target_time = time_step
def sub_step(time_values, target_time_ss):
    tss, vss = solver_settings.time_integrator(solver_settings, dynamics_ud, grid, *time_values, target_time_ss)
    return tss, vss

t, v = sub_step((time, iv), target_time)

values_sbs = iv + t * dvalues_dt

# interestingly, jax does math a little different than python!
print("Step-by-step update correct?", np.allclose(values_sbs, v))
val_print(iv, values_sbs)

# %%

time_values = (time, init_values)
num_steps = 0

while jnp.abs(target_time - time_values[0]) > 0:
    # print(f"Time {time_values[0]:>3.7f}")
    print(f"Time {time_values[0]:>3.7f} - LOWERL Value {time_values[1][0,0]}")
    time_values = sub_step(time_values, target_time)
    num_steps +=1

print(f"Time {time_values[0]:>3.7f}")
print(f"total time steps = {num_steps:2d}")


# %%

state_print()

val_print(iv, values_sbs)

grad_print()

ham()

diss_print()

diss_ham()

val_print(iv, values_sbs)

# %% STEP 2

iv = v.copy()

left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, iv)

time = 0.
dissipation_coefficients = solver_settings.artificial_dissipation_scheme(dynamics_ud.partial_max_magnitudes,
                                                                             grid.states, time, iv,
                                                                             left_grad_values, right_grad_values)

max_time_step=time_step - time
time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics_ud.hamiltonian(*args, **kwargs)

def lax_friedrichs_numerical_hamiltonian(hamiltonian, state, time, value, left_grad_value, right_grad_value,
                                         dissipation_coefficients):
    hamiltonian_value = hamiltonian(state, time, value, (left_grad_value + right_grad_value) / 2)
    dissipation_value = dissipation_coefficients @ (right_grad_value - left_grad_value) / 2
    return hamiltonian_value - dissipation_value

from hj_reachability import utils

dvalues_dt = -solver_settings.hamiltonian_postprocessor(time_direction * utils.multivmap(
        lambda state, value, left_grad_value, right_grad_value, dissipation_coefficients:
        (lax_friedrichs_numerical_hamiltonian(signed_hamiltonian, state, time, value,
                                              left_grad_value, right_grad_value, dissipation_coefficients)),
        np.arange(grid.ndim))(grid.states, iv, left_grad_values, right_grad_values, dissipation_coefficients))

target_time = time_step
def sub_step(time_values, target_time_ss):
    tss, vss = solver_settings.time_integrator(solver_settings, dynamics_ud, grid, *time_values, target_time_ss)
    return tss, vss

values_sbs2 = iv + t * dvalues_dt
t2, v2 = sub_step((time, iv), target_time)

print("Step-by-step update correct?", np.allclose(values_sbs2, v2))

val_print(iv, values_sbs2)

# %%

# val_print(iv, values_sbs2)

grad_print()

diss_print()

diss_ham()

ham()

# In[24]: ONE SHOT

max_u, max_d = 1., 0.5
tf = 1.0

# dynamics_ud = two_way_crash_x(flipped=False)
# dynamics_ud = two_way_crash_x(flipped=True)
dynamics_ud = linear_rotation(flipped=True)

ntimes = 5
fig, axes = plt.subplots(nrows=1, ncols=ntimes, figsize=(25, 5))

plt.jet()
tf = 5.0
times = np.linspace(0., tf, ntimes)

for ax, time in zip(axes, times):
    if time == 0.:
        values = init_values
    else:
        values = hj.step(solver_settings, dynamics_ud, grid, 0., values, -tf/(ntimes-1))
    ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], values[:, :].T)
    ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                values[:, :].T,
                levels=0,
                colors="black",
                linewidths=3)
    ax.set_title(f"t = {time:2.2f}")

plt.tight_layout()
plt.show()

# %%

# %%

# %%

# %% EXTRAPOLATE AWAY FROM ZERO tests

from hj_reachability import boundary_conditions

pad_width = 1.
# test_array = jnp.array([2.,1.,3.]) ## no difference
# test_array = jnp.array([2.,3.,4.]) ## one sided
test_array = jnp.array([2.,3.,2.]) ## two sided
# test_array = jnp.array([-2.,-3.,-2.]) ## two sided
# test_array = jnp.array([2.,3.,0.]) ## both sided but undesired (should be "away from zero")

print ("og arr", test_array, "\n")
extrapolated = boundary_conditions.extrapolate(test_array, pad_width)
extrapolaway = boundary_conditions.extrapolate_away_from_zero(test_array, pad_width)

print("Extrapolate", extrapolated)

print("ExtrapAWAY0", extrapolaway)

print("\nExtrapolated - diff", jnp.diff(extrapolated))
print("ExtrapoAWAY0 - diff", jnp.diff(extrapolaway))


# %% EXTRAPOLATE AWAY FROM ZERO tests

# test_array = jnp.array([2.,3.,2.]) ## two sided
test_array = jnp.array([-2.,-3.,-2.]) ## two sided

x = test_array

## EXTRAPOLATE
jnp.concatenate([
        x[0]  + (x[1] - x[0])   * jnp.arange(-pad_width, 0), 
        x, 
        x[-1] + (x[-1] - x[-2]) * jnp.arange(1, pad_width + 1)
    ])

## EXTRAPOLATE AWAY
jnp.concatenate([
        x[0]  - jnp.sign(x[0])  * jnp.abs(x[1]  - x[0])  * jnp.arange(-pad_width, 0), 
        x,
        x[-1] + jnp.sign(x[-1]) * jnp.abs(x[-1] - x[-2]) * jnp.arange(1, pad_width + 1)
    ])



# %%

# %%

# %%

# %%
# %%
# %%

# %%

class four_way_crash(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 evader_speed=5.,
                 pursuer_speed=5.,
                 evader_max_turn_rate=1.,
                 pursuer_max_turn_rate=1.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.evader_speed = evader_speed
        self.pursuer_speed = pursuer_speed
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-evader_max_turn_rate]), jnp.array([evader_max_turn_rate]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-pursuer_max_turn_rate]), jnp.array([pursuer_max_turn_rate]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        # _, _, psi = state
        # v_a, v_b = self.evader_speed, self.pursuer_speed
        # return jnp.array([-v_a + v_b * jnp.cos(psi), v_b * jnp.sin(psi)])
        x, y = state

        # if y > -x:
        #     if y > x:
        #         return jnp.array([0, 1])
        #     else:
        #         return jnp.array([1, 0])
        # else:
        #     if y > x:
        #         return jnp.array([-1, 0])
        #     else:
        #         return jnp.array([0, -1])
        
        return (y >= abs(x))  * jnp.array([0, -1]) + \
               (y < -abs(x)) * jnp.array([0, 1]) + \
               (x >= abs(y))  * jnp.array([-1, 0]) + \
               (x < -abs(y)) * jnp.array([1, 0]) + \
               (y == x) * jnp.array([-jnp.sign(y), -jnp.sign(y)]) + \
               (y == -x) * (x > 0) * jnp.array([-1, 1]) + \
               (y == -x) * (x < 0)  * jnp.array([1, -1])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])
    
# %%

c = np.array([0., 0.])
r = 0.5

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-2, -2]),
                                                                           np.array([2, 2])),
                                                               (1001, 1001))

backwards_reachable_set = lambda x : x
solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=backwards_reachable_set)
# solver_settings = hj.SolverSettings.with_accuracy("low", hamiltonian_postprocessor=backwards_reachable_set)
# solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

# init_values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 5
# init_values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(np.subtract(grid.states[..., :2], c))), axis=-1)) - r ** 2) * 0.5
init_values = (jnp.array(np.max(np.abs(np.multiply([1., 1.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r)
# init_values = (jnp.array(np.max(np.abs(np.multiply([1., 0.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r)

print("Max BC value:", init_values.max())
print("Min BC value:", init_values.min())

# In[24]:

tf = 1.0

dynamics_ud = four_way_crash()

ntimes = 5
times = np.linspace(0., tf, ntimes)
# time_step = -tf/(ntimes-1)
time_step = -0.3

time_values_tsteps = hj.step(solver_settings, dynamics_ud, grid, 0., init_values, time_step)

# %% STEP 1

iv = init_values

left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, iv)

time = 0.
dissipation_coefficients = solver_settings.artificial_dissipation_scheme(dynamics_ud.partial_max_magnitudes,
                                                                             grid.states, time, iv,
                                                                             left_grad_values, right_grad_values)

max_time_step=time_step - time
time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics_ud.hamiltonian(*args, **kwargs)

def lax_friedrichs_numerical_hamiltonian(hamiltonian, state, time, value, left_grad_value, right_grad_value,
                                         dissipation_coefficients):
    hamiltonian_value = hamiltonian(state, time, value, (left_grad_value + right_grad_value) / 2)
    dissipation_value = dissipation_coefficients @ (right_grad_value - left_grad_value) / 2
    return hamiltonian_value - dissipation_value

from hj_reachability import utils

dvalues_dt = -solver_settings.hamiltonian_postprocessor(time_direction * utils.multivmap(
        lambda state, value, left_grad_value, right_grad_value, dissipation_coefficients:
        (lax_friedrichs_numerical_hamiltonian(signed_hamiltonian, state, time, value,
                                              left_grad_value, right_grad_value, dissipation_coefficients)),
        np.arange(grid.ndim))(grid.states, iv, left_grad_values, right_grad_values, dissipation_coefficients))

target_time = time_step
def sub_step(time_values):
    t, v = solver_settings.time_integrator(solver_settings, dynamics_ud, grid, *time_values, target_time)
    return t, v

t, v = sub_step((time, iv))

values_sbs = iv + t * dvalues_dt
np.allclose(values_sbs, v)

# wrong_ix = np.argwhere(init_values + t * dvalues_dt != v) 
# fig, ax = plt.subplots()
# ax.scatter(wrong_ix[:,1]/1000, wrong_ix[:,0]/1000)
# ax.set_ylim([0,1])

# # hmm several bands in the middle are off, but what is this pattern?
# # after further investigation, this is precisely bc of jax usage! but allclose returns true
# t1_nj, values_nojax, temp = time_integration.first_order_total_variation_diminishing_runge_kutta_nojax(solver_settings, dynamics_ud, grid, time, init_values, target_time)
# print("Matches inner euler: ", np.all(values_nojax == v).item())
# print("Matches s-b-s euler: ", np.all(values_nojax == values_sbs).item())


# %%

time_values = (time, init_values)
num_steps = 0

while jnp.abs(target_time - time_values[0]) > 0:
    print(f"Time {time_values[0]:>3.7f}")
    time_values = sub_step(time_values)
    num_steps +=1

print(f"Time {time_values[0]:>3.7f}")
print(f"total time steps = {num_steps:2d}")

# %% STEP 2

iv = init_values

left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, iv)

time = 0.
dissipation_coefficients = solver_settings.artificial_dissipation_scheme(dynamics_ud.partial_max_magnitudes,
                                                                             grid.states, time, iv,
                                                                             left_grad_values, right_grad_values)

max_time_step=time_step - time
time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics_ud.hamiltonian(*args, **kwargs)

def lax_friedrichs_numerical_hamiltonian(hamiltonian, state, time, value, left_grad_value, right_grad_value,
                                         dissipation_coefficients):
    hamiltonian_value = hamiltonian(state, time, value, (left_grad_value + right_grad_value) / 2)
    dissipation_value = dissipation_coefficients @ (right_grad_value - left_grad_value) / 2
    return hamiltonian_value - dissipation_value

from hj_reachability import utils

dvalues_dt = -solver_settings.hamiltonian_postprocessor(time_direction * utils.multivmap(
        lambda state, value, left_grad_value, right_grad_value, dissipation_coefficients:
        (lax_friedrichs_numerical_hamiltonian(signed_hamiltonian, state, time, value,
                                              left_grad_value, right_grad_value, dissipation_coefficients)),
        np.arange(grid.ndim))(grid.states, iv, left_grad_values, right_grad_values, dissipation_coefficients))

target_time = time_step
def sub_step(time_values):
    t, v = solver_settings.time_integrator(solver_settings, dynamics_ud, grid, *time_values, target_time)
    return t, v

values_sbs = iv + t * dvalues_dt
t2, v2 = sub_step((time, iv))

np.allclose(values_sbs, v2)

# %%

grad_print()

diss_print()

diss_ham()

ham()

val_print()

# In[24]:

max_u, max_d = 1., 0.5
tf = 1.0

dynamics_ud = four_way_crash(evader_max_turn_rate=max_u, pursuer_max_turn_rate=max_d)

ntimes = 5
fig, axes = plt.subplots(nrows=1, ncols=ntimes, figsize=(25, 5))

plt.jet()
times = np.linspace(0., tf, ntimes)

for ax, time in zip(axes, times):
    if time == 0.:
        values = init_values
    else:
        values = hj.step(solver_settings, dynamics_ud, grid, 0., values, -tf/(ntimes-1))
    ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], values[:, :].T)
    ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                values[:, :].T,
                levels=0,
                colors="black",
                linewidths=3);
    ax.set_title(f"t = {time:2.2f}")

plt.tight_layout()
plt.show()

# %%
