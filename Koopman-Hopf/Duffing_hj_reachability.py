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


# In[8]:


class Duffing(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, alpha=1.0, beta=1.0, delta=0.1,
                max_u=1.,
                max_d=0.1,
                control_mode="min",
                disturbance_mode="max",
                control_space=None,
                disturbance_space=None,
                d_states=[0, 1]
                ):

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_u]),
                                        jnp.array([max_u]))
    
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(len(d_states)), max_d)
            self.d_states = d_states

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

        self.alpha, self.beta, self.delta = alpha, beta, delta

    def open_loop_dynamics(self, x, time):
        xdot1 = x[1]
        xdot2 = self.alpha * x[0] - self.beta * x[0] ** 3 - self.delta * x[1]
        return jnp.array([xdot1, xdot2])
    
    def control_jacobian(self, state, time):
        return jnp.array([
            [0.,],
            [1.,],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.vstack([np.eye(1,2,dsi) for dsi in self.d_states]).T


# In[27]:

r = 1.0

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-2.5 * r, -2.5 * r]),
                                                                           np.array([ 2.5 * r,  2.5 * r])),
                                                                           (100, 100))

values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(grid.states[..., :2])), axis=-1)) - r ** 2) * 0.5

backwards_reachable_set = lambda x : x # for BRS???
solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=backwards_reachable_set)


# In[24]:

d_states = [0, 1]
max_u, max_d = 1., 0.5
time, target_time_step, steps = 0., -0.33, 3

dynamics = Duffing(max_u=max_u, max_d=max_d, d_states=d_states)
target_values_33_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 1)
target_values_66_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 2)
target_values_99_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 3)
target_values_132_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 4)
target_values_165_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 5)
target_values_198_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 6)
target_values_231_dall = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 7)

d_states = [1]
dynamics = Duffing(max_u=max_u, max_d=max_d, d_states=d_states)
target_values_33_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 1)
target_values_66_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 2)
target_values_99_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 3)
target_values_132_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 4)
target_values_165_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 5)
target_values_198_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 6)
target_values_231_dc = hj.step(solver_settings, dynamics, grid, time, values, target_time_step * 7)


# %%
