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


class VanderPol(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mu=1.0, 
                max_u=1.,
                max_d=0.5,
                control_mode="min",
                disturbance_mode="max",
                control_space=None,
                disturbance_space=None,
                ):

        if control_space is None:
            control_space = hj.sets.Box(lo=jnp.array([-max_u]), hi=jnp.array([max_u]))
    
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(lo=jnp.array([-max_d]), hi=jnp.array([max_d]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

        self.mu = mu

    def open_loop_dynamics(self, x, time):
        xdot1 = x[1]
        xdot2 = self.mu * (1 - x[0] ** 2) * x[1] - x[0]
        return jnp.array([xdot1, xdot2])
    
    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [1.],
        ])

# In[27]:

c = np.array([0., 0.])
r = 0.5

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-1., -1.]),
                                                                           np.array([1., 1.])),
                                                                           (100, 100))

# backwards_reachable_set = lambda x : x
solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

# values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(np.subtract(grid.states[..., :2], c))), axis=-1)) - r ** 2) * 0.5
values = (jnp.array(np.max(np.abs(np.multiply([1., 1.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r) * 0.5

# In[]: Plot Values

plt.figure(figsize=(13, 8))
plt.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], values.T, levels=0, colors="black", linewidths=3)
plt.title("Initial Target");
plt.grid();

# In[24]:

max_u, max_d = 1., 0.5
tf = 1.0

dynamics = VanderPol(max_u=max_u, max_d=max_d)
target_values = hj.step(solver_settings, dynamics, grid, 0., values, -tf)

dynamics_nod = VanderPol(max_u=max_u, max_d=0.)
target_values_nod = hj.step(solver_settings, dynamics_nod, grid, 0., values, -tf)

dynamics_auto = VanderPol(max_u=0., max_d=0.)
target_values_auto = hj.step(solver_settings, dynamics_auto, grid, 0., values, -tf)


# %%

plt.figure(figsize=(13, 8))
plt.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], values.T, levels=0, colors="black", linewidths=3)

plt.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_auto.T, levels=0, colors="green", linewidths=3)
plt.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_nod.T, levels=0, colors="red", linewidths=3)
plt.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values.T, levels=0, colors="blue", linewidths=3)

plt.grid();
