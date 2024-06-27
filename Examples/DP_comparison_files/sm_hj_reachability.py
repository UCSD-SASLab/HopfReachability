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

class SlowManifold(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mu=-0.05, lambduh=-1.0,
                max_u=1.,
                max_d=0.5,
                control_mode="min",
                disturbance_mode="max",
                control_space=None,
                disturbance_space=None,
                ):

        if control_space is None:
            control_space = hj.sets.Ball(jnp.zeros(2), max_u)
    
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_d)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

        self.mu, self.lambduh = mu, lambduh

    def open_loop_dynamics(self, x, time):
        xdot1 = self.mu * x[0]
        xdot2 = self.lambduh * (x[1] - x[0] ** 2)
        return jnp.array([xdot1, xdot2])
    
    def control_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
        ])
    
class SlowManifoldAug(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, mu=-0.05, lambduh=-1.0,
                max_u=1.,
                max_d=0.5,
                control_mode="min",
                disturbance_mode="max",
                control_space=None,
                disturbance_space=None,
                ):

        if control_space is None:
            control_space = hj.sets.Ball(jnp.zeros(2), max_u)
    
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_d)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

        self.mu, self.lambduh = mu, lambduh

    def open_loop_dynamics(self, x, time):
        xdot1 = self.mu * x[0]
        xdot2 = self.lambduh * (x[1] - x[2])
        xdot3 = 2 * self.mu * x[2]
        return jnp.array([xdot1, xdot2, xdot3])
    
    def control_jacobian(self, x, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [2 * x[0], 0.],
        ])

    def disturbance_jacobian(self, x, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [2 * x[0], 0.],
        ])

# In[27]:

c = np.array([0, 0.5])
r = 1.0

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([0, 0]),
                                                                           np.array([4, 4])),
                                                                           (100, 100))

backwards_reachable_set = lambda x : x # for BRS???
solver_settings = hj.SolverSettings.with_accuracy("high", hamiltonian_postprocessor=backwards_reachable_set)

values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(np.subtract(grid.states, c))), axis=-1)) - r ** 2) * 0.5

# In[24]:

max_u, max_d = 1., 0.5

dynamics = SlowManifold(max_u=max_u, max_d=max_d)
target_values_p2 = hj.step(solver_settings, dynamics, grid, 0., values, -0.2)
target_values_2 = hj.step(solver_settings, dynamics, grid, 0., values, -2.0)

# In[27]:

c3 = np.array([0., 0.5, 0.])

grid3 = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([0, 0, 0]),
                                                                           np.array([4, 4, 4])),
                                                                           (100, 100, 100))

values3 = (jnp.array(np.sum(np.multiply([1., 1., 1.], np.square(np.subtract(grid3.states, c3))), axis=-1)) - r ** 2) * 0.5

# In[24]:

# dynamics_aug = SlowManifoldAug(max_u=max_u, max_d=max_d)
# target_values3_p2 = hj.step(solver_settings, dynamics_aug, grid3, 0., values3, -0.2)
# target_values_2 = hj.step(solver_settings, dynamics_aug, grid3, 0., values3, -2.0)
