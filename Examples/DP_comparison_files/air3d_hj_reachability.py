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


# %%

class Air3d(dynamics.ControlAndDisturbanceAffineDynamics):

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
        _, _, psi = state
        v_a, v_b = self.evader_speed, self.pursuer_speed
        return jnp.array([-v_a + v_b * jnp.cos(psi), v_b * jnp.sin(psi), 0.])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array([
            [y],
            [-x],
            [-1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [1.],
        ])
    
# In[8]:

class Air3d_lin(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 evader_speed=5.,
                 pursuer_speed=5.,
                 evader_max_turn_rate=1.,
                 pursuer_max_turn_rate=1.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None,
                 lin_point=[0., 0., 0.]):
        self.evader_speed = evader_speed
        self.pursuer_speed = pursuer_speed
        self.lin_point = lin_point
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-evader_max_turn_rate]), jnp.array([evader_max_turn_rate]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-pursuer_max_turn_rate]), jnp.array([pursuer_max_turn_rate]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v_a, v_b = self.evader_speed, self.pursuer_speed
        psi_l = self.lin_point[2]
        return jnp.array([-v_a - v_b * jnp.sin(psi_l) * psi, v_b * jnp.cos(psi_l) * psi, 0.])

    def control_jacobian(self, state, time):
        x_l, y_l = self.lin_point[0], self.lin_point[1]
        return jnp.array([
            [y_l],
            [-x_l],
            [-1.],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [1.],
        ])
    
# In[27]:

# c = np.array([0., 0.])
# r = 0.5

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([-6., -10., 0.]),
                                                                           np.array([20., 10., 2 * np.pi])),
                                                               (51, 40, 50),
                                                               periodic_dims=2)

backwards_reachable_set = lambda x : x
solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=backwards_reachable_set)
# solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

# values = (jnp.array(np.sum(np.multiply([1., 1.], np.square(np.subtract(grid.states[..., :2], c))), axis=-1)) - r ** 2) * 0.5
# values = (jnp.array(np.max(np.abs(np.multiply([1., 1., 1.], np.subtract(grid.states[..., :2], c))), axis=-1)) - r) * 0.5
values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 5

# In[24]:

max_u, max_d = 1., 0.5
tf = 1.0

dynamics_ud = Air3d(evader_max_turn_rate=max_u, pursuer_max_turn_rate=max_d)
target_values = hj.step(solver_settings, dynamics_ud, grid, 0., values, -tf)

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

theta_slices = [0, 12, 25, 12 + 25, 50]

plt.jet()
thetas = np.linspace(0., 2 * np.pi, 51)

for ax, theta_slice in zip(axes, theta_slices):
    ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values[:, :, theta_slice].T)
    ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_values[:, :, theta_slice].T,
                levels=0,
                colors="black",
                linewidths=3);
    ax.set_title(f"θ = {thetas[theta_slice]:2.2f}")

plt.tight_layout()
plt.show()

# %%

max_u, max_d = 1., 0.5
tf = 1.0
lin_point = [0., 0., 0.]

dynamics_ud_lin = Air3d_lin(evader_max_turn_rate=max_u, pursuer_max_turn_rate=max_d, lin_point=lin_point)
target_values_lin = hj.step(solver_settings, dynamics_ud_lin, grid, 0., values, -tf)

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))

theta_slices = [0, 12, 25, 12 + 25, 50]

plt.jet()
thetas = np.linspace(0., 2 * np.pi, 51)

for ax, theta_slice in zip(axes, theta_slices):
    ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_lin[:, :, theta_slice].T)
    ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_values_lin[:, :, theta_slice].T,
                levels=0,
                colors="black",
                linewidths=3);
    ax.set_title(f"θ = {thetas[theta_slice]:2.2f}")

plt.tight_layout()
plt.show()
# %%
