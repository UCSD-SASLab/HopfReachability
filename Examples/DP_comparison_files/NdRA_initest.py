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

## FIXED PARAMS
IH = 2 # Initial Height
G = 1 # Gravity
TC = np.sqrt(2 * IH / G) # Critical Time of Landing


# %%

class SkydiversDynamics(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 u_bd=0.0,
                 d_bd=0.0,
                 N=1,
                 alpha = 0.,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        self.N = N
        self.dim = N+1
        self.z1oN = jnp.array([np.concatenate(([0], np.ones(N)))])
        self.oN = jnp.ones(self.dim)
        self.alpha = alpha
        if control_space is None:
            control_space = hj.sets.Box(-u_bd * jnp.ones(N), u_bd * jnp.ones(N))
        if disturbance_space is None:
            disturbance_space  = hj.sets.Box(-d_bd * jnp.ones(N), d_bd * jnp.ones(N))
            # disturbance_space = hj.sets.Box(-d_bd * self.oN, d_bd * self.oN)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        # return jnp.concatenate([jnp.array([-1]), # Constant/terminal fall velocity
        #                         jnp.zeros(self.N)])
        # return jnp.concatenate([jnp.array([-(G * TC) + (G * -time)]), # NEGATIVE integ of accel (BRT computed backwards)
        #                                   jnp.zeros(self.N)])
        return jnp.concatenate([jnp.array([-1]),
                                self.alpha * state[1:] ** 3])

    def control_jacobian(self, state, time):
        return self.z1oN.T # all but shared controlled

    def disturbance_jacobian(self, state, time):
        return self.z1oN.T # all but shared disturbed
        # return self.oN.T # all disturbed
    
# In[27]: INIT

alpha = 0.
# dynamics_inst = SkydiversDynamics(u_bd=0., d_bd=0.0)
# dynamics_inst = SkydiversDynamics(u_bd=0.5, d_bd=0.0)
dynamics_inst = SkydiversDynamics(u_bd=0.5, d_bd=0.1, alpha=alpha)

## params
N = 1
x_width = 2.
eps = 0.3
ubs = np.concatenate(([IH + eps], x_width * np.ones(N))) 
lbs = np.concatenate(([-eps], -x_width * np.ones(N)))
grid_L = 501

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lbs, ubs), [grid_L for _ in range(N+1)])

rReach = 0.15
init_values = jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach

print("Max BC value:", init_values.max())
print("Min BC value:", init_values.min())

# In[24]: BRT - ONE SHOT

rReach = 0.15
init_values = jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach

solver_settings = hj.SolverSettings.with_accuracy("very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

# max_u, max_d = 1., 0.5
tf = TC

ntimes = 5
fig, axes = plt.subplots(nrows=1, ncols=ntimes, figsize=(25, ntimes))

plt.jet()
times = np.linspace(0., tf, ntimes)
levels = np.linspace(-0.5, 2)

for ax, time in zip(axes, times):
    if time == 0.:
        values = init_values
    else:
        values = hj.step(solver_settings, dynamics_inst, grid, 0., init_values, -time)
    ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :], levels=levels, extend="both")
    ax.contour(grid.coordinate_vectors[1],
               grid.coordinate_vectors[0],
                values[:, :],
                levels=0,
                colors="black",
                linewidths=3)
    ax.set_title(f"t = {time:2.2f}")

plt.tight_layout()
plt.show()

# %% BRAS/BRAT/BAT - Constant Obstacle

rReach = 0.15
# rReach = -1.
reach_values = jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach

rAvoid = 0.3
c = [1.0, 0.]
# avoid_values = -(jnp.array(np.max(np.abs(np.subtract(grid.states[..., :2], c)), axis=-1)) - rAvoid)
avoid_values = -(jnp.linalg.norm(np.subtract(grid.states[..., :2], c), axis=-1) - rAvoid)
print(f"Avoid BC value in [{avoid_values.min():2.2f}, {avoid_values.max():2.2f}]")

BRAS = lambda t, v : jnp.maximum(v, avoid_values)
BRAT = lambda t, v : jnp.maximum(jnp.minimum(v, reach_values), avoid_values)

lambdaR = 10
BRATlam = lambda t, v : jnp.maximum(jnp.minimum(v, reach_values + lambdaR), avoid_values)
# BAT = lambda t, v : jnp.minimum(v, -avoid_values)

solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=BRATlam)

# BRAT problem
dynamics_inst = SkydiversDynamics(u_bd=0.5, d_bd=0.1, alpha=0.)
init_values = jnp.maximum(reach_values + lambdaR, avoid_values)

# BAT problem
# dynamics_inst = SkydiversDynamics(u_bd=0.5, d_bd=0.1, alpha=0., disturbance_mode="min", control_mode="max")
# init_values = -avoid_values # BAT

## Plot
ntimes = 5
times = np.linspace(0., tf, ntimes)
fig, axes = plt.subplots(nrows=1, ncols=ntimes, figsize=(25, ntimes))

for ax, ti in zip(axes, range(len(times))):
    if ti == 0:
        values = init_values
    else:
        # print("Window", -times[ti-1], -times[ti])
        values = hj.step(solver_settings, dynamics_inst, grid, -times[ti-1], values, -times[ti], progress_bar=False)
    # try:
    #     ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :], 
    #                 levels=[values.min(), 0], colors="blue",
    #                 )
    # except:
    #     levels = np.linspace(-4, 1)
    ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :], levels=levels, extend="both")
    ax.contour(grid.coordinate_vectors[1],
               grid.coordinate_vectors[0],
                values[:, :],
                levels=0,
                colors="black",
                linewidths=3)
    ax.contour(grid.coordinate_vectors[1],
               grid.coordinate_vectors[0],
                avoid_values,
                levels=0,
                colors="black",
                linewidths=3)
    ax.contour(grid.coordinate_vectors[1],
               grid.coordinate_vectors[0],
                reach_values,
                levels=0,
                colors="black",
                linewidths=3)
    # ax.contourf(grid.coordinate_vectors[1],
    #            grid.coordinate_vectors[0],
    #             avoid_values,
    #             levels=[0, avoid_values.max()],
    #             colors="red")
    ax.set_title(f"t = {times[ti]:2.2f}")
    print(f"at t={times[ti]:2.2f}, value in [{values.min():2.2f},{values.max():2.2f}] ")

plt.tight_layout()
plt.show()
# %% BRAS/BRAT - Moving Target

alpha = 0
dynamics_inst = SkydiversDynamics(u_bd=0.5, d_bd=0.1, alpha=alpha) # , control_mode="max", disturbance_mode="min"
# dynamics_inst = SkydiversDynamics(u_bd=0.1, d_bd=0.5, alpha=alpha) # , control_mode="max", disturbance_mode="min"

rReach = 0.15
rReach = 1.5
reach_values = lambda t: jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach
# avoid_values = lambda t: -jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach

## Multi Rectangle Obstacle
rAvoid = 0.3
c = lambda i: jnp.array([0.5 * (i+1), 0.])
diag = jnp.array([5., 1.])
# path = lambda t: jnp.sin((2.*jnp.pi/2.)*t)
path = lambda time, phase: jnp.cos((2.*jnp.pi/2.) * time + phase)
# phases = [0., jnp.pi/2., jnp.pi]
phases = [0., jnp.pi, 0.]
avoid_values = lambda t: -jnp.minimum(jnp.minimum(
                            jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(0) + jnp.array([0., path(t, phases[0])])))), axis=-1)) - rAvoid, 
                            jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(1) + jnp.array([0., path(t, phases[1])])))), axis=-1)) - rAvoid),
                            jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(2) + jnp.array([0., path(t, phases[2])])))), axis=-1)) - rAvoid)

# reach_values = lambda t: jnp.minimum(jnp.minimum(
#                             jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(0) + jnp.array([0., path(t, phases[0])])))), axis=-1)) - rAvoid, 
#                             jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(1) + jnp.array([0., path(t, phases[1])])))), axis=-1)) - rAvoid),
#                             jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(2) + jnp.array([0., path(t, phases[2])])))), axis=-1)) - rAvoid)

init_values = jnp.maximum(reach_values(0.), avoid_values(0.))

BRAS = lambda t, v : jnp.maximum(v, avoid_values(t))
BRAT = lambda t, v : jnp.maximum(jnp.minimum(v, reach_values(t)), avoid_values(t))
solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=BRAT)

## Plot
ntimes = 5
times = np.linspace(0., tf, ntimes)
fig, axes = plt.subplots(nrows=1, ncols=ntimes, figsize=(25, ntimes))

for ax, ti in zip(axes, range(len(times))):
    if ti == 0:
        values = init_values
    else:
        # print("Window", -times[ti-1], -times[ti])
        values = hj.step(solver_settings, dynamics_inst, grid, -times[ti-1], values, -times[ti], progress_bar=False)
    # try:
    #     ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :], 
    #                 levels=[values.min(), 0], colors="blue",
    #                 )
    # except:
    #     levels = np.linspace(-4, 1)
    ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :]) #, levels=levels, extend="both")
    ax.contour(grid.coordinate_vectors[1],
               grid.coordinate_vectors[0],
                values[:, :],
                levels=0,
                colors="black",
                linewidths=3)
    # ax.contour(grid.coordinate_vectors[1],
    #            grid.coordinate_vectors[0],
    #             avoid_values(-times[ti]),
    #             levels=0,
    #             colors="black",
    #             linewidths=3)
    # ax.contourf(grid.coordinate_vectors[1],
    #            grid.coordinate_vectors[0],
    #             avoid_values(-times[ti]),
    #             levels=[0, avoid_values(-times[ti]).max()],
    #             colors="red")
    ax.set_title(f"t = {times[ti]:2.2f}")
    print(f"at t={times[ti]:2.2f}, value in [{values.min():2.2f},{values.max():2.2f}] ")

plt.tight_layout()
plt.show()

# %%

import matplotlib.animation as animation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(grid.coordinate_vectors[1].min(), grid.coordinate_vectors[1].max())
ax.set_ylim(grid.coordinate_vectors[0].min(), grid.coordinate_vectors[0].max())
ax.set_title("t = 0.00")

contourf_plot = None
contour_plot = None
avoid_contour = None
avoid_contourf = None

values = init_values.copy()

ntimes = 51  # Number of animation frames
times = np.linspace(0., tf, ntimes)

def update_frame(frame):
    global values, contourf_plot, contour_plot, avoid_contour, avoid_contourf
    # ax.clear()
    time = -times[frame]
    time_last = -times[frame-1]
    
    # Solve
    if time < 0.:
        values = hj.step(solver_settings, dynamics_inst, grid, time_last, values, time)
    
    # Remove previous contours before plotting new ones
    for c in [contourf_plot, contour_plot, avoid_contour, avoid_contourf]:
        if c is not None:
            for coll in c.collections:
                coll.remove()

    # Plot               
    contourf_plot = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :],
                                 levels=[values.min(), 0], colors="blue")
    contour_plot = ax.contour(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :],
                               levels=0, colors="black", linewidths=3)
    
    avoid_val = avoid_values(time)
    avoid_contour = ax.contour(grid.coordinate_vectors[1], grid.coordinate_vectors[0], avoid_val,
                                levels=0, colors="black", linewidths=3)
    avoid_contourf = ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], avoid_val,
                                 levels=[0, avoid_val.max()], colors="red", linewidths=3)
    
    ax.set_title(f"t = {time:2.2f}")
    return contourf_plot.collections + contour_plot.collections + avoid_contour.collections + avoid_contourf.collections

ani = animation.FuncAnimation(fig, update_frame, frames=ntimes, blit=False)
HTML(ani.to_jshtml())

# %%

class SkydiversProblem():

    def __init__(self, N=1, 
                    u_bd=0.3, d_bd=0.1,
                    rReach = 0.15, rAvoid = 0.25,
                    x_width = 1.5, eps = 0.3,
                    grid_L = 501,
                    obstacle_type="multi-rectangle",
                    phases=[0., jnp.pi/2., jnp.pi],
                    ):

        self.N = N
        self.dynamics = SkydiversDynamics(N=N, u_bd=u_bd, d_bd=d_bd)
        self.ubs = np.concatenate(([IH + eps], x_width * np.ones(N)))
        self.lbs = np.concatenate(([-eps], -x_width * np.ones(N)))
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(self.lbs, self.ubs), [grid_L for _ in range(N+1)])

        self.reach_values = jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach

        self.make_obstacles(rAvoid, obstacle_type, phases)

        self.BRS = lambda t, v : v
        self.BRT = lambda t, v : jnp.minimum(v, self.reach_values)
        self.BRAS = lambda t, v : jnp.maximum(v, self.avoid_values(t))
        self.BRAT = lambda t, v : jnp.maximum(jnp.minimum(v, self.reach_values), self.avoid_values(t))

    def make_obstacles(self, rAvoid, obstacle_type, phases=None, c=jnp.array([1., 0.])):
        
        if "rand" in obstacle_type or not phases:
            phases = 2 * jnp.pi * jnp.array(np.random.random(3))

        if "const" in obstacle_type:
            path = lambda time, phase: jnp.array([0., jnp.cos(phase)])

        elif "swing" in obstacle_type:
            path = lambda time, phase: jnp.array([0., jnp.cos((2.*jnp.pi/2.) * time + phase)])

        ## Multi-Obstacle
        if "multi" in obstacle_type:

            c = lambda i: jnp.array([0.5 * (i+1), 0.])

            if "box" in obstacle_type or "rectangle" in obstacle_type:
                diag = jnp.array([5., 1.])
                avoid_values = lambda t: -jnp.minimum(jnp.minimum(
                                        jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(0) + path(t, phases[0])))), axis=-1)) - rAvoid, 
                                        jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(1) + path(t, phases[1])))), axis=-1)) - rAvoid),
                                        jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], c(2) + path(t, phases[2])))), axis=-1)) - rAvoid)
            
            elif "ball" in obstacle_type or "circle" in obstacle_type:
                avoid_values = lambda t: -jnp.minimum(jnp.minimum(
                                            (jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c(0) + path(t, phases[0])), axis=-1) - rAvoid), 
                                            (jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c(1) + path(t, phases[1])), axis=-1) - rAvoid)),
                                            (jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c(2) + path(t, phases[2])), axis=-1) - rAvoid))

        else:

            if "box" in obstacle_type or "rectangle" in obstacle_type:
                avoid_values = lambda t: -(jnp.array(jnp.max(jnp.abs(jnp.subtract(grid.states[..., :2], c + path(t, 0))), axis=-1)) - rAvoid)

            else: ## "ball" in obstacle_type or "circle" in obstacle_type:
                avoid_values = lambda t: -(jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c + path(t, 0)), axis=-1) - rAvoid)

        self.avoid_values = avoid_values
    
    def solve(self, tf=TC, game="BRS", plot_type="none"):

        self.tf = tf

        if game == "BRS":
            processor = self.BRS
            self.init_values = self.reach_values.copy()
        elif game == "BRT":
            processor = self.BRT
            self.init_values = self.reach_values.copy()
        elif game == "BRAS":
            processor = self.BRAS
            self.init_values = jnp.maximum(self.reach_values, self.avoid_values(0.))
        elif game == "BRAT":
            processor = self.BRAT
            self.init_values = jnp.maximum(self.reach_values, self.avoid_values(0.))
        self.solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=processor)
        
        if plot_type == "panel":
            self.panel_plot(game)
        elif plot_type == "animation":
            self.anim_plot()
        else:
            values = hj.step(self.solver_settings, self.dynamics, self.grid, 0., self.init_values, -tf)
            return values

    def panel_plot(self, game="BRT", n_times=5):

        times = np.linspace(0., self.tf, n_times)
        fig, axes = plt.subplots(nrows=1, ncols=n_times, figsize=(25, n_times))

        for ax, ti in zip(axes, range(len(times))):
            if ti == 0:
                values = self.init_values
            else:
                # print("Window", -times[ti-1], -times[ti])
                values = hj.step(self.solver_settings, self.dynamics, self.grid, -times[ti-1], values, -times[ti])
            if values.min() < 0.:
                ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], values[:, :], 
                            levels=[values.min(), 0], colors="blue",
                            )
            # ax.contourf(grid.coordinate_vectors[1], grid.coordinate_vectors[0], values[:, :], 
            #             )
            ax.contour(self.grid.coordinate_vectors[1],
                    self.grid.coordinate_vectors[0],
                        values[:, :],
                        levels=0,
                        colors="black",
                        linewidths=3)
            if game == "BRAS" or game == "BRAT":
                ax.contour(self.grid.coordinate_vectors[1],
                        self.grid.coordinate_vectors[0],
                            self.avoid_values(-times[ti]),
                            levels=0,
                            colors="black",
                            linewidths=3)
                ax.contourf(self.grid.coordinate_vectors[1],
                        self.grid.coordinate_vectors[0],
                            self.avoid_values(-times[ti]),
                            levels=[0, self.avoid_values(-times[ti]).max()],
                            colors="red",
                        linewidths=3)
            ax.set_title(f"t = {times[ti]:2.2f}")

        plt.tight_layout()
        plt.show()

    def anim_plot(self, game, n_times=50):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(self.grid.coordinate_vectors[1].min(), self.grid.coordinate_vectors[1].max())
        ax.set_ylim(self.grid.coordinate_vectors[0].min(), self.grid.coordinate_vectors[0].max())
        ax.set_title("t = 0.00")

        self.contourf_plot = None
        self.contour_plot = None
        self.avoid_contour = None
        self.avoid_contourf = None

        self.curr_values = self.init_values.copy()

        times = np.linspace(0., tf, n_times)

        def update_frame(frame):
            time = -times[frame]
            time_last = -times[frame-1]
            
            # Solve
            if time < 0.:
                self.curr_values = hj.step(self.solver_settings, self.dynamics, self.grid, time_last, self.curr_values, time)
            
            # Remove previous contours before plotting new ones
            for c in [self.contourf_plot, self.contour_plot, self.avoid_contour, self.avoid_contourf]:
                if c is not None:
                    for coll in c.collections:
                        coll.remove()

            # Plot               
            self.contourf_plot = ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.curr_values[:, :],
                                        levels=[values.min(), 0], colors="blue")
            self.contour_plot = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], self.curr_values[:, :],
                                    levels=0, colors="black", linewidths=3)
            
            avoid_val = self.avoid_values(time)
            if game == "BRAS" or game == "BRAT":
                self.avoid_contour = ax.contour(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], avoid_val,
                                            levels=0, colors="black", linewidths=3)
                self.avoid_contourf = ax.contourf(self.grid.coordinate_vectors[1], self.grid.coordinate_vectors[0], avoid_val,
                                            levels=[0, avoid_val.max()], colors="red", linewidths=3)
                
            ax.set_title(f"t = {time:2.2f}")
            return contourf_plot.collections + contour_plot.collections + avoid_contour.collections + avoid_contourf.collections

        ani = animation.FuncAnimation(fig, update_frame, frames=n_times, blit=False)
        HTML(ani.to_jshtml())

# %%

prob = SkydiversProblem()

# %%

prob.solve(game="BRT", plot_type="panel", tf=1.)

# %%

prob.solve(game="BRAT", plot_type="panel")

# %%

# LEFTOVER TARGET TESTING


# cr = lambda i: jnp.array([0., -0.5 + 0.5 * i])
# cr = lambda i: jnp.array([0., -0.5 + 0.5 * i])
# reach_values = jnp.minimum(jnp.minimum(
#                     jnp.linalg.norm(jnp.subtract(grid.states[..., :], cr(0)), axis=-1) - rReach, 
#                     jnp.linalg.norm(jnp.subtract(grid.states[..., :], cr(1)), axis=-1) - rReach),
#                     jnp.linalg.norm(jnp.subtract(grid.states[..., :], cr(2)), axis=-1) - rReach)

## Box Obstacle
# rAvoid = 0.25
# c = jnp.array([1.0, 0.])
# avoid_values = lambda t: -(jnp.array(jnp.max(jnp.abs(jnp.subtract(grid.states[..., :2], c + jnp.array([0., 0.5 * jnp.sin((2.*jnp.pi/2.)*t)]))), axis=-1)) - rAvoid)

## Circle Obstacle (center start)
# rAvoid = 0.25
# c = jnp.array([1.0, 0.])
# avoid_values = lambda t: -(jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c + jnp.array([0., 0.5 * jnp.sin((2.*jnp.pi/2.)*t)])), axis=-1) - rAvoid)

## Circle Obstacle (side start)
# rAvoid = 0.25
# c = jnp.array([1.0, 0.])
# avoid_values = lambda t: -(jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c + jnp.array([0., jnp.cos((2.*jnp.pi/2.)*t)])), axis=-1) - rAvoid)

## Multi Circle Obstacle
# rAvoid = 0.2
# c = lambda i: jnp.array([0.5 * (i+1), 0.])
# avoid_values = lambda t: -jnp.minimum(jnp.minimum(
#                             (jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c(0) + jnp.array([0., jnp.cos((2.*jnp.pi/2.)*t)])), axis=-1) - rAvoid), 
#                             (jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c(1) + jnp.array([0., jnp.cos((2.*jnp.pi/2.)*t)])), axis=-1) - rAvoid)),
#                             (jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c(2) + jnp.array([0., jnp.cos((2.*jnp.pi/2.)*t)])), axis=-1) - rAvoid))
