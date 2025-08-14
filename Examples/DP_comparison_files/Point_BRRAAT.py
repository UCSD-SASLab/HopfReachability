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
from scipy import integrate as ode

import hj_reachability as hj

import hj_reachability.dynamics as dynamics
from hj_reachability import time_integration

import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(formatter={'float': '{: >9.10f}'.format})
jnp.set_printoptions(formatter={'float': '{: >9.10f}'.format}) 

# NOTE this file is made from canoe BRRT

WIND_H = 0.25
WIND_V = 0.1

# %%

class Point(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 u_bd=0.0,
                 d_bd=0.0,
                 N=1,
                 alpha = 0.,
                 control_mode="min",
                 disturbance_mode="max",
                 input_shape="ball"):
        self.N = N
        self.dim = 2*N
        self.eo = jnp.ravel(jnp.column_stack((jnp.ones(N), jnp.zeros(N))))
        self.oe = jnp.ravel(jnp.column_stack((jnp.zeros(N), jnp.ones(N))))
        self.alpha = alpha
        if input_shape == "box":
            control_space = hj.sets.Box(-u_bd * jnp.ones(2*N), u_bd * jnp.ones(2*N))
            disturbance_space  = hj.sets.Box(-d_bd * jnp.ones(2*N), d_bd * jnp.ones(2*N))
        else:
            control_space = hj.sets.Ball(center=jnp.zeros(2*N), radius=u_bd)
            disturbance_space = hj.sets.Ball(center=jnp.zeros(2*N), radius=d_bd)

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return -WIND_H * self.eo - WIND_V * self.oe
        # return -WIND_H

    def control_jacobian(self, state, time):
        return jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)

# %% POST-PROCESSORS

def BRT(t, v, reach_values_): return jnp.minimum(v, reach_values_(t))

def BAT(t, v, avoid_values): return jnp.maximum(v, avoid_values(t))

def BRAT(t, v, reach_values, avoid_values): return jnp.maximum(jnp.minimum(v, reach_values(t)), avoid_values(t))

def BRAAT(t, v, vA, times, rBC, aBC):
    i = jnp.argmin(jnp.abs(t - times)) # nearest ix
    return jnp.minimum(jnp.maximum(v, aBC), 
                       jnp.maximum(rBC, vA[i,...]))

def BRRT(t, v, V1, V2, times, rBC1, rBC2, lam=0.):
    i = jnp.argmin(jnp.abs(t - times))
    return jnp.minimum(v, 
                       jnp.minimum(jnp.maximum(rBC2, V1[i,...]), 
                                   jnp.maximum(rBC1, V2[i,...])))


# In[27]: INIT

u_bd, d_bd = 1., 0.
diffgame = Point(u_bd=u_bd, d_bd=d_bd, alpha=0., control_mode="min", disturbance_mode="max")

## params
N = 1
lb_x, ub_x, lb_y, ub_y = -1.25, 1.25, -1.25, 1.25
ubs = np.ravel(np.column_stack((ub_x * np.ones(N), ub_y * np.ones(N))))
lbs = np.ravel(np.column_stack((lb_x * np.ones(N), lb_y * np.ones(N))))

grid_L = 401

ntimes = 101
TF = 4.
times = -np.linspace(0., TF, ntimes)

grid_pad=0.5
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lbs-grid_pad, ubs+grid_pad), [grid_L for _ in range(2*N)])

## Reach Ball (static)
target_width = 1.
target_height = 1.
c1, c2 = jnp.array([-target_width, -target_height]), jnp.array([target_width, target_height])
# def c1_tv(t): return jnp.array([-1.0 - WIND_H * t, target_height])
rReach = 0.2

# NO TANH
def target_values_1(t): return (jnp.linalg.norm(jnp.subtract(grid.states[..., :], c1), axis=-1) - rReach)
def target_values_2(t): return (jnp.linalg.norm(jnp.subtract(grid.states[..., :], c2), axis=-1) - rReach)
# def target_values_1_tv(t): return (jnp.linalg.norm(jnp.subtract(grid.states[..., :], c1_tv(t)), axis=-1) - rReach)

## Avoid Ball (static)
rAvoid = 0.15
buildings = jnp.array([
    [0.7016235000, 0.3140618000],
    [0.2147154350, 0.5852965100],
    [-0.5801821450, 0.4490554650],
    [-0.4438824150, 0.7321038800],
    [0.0377818200, -0.6528376050],
    [0.3632435200, 0.0897887850],
    [-0.2655787100, -0.1513413500],
    [-0.6673516050, -0.6629686500],
])

def avoid_values_ball(t): # FIXME boxes not ball
    P = grid.states[..., :2]
    diffs = P[..., None, :] - buildings[None, :, :]
    dists_inf = jnp.max(jnp.abs(diffs), axis=-1)  
    phi = -(dists_inf - rAvoid)                   
    return jnp.max(phi, axis=-1)                  

init_values_avoid_ball = BAT(0., avoid_values_ball(0.), avoid_values_ball)

## Define BC with post-processors (with values = bc(0.))
init_values_1 = target_values_1(0.)
init_values_2 = target_values_2(0.)
init_values_RR = jnp.maximum(target_values_1(0.), target_values_2(0.))

# init_values_1_tv = target_values_1_tv(0.)
# init_values_tv = jnp.maximum(target_values_1_tv(0.), target_values_2(0.))

def reach_values_for_plotting(t): return jnp.minimum(target_values_1(0.), target_values_2(0.))

init_values_reachavoid_ball_1 = BRAT(0., target_values_1(0.), target_values_1, avoid_values_ball)
init_values_reachavoid_ball_2 = BRAT(0., target_values_2(0.), target_values_2, avoid_values_ball)

# %% 

## SOLVER & PLOTTER

def solveplot(diffgame, init_values, post_processor, title="", tf=TF, ntimes=5, 
                reach_values=reach_values_for_plotting, avoid_values=avoid_values_ball, progress_bar=True,
                plot_no_value=False, plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=False, plot_rBC=False, plot_aBC=False,
                one_shot=False, avoid_game=False, offset=0, vabs=0.075, xlims=(lbs[0], ubs[0]), ylims=(lbs[1], ubs[1])):

    times = np.linspace(0., tf, ntimes)

    # Scaled colormap
    cmap_name = "RdBu_r"
    vmin, vmax = -vabs, vabs
    levels = np.linspace(vmin, vmax)
    # levels = np.linspace(init_values.min(), init_values.max())
    # n_bins_high = round(256 * init_values.max()/(init_values.max() - init_values.min()))
    n_bins_high = round(256 * vmax/(vmax - vmin))
    scaled_colors = np.vstack((mpl.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), mpl.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
    RdWhBl_vscaled = mpl.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

    solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=post_processor)

    if one_shot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=ntimes, figsize=(25, ntimes))
    plt.rcParams['text.usetex'] = False

    for ax, ti in zip(axes, range(len(times))):

        ## Solve
        if one_shot:
            values = hj.solve(solver_settings, diffgame, grid, -times, init_values, progress_bar=progress_bar)
            plot_values = values[-1, ...].T
            ax.set_title(f"t = -{times[-1]:2.2f}")

        else:
            if ti == 0:
                values = init_values
            else:
                # print("Window", -times[ti-1], -times[ti])
                values = hj.step(solver_settings, diffgame, grid, -times[ti-1], values, -times[ti], progress_bar=progress_bar)
                # values = hj.solve(solver_settings, diffgame, grid, -times[ti-1], values, -times[ti], progress_bar=False)
            plot_values = values[:, :].T
            ax.set_title(f"t = -{times[ti]:2.2f}")

        ## Plot Value
        if not plot_no_value:
            cs = ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=levels, extend="both", cmap=RdWhBl_vscaled)
            ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=0, colors="black", linewidths=3)
        
        if plot_reach_solid:
            try: # if minval < 0
                ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=[values.min(), 0], colors="blue")
            except: 
                print(f"No Reach Contour to Plot at t={-times[ti]}")
        
        if plot_avoid_solid:
            try: # if minval < 0
                if avoid_game:
                    ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=[0, avoid_values(-times[ti]).max()], colors="red")
                ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], avoid_values(-times[ti]).T, levels=[0, avoid_values(-times[ti]).max()], colors="red")
            except: 
                print(f"No Obstacle Contour to Plot at t={-times[ti]}")
        
        ## Plot Zero Contours
        # ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=0, colors="black", linewidths=3)
        if plot_bcs or plot_rBC:
            ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], reach_values(-times[ti]).T, levels=0, colors="black", linewidths=3)
        if plot_bcs or plot_aBC:
            ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], avoid_values(-times[ti]).T, levels=0, colors="black", linewidths=3)

        ax.set_aspect('equal')

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    # plt.tight_layout()
    if not plot_no_value:
        plt.tight_layout(rect=[0, 0, 0.92, 0.875])
        cbar_ax = fig.add_axes([0.92, 0.075, 0.01, 0.7])
        # vmin, vmax = init_values.min(), init_values.max()
        if vmin < 0 < vmax:
            cbar = fig.colorbar(cs, cax=cbar_ax, ticks=[vmin, 0, vmax])
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            # tick_positions = cbar.ax.get_yticks()
            # zero_tick = tick_positions[np.argmin(np.abs(tick_positions - cs.norm(0)))]
            # cbar_ax.axhline(y=zero_tick, color='black', linewidth=2)
        else:
            fig.colorbar(cs, cax=cbar_ax, ticks=[vmin, vmax])
    # else:
    #     if not one_shot:
    #         plt.tight_layout(rect=[0, 0, 0.92, 0.])
    
    if one_shot:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(title, fontsize=25)
    plt.show()

    return values, fig


# %% 

## Ball - BAT
def BAT_ball_pp(t,vlast,v): return BAT(t, v, avoid_values_ball)
BAT_values_lin_ball_full, BAT_ball_lin_fig_last = solveplot(diffgame, init_values_avoid_ball, BAT_ball_pp, title="Point BAT Final", plot_avoid_solid=False, plot_no_value=False, plot_aBC=True, avoid_game=True, one_shot=True, ntimes=ntimes)
_, BAT_ball_lin_fig = solveplot(diffgame, init_values_avoid_ball, BAT_ball_pp, title="Point BAT", plot_avoid_solid=False, plot_no_value=False, plot_aBC=True, avoid_game=True)

# %%

## Ball - BRAAT
def BRAAT_ball_lin_l_pp_1(t,vlast,v): return BRAAT(t, v, BAT_values_lin_ball_full, times, target_values_1(t), avoid_values_ball(t))
BRAAT_values_1, BRAAT_fig_1_last = solveplot(diffgame, init_values_reachavoid_ball_1, BRAAT_ball_lin_l_pp_1, title=f"Point BRAAT Final", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False, one_shot=True, ntimes=ntimes)
_, BRAAT_fig_1 = solveplot(diffgame, init_values_reachavoid_ball_1, BRAAT_ball_lin_l_pp_1, title=f"Point BRAAT", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

def BRAAT_ball_lin_l_pp_2(t,vlast,v): return BRAAT(t, v, BAT_values_lin_ball_full, times, target_values_2(t), avoid_values_ball(t))
BRAAT_values_2, BRAAT_fig_2_last = solveplot(diffgame, init_values_reachavoid_ball_2, BRAAT_ball_lin_l_pp_2, title=f"Point BRAAT Final", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False, one_shot=True, ntimes=ntimes)
_, BRAAT_fig_2 = solveplot(diffgame, init_values_reachavoid_ball_2, BRAAT_ball_lin_l_pp_2, title=f"Point BRAAT", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

# %% BRRAAT

def BRRAAT(t, v, vA, V1AA, V2AA, times, rBC1, rBC2, aBC):
    i = jnp.argmin(jnp.abs(t - times)) # nearest ix
    return jnp.maximum(aBC, 
                       jnp.minimum(v,
                                jnp.minimum(jnp.maximum(rBC1, V2AA[i,...]), 
                                            jnp.maximum(rBC2, V1AA[i,...]))
                            )
                        )

## Two-Target - BRRAAT
def BRRAAT_pp(t,vlast,v): return BRRAAT(t, v, BAT_values_lin_ball_full, BRAAT_values_1, BRAAT_values_2, times, target_values_1(t), target_values_2(t), avoid_values_ball(t))

init_values_rraa = BRRAAT(0., init_values_RR, init_values_avoid_ball, init_values_reachavoid_ball_1, init_values_reachavoid_ball_2, times, target_values_1(0.), target_values_2(0.), avoid_values_ball(0.))
reach_values_stat, BRRAAT_fig = solveplot(diffgame, init_values_rraa, BRRAAT_pp, title="Point BRRAAT", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

