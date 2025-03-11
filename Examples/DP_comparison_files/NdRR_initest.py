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

import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(formatter={'float': '{: >9.10f}'.format})
jnp.set_printoptions(formatter={'float': '{: >9.10f}'.format}) 

CURRENT_H = 0.5
CURRENT_V = 0.5

# %%

class Canoe(dynamics.ControlAndDisturbanceAffineDynamics):

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
        return -(CURRENT_H + self.alpha * jnp.abs(state ** 3)) * self.eo * (state < jnp.zeros(self.dim)) + CURRENT_V * self.oe

    def control_jacobian(self, state, time):
        return jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)

# %% POST-PROCESSORS

def BRT(t, v, reach_values_): return jnp.minimum(v, reach_values_(t))

## nearest neighbor 
def BRRT(t, v, V1, V2, times, rBC1, rBC2, lam=0.):
    i = jnp.argmin(jnp.abs(t - times))
    # print("i", i, "times[i]", times[i]) 
    return jnp.minimum(v, 
                       jnp.minimum(jnp.maximum(rBC2, V1[i,...] + lam) - jnp.maximum(lam, 0), 
                                   jnp.maximum(rBC1 + lam, V2[i,...]) - jnp.maximum(lam, 0)))

# Ultimately these need to be wrapped, e.g.
# def vpp(t, v): return BRRT(t, v, values_reach_1, values_reach_2, times, target_values_1(t), target_values_2(t), lam)
    
# In[27]: INIT

u_bd, d_bd = 1.0, 0.1
diffgame_lin = Canoe(u_bd=u_bd, d_bd=d_bd, alpha=0., control_mode="min", disturbance_mode="max")
diffgame_nlin = Canoe(u_bd=u_bd, d_bd=d_bd, alpha=0.1, control_mode="min", disturbance_mode="max")

## params
N = 1
lb_x, ub_x, lb_y, ub_y = -1.25, 1.25, -0.5, 2.
ubs = np.ravel(np.column_stack((ub_x * np.ones(N), ub_y * np.ones(N))))
lbs = np.ravel(np.column_stack((lb_x * np.ones(N), lb_y * np.ones(N))))

grid_L = 501

ntimes = 101
TF = 2.
times = -np.linspace(0., TF, ntimes)

grid_pad=2.
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lbs-grid_pad, ubs+grid_pad), [grid_L for _ in range(2*N)])

## Reach Ball (static)
bc_alpha = 0.5
target_width = 0.75
target_height = 1.
c1, c2 = jnp.array([-target_width, target_height]), jnp.array([target_width, target_height])
rReach = 0.2
def target_values_1(t): return bc_alpha * jnp.tanh((jnp.linalg.norm(jnp.subtract(grid.states[..., :], c1), axis=-1) - rReach))
def target_values_2(t): return bc_alpha * jnp.tanh((jnp.linalg.norm(jnp.subtract(grid.states[..., :], c2), axis=-1) - rReach))

def c1_tv(t): return jnp.array([-1.0 - CURRENT_H * t, target_height])
def target_values_1_tv(t): return bc_alpha * jnp.tanh((jnp.linalg.norm(jnp.subtract(grid.states[..., :], c1_tv(t)), axis=-1) - rReach))

## Define BC with post-processors (with values = bc(0.))
init_values_1 = target_values_1(0.)
init_values_2 = target_values_2(0.)
init_values = jnp.minimum(target_values_1(0.), target_values_2(0.))

init_values_1_tv = target_values_1_tv(0.)
init_values_tv = jnp.minimum(target_values_1_tv(0.), target_values_2(0.))

def init_values_lam(lam): return BRRT(0., jnp.inf * jnp.abs(target_values_1(0.)), target_values_1(0.), target_values_2(0.), times, target_values_1(0.), target_values_2(0.), lam)
def init_values_lam_tv(lam): return BRRT(0., jnp.inf * jnp.abs(target_values_1_tv(0.)), target_values_1(0.), target_values_2(0.), times, target_values_1_tv(0.), target_values_2(0.), lam)

# %% 

# SOLVER

def solveplot(diffgame, init_values, post_processor, title="", tf=TF, nplots=5, ntimes=ntimes,
                target_values_1_=target_values_1, target_values_2_=target_values_2, progress_bar=True,
                plot_no_value=False, plot_bcs=True, one_shot=False,
                offset=0, vabs=0.075, xlims=(lbs[0], ubs[0]), ylims=(lbs[1], ubs[1])):

    ## Solve
    times = np.linspace(0., tf, ntimes)
    solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=post_processor)
    values = hj.solve(solver_settings, diffgame, grid, -times, init_values, progress_bar=progress_bar)

    ## Plot Init
    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=(nplots**2, nplots))
    plt.rcParams['text.usetex'] = False
    cmap_name = "RdBu_r"
    vmin, vmax = -vabs, vabs
    levels = np.linspace(vmin, vmax)
    n_bins_high = round(256 * vmax/(vmax - vmin))
    scaled_colors = np.vstack((mpl.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), mpl.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
    RdWhBl_vscaled = mpl.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

    ## Plot
    for ax, i in zip(axes, range(nplots)):

        # Find nearest ix
        tix = jnp.argmin(jnp.abs(i*(tf/(nplots-1)) - times))
        plot_values = values[tix, ...].T
        ax.set_title(f"t = -{times[tix]:2.2f}")

        ## Plot Value
        if not plot_no_value:
            cs = ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=levels, extend="both", cmap=RdWhBl_vscaled)
            if plot_values.min() < 0. < plot_values.max():
                ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=0, colors="black", linewidths=3)
        
        ## Plot Zero Contours
        # ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=0, colors="black", linewidths=3)
        if plot_bcs:
            ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_1_(-times[tix]).T, levels=0, colors="black", linewidths=3)
            ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_2_(-times[tix]).T, levels=0, colors="black", linewidths=3)

        ax.set_aspect('equal')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    # plt.tight_layout()
    if not plot_no_value:
        plt.tight_layout(rect=[0, 0, 0.92, 0.875])
        cbar_ax = fig.add_axes([0.92, 0.075, 0.01, 0.7])
        if vmin < 0 < vmax:
            cbar = fig.colorbar(cs, cax=cbar_ax, ticks=[vmin, 0, vmax])
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        else:
            fig.colorbar(cs, cax=cbar_ax, ticks=[vmin, vmax])
    
    fig.suptitle(title, fontsize=25)
    plt.show()

    return values, fig

# %% 

## Target 1 - BRT 
def BRT1_pp(t,vlast,v): return BRT(t, v, target_values_1)

# Linear
reach_values_1, BRT_fig_1 = solveplot(diffgame_lin, init_values_1, BRT1_pp, ntimes=ntimes, title="Canoe BRT 1 (Linear)")

# Nonlinear
# BRT_values, BRT_fig = solveplot(diffgame_nlin, init_values_reach, BRT_pp, title="Canoe BRT 1 (Nonlinear)", plot_reach_solid=False, plot_rBC=True)

## Target 2 - BRT 
def BRT2_pp(t,vlast,v): return BRT(t, v, target_values_2)

# Linear
reach_values_2, BRT_fig_2 = solveplot(diffgame_lin, init_values_2, BRT2_pp, ntimes=ntimes, title="Canoe BRT 2 (Linear)")

# Nonlinear
# BRT_values, BRT_fig = solveplot(diffgame_nlin, init_values_2, BRT2_pp, title="Canoe BRT 2 (Nonlinear)", plot_reach_solid=False, plot_rBC=True)

# %% 

## Two-Target - BRRT
def BRRT_stat_pp(t,vlast,v): return BRRT(t, v, reach_values_1, reach_values_2, times, target_values_1(t), target_values_2(t))

# def vpp(t, v): return BRRT(t, v, values_reach_1, values_reach_2, times, target_values_1(t), target_values_2(t), lam)

# Linear
reach_values_stat, BRRT_fig = solveplot(diffgame_lin, init_values_lam(0.), BRRT_stat_pp, ntimes=ntimes, title="Canoe BRRT (Linear)", plot_bcs=True)

# Nonlinear
# BRAT_values, BRAT_fig = solveplot(diffgame_nlin, init_values, BRAT_ball_pp, title="Canoe BRAT Ball (Nonlinear)", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

# %%

## Two-Target - BRRT lambda variations

for lam in [-1., -0.2, 0., 0.2, 1.]:
    
    # Linear
    def BRRT_stat_pp(t,vlast,v): return BRRT(t, v, reach_values_1, reach_values_2, times, target_values_1(t), target_values_2(t), lam=lam)
    reach_values_stat, BRRT_fig = solveplot(diffgame_lin, init_values_lam(lam), BRRT_stat_pp, tf=4., ntimes=ntimes, title=f"Canoe BRRT - Lambda {lam:2.2f} (Linear)", plot_bcs=True)
    
    # # Nonlinear
    # def BRRT_stat_nlin_pp(t,vlast,v): return BRRT(t, v, reach_values_nlin_1, reach_values_nlin_2, times, target_values_1(t), target_values_2(t))
    # reach_values_stat, BRRT_fig = solveplot(diffgame_nlin, init_values_lam(0.), BRRT_stat_nlin_pp, tf=4., ntimes=ntimes, title="Canoe BRRT (Linear)", plot_bcs=True)

# %%

def solve_BRRTpanel(diffgame, init_values_lam, v1sol, v2sol, lams, target_value_fn_1, target_value_fn_2, ti=0., tf=TF, ntimes=ntimes):
    
    times = -np.linspace(ti, tf, ntimes)
    values_lams = [v1sol.copy() for _ in lams]

    for li, lam in enumerate(lams):

        lam_fixed = lam
        target_fn_1 = target_value_fn_1
        target_fn_2 = target_value_fn_2
        reach_sol_1 = v1sol.copy()
        reach_sol_2 = v2sol.copy()
        timez=times.copy()

        def BRRT_stat_pp_lam(t,vlast,v): 
            return BRRT(t, v, reach_sol_1, reach_sol_2, timez, target_fn_1(t), target_fn_2(t), lam=lam_fixed)

        solver_sets = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=BRRT_stat_pp_lam)

        ## Solve
        values = hj.solve(solver_sets, diffgame, grid, times, init_values_lam(lam).copy())
        values_lams[li] = values

    return values_lams, times

# %%

def plot_BRRTpanel(values_lams, lams, times, i, target_values_1, target_values_2, vabs=0.075, xlims=(lbs[0], ubs[0]), ylims=(lbs[1], ubs[1])):

    # times = np.linspace(ti, tf, ntimes)
    vmin, vmax = -vabs, vabs
    levels = np.linspace(vmin, vmax)
    ti = times[i]

    # Scaled colormap
    cmap_name = "RdBu_r"
    n_bins_high = round(256 * vmax/(vmax - vmin))
    scaled_colors = np.vstack((mpl.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high)), mpl.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high))))
    RdWhBl_vscaled = mpl.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    plt.rcParams['text.usetex'] = False

    title="BRRT:  " + r"$V(x,t) = \sup\,_{d} \inf\,_{u} \: \max\{\min_{\tau\in[t,T]} \: R_1(\mathsf{x}(\tau)) + \lambda, \min_{\tau'\in[t,T]} \: R_2 (\mathsf{x}(\tau'))\} - max\{λ, 0\}$"

    for ax, li in zip(axes, range(len(values_lams))):

        plot_values = values_lams[li][i, ...].T

        ## Plot Value
        ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=levels, extend="both", cmap=RdWhBl_vscaled)
        if plot_values.min() < 0. < plot_values.max():
            ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=0, colors="black", linewidths=3)
        
        ## Plot Zero Contours
        ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_1(ti).T, levels=0, colors="black", linewidths=3)
        ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], target_values_2(ti).T, levels=0, colors="black", linewidths=3)

        ## Finish Plot
        ax.set_title(f"t = -{ti:2.2f}, λ={lams[li]:2.2f}")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ann_fontsize = 20
        xy=(-1., -1.4)
        fontweight='bold'
        ha='center'
        if li == 0:
            ax.annotate("BRT 1", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 2:
            ax.annotate("BRRT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 4:
            ax.annotate("BRT 2", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        ax.set_aspect('equal')

    # plt.tight_layout()
    # plt.tight_layout(rect=[0.0, 0.0, 0., 0.7])
    fig.suptitle(title, fontsize=20)
    # plt.show()

    return fig

# %%

def anim_BRRTpanel(values_lams, lams, times, i, target_values_1, target_values_2, bounce=False, vabs=0.075, xlims=(lbs[0], ubs[0]), ylims=(lbs[1], ubs[1])):

    # times = np.linspace(ti, tf, ntimes)
    vmin, vmax = -vabs, vabs
    levels = np.linspace(vmin, vmax)
    ti = times[i]

    # Scaled colormap
    cmap_name = "RdBu_r"
    n_bins_high = round(256 * vmax/(vmax - vmin))
    scaled_colors = np.vstack((mpl.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high)), mpl.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high))))
    RdWhBl_vscaled = mpl.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    plt.rcParams['text.usetex'] = False

    title="BRRT:  " + r"$V(x,t) = \sup\,_{d} \inf\,_{u} \: \max\{\min_{\tau\in[t,T]} \: R_1(\mathsf{x}(\tau)) + \lambda, \min_{\tau'\in[t,T]} \: R_2 (\mathsf{x}(\tau'))\} - max\{λ, 0\}$"

    def draw_panel(ax, li, i, ti, levels, RdWhBl_vscaled):
        # Draw the panel for one subplot
        plot_values = values_lams[li][i, ...].T
        ax.set_title(f"t = -{ti:2.2f}, λ={lams[li]:2.2f}")

        # Plot Value
        ax.contourf(grid.coordinate_vectors[0],
                    grid.coordinate_vectors[1],
                    plot_values,
                    levels=levels,
                    extend="both",
                    cmap=RdWhBl_vscaled)
        
        if plot_values.min() < 0. < plot_values.max():
            ax.contour(grid.coordinate_vectors[0], 
                       grid.coordinate_vectors[1], 
                       plot_values, 
                       levels=0, 
                       colors="black", 
                       linewidths=3)
        
        # Plot Zero Contours
        ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_values_1(ti).T,
                levels=0,
                colors="black",
                linewidths=3)
        ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                target_values_2(ti).T,
                levels=0,
                colors="black",
                linewidths=3)

        # Annotations
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ann_fontsize = 20
        xy = (-0.0, -0.4)
        fontweight = 'bold'
        ha = 'center'
        if li == 0:
            ax.annotate("BRT 1", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 2:
            ax.annotate("BRRT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 4:
            ax.annotate("BRT 2", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        ax.set_aspect('equal')

    def update_frame(i):
        # Clear the figure if you're redrawing everything
        for ax in axes:  
            ax.cla()
        
        ti = times[i]
        # Optionally, recompute any values that depend on time
        for ax, li in zip(axes, range(len(values_lams))):
            draw_panel(ax, li, i, ti, levels, RdWhBl_vscaled)
        
        fig.suptitle(title, fontsize=20)

    if not bounce:
        ani = animation.FuncAnimation(fig, update_frame, frames=len(times), interval=200)
    else:
        N = len(times)
        bounce_frames = list(range(N)) + list(range(N-2, -1, -1)) + list(range(N))
        ani = animation.FuncAnimation(fig, update_frame, frames=bounce_frames, interval=200)

    return ani

# %%

## Ball - BRAAT lambda panel
lams=[-1., -0.2, 0., 0.2, 1.]

## Linear
BRRT_values_lam_lin, times = solve_BRRTpanel(diffgame_lin, init_values_lam, reach_values_1, reach_values_2, lams, target_values_1, target_values_2) 

# ## Nonlnear
# BRAAT_values_lam_nlin, times = solve_BRAATpanel(diffgame_nlin, init_values_reachavoid_ball_lam, BAT_values_nlin_ball_full, lams, reach_values, avoid_values_ball)

## Plot
BRRT_lam_lin_fig = plot_BRRTpanel(BRRT_values_lam_lin, lams, times, -1, target_values_1, target_values_2)
# BRRT_lam_nlin_fig = plot_BRRTpanel(BRRT_values_lam_nlin, lams, times, -1)

# %%

## Two-Target - BRRT lambda panel anim

## Linear
BRRT_lam_lin_stat_anim = anim_BRRTpanel(BRRT_values_lam_lin, lams, times, -1, target_values_1, target_values_2, bounce=True)
# HTML(BRRT_lam_lin_stat_anim.to_jshtml())

# BRRT_lam_lin_stat_anim.save('BRRT_lam_lin_stat_anim.mp4', writer='ffmpeg', fps=25)

# ## Nonlinear
# BRAAT_lam_nlin_anim = anim_BRRTpanel(BRAAT_values_lam_nlin, lams, times, -1, bounce=True)
# HTML(BRAAT_lam_nlin_anim.to_jshtml())

# # BRAAT_lam_nlin_anim.save('BRAAT_lam_ball_nlin_anim.mp4', writer='ffmpeg', fps=25)

# %% 

## Target 1 (TV) - BRT 
def BRT1_tv_pp(t,vlast,v): return BRT(t, v, target_values_1_tv)

# Linear
reach_values_1_tv, BRT_fig_1_tv = solveplot(diffgame_lin, init_values_1_tv, BRT1_tv_pp, target_values_1_=target_values_1_tv, ntimes=ntimes, title="Canoe BRT 1 TV (Linear)")

# Nonlinear
# BRT_values, BRT_fig = solveplot(diffgame_nlin, init_values_reach, BRT_pp, title="Canoe BRT 1 (Nonlinear)", plot_reach_solid=False, plot_rBC=True)

# %% 

## Two-Target (TV) - BRRT
def BRRT_tv_pp(t,vlast,v): return BRRT(t, v, reach_values_1_tv, reach_values_2, times, target_values_1_tv(t), target_values_2(t))

# Linear
reach_values_tv, BRRT_fig = solveplot(diffgame_lin, init_values_lam_tv(0.), BRRT_tv_pp, target_values_1_=target_values_1_tv, ntimes=ntimes, title="Canoe BRRT TV (Linear)", plot_bcs=True)

# Nonlinear
# BRAT_values, BRAT_fig = solveplot(diffgame_nlin, init_values, BRAT_ball_pp, title="Canoe BRAT Ball (Nonlinear)", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

# %%

## Two-Target (TV) - BRRT lambda panel
lams=[-1., -0.2, 0., 0.2, 1.]

## Linear
BRRT_values_lam_lin_tv, times = solve_BRRTpanel(diffgame_lin, init_values_lam_tv, reach_values_1_tv, reach_values_2, lams, target_values_1_tv, target_values_2) 

# ## Nonlnear
# BRAAT_values_lam_nlin_tv, times = solve_BRAATpanel(diffgame_nlin, init_values_reachavoid_ball_lam, BAT_values_nlin_ball_full, lams, reach_values, avoid_values_ball)

## Plot
BRRT_lam_lin_tv_fig = plot_BRRTpanel(BRRT_values_lam_lin_tv, lams, times, -1, target_values_1_tv, target_values_2)
# BRRT_lam_nlin_fig = plot_BRRTpanel(BRRT_values_lam_nlin, lams, times, -1)

# %%

## Two-Target (TV) - BRRT lambda panel anim

## Linear
BRRT_lam_lin_tv_anim = anim_BRRTpanel(BRRT_values_lam_lin_tv, lams, times, -1, target_values_1_tv, target_values_2, bounce=True)
# HTML(BRRT_lam_lin_tv_anim.to_jshtml())

BRRT_lam_lin_tv_anim.save('BRRT_lam_lin_tv2_anim.mp4', writer='ffmpeg', fps=25)

# ## Nonlinear
# BRAAT_lam_nlin_anim = anim_BRRTpanel(BRAAT_values_lam_nlin, lams, times, -1, bounce=True)
# HTML(BRAAT_lam_nlin_anim.to_jshtml())

# # BRAAT_lam_nlin_anim.save('BRAAT_lam_ball_nlin_anim.mp4', writer='ffmpeg', fps=25)

# %%
