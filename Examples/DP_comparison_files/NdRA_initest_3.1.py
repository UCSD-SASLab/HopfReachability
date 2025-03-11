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

## FIXED PARAMS
IH = 2 # Initial Height
G = 1 # Gravity
TC = np.sqrt(2 * IH / G) # Critical Time of Landing


# %%

class Conveyor(dynamics.ControlAndDisturbanceAffineDynamics):

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
        return jnp.concatenate([jnp.array([1]),
                                self.alpha * jnp.cos(5. * state[0]) * jnp.ones(self.N)])

    def control_jacobian(self, state, time):
        return self.z1oN.T # all but shared controlled

    def disturbance_jacobian(self, state, time):
        return self.z1oN.T # all but shared disturbed
        # return self.oN.T # all disturbed

# %% POST-PROCESSORS

def BRT(t, v, reach_values): return jnp.minimum(v, reach_values(t))

def BAT(t, v, avoid_values): return jnp.maximum(v, avoid_values(t))

def BRAT(t, v, reach_values, avoid_values, lam): return jnp.maximum(jnp.minimum(v, reach_values(t)+lam), avoid_values(t))

## linear ITP not working
# def BRAAT(t, v, vA, times, rBC, aBC, lam=0.):
#     i = jnp.argmin(jnp.abs(t - times)) # nearest ix
#     j = i + jnp.sign((t - times)[i]).astype(int) # across nearest-gap ix
#     vA_itp = vA[i, ...] + jnp.sign(i - j) * (vA[i, ...] - vA[j, ...]) / jnp.abs((t - times)[i])
#     return jnp.minimum(jnp.maximum(v, aBC), jnp.maximum(rBC+lam, vA_itp)) - jnp.maximum(lam, 0)

## nearest neighbor 
def BRAAT(t, v, vA, times, rBC, aBC, lam=0.):
    i = jnp.argmin(jnp.abs(t - times)) # nearest ix
    return jnp.minimum(jnp.maximum(v, aBC - jnp.maximum(lam, 0)), jnp.maximum(rBC+lam, vA[i,...]) - jnp.maximum(lam, 0))

# Ultimately these need to be wrapped, e.g.
# def vpp(t, v): return BRAAT_value_postprocessor(t, v, values_avoid, times, reach_values(t), avoid_values(t), lam)
    
# In[27]: INIT

u_bd, d_bd = 0.5, 0.2
diffgame_lin = Conveyor(u_bd=u_bd, d_bd=d_bd, alpha=0., control_mode="min", disturbance_mode="max")
diffgame_nlin = Conveyor(u_bd=u_bd, d_bd=d_bd, alpha=1., control_mode="min", disturbance_mode="max")

## params
N = 1
width_half = 1.75
eps = 0.5
ubs = np.concatenate(([eps], width_half * np.ones(N))) 
lbs = np.concatenate(([-(IH + 2*eps)], -width_half * np.ones(N)))
grid_L = 501

grid_scale=2
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(grid_scale*lbs, grid_scale*ubs), [grid_L for _ in range(N+1)])

## Reach Ball (static)
bc_alpha = 0.5
rReach = 0.3
def reach_values(t): return bc_alpha * jnp.tanh((jnp.linalg.norm(grid.states[..., :], axis=-1) - rReach))

## Avoid Ball (static)
rAvoid = 0.3
c = jnp.array([-1.0, 0.])
def avoid_values_ball(t): return bc_alpha * -jnp.tanh((jnp.linalg.norm(jnp.subtract(grid.states[..., :2], c), axis=-1) - rAvoid))

## Multi Rectangle Obstacle (moves)
def ci(i): return jnp.array([-0.5 * (i+1), 0.])
diag = jnp.array([5., 1.])
freq = (2.*jnp.pi/2.)
def path(time, phase): return jnp.cos(freq * time + phase)
phases = [0., jnp.pi, 0.] # staggered
def avoid_values_axes(t): return bc_alpha * -jnp.tanh(jnp.minimum(jnp.minimum(
                            jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], ci(0) + jnp.array([0., path(t, phases[0])])))), axis=-1)) - rAvoid, 
                            jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], ci(1) + jnp.array([0., path(t, phases[1])])))), axis=-1)) - rAvoid),
                            jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], ci(2) + jnp.array([0., path(t, phases[2])])))), axis=-1)) - rAvoid))

# def avoid_values_axes(t): return bc_alpha * -jnp.tanh(jnp.array(jnp.max(jnp.abs(jnp.multiply(diag, jnp.subtract(grid.states[..., :2], ci(0) + jnp.array([0., path(t, phases[0])])))), axis=-1)) - rAvoid)

## Define BC with post-processors (with values = bc(0.))
init_values_reach = BRT(0., reach_values(0.), reach_values)

init_values_avoid_ball = BAT(0., avoid_values_ball(0.), avoid_values_ball)
init_values_avoid_axes = BAT(0., avoid_values_axes(0.), avoid_values_axes)

init_values_reachavoid_ball = BRAT(0., reach_values(0.), reach_values, avoid_values_ball, 0.)
init_values_reachavoid_axes = BRAT(0., reach_values(0.), reach_values, avoid_values_axes, 0.)

def init_values_reachavoid_ball_lam(lam): return BRAT(0., reach_values(0.), reach_values, avoid_values_ball, lam)
def init_values_reachavoid_axes_lam(lam): return BRAT(0., reach_values(0.), reach_values, avoid_values_axes, lam)

# %% 

# SOLVER

def solveplot(diffgame, init_values, post_processor, title="", tf=TC, ntimes=5, 
                reach_values=reach_values, avoid_values=avoid_values_ball, progress_bar=True,
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

## Ball - BRT 
def BRT_pp(t,vlast,v): return BRT(t, v, reach_values)

# Linear
BRT_values, BRT_fig = solveplot(diffgame_lin, init_values_reach, BRT_pp, title="Conveyor BRT (Linear)", plot_reach_solid=False, plot_rBC=True)

# Nonlinear
BRT_values, BRT_fig = solveplot(diffgame_nlin, init_values_reach, BRT_pp, title="Conveyor BRT (Nonlinear)", plot_reach_solid=False, plot_rBC=True)

# %% 

## Ball - BAT 
def BAT_ball_pp(t,vlast,v): return BAT(t, v, avoid_values_ball)

# Linear
BAT_values, BAT_fig = solveplot(diffgame_lin, init_values_avoid_ball, BAT_ball_pp, title="Conveyor BAT Ball (Linear)", plot_avoid_solid=True, plot_aBC=True, avoid_game=True)

# Nonlinear
BAT_values, BAT_fig = solveplot(diffgame_nlin, init_values_avoid_ball, BAT_ball_pp, title="Conveyor BAT Ball (Nonlinear)", plot_avoid_solid=True, plot_aBC=True, avoid_game=True)

# %% 

## Ball - BRAT
def BRAT_ball_pp(t,vlast,v): return BRAT(t,v, reach_values, avoid_values_ball, 0.)

# Linear
BRAT_values, BRAT_fig = solveplot(diffgame_lin, init_values_reachavoid_ball, BRAT_ball_pp, title="Conveyor BRAT Ball (Linear)", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

# Nonlinear
BRAT_values, BRAT_fig = solveplot(diffgame_nlin, init_values_reachavoid_ball, BRAT_ball_pp, title="Conveyor BRAT Ball (Nonlinear)", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

# %%

## Ball - BAT (full) for BRAAT

ntimes = 100
times = -np.linspace(0., TC, ntimes)

BAT_values_lin_ball_full, BAT_fig = solveplot(diffgame_lin, init_values_avoid_ball, BAT_ball_pp, title="Conveyor BAT Ball (Linear)", plot_avoid_solid=False, plot_no_value=False, plot_aBC=True, avoid_game=True, one_shot=True, ntimes=ntimes)
BAT_values_nlin_ball_full, BAT_nlin_fig = solveplot(diffgame_nlin, init_values_avoid_ball, BAT_ball_pp, title="Conveyor BAT Ball (Nonlinear)", plot_avoid_solid=False, plot_no_value=False, plot_aBC=True, avoid_game=True, one_shot=True, ntimes=ntimes)

# %%

## Ball - BRAAT lambda variations

for lam in [0., -0.25, 0.1]:
    
    # Linear
    def BRAAT_ball_lin_l_pp(t,vlast,v): return BRAAT(t, v, BAT_values_lin_ball_full, times, reach_values(t), avoid_values_ball(t), lam)
    BRAAT_values, BRAAT_fig = solveplot(diffgame_lin, init_values_reachavoid_ball_lam(lam), BRAAT_ball_lin_l_pp, title=f"Conveyor BRAAT Ball - Lambda {lam:2.1f} (Linear)", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)
    
    # Nonlinear
    def BRAAT_ball_nlin_l_pp(t,vlast,v): return BRAAT(t, v, BAT_values_nlin_ball_full, times, reach_values(t), avoid_values_ball(t), lam)
    BRAAT_values_nlin, BRAAT_fig = solveplot(diffgame_nlin, init_values_reachavoid_ball_lam(lam), BRAAT_ball_lin_l_pp, title=f"Conveyor BRAAT Ball - Lambda {lam:2.1f} (Nonlinear)", plot_reach_solid=False, plot_avoid_solid=False, plot_bcs=True, plot_no_value=False)

# %%

def solve_BRAATpanel(diffgame, init_values_lam, vAsol, lams, reach_value_fn, avoid_value_fn, ti=0., tf=TC, ntimes=100):
    
    times = -np.linspace(ti, tf, ntimes)
    values_lams = [vAsol.copy() for _ in lams]

    for li, lam in enumerate(lams):

        lam_fixed = lam
        reach_fn = reach_value_fn
        avoid_fn = avoid_value_fn
        avoid_sol = vAsol.copy()
        timez=times.copy()

        def BRAAT_p(t, v, vA, times, rBC, aBC, lam=0.):
            i = jnp.argmin(jnp.abs(t - times)) # nearest ix
            return jnp.minimum(jnp.maximum(v, aBC - jnp.maximum(lam, 0)), jnp.maximum(rBC+lam, vA[i,...]) - jnp.maximum(lam, 0))

        # BRAAT_post_processor = lambda t,v: BRAAT(t, v, vA, times, reach_values(t), avoid_values(t), lam)
        def BRAAT_post_processor(t, vlast, v, timez=timez, avoid_sol=avoid_sol, reach_fn=reach_fn, avoid_fn=avoid_fn, lamb=lam_fixed):
            return BRAAT_p(t, v, avoid_sol, timez, reach_fn(t), avoid_fn(t), lamb)
        
        solver_sets = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=BRAAT_post_processor)

        ## Solve
        values = hj.solve(solver_sets, diffgame, grid, times, init_values_lam(lam).copy())
        values_lams[li] = values

    return values_lams, times

# %%

def plot_BRAATpanel(values_lams, lams, times, i, reach_values=reach_values, avoid_values=avoid_values_ball, vabs=0.075, xlims=(lbs[0], ubs[0]), ylims=(lbs[1], ubs[1])):

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

    title="BRAAT:  " + r"$V(x,t) = \sup\,_{d} \inf\,_{u} \: \max\{\min_{\tau\in[t,T]} \: R(\mathsf{x}(\tau)) + \lambda, \max_{\tau'\in[t,T]}\: A(\mathsf{x}(\tau'))\}- max\{λ, 0\}$"

    for ax, li in zip(axes, range(len(values_lams))):

        plot_values = values_lams[li][i, ...].T

        ## Plot Value
        ax.contourf(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=levels, extend="both", cmap=RdWhBl_vscaled)
        ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], plot_values, levels=0, colors="black", linewidths=3)
        
        ## Plot Zero Contours
        ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], reach_values(ti).T, levels=0, colors="black", linewidths=3)
        ax.contour(grid.coordinate_vectors[0], grid.coordinate_vectors[1], avoid_values(ti).T, levels=0, colors="black", linewidths=3)

        ## Finish Plot
        ax.set_title(f"t = -{ti:2.2f}, λ={lams[li]:2.1f}")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ann_fontsize = 20
        xy=(-1., -1.4)
        fontweight='bold'
        ha='center'
        if li == 0:
            ax.annotate("BAT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 2:
            ax.annotate("BRAAT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 4:
            ax.annotate("BRT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        ax.set_aspect('equal')

    # plt.tight_layout()
    # plt.tight_layout(rect=[0.0, 0.0, 0., 0.7])
    fig.suptitle(title, fontsize=20)
    # plt.show()

    return fig

# %%

def anim_BRAATpanel(values_lams, lams, times, i, reach_values=reach_values, avoid_values=avoid_values_ball, bounce=False, vabs=0.075):

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

    title="BRAAT:  " + r"$V(x,t) = \sup\,_{d} \inf\,_{u} \: \max\{\min_{\tau\in[t,T]} \: R(\mathsf{x}(\tau)) + \lambda, \max_{\tau'\in[t,T]}\: A(\mathsf{x}(\tau'))\}- max\{λ, 0\}$"

    def draw_panel(ax, li, i, ti, levels, RdWhBl_vscaled):
        # Draw the panel for one subplot
        plot_values = values_lams[li][i, ...].T
        ax.set_title(f"t = -{ti:2.2f}, λ={lams[li]:2.1f}")

        # Plot Value
        ax.contourf(grid.coordinate_vectors[0],
                    grid.coordinate_vectors[1],
                    plot_values,
                    levels=levels,
                    extend="both",
                    cmap=RdWhBl_vscaled)
        
        # Plot Zero Contours
        ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                reach_values(ti).T,
                levels=0,
                colors="black",
                linewidths=3)
        ax.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                avoid_values(ti).T,
                levels=0,
                colors="black",
                linewidths=3)

        # Annotations
        ann_fontsize = 20
        xy = (-1., -1.4)
        fontweight = 'bold'
        ha = 'center'
        if li == 0:
            ax.annotate("BAT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 2:
            ax.annotate("BRAAT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
        elif li == 4:
            ax.annotate("BRT", xy=xy, ha=ha, fontweight=fontweight, fontsize=ann_fontsize)
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
lams=[-1., -0.2, 0., 0.1, 1.]

## Linear
BRAAT_values_lam_lin, times = solve_BRAATpanel(diffgame_lin, init_values_reachavoid_ball_lam, BAT_values_lin_ball_full, lams, reach_values, avoid_values_ball) 

## Nonlnear
BRAAT_values_lam_nlin, times = solve_BRAATpanel(diffgame_nlin, init_values_reachavoid_ball_lam, BAT_values_nlin_ball_full, lams, reach_values, avoid_values_ball)

## Plot
BRAAT_lam_lin_fig = plot_BRAATpanel(BRAAT_values_lam_lin, lams, times, -1)
BRAAT_lam_nlin_fig = plot_BRAATpanel(BRAAT_values_lam_nlin, lams, times, -1)

# %%

## Ball - BRAAT lambda panel anim

# ## Linear
# BRAAT_lam_lin_anim = anim_BRAATpanel(BRAAT_values_lam_lin, lams, times, -1, bounce=True)
# HTML(BRAAT_lam_lin_anim.to_jshtml())

# # BRAAT_lam_lin_anim.save('BRAAT_lam_ball_lin_anim.mp4', writer='ffmpeg', fps=25)

# ## Nonlinear
# BRAAT_lam_nlin_anim = anim_BRAATpanel(BRAAT_values_lam_nlin, lams, times, -1, bounce=True)
# HTML(BRAAT_lam_nlin_anim.to_jshtml())

# # BRAAT_lam_nlin_anim.save('BRAAT_lam_ball_nlin_anim.mp4', writer='ffmpeg', fps=25)

# %% 

## Axes - BAT (full) for BRAAT

def BAT_axes_pp(t,vlast,v): return BAT(t, v, avoid_values_axes)
BAT_values_lin_axes_full, BAT_axe_fig = solveplot(diffgame_lin, init_values_avoid_axes, BAT_axes_pp, title="Conveyor BAT Axes (Linear)", plot_avoid_solid=False, plot_no_value=False, plot_aBC=True, avoid_game=True, one_shot=True, ntimes=100, avoid_values=avoid_values_axes, offset=0)
BAT_values_nlin_axes_full, BAT_axe_fig = solveplot(diffgame_nlin, init_values_avoid_axes, BAT_axes_pp, title="Conveyor BAT Axes (Nonlinear)", plot_avoid_solid=False, plot_no_value=False, plot_aBC=True, avoid_game=True, one_shot=True, ntimes=100, avoid_values=avoid_values_axes, offset=0)

# %% 

## Axes - BRAAT/BRAT

for lam in [-1., 0., 1.]:

    # Linear
    def BRAAT_axes_lin_l_pp(t,vlast,v): return BRAAT(t, v, BAT_values_lin_axes_full, times, reach_values(t), avoid_values_axes(t), lam)
    BRAAT_values_lin_axes, BRAAT_fig = solveplot(diffgame_lin, init_values_reachavoid_axes_lam(lam), BRAAT_axes_lin_l_pp, title=f"Conveyor BRAAT Ball - Lambda {lam:2.1f} (Linear)", plot_bcs=True, avoid_values=avoid_values_axes)

    # Nonlinear
    def BRAAT_axes_nlin_l_pp(t,vlast,v): return BRAAT(t, v, BAT_values_lin_axes_full, times, reach_values(t), avoid_values_axes(t), lam)
    BRAAT_values_nlin_axes, BRAAT_fig = solveplot(diffgame_nlin, init_values_reachavoid_axes_lam(lam), BRAAT_axes_nlin_l_pp, title=f"Conveyor BRAAT Ball - Lambda {lam:2.1f} (Nonlinear)", plot_bcs=True, avoid_values=avoid_values_axes)

# %% 

## Axes - BRAAT lambda solve

lams=[-1., -0.2, 0., 0.1, 1.]

## Linear
BRAAT_values_axes_lam_lin, times = solve_BRAATpanel(diffgame_lin, init_values_reachavoid_axes_lam, BAT_values_lin_axes_full, lams, reach_values, avoid_values_axes) #, tf=0.5)

## Nonlnear
BRAAT_values_axes_lam_nlin, times = solve_BRAATpanel(diffgame_nlin, init_values_reachavoid_axes_lam, BAT_values_lin_axes_full, lams, reach_values, avoid_values_axes) #, tf=0.5)

## Plot
BRAAT_lam_axes_lin_fig = plot_BRAATpanel(BRAAT_values_axes_lam_lin, lams, times, -1, avoid_values=avoid_values_axes)
BRAAT_lam_axes_nlin_fig = plot_BRAATpanel(BRAAT_values_axes_lam_nlin, lams, times, -1, avoid_values=avoid_values_axes)

# %% 

## Axes - BRAAT lambda panel anim

# ## Linear
# BRAAT_lam_axes_lin_anim = anim_BRAATpanel(BRAAT_values_axes_lam_lin, lams, times, -1, avoid_values=avoid_values_axes, bounce=True)
# HTML(BRAAT_lam_axes_lin_anim.to_jshtml())

# # BRAAT_lam_axes_lin_anim.save('BRAAT_lam_axes_lin_anim.mp4', writer='ffmpeg', fps=25)

# ## Nonlinear
# BRAAT_lam_axes_nlin_anim = anim_BRAATpanel(BRAAT_values_axes_lam_nlin, lams, times, -1, avoid_values=avoid_values_axes, bounce=True)
# HTML(BRAAT_lam_axes_nlin_anim.to_jshtml())

# # BRAAT_lam_axes_nlin_anim.save('BRAAT_lam_axes_nlin_anim.mp4', writer='ffmpeg', fps=25)
