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


class Linear2D(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, 
                a11=0., a12=.5, a21=-1., a22=-1.,
                b1=.4, b2=.1, c1=0., c2=.1,
                max_u=0.5,
                max_d=0.3,
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

        self.a11, self.a12, self.a21, self.a22 = a11, a12, a21, a22
        self.b1, self.b2, self.c1, self.c2 = b1, b2, c1, c2

    def open_loop_dynamics(self, x, time):
        xdot1 = self.a11 * x[0] + self.a12 * x[1]
        xdot2 = self.a21 * x[0] + self.a22 * x[1]
        return jnp.array([xdot1, xdot2])
    
    def control_jacobian(self, state, time):
        return jnp.array([
            [self.b1],
            [self.b2],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [self.c1],
            [self.c2],
        ])
    

class LessLinear2D(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, 
                a11=0., a12=.5, a21=-1., a22=-1.,
                b1=.4, b2=.1, c1=0., c2=.1,
                max_u=0.5,
                max_d=0.3,
                control_mode="min",
                disturbance_mode="max",
                control_space=None,
                disturbance_space=None,
                gamma=0.,
                mu=0.,
                alpha=1.
                ):

        if control_space is None:
            control_space = hj.sets.Box(lo=jnp.array([-max_u]), hi=jnp.array([max_u]))
    
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(lo=jnp.array([-max_d]), hi=jnp.array([max_d]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

        self.a11, self.a12, self.a21, self.a22 = a11, a12, a21, a22
        self.b1, self.b2, self.c1, self.c2 = b1, b2, c1, c2
        self.gamma, self.mu, self.alpha = gamma, mu, alpha

    def open_loop_dynamics(self, x, time):
        xdot1 = self.a11 * x[0] + self.a12 * x[1] + self.mu * jnp.sin(self.alpha * x[0]) * x[1] * x[1]
        xdot2 = self.a21 * x[0] + self.a22 * x[1] - self.gamma * x[1] * x[0] * x[0]
        return jnp.array([xdot1, xdot2])
    
    def control_jacobian(self, state, time):
        return jnp.array([
            [self.b1],
            [self.b2],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [self.c1],
            [self.c2],
        ])