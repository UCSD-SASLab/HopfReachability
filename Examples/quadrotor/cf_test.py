
import sys, os
sys.path.append(os.getcwd() + "/Examples/quadrotor")
sys.path.append("/Users/willsharpless/crazyflie-firmware/build")
import numpy as np
from crazyswarm_model import Quadrotor
from sim_data_types import Action, State
from crazyflie_sil import CrazyflieSIL
import rowan

dt, tf = 0.0005, 10.0
p0 = [0., 0., 0.]

model = Quadrotor(State(p0)) # normally in Backend in CrazyflieServer

ll_ctrl_name = 'mellinger'
uav = CrazyflieSIL("uav", p0, ll_ctrl_name, model.time) # usually self.backend.time
uav2 = CrazyflieSIL("uav", State(p0), ll_ctrl_name, model.time, initialState=State(p0))
u = lambda t: [0., 0., 0., 60000]

for ti in np.linspace(0., tf, 1+int(tf/dt)):
    uav.cmdVelLegacy(*u(ti))            # sets mode & set_point
    action = uav.executeController()    # calls controller, powerDist, pwm_to_rpm
    model.step(action, dt)              # evolves 13D model, uses rown
    uav.setState(model.state)           # updates uav state

model.fullstate()
