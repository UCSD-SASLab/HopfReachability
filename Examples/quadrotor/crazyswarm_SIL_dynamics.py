#!/usr/bin/env python3
"""
Crazyflie Software-In-The-Loop Wrapper that uses the firmware Python bindings.

    2022 - Wolfgang Hönig (TU Berlin)

    2024 - Will Sharpless: combined for simulation sin ROS (this is self-contained except firmware, np and rowan)
      Note, I made a few edits to remove ROS & the Backend class. Also, added a method at the end for sim.
"""
from __future__ import annotations

import numpy as np
# from rclpy.node import Node
# from rclpy.time import Time
# from rosgraph_msgs.msg import Clock
import cffirmware as firm
import rowan

class State:
    """Class that stores the state of a UAV as used in the simulator interface."""

    def __init__(self, pos=np.zeros(3), vel=np.zeros(3), euler=None,
                 quat=np.array([1, 0, 0, 0]), omega=np.zeros(3)):
        # internally use one numpy array
        self._state = np.empty(13)
        self.pos = pos
        self.vel = vel
        self.omega = omega
        if euler is None:
            self.quat = quat
        else:
            self.quat = rowan.from_euler(*euler)            

    @property
    def pos(self):
        """Position [m; world frame]."""
        return self._state[0:3]

    @pos.setter
    def pos(self, value):
        self._state[0:3] = value

    @property
    def vel(self):
        """Velocity [m/s; world frame]."""
        return self._state[3:6]

    @vel.setter
    def vel(self, value):
        self._state[3:6] = value

    @property
    def quat(self):
        """Quaternion [qw, qx, qy, qz; body -> world]."""
        return self._state[6:10]

    @quat.setter
    def quat(self, value):
        self._state[6:10] = value

    @property
    def omega(self):
        """Angular velocity [rad/s; body frame]."""
        return self._state[10:13]

    @omega.setter
    def omega(self, value):
        self._state[10:13] = value

    def __repr__(self) -> str:
        return 'State pos={}, vel={}, quat={}, omega={}'.format(
            self.pos, self.vel, self.quat, self.omega)


class Action:
    """Class that stores the action of a UAV as used in the simulator interface."""

    def __init__(self, rpm):
        # internally use one numpy array
        self._action = np.empty(4)
        self.rpm = rpm

    @property
    def rpm(self):
        """Rotation per second [rpm]."""
        return self._action

    @rpm.setter
    def rpm(self, value):
        self._action = value

    def __repr__(self) -> str:
        return 'Action rpm={}'.format(self.rpm)

# class Backend:
#     """Backend that uses newton-euler rigid-body dynamics implemented in numpy."""

#     def __init__(self, node: Node, names: list[str], states: list[State]):
#         self.node = node
#         self.names = names
#         self.clock_publisher = node.create_publisher(Clock, 'clock', 10)
#         self.t = 0
#         self.dt = 0.0005

#         self.uavs = []
#         for state in states:
#             uav = Quadrotor(state)
#             self.uavs.append(uav)

#     def time(self) -> float:
#         return self.t

#     def step(self, states_desired: list[State], actions: list[Action]) -> list[State]:
#         # advance the time
#         self.t += self.dt

#         next_states = []

#         for uav, action in zip(self.uavs, actions):
#             uav.step(action, self.dt)
#             next_states.append(uav.state)

#         # print(states_desired, actions, next_states)
#         # publish the current clock
#         clock_message = Clock()
#         clock_message.clock = Time(seconds=self.time()).to_msg()
#         self.clock_publisher.publish(clock_message)

#         return next_states

#     def shutdown(self):
#         pass

class Quadrotor:
    """Basic rigid body quadrotor model (no drag) using numpy and rowan."""
    # WAS: added time to shed Backend wrapper

    def __init__(self, state):
        self.t = 0
        # parameters (Crazyflie 2.0 quadrotor)
        self.mass = 0.034  # kg
        # self.J = np.array([
        # 	[16.56,0.83,0.71],
        # 	[0.83,16.66,1.8],
        # 	[0.72,1.8,29.26]
        # 	]) * 1e-6  # kg m^2
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

        # Note: we assume here that our control is forces
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ])
        self.g = 9.81  # not signed

        if self.J.shape == (3, 3):
            self.inv_J = np.linalg.pinv(self.J)  # full matrix -> pseudo inverse
        else:
            self.inv_J = 1 / self.J  # diagonal matrix -> division

        self.state = state

    def step(self, action, dt, f_a=np.zeros(3)):

        self.t += dt

        # convert RPM -> Force
        def rpm_to_force(rpm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [2.55077341e-08, -4.92422570e-05, -1.51910248e-01]
            force_in_grams = np.polyval(p, rpm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        force = rpm_to_force(action.rpm)

        # compute next state
        eta = np.dot(self.B0, force)
        f_u = np.array([0, 0, eta[0]])
        tau_u = np.array([eta[1], eta[2], eta[3]])

        # dynamics
        # dot{p} = v
        pos_next = self.state.pos + self.state.vel * dt
        # mv = mg + R f_u + f_a
        vel_next = self.state.vel + (
            np.array([0, 0, -self.g]) +
            (rowan.rotate(self.state.quat, f_u) + f_a) / self.mass) * dt

        # dot{R} = R S(w)
        # to integrate the dynamics, see
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
        # https://arxiv.org/pdf/1604.08139.pdf
        # Sec 4.5, https://arxiv.org/pdf/1711.02508.pdf
        omega_global = rowan.rotate(self.state.quat, self.state.omega)
        q_next = rowan.normalize(
            rowan.calculus.integrate(
                self.state.quat, omega_global, dt))

        # mJ = Jw x w + tau_u
        omega_next = self.state.omega + (
            self.inv_J * (np.cross(self.J * self.state.omega, self.state.omega) + tau_u)) * dt

        self.state.pos = pos_next
        self.state.vel = vel_next
        self.state.quat = q_next
        self.state.omega = omega_next

        # if we fall below the ground, set velocities to 0
        if self.state.pos[2] < 0:
            self.state.pos[2] = 0
            self.state.vel = [0, 0, 0]
            self.state.omega = [0, 0, 0]

    def time(self) -> float:
        return self.t
    
    def fullstate(self):
        return np.concatenate([self.state.pos, 
                               self.state.vel, 
                               rowan.to_euler(self.state.quat, convention='xyz'), # tried 'xyz' & 'zyx', neither fixes
                               self.state.omega])


class TrajectoryPolynomialPiece:

    def __init__(self, poly_x, poly_y, poly_z, poly_yaw, duration):
        self.poly_x = poly_x
        self.poly_y = poly_y
        self.poly_z = poly_z
        self.poly_yaw = poly_yaw
        self.duration = duration


def copy_svec(v):
    return firm.mkvec(v.x, v.y, v.z)


class CrazyflieSIL:

    # Flight modes.
    MODE_IDLE = 0
    MODE_HIGH_POLY = 1
    MODE_LOW_FULLSTATE = 2
    MODE_LOW_POSITION = 3
    MODE_LOW_VELOCITY = 4

    def __init__(self, name, initialPosition, controller_name, time_func, initialState=None):
        # Core.
        self.name = name
        self.groupMask = 0
        if initialState == None:
            self.initialPosition = np.array(initialPosition)
        else:
            self.initialPosition = initialState.pos
        self.time_func = time_func

        # Commander.
        self.mode = CrazyflieSIL.MODE_IDLE
        self.planner = firm.planner()
        firm.plan_init(self.planner)
        self.trajectories = {}

        # previous state for HL commander
        self.cmdHl_pos = firm.mkvec(*self.initialPosition)
        self.cmdHl_vel = firm.vzero()
        self.cmdHl_yaw = 0
        # FIXME: WAS: fix for full initial state (doesn't matter w LL ctrl)

        # current setpoint
        self.setpoint = firm.setpoint_t()

        # latest sensor values.
        self.state = firm.state_t()
        self.state.position.x = self.initialPosition[0]
        self.state.position.y = self.initialPosition[1]
        self.state.position.z = self.initialPosition[2]
        self.state.velocity.x = 0
        self.state.velocity.y = 0
        self.state.velocity.z = 0
        self.state.attitude.roll = 0
        self.state.attitude.pitch = -0  # WARNING: this is in the legacy coordinate system
        self.state.attitude.yaw = 0

        self.sensors = firm.sensorData_t()
        self.sensors.gyro.x = 0
        self.sensors.gyro.y = 0
        self.sensors.gyro.z = 0

        if initialState != None:
            self.setState(initialState) # WAS: allow full initial state

        # current controller output
        self.control = firm.control_t()
        self.motors_thrust_uncapped = firm.motors_thrust_uncapped_t()
        self.motors_thrust_pwm = firm.motors_thrust_pwm_t()

        self.controller_name = controller_name

        # set up controller
        if controller_name == 'none':
            self.controller = None
        elif controller_name == 'pid':
            firm.controllerPidInit()
            self.controller = firm.controllerPid
        elif controller_name == 'mellinger':
            self.mellinger_control = firm.controllerMellinger_t()
            firm.controllerMellingerInit(self.mellinger_control)
            self.controller = firm.controllerMellinger
        elif controller_name == 'brescianini':
            firm.controllerBrescianiniInit()
            self.controller = firm.controllerBrescianini
        else:
            raise ValueError('Unknown controller {}'.format(controller_name))

    def setGroupMask(self, groupMask):
        self.groupMask = groupMask

    def takeoff(self, targetHeight, duration, groupMask=0):
        if self._isGroup(groupMask):
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            targetYaw = 0.0
            firm.plan_takeoff(
                self.planner,
                self.cmdHl_pos,
                self.cmdHl_yaw,
                targetHeight, targetYaw, duration, self.time_func())

    def land(self, targetHeight, duration, groupMask=0):
        if self._isGroup(groupMask):
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            targetYaw = 0.0
            firm.plan_land(
                self.planner,
                self.cmdHl_pos,
                self.cmdHl_yaw,
                targetHeight, targetYaw, duration, self.time_func())

    # def stop(self, groupMask = 0):
    #     if self._isGroup(groupMask):
    #         self.mode = CrazyflieSIL.MODE_IDLE
    #         firm.plan_stop(self.planner)

    def goTo(self, goal, yaw, duration, relative=False, groupMask=0):
        if self._isGroup(groupMask):
            if self.mode != CrazyflieSIL.MODE_HIGH_POLY:
                # We need to update to the latest firmware that has go_to_from.
                raise ValueError('goTo from low-level modes not yet supported.')
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            firm.plan_go_to(
                self.planner,
                relative,
                firm.mkvec(*goal),
                yaw, duration, self.time_func())

    def uploadTrajectory(self,
                         trajectoryId: int,
                         pieceOffset: int,
                         pieces: list[TrajectoryPolynomialPiece]):
        traj = firm.piecewise_traj()
        traj.t_begin = 0
        traj.timescale = 1.0
        traj.shift = firm.mkvec(0, 0, 0)
        traj.n_pieces = len(pieces)
        traj.pieces = firm.poly4d_malloc(traj.n_pieces)
        for i, piece in enumerate(pieces):
            fwpiece = firm.piecewise_get(traj, i)
            fwpiece.duration = piece.duration
            for coef in range(0, 8):
                firm.poly4d_set(fwpiece, 0, coef, piece.poly_x[coef])
                firm.poly4d_set(fwpiece, 1, coef, piece.poly_y[coef])
                firm.poly4d_set(fwpiece, 2, coef, piece.poly_z[coef])
                firm.poly4d_set(fwpiece, 3, coef, piece.poly_yaw[coef])
        self.trajectories[trajectoryId] = traj

    def startTrajectory(self,
                        trajectoryId: int,
                        timescale: float = 1.0,
                        reverse: bool = False,
                        relative: bool = True,
                        groupMask: int = 0):
        if self._isGroup(groupMask):
            self.mode = CrazyflieSIL.MODE_HIGH_POLY
            traj = self.trajectories[trajectoryId]
            traj.t_begin = self.time_func()
            traj.timescale = timescale
            startfrom = self.cmdHl_pos
            firm.plan_start_trajectory(self.planner, traj, reverse, relative, startfrom)

    # def notifySetpointsStop(self, remainValidMillisecs=100):
    #     # No-op - the real Crazyflie prioritizes streaming setpoints over
    #     # high-level commands. This tells it to stop doing that. We don't
    #     # simulate this behavior.
    #     pass

    def cmdVelLegacy(self, roll, pitch, yawrate, thrust): 
        self.mode = CrazyflieSIL.MODE_LOW_VELOCITY
        self.setpoint.attitude.roll = roll
        # self.setpoint.attitude.pitch = -pitch
        self.setpoint.attitude.pitch = pitch
        self.setpoint.attitudeRate.yaw = yawrate  # rad/s -> deg/s
        self.setpoint.thrust = thrust
        
        self.setpoint.mode.x = firm.modeDisable
        self.setpoint.mode.y = firm.modeDisable
        self.setpoint.mode.z = firm.modeDisable
        self.setpoint.mode.quat = firm.modeDisable
        self.setpoint.mode.roll = firm.modeAbs
        self.setpoint.mode.pitch = firm.modeAbs
        self.setpoint.mode.yaw = firm.modeVelocity

    def cmdFullState(self, pos, vel, acc, yaw, omega):
        self.mode = CrazyflieSIL.MODE_LOW_FULLSTATE
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.attitude.yaw = np.degrees(yaw)
        self.setpoint.attitudeRate.roll = np.degrees(omega[0])
        self.setpoint.attitudeRate.pitch = np.degrees(omega[1])
        self.setpoint.attitudeRate.yaw = np.degrees(omega[2])
        self.setpoint.mode.x = firm.modeAbs
        self.setpoint.mode.y = firm.modeAbs
        self.setpoint.mode.z = firm.modeAbs
        self.setpoint.mode.roll = firm.modeDisable
        self.setpoint.mode.pitch = firm.modeDisable
        self.setpoint.mode.yaw = firm.modeAbs
        self.setpoint.mode.quat = firm.modeDisable
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.cmdHl_pos = copy_svec(self.setpoint.position)
        self.cmdHl_vel = copy_svec(self.setpoint.velocity)
        self.cmdHl_yaw = yaw

    # def cmdPosition(self, pos, yaw = 0):
    #     self.mode = CrazyflieSIL.MODE_LOW_POSITION
    #     self.setState.pos = firm.mkvec(*pos)
    #     self.setState.yaw = yaw
    #     # TODO: should we set vel, acc, omega to zero, or rely on modes to not read them?

    # def cmdVelocityWorld(self, vel, yawRate):
    #     self.mode = CrazyflieSIL.MODE_LOW_VELOCITY
    #     self.setState.vel = firm.mkvec(*vel)
    #     self.setState.omega = firm.mkvec(0.0, 0.0, yawRate)
    #     # TODO: should we set pos, acc, yaw to zero, or rely on modes to not read them?

    # def cmdStop(self):
    #     # TODO: set mode to MODE_IDLE?
    #     pass

    def getSetpoint(self):
        if self.mode == CrazyflieSIL.MODE_HIGH_POLY:
            # See logic in crtp_commander_high_level.c
            ev = firm.plan_current_goal(self.planner, self.time_func())
            if firm.is_traj_eval_valid(ev):
                self.setpoint.position.x = ev.pos.x
                self.setpoint.position.y = ev.pos.y
                self.setpoint.position.z = ev.pos.z
                self.setpoint.velocity.x = ev.vel.x
                self.setpoint.velocity.y = ev.vel.y
                self.setpoint.velocity.z = ev.vel.z
                self.setpoint.attitude.yaw = np.degrees(ev.yaw)
                self.setpoint.attitudeRate.roll = np.degrees(ev.omega.x)
                self.setpoint.attitudeRate.pitch = np.degrees(ev.omega.y)
                self.setpoint.attitudeRate.yaw = np.degrees(ev.omega.z)
                self.setpoint.mode.x = firm.modeAbs
                self.setpoint.mode.y = firm.modeAbs
                self.setpoint.mode.z = firm.modeAbs
                self.setpoint.mode.roll = firm.modeDisable
                self.setpoint.mode.pitch = firm.modeDisable
                self.setpoint.mode.yaw = firm.modeAbs
                self.setpoint.mode.quat = firm.modeDisable
                self.setpoint.acceleration.x = ev.acc.x
                self.setpoint.acceleration.y = ev.acc.y
                self.setpoint.acceleration.z = ev.acc.z

                self.cmdHl_pos = copy_svec(ev.pos)
                self.cmdHl_vel = copy_svec(ev.vel)
                self.cmdHl_yaw = ev.yaw

        return self._fwsetpoint_to_sim_data_types_state(self.setpoint)

        # # else:
        #     # return self._fwstate_to_sim_data_types_state(self.setState)
        # setState = firm.traj_eval(self.setState)
        # if not firm.is_traj_eval_valid(setState):
        #     return self._fwstate_to_sim_data_types_state(self.state)

        # if self.mode == CrazyflieSIL.MODE_IDLE:
        #     return self._fwstate_to_sim_data_types_state(self.state)

        # self.state = setState
        # return self._fwstate_to_sim_data_types_state(setState)

    def setState(self, state: State):
        self.state.position.x = state.pos[0]
        self.state.position.y = state.pos[1]
        self.state.position.z = state.pos[2]

        self.state.velocity.x = state.vel[0]
        self.state.velocity.y = state.vel[1]
        self.state.velocity.z = state.vel[2]

        rpy = np.degrees(rowan.to_euler(state.quat, convention='xyz'))
        # Note, legacy coordinate system, so invert pitch
        self.state.attitude.roll = rpy[0]
        self.state.attitude.pitch = -rpy[1]
        self.state.attitude.yaw = rpy[2]

        self.state.attitudeQuaternion.w = state.quat[0]
        self.state.attitudeQuaternion.x = state.quat[1]
        self.state.attitudeQuaternion.y = state.quat[2]
        self.state.attitudeQuaternion.z = state.quat[3]

        # omega is part of sensors, not of the state
        self.sensors.gyro.x = np.degrees(state.omega[0])
        self.sensors.gyro.y = np.degrees(state.omega[1])
        self.sensors.gyro.z = np.degrees(state.omega[2])

        # TODO: state technically also has acceleration, but sim_data_types does not

    def executeController(self):
        if self.controller is None:
            return None

        if self.mode == CrazyflieSIL.MODE_IDLE:
            return Action([0, 0, 0, 0])

        time_in_seconds = self.time_func()
        # ticks is essentially the time in milliseconds as an integer
        tick = int(time_in_seconds * 1000)
        if self.controller_name != 'mellinger':
            self.controller(self.control, self.setpoint, self.sensors, self.state, tick)
        else:
            self.controller(
                self.mellinger_control,
                self.control,
                self.setpoint,
                self.sensors,
                self.state,
                tick)
        return self._fwcontrol_to_sim_data_types_action()

    # 'private' methods
    def _isGroup(self, groupMask):
        return groupMask == 0 or (self.groupMask & groupMask) > 0

    def _fwcontrol_to_sim_data_types_action(self):

        firm.powerDistribution(self.control, self.motors_thrust_uncapped)
        firm.powerDistributionCap(self.motors_thrust_uncapped, self.motors_thrust_pwm)

        # self.motors_thrust_pwm.motors.m{1,4} contain the PWM
        # convert PWM -> RPM
        def pwm_to_rpm(pwm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            if pwm < 10000:
                return 0
            p = [3.26535711e-01, 3.37495115e+03]
            return np.polyval(p, pwm)

        def pwm_to_force(pwm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [1.71479058e-09,  8.80284482e-05, -2.21152097e-01]
            force_in_grams = np.polyval(p, pwm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        return Action(
            [pwm_to_rpm(self.motors_thrust_pwm.motors.m1),
             pwm_to_rpm(self.motors_thrust_pwm.motors.m2),
             pwm_to_rpm(self.motors_thrust_pwm.motors.m3),
             pwm_to_rpm(self.motors_thrust_pwm.motors.m4)])

    @staticmethod
    def _fwsetpoint_to_sim_data_types_state(fwsetpoint):
        pos = np.array([fwsetpoint.position.x, fwsetpoint.position.y, fwsetpoint.position.z])
        vel = np.array([fwsetpoint.velocity.x, fwsetpoint.velocity.y, fwsetpoint.velocity.z])
        acc = np.array([
            fwsetpoint.acceleration.x,
            fwsetpoint.acceleration.y,
            fwsetpoint.acceleration.z])
        omega = np.radians(np.array([
            fwsetpoint.attitudeRate.roll,
            fwsetpoint.attitudeRate.pitch,
            fwsetpoint.attitudeRate.yaw]))

        if fwsetpoint.mode.quat == firm.modeDisable:
            # compute rotation based on differential flatness
            thrust = acc + np.array([0, 0, 9.81])
            z_body = thrust / np.linalg.norm(thrust)
            yaw = np.radians(fwsetpoint.attitude.yaw)
            x_world = np.array([np.cos(yaw), np.sin(yaw), 0])
            y_body = np.cross(z_body, x_world)
            # Mathematically not needed. This addresses numerical issues to ensure R is orthogonal
            y_body /= np.linalg.norm(y_body)
            x_body = np.cross(y_body, z_body)
            # Mathematically not needed. This addresses numerical issues to ensure R is orthogonal
            x_body /= np.linalg.norm(x_body)
            R = np.column_stack([x_body, y_body, z_body])
            quat = rowan.from_matrix(R)
        else:
            quat = fwsetpoint.attitudeQuaternion

        return State(pos, vel, quat, omega)

def simulate(x0, u, tf, dt=5e-4, ll_ctrl_name="pid"):
    """
    Fn for wrapping the above methods and numerically integrating the dynamcis.
    
    x0 : assumed to be a 12D array of [position, velocity, euler angles (world), angular velocity (body)]
    u  : feedback law, which is a function of state (12D array) and time (scalar)
    tf : scalar for final time
    dt : temporal step size of integrator (significant error for values larger than 5e-4)
    ll_ctrl_name : the controller for the firmware to use to thrust motors
    """

    steps = 1+int(tf/dt)
    X = np.zeros((12, steps))
    X[:,0] = x0

    x0s = State(pos=x0[0:3], vel=x0[3:6], euler=x0[6:9], omega=x0[9:12])
    model = Quadrotor(x0s) # normally in Backend in CrazyflieServer
    uav = CrazyflieSIL("uav", x0s, ll_ctrl_name, model.time, initialState=x0s) # usually self.backend.time

    for tix, ti in enumerate(np.linspace(0., tf, steps)):
        uav.cmdVelLegacy(*u(model.fullstate(), ti))         # sets mode & set_point
        action = uav.executeController()                    # calls controller, powerDist, pwm_to_rpm
        model.step(action, dt)                              # evolves 13D model, uses rown
        uav.setState(model.state)                           # updates uav state
        X[:,tix] = model.fullstate()
    
    return X