#!/usr/bin/env python
from crazyflie_interfaces.srv import GoTo, Land, Takeoff, NotifySetpointsStop
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
import rclpy.logging
from std_msgs.msg import Bool, String
import numpy as np
import time
from rclpy.node import Node
import rclpy 
# import rclpy.node
import rowan
import logging
from crazyflie_py import Crazyswarm


K_matrix = np.array([[0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0],
                     [0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], 
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  
                     [0.0, 0.0, -5.4772, 0.0, 0.0, -5.5637, 0.0]])

def make_fig8_traj(center=np.array([0., 0.]), offset=np.array([1., 2.]), h=1.): ## for making a figure 8 waypoint list
    traj = np.array([[center[0], center[1], h, 0. ,0. ,0. ,0.],
                    [center[0] - offset[0], center[1] + offset[1]/2, h, 0.0, 0.0, 0.0, 0.0], 
                    [center[0], center[1] + offset[1], h, 0.0, 0.0, 0.0, 0.0], 
                    [center[0] + offset[0], center[1] + offset[1]/2, h, 0.0, 0.0, 0.0, 0.0], 
                    [center[0], center[1], h, 0.0, 0.0, 0.0, 0.0], 
                    [center[0] - offset[0], center[1] - offset[1]/2, h, 0.0, 0.0, 0.0, 0.0], 
                    [center[0], center[1] - offset[1], h, 0.0, 0.0, 0.0, 0.0], 
                    [center[0] + offset[0], center[1] - offset[1]/2, h, 0.0, 0.0, 0.0, 0.0],
                    [center[0], center[1], h, 0.0, 0.0, 0.0, 0.0]])
    return traj

def make_jump_traj(center=np.array([0., 0., 1.]), step_max=3, iter=1, jumps=3): 
    traj = [np.concatenate([center, np.zeros(4)])]

    x_max = 3.5
    y_max = 2.0
    for i in range(iter):
        for j in range(jumps):
            d = step_max * 2 * np.random.rand() # deviation param
            dev = np.subtract(d * np.random.rand(center.shape[0]), d/2) # deviation
            traj.append(np.concatenate([traj[-1][:center.shape[0]] + dev, np.zeros(4)]))

            if abs(traj[-1][0]) > x_max: traj[-1][1] = np.sign(traj[-1][1])*x_max
            if abs(traj[-1][1]) > y_max: traj[-1][1] = np.sign(traj[-1][1])*y_max
            if traj[-1][2] < 0.5: traj[-1][2] = 0.5
            if traj[-1][2] > 1.5: traj[-1][2] = 1.5

        traj.append(np.concatenate([center, np.zeros(4)]))
    return traj

u_target = np.array([0.0, 0.0, 0.0, 10.5])
x_targets = make_jump_traj()

# center = [-4.5, 0.] # center of fig 8, back area
# offset = [0.6, 1.7] #

class FeedbackController_Fig8(Crazyswarm):
    def __init__(self):
        super().__init__()  # Inits Crazyswarm
        self.rclpy_node = self.allcfs
        self.rclpy_node.estimated_state = np.zeros(7) # TODO: Customize for different state fidelities
        self.rclpy_node.lqr_active = False
        self.i = -1
        self.delay = 1.5
        self.control_angle_bound = np.pi/6 # higher bound allows more aggressive control
        
        self.pub = self.rclpy_node.create_publisher(
                    String,
                    'cf231/test', 10)
        
        self.control_pub = self.rclpy_node.create_publisher(
                    String,
                    'cf231/rpyt', 10)
        
        # self.rclpy_node.get_logger().info("TEST")

        self.t = None
        assert len(self.rclpy_node.crazyflies) == 1, "Feedback controller only supports one drone"
        for cf in self.rclpy_node.crazyflies:
            cf_name = cf.prefix
            self.odomSubscriber = self.rclpy_node.create_subscription(Odometry, f'{cf_name}/odom', self.odom_callback, 10)
        self.num_zeros_sent = 0.0
        self.start_lqr_subscriber = self.rclpy_node.create_subscription(String, 'start_custom_controller', self.start_lqr_callback, 10)
    
    def update_i(self):
        if self.i < len(x_targets)-1: 
            self.i += 1

            msg = String()
            msg.data = f"{self.i}"
            self.pub.publish(msg)
            # self.rclpy_node.get_logger().info(f"New Waypoint : [{x_targets[self.i][0]:.2f}, {x_targets[self.i][1]:.2f}, {x_targets[self.i][2]:.2f}]")
            print(f"New Waypoint : [{x_targets[self.i][0]:.2f}, {x_targets[self.i][1]:.2f}, {x_targets[self.i][2]:.2f}]")

    def start_lqr_callback(self, msg):
        self.t = self.rclpy_node.create_timer(self.delay, self.update_i)
        self.rclpy_node.lqr_active = not self.rclpy_node.lqr_active
        if self.rclpy_node.lqr_active:
            print("LQR activated")
        else:
            print("End of LQR, landing")
            for cf in self.rclpy_node.crazyflies:
                cf.notifySetpointsStop(10)
            # self.rclpy_node.goTo([0.0, 0.0, 0.25], 0.0, 5.0)
            self.rclpy_node.land(targetHeight=0.06, duration=5.0)
            self.timeHelper.sleep(0.1)
            self.rclpy_node.land(targetHeight=0.06, duration=5.0)
   
    def odom_callback(self, msg):
        # Read odom msg and update estimated state (x, y, z, xdot, ydot, zdot, yaw)
        euler_angles = rowan.to_euler(([msg.pose.pose.orientation.w, 
                                        msg.pose.pose.orientation.x, 
                                        msg.pose.pose.orientation.y, 
                                        msg.pose.pose.orientation.z]), "xyz")
        new_state = [msg.pose.pose.position.x, 
                     msg.pose.pose.position.y, 
                     msg.pose.pose.position.z, 
                     msg.twist.twist.linear.x, 
                     msg.twist.twist.linear.y, 
                     msg.twist.twist.linear.z,
                     euler_angles[2]]
        self.rclpy_node.estimated_state = new_state
        self.update_control()  # TODO: make such that this is controlled with separate timer

    def update_control(self):
        # Calculate control input based on estimated state
        # Publish control input

        # self.rclpy_node.get_logger().info("LQR active? {self.rclpy_node.lqr_active}")

        if self.rclpy_node.lqr_active:
            if self.num_zeros_sent < 5:
                converted_control = np.zeros(4)
                control_bounded = np.zeros(4)
                self.num_zeros_sent += 1

            else:
                control = K_matrix @ (self.rclpy_node.estimated_state - x_targets[self.i]) + u_target
                control_bounded = np.concatenate((np.clip(control[:3], -self.control_angle_bound, self.control_angle_bound), [control[3]])) # CONTROL ANGLE BOUNDING
                converted_control = self.convert_control(control_bounded)

            for cf in self.rclpy_node.crazyflies:
                # self.rclpy_node.get_logger().info(f"RPYT = [{converted_control[0]}, {converted_control[1]}, {converted_control[2]}, {converted_control[3]}]")
                cf.cmdVel(converted_control[0], converted_control[1], converted_control[2], converted_control[3])

                msg = String()
                msg.data = f"RPYT = [{control_bounded[0]}, {control_bounded[1]}, {control_bounded[2]}, {control_bounded[3]}]"
                self.control_pub.publish(msg)
        
    @staticmethod
    def convert_control(control):
        # rpy to degrees, thrust scaled to 0-65535 from 0-16
        rpy = np.degrees(control[:3])
        # rpy = np.clip(rpy, -30.0, 30.0) # WAS : moved out since bounding isnt conversion, important to know about
        thrust = control[3] * 4096
        return np.array([rpy[0], rpy[1], rpy[2], thrust]) # roll, pitch, yaw, thrust