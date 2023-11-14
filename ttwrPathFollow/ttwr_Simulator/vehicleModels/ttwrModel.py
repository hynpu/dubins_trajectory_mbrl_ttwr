import numpy as np
import matplotlib.pyplot as plt

from ttwrPathFollow.ttwr_Simulator.vehicleModels.ttwrParams import *
from ttwrPathFollow.ttwr_Simulator.vehicleModels.ttwrRender import *
import ttwrPathFollow.ttwr_Simulator.utils.angleConversion as helper

class TtwrModel():

    def __init__(self, render_mode=None):

        self.L1 = ttwrParams.L1
        self.L2 = ttwrParams.L2
        self.L3 = ttwrParams.L3
        self.dt = ttwrParams.dt

        self.length = ttwrParams.host_length
        self.width = ttwrParams.host_width

        self.x_lim = ttwrParams.map_x_limit
        self.y_lim = ttwrParams.map_y_limit
        self.str_lim = ttwrParams.maxSteeringAngle

        # init input
        self.velo = 0
        self.delta = 0

        self.state = np.zeros(7)
        self.reset()
    
    # input is x2, y2, theta2, phi
    def reset(self, trailerState = np.zeros(4), velo = 0):
        # reset velocity first
        self.v1 = velo
        # trailer states from env
        self.get_state(trailerState)

        return self.state

    def get_state(self, trailerState = np.zeros(4)):
        # trailer states from env
        self.x2     = trailerState[0]
        self.y2     = trailerState[1]
        self.theta2 = trailerState[2]
        self.phi    = trailerState[3]

        self.theta1 = self.theta2 - self.phi
        self.x1     = self.x2 + self.L2 * np.cos(self.theta1) + self.L3 * np.cos(self.theta2)
        self.y1     = self.y2 + self.L2 * np.sin(self.theta1) + self.L3 * np.sin(self.theta2)

        self.state = np.array([self.x1, self.y1, self.theta1, self.x2, self.y2, self.theta2, self.phi, self.v1])

        return self.state
        
    def move(self, acc, delta):
        # self.v1 = v1
        self.acc = acc
        self.delta = delta
        
        # # host vehicle state update; no use for now
        # x1_dot = self.v1 * np.cos( self.theta1 )
        # y1_dot = self.v1 * np.sin( self.theta1 )
        # theta1_dot = self.v1 * np.tan( delta ) / self.L1
        # trailer vehicle state update
        x2_dot = self.v1 * np.cos(self.phi) * \
            (1 - self.L2 / self.L1 * np.tan(self.phi) * np.tan(delta)) * np.cos(self.theta2)
        y2_dot = self.v1 * np.cos(self.phi) * \
            (1 - self.L2 / self.L1 * np.tan(self.phi) * np.tan(delta)) * np.sin(self.theta2)
        theta2_dot = -self.v1 * ( np.sin(self.phi)/self.L3 + self.L2/(self.L1*self.L3) * np.cos(self.phi) * np.tan(delta) )
        # host trailer angle update
        # phi = theta2 - theta1
        phi_dot = -self.v1 * \
            (np.sin(self.phi)/self.L3 + self.L2/(self.L1*self.L3) * np.cos(self.phi) * np.tan(delta)) \
            -self.v1 * np.tan( delta ) / self.L1

        # ttwr states
        self.x2 += x2_dot * self.dt
        self.y2 += y2_dot * self.dt
        self.theta2 = helper.wrapToPi(theta2_dot * self.dt + self.theta2)
        self.phi += phi_dot * self.dt
        
        self.v1 = self.v1 + acc * self.dt

        # get ttwr based on trailer states and phi
        self.get_state(np.array([self.x2, self.y2, self.theta2, self.phi]))

        return self.state
    
