import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from ttwrPathFollow.pathPlanner.utils.utils import *

class TurnType(Enum):
    LSL = 1
    LSR = 2
    RSL = 3
    RSR = 4
    RLR = 5
    LRL = 6

class Param:
    def __init__(self, p_init, seg_final, turn_radius,):
        self.p_init = p_init
        self.seg_final = seg_final
        self.turn_radius = turn_radius
        self.type = 0

# Output:
# param: a struct that includes 4 field:
#     p_init: Initial pose, equals to input p1
#     type: One of the 6 types of the dubins curve
#     r: Turning radius, same as input r, also the scaling factor for the dubins paramaters
#     seg_param: angle or normalized length, in row vector [pr1, pr2, pr3]
# Reference:
# Shkel, A. M. and Lumelsky, V. (2001). "Classification of the Dubins set". Robotics and Autonomous Systems 34 (2001) 179¡V202
def calcDubinsPath(wpt1, wpt2, turning_radius):
    # Calculate a dubins path between two waypoints
    param = Param(wpt1, 0, 0)
    tz        = [0, 0, 0, 0, 0, 0]
    pz        = [0, 0, 0, 0, 0, 0]
    qz        = [0, 0, 0, 0, 0, 0]
    param.seg_final = [0, 0, 0]
    # Convert the headings from NED to standard unit cirlce, and then to radians
    psi1 = wrapToPi(wpt1[2]) # / 180 * np.pi
    psi2 = wrapToPi(wpt2[2]) # / 180 * np.pi

    # Do math
    param.turn_radius = turning_radius
    dx = wpt2[0] - wpt1[0]
    dy = wpt2[1] - wpt1[1]
    D = np.sqrt(dx*dx + dy*dy)
    d = D/param.turn_radius # Normalize by turn radius...makes length calculation easier down the road.

    # Angles defined in the paper
    theta = np.arctan2(dy,dx) % (2*np.pi)
    alpha = (psi1 - theta) % (2*np.pi)
    beta  = (psi2 - theta) % (2*np.pi)
    best_word = -1
    best_cost = -1

    # Calculate all dubin's paths between points
    tz[0], pz[0], qz[0] = dubinsLSL(alpha,beta,d)
    tz[1], pz[1], qz[1] = dubinsLSR(alpha,beta,d)
    tz[2], pz[2], qz[2] = dubinsRSL(alpha,beta,d)
    tz[3], pz[3], qz[3] = dubinsRSR(alpha,beta,d)
    tz[4], pz[4], qz[4] = dubinsRLR(alpha,beta,d)
    tz[5], pz[5], qz[5] = dubinsLRL(alpha,beta,d)

    # Now, pick the one with the lowest cost
    for x in range(6):
        if(tz[x]!=-1):
            cost = tz[x] + pz[x] + qz[x]
            if(cost<best_cost or best_cost==-1):
                best_word = x+1
                best_cost = cost
                param.seg_final = [tz[x],pz[x],qz[x]]

    param.type = TurnType(best_word)
    return param

# Here's all of the dubins path math
def dubinsLSL(alpha, beta, d):
    tmp0      = d + np.sin(alpha) - np.sin(beta)
    tmp1      = np.arctan2((np.cos(beta)-np.cos(alpha)),tmp0)
    p_squared = 2 + d*d - (2*np.cos(alpha-beta)) + (2*d*(np.sin(alpha)-np.sin(beta)))
    if p_squared<0:
        # print('No LSL Path')
        p=-1
        q=-1
        t=-1
    else:
        t         = (tmp1-alpha) % (2*np.pi)
        p         = np.sqrt(p_squared)
        q         = (beta - tmp1) % (2*np.pi)
    return t, p, q

def dubinsRSR(alpha, beta, d):
    tmp0      = d - np.sin(alpha) + np.sin(beta)
    tmp1      = np.arctan2((np.cos(alpha)-np.cos(beta)),tmp0)
    p_squared = 2 + d*d - (2*np.cos(alpha-beta)) + 2*d*(np.sin(beta)-np.sin(alpha))
    if p_squared<0:
        # print('No RSR Path')
        p=-1
        q=-1
        t=-1
    else:
        t         = (alpha - tmp1 ) % (2*np.pi)
        p         = np.sqrt(p_squared)
        q         = (-1*beta + tmp1) % (2*np.pi)
    return t, p, q

def dubinsRSL(alpha,beta,d):
    tmp0      = d - np.sin(alpha) - np.sin(beta)
    p_squared = -2 + d*d + 2*np.cos(alpha-beta) - 2*d*(np.sin(alpha) + np.sin(beta))
    if p_squared<0:
        # print('No RSL Path')
        p=-1
        q=-1
        t=-1
    else:
        p         = np.sqrt(p_squared)
        tmp2      = np.arctan2((np.cos(alpha)+np.cos(beta)),tmp0) - np.arctan2(2,p)
        t         = (alpha - tmp2) % (2*np.pi)
        q         = (beta - tmp2) % (2*np.pi)
    return t, p, q

def dubinsLSR(alpha, beta, d):
    tmp0      = d + np.sin(alpha) + np.sin(beta)
    p_squared = -2 + d*d + 2*np.cos(alpha-beta) + 2*d*(np.sin(alpha) + np.sin(beta))
    if p_squared<0:
        # print('No LSR Path')
        p=-1
        q=-1
        t=-1
    else:
        p         = np.sqrt(p_squared)
        tmp2      = np.arctan2((-1*np.cos(alpha)-np.cos(beta)),tmp0) - np.arctan2(-2,p)
        t         = (tmp2 - alpha) % (2*np.pi)
        q         = (tmp2 - beta) % (2*np.pi)
    return t, p, q

def dubinsRLR(alpha, beta, d):
    tmp_rlr = (6 - d*d + 2*np.cos(alpha-beta) + 2*d*(np.sin(alpha)-np.sin(beta)))/8
    if(abs(tmp_rlr)>1):
        # print('No RLR Path')
        p=-1
        q=-1
        t=-1
    else:
        p = (2*np.pi - np.arccos(tmp_rlr)) % (2*np.pi)
        t = (alpha - np.arctan2((np.cos(alpha)-np.cos(beta)), d-np.sin(alpha)+np.sin(beta)) + p/2 % (2*np.pi)) % (2*np.pi)
        q = (alpha - beta - t + (p % (2*np.pi))) % (2*np.pi)

    return t, p, q

def dubinsLRL(alpha, beta, d):
    tmp_lrl = (6 - d*d + 2*np.cos(alpha-beta) + 2*d*(-1*np.sin(alpha)+np.sin(beta)))/8
    if(abs(tmp_lrl)>1):
        # print('No LRL Path')
        p=-1
        q=-1
        t=-1
    else:
        p = (2*np.pi - np.arccos(tmp_lrl)) % (2*np.pi)
        t = (-1*alpha - np.arctan2((np.cos(alpha)-np.cos(beta)), d+np.sin(alpha)-np.sin(beta)) + p/2) % (2*np.pi)
        q = ((beta % (2*np.pi))-alpha-t+(p % (2*np.pi))) % (2*np.pi)
        # print(t,p,q,beta,alpha)
    return t, p, q

def dubins_traj(param,step):
    # Build the trajectory from the lowest-cost path
    x = 0
    i = 0
    length = (param.seg_final[0]+param.seg_final[1]+param.seg_final[2])*param.turn_radius
    path_len = np.floor(length/step)
    # path = -1 * np.ones((path_len,3))
    path = []
    while x < length:
        # path[i] = dubins_path(param,x)
        path.append(dubins_path(param,x))
        x += step
        i+=1
    return np.array(path)


def dubins_path(param, t):
    # Helper function for curve generation
    tprime = t/param.turn_radius
    p_init = np.array([0,0,wrapTo2Pi(param.p_init[2])])
    #
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    DIRDATA = np.array([[L_SEG,S_SEG,L_SEG],[L_SEG,S_SEG,R_SEG],[R_SEG,S_SEG,L_SEG],[R_SEG,S_SEG,R_SEG],[R_SEG,L_SEG,R_SEG],[L_SEG,R_SEG,L_SEG]])
    #
    types = DIRDATA[param.type.value-1][:]
    param1 = param.seg_final[0]
    param2 = param.seg_final[1]
    mid_pt1 = dubins_segment(param1,p_init,types[0])
    mid_pt2 = dubins_segment(param2,mid_pt1,types[1])

    if(tprime<param1):
        end_pt = dubins_segment(tprime,p_init,types[0])
    elif(tprime<(param1+param2)):
        end_pt = dubins_segment(tprime-param1,mid_pt1,types[1])
    else:
        end_pt = dubins_segment(tprime-param1-param2, mid_pt2, types[2])

    end_pt[0] = end_pt[0] * param.turn_radius + param.p_init[0]
    end_pt[1] = end_pt[1] * param.turn_radius + param.p_init[1]
    end_pt[2] = wrapToPi( end_pt[2] % (2*np.pi) )

    return end_pt

def dubins_segment(seg_param, seg_init, seg_type):
    # Helper function for curve generation
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    seg_end = np.array([0.0,0.0,0.0])
    if( seg_type == L_SEG ):
        seg_end[0] = seg_init[0] + np.sin(seg_init[2]+seg_param) - np.sin(seg_init[2])
        seg_end[1] = seg_init[1] - np.cos(seg_init[2]+seg_param) + np.cos(seg_init[2])
        seg_end[2] = seg_init[2] + seg_param
    elif( seg_type == R_SEG ):
        seg_end[0] = seg_init[0] - np.sin(seg_init[2]-seg_param) + np.sin(seg_init[2])
        seg_end[1] = seg_init[1] + np.cos(seg_init[2]-seg_param) - np.cos(seg_init[2])
        seg_end[2] = seg_init[2] - seg_param
    elif( seg_type == S_SEG ):
        seg_end[0] = seg_init[0] + np.cos(seg_init[2]) * seg_param
        seg_end[1] = seg_init[1] + np.sin(seg_init[2]) * seg_param
        seg_end[2] = seg_init[2]

    return seg_end

def dubinsPathGenerator(startState, endState, turn_radius, stepSize):
    startState[2] = wrapToPi(startState[2] + np.pi)
    endState[2] = wrapToPi(endState[2] + np.pi)
    
    param = calcDubinsPath(startState,endState, turn_radius) 
    path_ref_pnts = dubins_traj(param, stepSize)
    
    for i in range(len(path_ref_pnts[:,2])):
        path_ref_pnts[i,2] = wrapToPi(path_ref_pnts[i,2] + np.pi)
    return path_ref_pnts

# https://github.com/fgabbert/dubins_py/tree/master
# https://github.com/EwingKang/Dubins-Curve-For-MATLAB/blob/master/dubins_core.m
# Shkel, A. M. and Lumelsky, V. (2001). "Classification of the Dubins set". Robotics and Autonomous Systems 34 (2001) 179¡V202