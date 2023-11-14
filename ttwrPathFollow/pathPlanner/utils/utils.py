import numpy as np

def eucDistCompt(pnt1, pnt2):
    return np.sqrt((pnt2[0] - pnt1[0]) ** 2 + (pnt2[1] - pnt1[1]) ** 2)

def wrapTo2Pi(angle):
    posIn = angle>0
    angle = angle % (2*np.pi)
    if angle == 0 and posIn:
        angle = 2*np.pi
    return angle

def wrapToPi(angle):
    q = (angle < -np.pi) or (np.pi < angle)
    if(q):
        angle = wrapTo2Pi(angle + np.pi) - np.pi
    return angle