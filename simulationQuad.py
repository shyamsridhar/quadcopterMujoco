#!/usr/bin/env python3

"""
Quadcopter simulation in matlab
"""

from doctest import debug
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


Tz = 1.14
Tx = 0.3066
Ty = 0.3066
timestep = 0.01
z_cur = 0
massDrone = 1.5

simtime = np.linspace(0,20,100, endpoint=True)

#declare  pid values
kpz = 10; kpx = 10; kpy = 10
kpvz = 10; kdvz = 5
kpvx = 10; kdvx = 10; kpvy = 10; kdvy = 10


kxy = np.array([[kpx,0],[0,kpy]], ndmin=2)
pxy = np.array([0,0], ndmin=2).T
p_xy_des = np.array([0,0], ndmin=2).T

#

def zfunc():
    pass
    #contains z coordinate position equation

def xyfunc():
    pass
    #contains xy coordinate position equation

def attfunc():
    pass
    #contains attitude differential equation


for i in simtime: 

    vz_des = kpz*(z_des - z_cur)
    thrInput = massDrone*(kpvz(vz_des - vz) - kdvz(az) + 9.81)

    v_xy_des = np.matmul(kxy, (p_xy_des - pxy))
    pry_des = massDrone*(kpvz(v_xy_des - v_xy) - kdvz(a_xy) + 9.81)



    #Integrate differential equation
    z2 = solve_ivp(zfunc, [t0, t1], y0, method='RK45', vectorized=True)

    #Integrate xy differential equation


    #Integrate attitude differential equations


    # find current position, velocity, acceleration, attitude of the drone

    #store values of the state in separate numpy variable








#plot the results

