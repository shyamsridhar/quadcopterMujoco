#!/usr/bin/env python3

"""
This file is used for generating path provided a current package position and target package position is given.
Returns a numpy array containing the desired shortest straight line path

Assumptions:
    1. Target position of package will be same height as starting postion
    2. Separataion distance between drones in 4 metres
    3. Movement altitude is 10 metres
    4. Path required is always straight line path
"""

import numpy as np

def generatePath(curPackagePos, targetPackagePos, quadstartPos):

    path = np.zeros((40,3,4))

    waypoint1 = quadPositions(curPackagePos, 10)
    waypoint2 = quadPositions(targetPackagePos, 10)
    landingPos = quadPositions(targetPackagePos, 0)

    for i in range(4):
        path[0:10,:,i] = np.linspace(quadstartPos[i,:], waypoint1[i,:], num=10, endpoint=True)
        path[10:30,:,i] = np.linspace(waypoint1[i,:], waypoint2[i,:], num=20, endpoint=True)
        path[30:40,:,i] = np.linspace(waypoint2[i,:], landingPos[i,:], num=10, endpoint=True)

    return path



def quadPositions(packagePos, altitude):
    quad = np.zeros((4,3))
    quad[0,:] = packagePos + np.array([2, 2, altitude], ndmin=2)
    quad[1,:] = packagePos + np.array([2, -2, altitude], ndmin=2)
    quad[2,:] = packagePos + np.array([-2, -2, altitude], ndmin=2)
    quad[3,:] = packagePos + np.array([-2, 2, altitude], ndmin=2)
    return quad


if __name__ == "__main__":
    curPackagePos = np.array([0, 0, 0], ndmin=2)
    targetPackagePos = np.array([20, 0, 0], ndmin=2)
    quadstartPos = np.array([[5.65, 5.65, 0],[5.65, -5.65, 0],[-5.65, -5.65, 0],[-5.65, 5.65, 0]], ndmin=2)
    path = generatePath(curPackagePos, targetPackagePos, quadstartPos)

    print(path)