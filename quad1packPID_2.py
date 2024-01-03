#!/usr/bin/env python3


#qpos and qvel represent the position and velocity of joints and not bodies. for a 6 dof joint qpos first 3 represent position and next 4 represent unit quaternion.
#obtain qpos address from mjModel.jnt_qposadr. qpos contains data according to number of generalized coordinates nq. For a 6 degree of freedom joint qpos contains 7*1 data. This depends on joint type
#for example for drone 2 we know that it is connected to world body using root2 free joint. This has jnt_qposadr 11. Therefore 14:18 gives orientation
#qvel contains generalized velocities of joints depending on number of degrees of freedom
#use mjData.jnt_dofadr to get the qvel position corresponding to the joint.
#qvel data is usually 6 numbers for 6 dof joint like root. first 3 for linear velocity and next 3 for angular velocity

#The orientation of a free joint (a quaternion) is also in the global frame. However, the rotational velocities of a free joint are in the local frame. --- Explained by Yuval Tassa as a response to a query
#For rest of the joints it is always local and expressed with respect to parent.

#use mjmodel.sensor_adr to get the position of sensor values in sensor data

#refer to mujoco rotation.py file for comments on quaternions and rotation matrix. It follows x-y'-z" intrinsic/z-y-x extrinsic rotation matrix

#roll and pitch behave separately for position error. When poserror is positive current roll is also positive while poserror is positive for pitch current pich is negative. Therefore we multiply -1 for roll

#calculate desired rp angles in world frame based on PD control. Target is fixed i.e. does not change with time. Derivative kick is avoided. 
from doctest import debug
from typing import Concatenate
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from matplotlib import pyplot as plt

xml_path = 'quad1pack.xml'
simend = 20
print_camera_config = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

#target for quad1
target1 = np.array([2, 1, 10], ndmin=2).T


# kthr = 0.00005
# ktrq = kthr/5
kthr = 1.08E-05
ktrq = 1.62E-07

kp=0.2*50
kd=6
ki = 5
kpr=20
kdr=20
movement = 0
package = 1

z = np.array([])
x = np.array([])
y = np.array([])
w1 = np.array([])
w2 = np.array([])
w3 = np.array([])

timelog = np.array([])



#declaring body thrust and gyro torques as global
th1, tq1 = np.zeros((3,4)), np.zeros((3,4))


def init_controller(model, data):
    data.qpos[2] = 0.1
    data.qvel[:] = 0
    pass

def controller(model, data):
    """
    Applies force on to the correct propeller depending on the speed of the motor
    """
    R = rotation_m(data, 3, 7)
    R2 = rotation_m(data, 3, 7)

    propvel = data.qvel[6:10]; propvel2 = data.qvel[6:10]

    # propvel[0] = max(min(2000, propvel[0]), 0) 
    # propvel[1] = max(min(2000, propvel[1]), 0) 
    # propvel[2] = max(min(0, propvel[2]), -2000) 
    # propvel[3] = max(min(0, propvel[3]), -2000) 

    thrust = kthr*propvel**2 
    torque = ktrq*propvel**2 

    print(propvel)

    global th1, tq1
    th1, tq1 = np.zeros((3,4)), np.zeros((3,4))


    th1[2,:] = thrust
    tq1[2,:] = torque
 
    #thrust and torques in global coordinate frame
    th1a = np.matmul(R,th1)
    tq1a = np.matmul(R,tq1)


    #xfrc. Find XFRC position based on body id of propellers
    data.xfrc_applied[11][:] = np.concatenate((th1a[:,0],-tq1a[:,0]),axis=0).flatten()
    data.xfrc_applied[13][:] = np.concatenate((th1a[:,1],-tq1a[:,1]),axis=0).flatten()
    data.xfrc_applied[15][:] = np.concatenate((th1a[:,2],tq1a[:,2]),axis=0).flatten()
    data.xfrc_applied[17][:] = np.concatenate((th1a[:,3],tq1a[:,3]),axis=0).flatten()



def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)



def rotation_m(data, start, end):
    """
    Returns the orientation of the quadcopter. 

    Args:
        start: Starting position of qpos
        end: Ending position of qpos
    """
    res = np.zeros((9,1))  
    mj.mju_quat2Mat(res,data.qpos[start:end])
    R = res.reshape((3,3))
    return R


def calculate_pry(R_M):
    """
    Calculates the pitch roll and yaw from rotation matrix
    """
    roll = np.arctan2(R_M[2][1],R_M[2][2]) 
    pitch = np.arcsin(-R_M[2][0]) 
    yaw = np.arctan2(R_M[1][0],R_M[0][0])
    current_pry = np.array([pitch, roll, yaw], ndmin=2).T
    return current_pry


def calculate_R_rate(pry):
    """
    Calculate rate change of rotation matrix
    """
    pitch = pry[0]; roll = pry[1]; yaw = pry[2]
    R_rate = np.array([[1,np.sin(roll[0])*np.tan(pitch[0]),np.cos(roll[0])*np.tan(pitch[0])]
                       ,[0,np.cos(roll[0]),-np.sin(roll[0])]
                        ,[0,np.sin(roll[0])/np.cos(pitch[0]),np.cos(roll[0])/np.cos(pitch[0])]])

    return R_rate


def calculate_ang_vel(R_M, pry, qvelStart, qvelEnd):
    """
    Calculate angular velocity of the quadcopter in body coordinate frame based on qvel data of the root joint

    Args:
        R_M: Rotation matrix for orientation
        pry: Current pitch roll and yaw
        qvelStart: Starting position of qvel of the joint
        qvelEnd: Ending position of qvel of the joint
    """
    absangvel = data.qvel[qvelStart:qvelEnd].reshape((3,1))
    bdyangvel = np.matmul(R_M.T,absangvel)
    Tmatinv = calculate_R_rate(pry)
    rpyrates = np.matmul(Tmatinv,bdyangvel)
    rollrate, pitchrate, yawrate = rpyrates.flatten()

    bodyangvel = np.array([pitchrate, rollrate, yawrate], ndmin=2).T
    return bodyangvel


def sensor_data():
    pass


def cntrl(data, target, quad: str):
    """
    Function which contains the flight controller PID code. MjData positions have been hardcoded
    
    """
    
    #initializing variables for retrieving data
    if quad == "quad1":
        quatStart, quatEnd, qvelStart, qvelEnd  = 3, 7, 3, 6
        qposStart, qposEnd, linVelStart, linVelEnd = 0, 3, 0, 3
        linaccStart, linaccEnd = 3, 6
    elif quad == "quad2":
        quatStart = 3; quatEnd = 7; qvelStart = 3; qvelEnd = 6
        qposStart = 0; qposEnd = 3; linVelStart = 0; linVelEnd = 3
        linaccStart = 3; linaccEnd = 6
    elif quad == "quad3":
        quatStart = 3; quatEnd = 7; qvelStart = 3; qvelEnd = 6
        qposStart = 0; qposEnd = 3; linVelStart = 0; linVelEnd = 3
        linaccStart = 3; linaccEnd = 6
    elif quad == "quad4":
        quatStart = 3; quatEnd = 7; qvelStart = 3; qvelEnd = 6
        qposStart = 0; qposEnd = 3; linVelStart = 0; linVelEnd = 3
        linaccStart = 3; linaccEnd = 6


    R = rotation_m(data, quatStart, quatEnd)
    cur_pry = calculate_pry(R)
    bodyangvel = calculate_ang_vel(R, cur_pry, qvelStart, qvelEnd)

    currentpos = np.array(data.qpos[qposStart:qposEnd], ndmin=2).T 
    currentvel = np.array(data.sensordata[linVelStart:linVelEnd], ndmin=2).T 
    currentacc = np.array(data.sensordata[linaccStart:linaccEnd], ndmin=2).T

    #Altitude model
    des_vel_z = 1.1*(target[2] - currentpos[2])
    des_thrust = 10*(des_vel_z - currentvel[2]) + 0.001*(0-currentacc[2])

    #horizontal model
    des_xy_vel = 1.1*(target[0:2]-currentpos[0:2])
    current_xy_vel = currentvel[0:2]
    current_xy_acc = currentacc[0:2]
    des_xy_ang = 0.5*(des_xy_vel - current_xy_vel) + 0.005*(0-current_xy_acc)

    des_xy_ang[1] =  des_xy_ang[1]*-1


    #saturation limit is 18 degrees
    max_ang = np.linalg.norm(des_xy_ang[0:2,:], ord=np.inf)
    if max_ang > np.pi/10:
        des_xy_ang =  (des_xy_ang*np.pi/10)/max_ang

    # p d controller for attitude model
    des_torq = 10*(des_xy_ang - cur_pry[0:2]) + 2*(0 - bodyangvel[0:2])

    # zf = 50 + dpr1[2] 
    zf = 50 + des_thrust
    rf = des_torq[1]
    pf = des_torq[0]
    yf = 5*(0-cur_pry[2]) + 1*(0-bodyangvel[2])

    return zf, rf, pf, yf


# def jntVel(model, data, name: str, dataReq: str, objType: int):
#     """
#     Fetches the data from mjData.

#     Args:
#         model: mjModel
#         data: mjData
#         name: Name of the body or joint
#         dataReq: Data required like velocity, position
#         objType: Type of object mjtObj
#     """
    
#     if name is None:
#         return -1
    
#     id = mj.mjname2id(model,objType,name)

#     if dataReq == "pos":
#         adr_start = model.qposadr[id]
#         adr_end = model.qposadr[id+1]
#         return data.qpos[adr_start:adr_start+3]
#     elif dataReq == "vel":
#         adr_start = model.dofadr[id]
#         # adr_end = model.dofadr[id+1]
#         jnt_type = model.jnt_type[id]
#         if jnt_type == 0:
#             adr_end = adr_start + 6
#         elif jnt_type == 1:
#             adr_end = adr_start + 3
#         elif jnt_type == 3:
#             adr_end = adr_start + 1

#         vel = data.qpos[adr_start:adr_end]
#     elif dataReq == "quat":
#         pass


#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# cam.type=1; cam.trackbodyid=53
cam.type=1; cam.trackbodyid=1
cam.azimuth = 90 ; cam.elevation = -45 ; cam.distance =  10

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

error = np.array([0, 0, 0], ndmin=2).T
bodyangvel1_prev = np.array([0, 0], ndmin=2).T

while not glfw.window_should_close(window):

    # id = mj.mjname2id(model,3,"root1")

    
    zf, rf, pf, yf = cntrl(data, target1, "quad1")


    data.ctrl[0] = zf[0]-rf[0]-pf[0]-yf
    data.ctrl[1] = zf[0]+rf[0]+pf[0]-yf
    data.ctrl[2] = zf[0]+rf[0]-pf[0]+yf
    data.ctrl[3] = zf[0]-rf[0]+pf[0]+yf

    for _ in range(2):
      mj.mj_step(model, data)


    # z = np.append(z, currentpos1[2])
    # x = np.append(x, currentpos1[0])
    # y = np.append(y, currentpos1[1])
    # w1 = np.append(w1, currentpr1[0])
    # w2 = np.append(w2, currentpr1[1])
    # w3 = np.append(w3, currentpr1[2])
    # timelog = np.append(timelog, data.time)

    if (data.time>=simend):
        break
      

    # if np.all(abs(target1 - currentpos1) <= tol) and data.time>=10:
    #     print("setting new target")
    #     target1 = target1 + np.array([0.1,0,0], ndmin=2).T
    #     # target2 = target2 + np.array([0.1,0,0], ndmin=2).T
    #     # target3 = target3 + np.array([0.1,0,0], ndmin=2).T
    #     # target4 = target4 + np.array([0.1,0,0], ndmin=2).T
    
    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)


    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()


glfw.terminate()


# plt.scatter(timelog,z)
# plt.scatter(timelog,x)
# plt.scatter(timelog,y)
# plt.show()
