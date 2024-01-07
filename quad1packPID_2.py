#!/usr/bin/env python3

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

kthr = 1.08E-05
ktrq = 1.62E-07

numQuads = 4

target1, target2, target3, target4 = np.zeros((10,3)), np.zeros((10,3)), np.zeros((10,3)), np.zeros((10,3))
ti = np.linspace(0, 1, num=10, endpoint=True).T

#takeoff
for i in range(len(ti)):
    target1[i,0] = -ti[i]*3.65 + 5.65; target2[i,0] = -ti[i]*3.65 + 5.65; target3[i,0] = ti[i]*3.65 - 5.65; target4[i,0] = ti[i]*3.65 - 5.65
    target1[i,1] = -ti[i]*3.65 + 5.65; target2[i,1] = ti[i]*3.65 - 5.65 ; target3[i,1] = ti[i]*3.65 - 5.65; target4[i,1] = -ti[i]*3.65 + 5.65
    target1[i,2] = ti[i]*10; target2[i,2] = ti[i]*10; target3[i,2] = ti[i]*10; target4[i,2] = ti[i]*10 

#horizontal travel
horzTargets1 = np.zeros((40,3)); horzTargets2 = np.zeros((40,3)); horzTargets3 = np.zeros((40,3)); horzTargets4 = np.zeros((40,3))
horzTargets1[:,1:3] = [2,10]; horzTargets2[:,1:3] = [-2,10]; horzTargets3[:,1:3] = [-2,10]; horzTargets4[:,1:3] = [2,10]
horzTargets1[:,0] = np.linspace(2, 15, num=40, endpoint=True).T; horzTargets2[:,0] = np.linspace(2, 15, num=40, endpoint=True).T
horzTargets3[:,0] = horzTargets2[:,0] - 4; horzTargets4[:,0] = horzTargets1[:,0] - 4

target1 = np.append(target1, horzTargets1, axis=0); target2 = np.append(target2, horzTargets2, axis=0)
target3 = np.append(target3, horzTargets3, axis=0); target4 = np.append(target4, horzTargets4, axis=0)

targetquad = np.zeros((50,3,4)) 
targetquad[:,:,0] =  target1; targetquad[:,:,1] =  target2
targetquad[:,:,2] = target3; targetquad[:,:,3] = target4


#Declaring global logger variables. number indicates the quadcopter
g_pos_log = np.zeros((1000000,3,4)); g_pry_log = np.zeros((1000000,3,4))
timelog = np.zeros((1000000)) #np.array([])


def init_controller(model, data):
    data.qpos[2] = 0.1; data.qpos[13] = 0.1; data.qpos[24] = 0.1; data.qpos[35] = 0.1
    data.qvel[:] = 0


def controller(model, data):
    """
    Applies force on to the correct propeller depending on the speed of the motor
    """
    for j in range(numQuads):
        apply_force(model, data, j+1)


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


def apply_force(model, data, quadNo):

    """ Applies force on the propellers."""

    jntId = mj.mj_name2id(model, 3, f"root{quadNo}")
    adr_start = model.jnt_qposadr[jntId]
    R = rotation_m(data, adr_start+3, adr_start+7)

    dofId = mj.mj_name2id(model, 3, f"motor{quadNo}1")
    adr_start = model.jnt_dofadr[dofId]

    propvel = data.qvel[adr_start:adr_start+4]

    thrust = kthr*propvel**2 
    torque = ktrq*propvel**2 

    # print(propvel)

    id1 = mj.mj_name2id(model, 1, f"propeller_ccw_{quadNo}1")
    id2 = mj.mj_name2id(model, 1, f"propeller_ccw_{quadNo}2")
    id3 = mj.mj_name2id(model, 1, f"propeller_cw_{quadNo}3")
    id4 = mj.mj_name2id(model, 1, f"propeller_cw_{quadNo}4")

    th1, tq1 = np.zeros((3,4)), np.zeros((3,4))
    th1[2,:] = thrust
    tq1[2,:] = torque

    th1a = np.matmul(R,th1)
    tq1a = np.matmul(R,tq1)


    #xfrc. Find XFRC position based on body id of propellers
    data.xfrc_applied[id1][:] = np.concatenate((th1a[:,0],-tq1a[:,0]),axis=0).flatten()
    data.xfrc_applied[id2][:] = np.concatenate((th1a[:,1],-tq1a[:,1]),axis=0).flatten()
    data.xfrc_applied[id3][:] = np.concatenate((th1a[:,2],tq1a[:,2]),axis=0).flatten()
    data.xfrc_applied[id4][:] = np.concatenate((th1a[:,3],tq1a[:,3]),axis=0).flatten()


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


def cntrl(data, target, quad):
    """
    Function which contains the flight controller PID code. MjData positions have been hardcoded
    
    Variables:
        qvelStart: Starting postion of angular velocity of root joints
        qvelEng: Ending position of angular velocity of root joints
    """
    
    #initializing variables for retrieving sensor data
    linaccEnd = 6*quad; linaccStart = 6*quad-3
    linVelStart = 6*quad-6; linVelEnd = linaccStart

    jntId = mj.mj_name2id(model, 3, f"root{quad}")
    qposStart = model.jnt_qposadr[jntId]; qvelStart = model.jnt_dofadr[jntId] + 3
    quatStart = qposStart + 3; quatEnd = qposStart + 7; qposEnd = qposStart + 3
    qvelEnd = qvelStart + 3

    target = np.reshape(target, (3,1))
    global g_x, g_y, g_z, timelog

    R = rotation_m(data, quatStart, quatEnd)
    cur_pry = calculate_pry(R)
    bodyangvel = calculate_ang_vel(R, cur_pry, qvelStart, qvelEnd)

    currentpos = np.array(data.qpos[qposStart:qposEnd], ndmin=2).T 
    currentvel = np.array(data.sensordata[linVelStart:linVelEnd], ndmin=2).T 
    currentacc = np.array(data.sensordata[linaccStart:linaccEnd], ndmin=2).T

    print(data.qpos[0:3])
    #Altitude model
    des_vel_z = 1.1*(target[2] - currentpos[2])
    des_thrust = 10*(des_vel_z - currentvel[2]) + 0.001*(0-currentacc[2])

    #horizontal model
    des_xy_vel = 1.1*(target[0:2]-currentpos[0:2])
    current_xy_vel = currentvel[0:2]
    current_xy_acc = currentacc[0:2]
    des_xy_ang = 0.5*(des_xy_vel - current_xy_vel) + 0.005*(0-current_xy_acc)

    des_xy_ang[1] =  des_xy_ang[1]*-1

    #Current controller does not put any constraint on des_xy_ang. This is being calculated based on velocity error.
    #As a reult des_xy_ang could still be considerable amount even though quadcopter is tilted heavily.
    #The following saturation only ensures that des_xy_ang does not exceed 18 deg but there is no check to see if
    #quadcopter is already tilted to max or not

    #saturation limit is 18 degrees
    max_ang = np.linalg.norm(des_xy_ang[0:2,:], ord=np.inf)
    if max_ang > np.pi/10:
        des_xy_ang =  (des_xy_ang*np.pi/10)/max_ang

    # p d controller for attitude model
    des_torq = 10*(des_xy_ang - cur_pry[0:2]) + 2*(0 - bodyangvel[0:2])

    # zf = 50 + dpr1[2] 
    zf_l = 50 + des_thrust
    rf_l = des_torq[1]
    pf_l = des_torq[0]
    yf_l = 5*(0-cur_pry[2]) + 1*(0-bodyangvel[2])

    return zf_l, rf_l, pf_l, yf_l, currentpos, cur_pry


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

i = 1; logcount = 0
while not glfw.window_should_close(window):

    for j in range(numQuads):
        zf, rf, pf, yf, poslog, prylog = cntrl(data, targetquad[i,:,j], j+1)

        data.ctrl[4*j] = zf[0]-rf[0]-pf[0]-yf[0]
        data.ctrl[4*j+1] = zf[0]+rf[0]+pf[0]-yf[0]
        data.ctrl[4*j+2] = zf[0]+rf[0]-pf[0]+yf[0]
        data.ctrl[4*j+3] = zf[0]-rf[0]+pf[0]+yf[0]

        g_pos_log[logcount,:,j] = poslog.T
        g_pry_log[logcount,:,j] = prylog.T


    timelog[logcount] = data.time
    logcount = logcount + 1

    for _ in range(2):
      mj.mj_step(model, data)


    if i < len(target1) and np.all(abs(target1[i,:]-np.array(data.qpos[0:3])) <= 7e-1) \
            and np.all(abs(target2[i,:]-np.array(data.qpos[11:14])) <= 7e-1) \
            and np.all(abs(target3[i,:]-np.array(data.qpos[22:25])) <= 7e-1) \
            and np.all(abs(target4[i,:]-np.array(data.qpos[33:36])) <= 7e-1):
        # print(target1[i,:]-np.array(data.qpos[0:3]))
        i = i + 1

    if (data.time>=simend):
        break
      
    # get framebuffer viewport
    # viewport_width, viewport_height = glfw.get_framebuffer_size(
    #     window)
    # viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)


    # # Update scene and render
    # mj.mjv_updateScene(model, data, opt, None, cam,
    #                    mj.mjtCatBit.mjCAT_ALL.value, scene)
    # mj.mjr_render(viewport, scene, context)

    # # swap OpenGL buffers (blocking call due to v-sync)
    # glfw.swap_buffers(window)

    # # process pending GUI events, call GLFW callbacks
    # glfw.poll_events()


glfw.terminate()

fig, pos = plt.subplots(numQuads,3)
fig, pry = plt.subplots(numQuads,3)
for j in range(numQuads):
    for i in range(3):
        if numQuads == 1:
            pos[i].plot(timelog[0:logcount], g_pos_log[0:logcount,i,j].T)
            pry[i].plot(timelog[0:logcount], g_pry_log[0:logcount,i,j].T)
        else:
            pos[j,i].plot(timelog[0:logcount], g_pos_log[0:logcount,i,j].T)
            pry[j,i].plot(timelog[0:logcount], g_pry_log[0:logcount,i,j].T)

pos[0,0].set_title("x vs Time plot")
pos[0,1].set_title("y vs Time plot")
pos[0,2].set_title("z vs Time plot")
pry[0,0].set_title("pitch vs Time plot")
pry[0,1].set_title("roll vs Time plot")
pry[0,2].set_title("yaw vs Time plot")

plt.show()

