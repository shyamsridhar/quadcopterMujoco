#!/usr/bin/env python3

from doctest import debug
from typing import Concatenate
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from matplotlib import pyplot as plt

xml_path = 'quad1pack.xml'
simend = 25
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

#attitude control pid constants matrix
kpvel = np.array([[8,0,0],[0,8,0],[0,0,5]])
kptorq = np.array([[1,0,0],[0,1,0],[0,0,3]])
kdtorq = np.array([[0.001,0,0],[0,0.001,0],[0,0,0.9]])

currentPos_g = np.zeros((3,4)); currentPry_g = np.zeros((3,4))
des_xy_ang_g = np.zeros((2,4)); zf_g = np.zeros((1,4))
rf_g = np.zeros((1,4)); pf_g = np.zeros((1,4)); yf_g = np.zeros((1,4))
actuatorSpeed = np.zeros((4,4))
targetquad = np.zeros((60,3,4)); curPackage = np.array([0, 0, 0], ndmin=2)
targetPackage = np.array([20, 0, 0], ndmin=2)
quadPos = np.array([[5.65, 5.65, 0],[5.65, -5.65, 0]
                         ,[-5.65, -5.65, 0],[-5.65, 5.65, 0]], ndmin=2)

#Declaring global logger variables. number indicates the quadcopter
g_pos_log = np.zeros((1000000,3,4)); g_pry_log = np.zeros((1000000,3,4))
timelog = np.zeros((1000000)) #np.array([])

#Control allocation matrix for X configuration quadcopter
ca_mat = np.array([[kthr, kthr, kthr, kthr],[-0.159*kthr, 0.159*kthr, 0.159*kthr, -0.159*kthr],[-0.159*kthr, 0.159*kthr, -0.159*kthr, 0.159*kthr],[-ktrq, -ktrq, ktrq, ktrq]], ndmin=2)
ca_mat_inv = np.linalg.inv(ca_mat)

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


def generatePath(curPackagePos, targetPackagePos, quadstartPos):

    path = np.zeros((60,3,4))

    waypoint1 = quadPositions(curPackagePos, 10)
    waypoint2 = quadPositions(targetPackagePos, 10)
    landingPos = quadPositions(targetPackagePos, 0)

    for i in range(4):
        path[0:10,:,i] = np.linspace(quadstartPos[i,:], waypoint1[i,:], num=10, endpoint=True)
        path[10:50,:,i] = np.linspace(waypoint1[i,:], waypoint2[i,:], num=40, endpoint=True)
        path[50:60,:,i] = np.linspace(waypoint2[i,:], landingPos[i,:], num=10, endpoint=True)

    return path


def quadPositions(packagePos, altitude):
    quad = np.zeros((4,3))
    quad[0,:] = packagePos + np.array([2, 2, altitude], ndmin=2)
    quad[1,:] = packagePos + np.array([2, -2, altitude], ndmin=2)
    quad[2,:] = packagePos + np.array([-2, -2, altitude], ndmin=2)
    quad[3,:] = packagePos + np.array([-2, 2, altitude], ndmin=2)
    # quad[0,:] = packagePos + np.array([5.65, 5.65, altitude], ndmin=2)
    # quad[1,:] = packagePos + np.array([5.65, -5.65, altitude], ndmin=2)
    # quad[2,:] = packagePos + np.array([-5.65, -5.65, altitude], ndmin=2)
    # quad[3,:] = packagePos + np.array([-5.65, 5.65, altitude], ndmin=2)
    return quad


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


def pos_cntrl(data, target, quad):
    """ 
    Function which contains the position controller
    """
    #initializing variables for retrieving sensor data
    linaccEnd = 6*quad; linaccStart = 6*quad-3
    linVelStart = 6*quad-6; linVelEnd = linaccStart

    jntId = mj.mj_name2id(model, 3, f"root{quad}")
    qposStart = model.jnt_qposadr[jntId]; qposEnd = qposStart + 3

    target = np.reshape(target, (3,1))

    currentpos = np.array(data.qpos[qposStart:qposEnd], ndmin=2).T #+ np.random.normal(loc=0.0, scale=0.1, size=(3,1))
    currentvel = np.array(data.sensordata[linVelStart:linVelEnd], ndmin=2).T #+ np.random.normal(loc=0.0, scale=0.05, size=(3,1))
    currentacc = np.array(data.sensordata[linaccStart:linaccEnd], ndmin=2).T #+ np.random.normal(loc=0.0, scale=0.05, size=(3,1))

    # print(data.qpos[0:3])
    #Altitude model
    des_vel_z = 3*(target[2] - currentpos[2])
    # des_vel_z = des_vel_z/np.linalg.norm(des_vel_z)
    des_thrust = 5*(des_vel_z - currentvel[2]) + 0.01*(des_vel_z-currentvel[2])/0.01

    if data.time>9:
        breakpoint
    #horizontal model
    des_xy_vel = 1.1*(target[0:2]-currentpos[0:2])
    # des_xy_vel = 1*des_xy_vel/np.linalg.norm(des_xy_vel)
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

    zf_l = (10 + des_thrust)*1.5

    #saturating max thrust desired to 50.73N for air 2216 motors
    zf_l = max(min(zf_l, 50),0)
    
    return des_xy_ang, zf_l, currentpos


def att_cntrl(data, des_ang, quad):

    des_ang = np.reshape(des_ang,(2,1))
    des_ang = np.append(des_ang, [[0]], axis=0)
    jntId = mj.mj_name2id(model, 3, f"root{quad}")
    qposStart = model.jnt_qposadr[jntId]; qvelStart = model.jnt_dofadr[jntId] + 3
    quatStart = qposStart + 3; quatEnd = qposStart + 7; qvelEnd = qvelStart + 3

    R = rotation_m(data, quatStart, quatEnd)
    cur_pry = calculate_pry(R)
    bodyangvel = calculate_ang_vel(R, cur_pry, qvelStart, qvelEnd)

    # p d controller for attitude model
    # des_torq = 10*(des_xy_ang - cur_pry[0:2]) + 2*(0 - bodyangvel[0:2])
    # des_angvel = 8*(des_ang - cur_pry)
    # des_torq = (des_angvel - bodyangvel) + 0.0005*(des_angvel-bodyangvel)/0.002
    des_angvel = np.matmul(kpvel, (des_ang - cur_pry))
    des_torq = np.matmul(kptorq, (des_angvel - bodyangvel)) + np.matmul(kdtorq, (des_angvel-bodyangvel)/0.002)

    rf_l = des_torq[1]
    pf_l = des_torq[0]
    yf_l = des_torq[2]

    #saturating max yaw torque 0.18N-m per motor
    yf_l = max(min(yf_l, 0.10),-0.10)

    #direction guaranteed saturation for roll and pitch
    max_torq = np.linalg.norm(des_torq[0:2,:], ord=np.inf)
    if max_torq > 1:
        des_torq = (des_torq[0:2,:]*1)/max_torq

    rf_l = des_torq[1]
    pf_l = des_torq[0]

    # yf_l = 5*(0-cur_pry[2]) + 0.5*(0-bodyangvel[2])



    return rf_l, pf_l, yf_l, cur_pry


def cntrl(data, target, quad):
    """
    Function which contains the flight controller PID code. 
    
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

    currentpos = np.array(data.qpos[qposStart:qposEnd], ndmin=2).T #+ np.random.normal(loc=0.0, scale=0.1, size=(3,1))
    currentvel = np.array(data.sensordata[linVelStart:linVelEnd], ndmin=2).T #+ np.random.normal(loc=0.0, scale=0.05, size=(3,1))
    currentacc = np.array(data.sensordata[linaccStart:linaccEnd], ndmin=2).T #+ np.random.normal(loc=0.0, scale=0.05, size=(3,1))

    print(data.qpos[0:3])
    #Altitude model
    des_vel_z = 1.1*(target[2] - currentpos[2])
    # des_vel_z = des_vel_z/np.linalg.norm(des_vel_z)
    des_thrust = 10*(des_vel_z - currentvel[2]) + 0.001*(des_vel_z-currentvel[2])/0.002

    if data.time>9:
        breakpoint
    #horizontal model
    des_xy_vel = 1.1*(target[0:2]-currentpos[0:2])
    # des_xy_vel = 1*des_xy_vel/np.linalg.norm(des_xy_vel)
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
    # des_torq = 10*(des_xy_ang - cur_pry[0:2]) + 2*(0 - bodyangvel[0:2])
    des_angvel = 8*(des_xy_ang - cur_pry[0:2])
    des_torq = (des_angvel - bodyangvel[0:2]) + 0.0005*(des_angvel-bodyangvel[0:2])/0.002

    # zf = 50 + dpr1[2] 
    zf_l = 50 + des_thrust
    rf_l = des_torq[1]
    pf_l = des_torq[0]
    yf_l = 5*(0-cur_pry[2]) + 0.5*(0-bodyangvel[2])

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
cam.type=1; cam.trackbodyid=5
cam.azimuth = 90 ; cam.elevation = -45 ; cam.distance =  10

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

#generate path for all the quadcopters
targetquad = generatePath(curPackage, targetPackage, quadPos)

i = 1; logcount = 0
while not glfw.window_should_close(window):

    #position controller
    for j in range(numQuads):
        des_xy_ang, zf_g[:,j], poslog = pos_cntrl(data, targetquad[i,:,j], j+1)
        des_xy_ang_g[:,j] = des_xy_ang[:,0]; currentPos_g[:,j] = poslog[:,0] 
    
    for _ in range(5):
        #attitude controller
        for j in range(numQuads):
            rf_g[:,j], pf_g[:,j], yf_g[:,j], prylog = att_cntrl(data, des_xy_ang_g[:,j], j+1)
            ft_mat = np.array([zf_g[:,j], rf_g[:,j], pf_g[:,j], yf_g[:,j]], ndmin=2)
            actuatorSpeed[:,j] = np.sqrt(np.matmul(ca_mat_inv, ft_mat))[:,0]
            g_pos_log[logcount,:,j] = currentPos_g[:,j].T
            g_pry_log[logcount,:,j] = prylog.T

            #linear relation assumption based on manufacturers data
            actuatorCtrl = 0.10622*actuatorSpeed[:,j] - 18.60
            print(actuatorCtrl)

            #actuator saturation needs to be implemented


            data.ctrl[4*j] = actuatorCtrl[0]
            data.ctrl[4*j+1] = actuatorCtrl[1]
            data.ctrl[4*j+2] = actuatorCtrl[2]
            data.ctrl[4*j+3] = actuatorCtrl[3]


        timelog[logcount] = data.time
        logcount = logcount + 1

        for _ in range(2):
            mj.mj_step(model, data)

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


    # # # # # # # for j in range(numQuads):
    # # # # # # #     zf, rf, pf, yf, poslog, prylog = cntrl(data, targetquad[i,:,j], j+1)

    # # # # # # #     # ft_mat = np.array([zf, rf, pf, yf], ndmin=2)
    # # # # # # #     # actuatorSpeed = np.matmul(ca_mat_inv, ft_mat)



    # # # # # # #     data.ctrl[4*j] = zf[0]-rf[0]-pf[0]-yf[0]
    # # # # # # #     data.ctrl[4*j+1] = zf[0]+rf[0]+pf[0]-yf[0]
    # # # # # # #     data.ctrl[4*j+2] = zf[0]+rf[0]-pf[0]+yf[0]
    # # # # # # #     data.ctrl[4*j+3] = zf[0]-rf[0]+pf[0]+yf[0]

    # # # # # # #     # print(data.ctrl[4*j], data.ctrl[4*j+1], data.ctrl[4*j+2], data.ctrl[4*j+3])

    # # # # # # #     g_pos_log[logcount,:,j] = poslog.T
    # # # # # # #     g_pry_log[logcount,:,j] = prylog.T


    # # # # # # # timelog[logcount] = data.time
    # # # # # # # logcount = logcount + 1

    # # # # # # # for _ in range(2):
    # # # # # # #   mj.mj_step(model, data)


    if i < len(targetquad)-1 and np.all(abs(targetquad[i,:,0]-np.array(data.qpos[0:3])) <= 15e-1) \
            and np.all(abs(targetquad[i,:,1]-np.array(data.qpos[11:14])) <= 15e-1) \
            and np.all(abs(targetquad[i,:,2]-np.array(data.qpos[22:25])) <= 15e-1) \
            and np.all(abs(targetquad[i,:,3]-np.array(data.qpos[33:36])) <= 15e-1):
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

