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
from matplotlib import pyplot

xml_path = 'quad1pack.xml'
simend = 20
print_camera_config = 0
mot1 = []; mot2 = []; mot3 = []; mot4= []

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

#target for quad1
target1 = np.array([1, 0, 7], ndmin=2).T

# kthr = 0.00005
# ktrq = kthr/5
kthr = 1.08E-05
ktrq = 1.62E-07

kp=0.2*50
kd=6
kpr=20
kdr=20


#arm positions
r1 = np.array([0.1591,-0.1591,0],ndmin=2).T
r2 =  np.array([-0.1591,0.1591,0],ndmin=2).T
r3 =  np.array([0.1591,0.1591,0],ndmin=2).T
r4 =  np.array([-0.1591,-0.1591,0],ndmin=2).T

#declaring body thrust and gyro torques as global
th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))

def init_controller(model, data):
    data.qpos[2] = 0#; data.qpos[0] = 0.5
    data.qvel[:] = 0
    pass

def controller(model, data):
    res = np.zeros((9,1))
    mj.mju_quat2Mat(res,data.qpos[3:7])

    R = res.reshape((3,3))

    propvel = data.qvel[6:10]

    propvel[0] = max(min(2000, propvel[0]), 0)
    propvel[1] = max(min(2000, propvel[1]), 0)
    propvel[2] = max(min(0, propvel[2]), -2000)
    propvel[3] = max(min(0, propvel[3]), -2000)

    thrust = kthr*propvel**2
    torque = ktrq*propvel**2

    print(propvel)

    global th1, th2, th3, th4, tq1, tq2, tq3, tq4
    th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))

    th1[2], th2[2], th3[2], th4[2] = thrust

    tq1[2], tq2[2], tq3[2], tq4[2] = torque
 

    th1a = np.matmul(R,th1)
    th2a = np.matmul(R,th2)
    th3a = np.matmul(R,th3)
    th4a = np.matmul(R,th4)
    tq1a = np.matmul(R,tq1)
    tq2a = np.matmul(R,tq2)
    tq3a = np.matmul(R,tq3)
    tq4a = np.matmul(R,tq4)

    #xfrc. Find XFRC position based on body id
    data.xfrc_applied[7][:] = np.concatenate((th1a,-tq1a),axis=0).flatten()
    data.xfrc_applied[9][:] = np.concatenate((th2a,-tq2a),axis=0).flatten()
    data.xfrc_applied[11][:] = np.concatenate((th3a,tq3a),axis=0).flatten()
    data.xfrc_applied[13][:] = np.concatenate((th4a,tq4a),axis=0).flatten()


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

while not glfw.window_should_close(window):
    res = np.zeros((9,1))

    mj.mju_quat2Mat(res,data.qpos[3:7])
    R = res.reshape((3,3))

    roll = np.arctan2(R[2][1],R[2][2])
    pitch = np.arcsin(-R[2][0])
    yaw = np.arctan2(R[1][0],R[0][0])

    absangvel = data.qvel[3:6].reshape((3,1))
    bdyangvel = np.matmul(R.T,absangvel)
    Tmatinv = np.array([[1,np.sin(roll)*np.tan(pitch),np.cos(roll)*np.tan(pitch)],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll)/np.cos(pitch),np.cos(roll)/np.cos(pitch)]])
    rpyrates = np.matmul(Tmatinv,bdyangvel)
    rollrate, pitchrate, yawrate = rpyrates.flatten()

    bodyangvel1 = np.array([pitchrate, rollrate, yawrate], ndmin=2).T
    # bodyangvel1 = np.array([data.qvel[4], data.qvel[3], data.qvel[5]], ndmin=2).T
    
    t = data.time

    ##based on quadcopter matlab
    # currentpos1 = np.array(data.sensordata[24:27], ndmin=2).T; currentpos2 = np.array(data.sensordata[27:30], ndmin=2).T; currentpos3 = np.array(data.sensordata[30:33], ndmin=2).T; currentpos4 = np.array(data.sensordata[33:36], ndmin=2).T
    currentpos1 = np.array(data.qpos[0:3], ndmin=2).T
    currentvel1 = np.array(data.sensordata[12:15], ndmin=2).T
    currentpr1 = np.array([pitch, roll, yaw], ndmin=2).T
    # currentrp1vel = np.array(data.sensordata[0:3], ndmin=2); currentrp2vel = np.array(data.sensordata[3:6], ndmin=2); currentrp3vel = np.array(data.sensordata[6:9], ndmin=2); currentrp4vel = np.array(data.sensordata[9:12], ndmin=2)
    dpr1 = kp*(target1-currentpos1) + kd*(0-currentvel1)
    
    #calculate desired rp angles in body frame
    # dpr1body = np.matmul(R.T, target1-currentpos1)
    dpr1body = (target1-currentpos1)*0.1

    dpr1body[1] = dpr1body[1]*(-1)
    

    #saturation limit to 60 degrees
    dpr1body[0] = max(min(np.pi/3, dpr1body[0]), -np.pi/3); dpr1body[1] = max(min(np.pi/3, dpr1body[1]), -np.pi/3)

    dpr1error = dpr1body-currentpr1
    prf1 = 10*dpr1error - 0.5*(bodyangvel1)

    zf = 50 + dpr1[2] #+ prf1[1] + prf1[0]
    rf = prf1[1]
    pf = prf1[0]
    yf = 5*(0-yaw) + 1*(0-yawrate)
    # yf = kpr*(0 - yaw) - kdr*(bodyangvel1[2])

    print(currentpos1)
    #quad1
    data.ctrl[0] = zf-rf-pf-yf
    data.ctrl[1] = zf+rf+pf-yf
    data.ctrl[2] = zf+rf-pf+yf
    data.ctrl[3] = zf-rf+pf+yf

    # pyplot.figure(1)
    # pyplot.plot(data.ctrl[0], data.time)
    # pyplot.show()

    for _ in range(2):
      mj.mj_step(model, data)


    if (data.time>=simend):
        break

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


# pyplot.plot(mot1)
# pyplot.show()

glfw.terminate()
