#!/usr/bin/env python3

from doctest import debug
from typing import Concatenate
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = 'quad1pack2.xml'
simend = 20
print_camera_config = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# kthr = 0.00005
# ktrq = kthr/5
kthr = 1.08E-05
ktrq = 1.62E-07


#declaring body thrust and gyro torques as global
th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))


def init_controller(model, data):
    data.qpos[2] = 0
    data.qvel[:] = 0
    pass

def controller(model, data):
    res = np.zeros((9,1))
    mj.mju_quat2Mat(res,data.qpos[3:7])
    R = res.reshape((3,3))
    propvel = data.qvel[6:10]
    print(propvel)
    thrust = kthr*propvel**2
    torque = ktrq*propvel**2
    global th1, th2, th3, th4, tq1, tq2, tq3, tq4 
    th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th1[2], th2[2], th3[2], th4[2] = thrust
    tq1[2], tq2[2], tq3[2], tq4[2] = torque
    #print(tq1,tq2,tq3,tq4,torque)
    th1a = np.matmul(R,th1)
    th2a = np.matmul(R,th2)
    th3a = np.matmul(R,th3)
    th4a = np.matmul(R,th4)
    tq1a = np.matmul(R,tq1)
    tq2a = np.matmul(R,tq2)
    tq3a = np.matmul(R,tq3)
    tq4a = np.matmul(R,tq4)
    
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

cam.azimuth = 90 ; cam.elevation = -45 ; cam.distance =  5
cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])

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
    t = data.time
    # dr, dp, dy = 20.0*np.pi/180*np.sin(10*t), 20.0*np.pi/180*np.sin(2*t), 20.0*np.pi/180*np.sin(2*t)
    dr, dp, dy = 0, 0, 0
    zf = 50 + 10*(5-data.qpos[2])+6*(0-data.qvel[2])
    rf = 10*(dr-roll)+5*(0-rollrate)
    pf = 10*(dp-pitch)+5*(0-pitchrate)
    yf = 10*(dy-yaw)+5*(0-yawrate)
    data.ctrl[0] = zf-rf-pf-yf
    data.ctrl[1] = zf+rf+pf-yf
    data.ctrl[2] = zf+rf-pf+yf
    data.ctrl[3] = zf-rf+pf+yf
    for _ in range(5):
      mj.mj_step(model, data)
    
    print(data.qpos[2])

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
    

    
glfw.terminate()