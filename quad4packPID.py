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
import ahrs

xml_path = 'quad1pack.xml'
simend = 30
print_camera_config = 0
mot1 = []; mot2 = []; mot3 = []; mot4= []

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

#target for quad1
target1 = np.array([0.98, 0.98, 5], ndmin=2).T

# tol = np.array([3*0.1, 3*0.1, 3*0.1], ndmin=2).T

# kthr = 0.00005
# ktrq = kthr/5
kthr = 1.08E-05
ktrq = 1.62E-07

k_att = 5
k_par = 1
k_dar = 0.5



#declaring body thrust and gyro torques as global
th1, tq1 = np.zeros((3,4)), np.zeros((3,4))

def init_controller(model, data):
    data.qpos[2] = 0.1
    data.qvel[:] = 0
    pass

def controller(model, data):
    res = np.zeros((9,1))
    mj.mju_quat2Mat(res,data.qpos[3:7])

    R = res.reshape((3,3))

    propvel = data.qvel[6:10]

    thrust = kthr*propvel**2 
    torque = ktrq*propvel**2 

    # print(propvel)

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

while not glfw.window_should_close(window):

    currentpos1 = np.array(data.qpos[0:3], ndmin=2).T
    currentvel1 = np.array(data.sensordata[0:3], ndmin=2).T
    currentaccel1 = np.array(data.sensordata[3:6], ndmin=2).T

    
    #Position controller P-PID controle
    velocity_sp = (target1 - currentpos1)*5 
    acceleration_sp = (velocity_sp-currentvel1)*5 - (currentaccel1)*1 # need to add integral controller


    #finding quaternion setpoint from acceleration set point using reduced attitude control

    des_thrust_dir = acceleration_sp/np.linalg.norm(acceleration_sp)
    des_thrust = np.linalg.norm(acceleration_sp)

    if des_thrust_dir[2] < 0:
        des_thrust = des_thrust*-1
 
    #limiting maximum tilt angle
    


    #obtaining desired roll pitch and yaw angles from des_trhust_dir
    #rotation about y axis causes pitch and x axis causes roll
    des_yaw = 0
    des_pitch = np.arctan(des_thrust_dir[0],des_thrust_dir[2])

    #rotate des_thrust_dir in pitch direction

    R_pitch = np.array([[np.cos(des_pitch[0]), 0, -np.sin(des_pitch[0])],[0, 1, 0],[np.sin(des_pitch[0]), 0, np.cos(des_pitch[0])]])
    new_direction = np.matmul(R_pitch,des_thrust_dir)

    des_roll = np.arctan2(-new_direction[1],new_direction[2])
    euler = np.array([des_roll[0], des_pitch[0], des_yaw], ndmin=2)
    
    des_att_quat = np.array([np.cos(0.5*des_roll[0])*np.cos(0.5*des_pitch[0])*np.cos(0.5*des_yaw) + np.sin(0.5*des_roll[0])*np.sin(0.5*des_pitch[0])*np.sin(0.5*des_yaw)
                             ,np.sin(0.5*des_roll[0])*np.cos(0.5*des_pitch[0])*np.cos(0.5*des_yaw) - np.cos(0.5*des_roll[0])*np.sin(0.5*des_pitch[0])*np.sin(0.5*des_yaw)
                             ,np.cos(0.5*des_roll[0])*np.sin(0.5*des_pitch[0])*np.cos(0.5*des_yaw) + np.sin(0.5*des_roll[0])*np.cos(0.5*des_pitch[0])*np.sin(0.5*des_yaw)
                             ,np.cos(0.5*des_roll[0])*np.cos(0.5*des_pitch[0])*np.sin(0.5*des_yaw) - np.sin(0.5*des_roll[0])*np.sin(0.5*des_pitch[0])*np.cos(0.5*des_yaw)])

    cur_att_quat = np.array(data.qpos[3:7])

    #attitude controller
    cur_quat = ahrs.Quaternion(cur_att_quat)
    des_quat = ahrs.Quaternion(des_att_quat)
    cur_quat_inv = ahrs.Quaternion(cur_quat.inverse)

    error_att_quat = cur_quat_inv.product(des_quat)

    if error_att_quat[0] < 0:
        des_ang = np.array(error_att_quat[1:4] * -1 * k_att, ndmin=2).T
    else:
        des_ang = np.array(error_att_quat[1:4] * k_att, ndmin=2).T


    #attitude rate controller
    res = np.zeros((9,1))
    mj.mju_quat2Mat(res,data.qpos[3:7])
    R = res.reshape((3,3))
    roll = np.arctan2(R[2][1],R[2][2])
    pitch = np.arcsin(-R[2][0])
    yaw = np.arctan2(R[1][0],R[0][0])

    currentpr1 = np.array([pitch, roll, yaw], ndmin=2).T


    absangvel = data.qvel[3:6].reshape((3,1))
    bdyangvel = np.matmul(R.T,absangvel)
    Tmatinv = np.array([[1,np.sin(roll)*np.tan(pitch),np.cos(roll)*np.tan(pitch)],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll)/np.cos(pitch),np.cos(roll)/np.cos(pitch)]])
    rpyrates = np.matmul(Tmatinv,bdyangvel)
    rollrate, pitchrate, yawrate = rpyrates.flatten()

    bodyangvel1 = np.array([pitchrate, rollrate, yawrate], ndmin=2).T


    des_ang_rate = (des_ang-currentpr1)*k_par - bodyangvel1*k_dar

    zf = 50 + des_thrust
    rf = des_ang_rate[1]
    pf = des_ang_rate[0]
    yf = des_ang_rate[2]

    print(currentpos1[0], currentpos1[1], currentpos1[2])
    #quad1
    data.ctrl[0] = zf-rf[0]-pf[0]-yf[0]
    data.ctrl[1] = zf+rf[0]+pf[0]-yf[0]
    data.ctrl[2] = zf+rf[0]-pf[0]+yf[0]
    data.ctrl[3] = zf-rf[0]+pf[0]+yf[0]


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
