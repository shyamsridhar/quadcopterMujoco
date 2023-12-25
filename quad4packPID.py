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

xml_path = 'quad4pack.xml'
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
target1 = np.array([1, 0, 5], ndmin=2).T
target2 = np.array([-1, 0, 5], ndmin=2).T
target3 = np.array([0, 1, 5], ndmin=2).T
target4 = np.array([0, -1, 5], ndmin=2).T

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



#declaring body thrust and gyro torques as global
th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
th21, th22, th23, th24, tq21, tq22, tq23, tq24 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
th31, th32, th33, th34, tq31, tq32, tq33, tq34 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
th41, th42, th43, th44, tq41, tq42, tq43, tq44 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))


def init_controller(model, data):
    data.qpos[2] = 0; data.qpos[13] = 0; data.qpos[24] = 0; data.qpos[35] = 0
    data.qvel[:] = 0
    pass

def controller(model, data):
    res = np.zeros((9,1)); res2 = np.zeros((9,1)); res3 = np.zeros((9,1)); res4 = np.zeros((9,1))
    mj.mju_quat2Mat(res,data.qpos[3:7]); mj.mju_quat2Mat(res2,data.qpos[14:18]); mj.mju_quat2Mat(res3,data.qpos[25:29]); mj.mju_quat2Mat(res4,data.qpos[36:40])

    R = res.reshape((3,3)); R2 = res2.reshape((3,3)); R3 = res3.reshape((3,3)); R4 = res4.reshape((3,3))

    propvel = data.qvel[6:10]; propvel2 = data.qvel[16:20]; propvel3 = data.qvel[26:30]; propvel4 = data.qvel[36:40]

    propvel[0] = max(min(2000, propvel[0]), 0); propvel2[0] = max(min(2000, propvel2[0]), 0); propvel3[0] = max(min(2000, propvel3[0]), 0); propvel4[0] = max(min(2000, propvel4[0]), 0)
    propvel[1] = max(min(2000, propvel[1]), 0); propvel2[1] = max(min(2000, propvel2[1]), 0); propvel3[1] = max(min(2000, propvel3[1]), 0); propvel4[1] = max(min(2000, propvel4[1]), 0)
    propvel[2] = max(min(0, propvel[2]), -2000); propvel2[2] = max(min(0, propvel2[2]), -2000); propvel3[2] = max(min(0, propvel3[2]), -2000); propvel4[2] = max(min(0, propvel4[2]), -2000)
    propvel[3] = max(min(0, propvel[3]), -2000); propvel2[3] = max(min(0, propvel2[3]), -2000); propvel3[3] = max(min(0, propvel3[3]), -2000); propvel4[3] = max(min(0, propvel4[3]), -2000)

    thrust = kthr*propvel**2; thrust2 = kthr*propvel2**2; thrust3 = kthr*propvel3**2; thrust4 = kthr*propvel4**2
    torque = ktrq*propvel**2; torque2 = ktrq*propvel2**2; torque3 = ktrq*propvel3**2; torque4 = ktrq*propvel4**2

    # print(propvel)

    global th1, th2, th3, th4, tq1, tq2, tq3, tq4, th21, th22, th23, th24, tq21, tq22, tq23, tq24, th31, th32, th33, th34, tq31, tq32, tq33, tq34, th41, th42, th43, th44, tq41, tq42, tq43, tq44
    th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th21, th22, th23, th24, tq21, tq22, tq23, tq24 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th31, th32, th33, th34, tq31, tq32, tq33, tq34 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th41, th42, th43, th44, tq41, tq42, tq43, tq44 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))

    th1[2], th2[2], th3[2], th4[2] = thrust
    tq1[2], tq2[2], tq3[2], tq4[2] = torque
    th21[2], th22[2], th23[2], th24[2] = thrust2
    tq21[2], tq22[2], tq23[2], tq24[2] = torque2
    th31[2], th32[2], th33[2], th34[2] = thrust3
    tq31[2], tq32[2], tq33[2], tq34[2] = torque3
    th41[2], th42[2], th43[2], th44[2] = thrust4
    tq41[2], tq42[2], tq43[2], tq44[2] = torque4
 

    th1a = np.matmul(R,th1)
    th2a = np.matmul(R,th2)
    th3a = np.matmul(R,th3)
    th4a = np.matmul(R,th4)
    tq1a = np.matmul(R,tq1)
    tq2a = np.matmul(R,tq2)
    tq3a = np.matmul(R,tq3)
    tq4a = np.matmul(R,tq4)

    th21a = np.matmul(R2,th21)
    th22a = np.matmul(R2,th22)
    th23a = np.matmul(R2,th23)
    th24a = np.matmul(R2,th24)
    tq21a = np.matmul(R2,tq21)
    tq22a = np.matmul(R2,tq22)
    tq23a = np.matmul(R2,tq23)
    tq24a = np.matmul(R2,tq24)

    th31a = np.matmul(R3,th31)
    th32a = np.matmul(R3,th32)
    th33a = np.matmul(R3,th33)
    th34a = np.matmul(R3,th34)
    tq31a = np.matmul(R3,tq31)
    tq32a = np.matmul(R3,tq32)
    tq33a = np.matmul(R3,tq33)
    tq34a = np.matmul(R3,tq34)

    th41a = np.matmul(R4,th41)
    th42a = np.matmul(R4,th42)
    th43a = np.matmul(R4,th43)
    th44a = np.matmul(R4,th44)
    tq41a = np.matmul(R4,tq41)
    tq42a = np.matmul(R4,tq42)
    tq43a = np.matmul(R4,tq43)
    tq44a = np.matmul(R4,tq44)

    #xfrc. Find XFRC position based on body id
    data.xfrc_applied[7][:] = np.concatenate((th1a,-tq1a),axis=0).flatten()
    data.xfrc_applied[9][:] = np.concatenate((th2a,-tq2a),axis=0).flatten()
    data.xfrc_applied[11][:] = np.concatenate((th3a,tq3a),axis=0).flatten()
    data.xfrc_applied[13][:] = np.concatenate((th4a,tq4a),axis=0).flatten()

    data.xfrc_applied[20][:] = np.concatenate((th21a,-tq21a),axis=0).flatten()
    data.xfrc_applied[22][:] = np.concatenate((th22a,-tq22a),axis=0).flatten()
    data.xfrc_applied[24][:] = np.concatenate((th23a,tq23a),axis=0).flatten()
    data.xfrc_applied[26][:] = np.concatenate((th24a,tq24a),axis=0).flatten()

    data.xfrc_applied[33][:] = np.concatenate((th31a,-tq31a),axis=0).flatten()
    data.xfrc_applied[35][:] = np.concatenate((th32a,-tq32a),axis=0).flatten()
    data.xfrc_applied[37][:] = np.concatenate((th33a,tq33a),axis=0).flatten()
    data.xfrc_applied[39][:] = np.concatenate((th34a,tq34a),axis=0).flatten()

    data.xfrc_applied[46][:] = np.concatenate((th41a,-tq41a),axis=0).flatten()
    data.xfrc_applied[48][:] = np.concatenate((th42a,-tq42a),axis=0).flatten()
    data.xfrc_applied[50][:] = np.concatenate((th43a,tq43a),axis=0).flatten()
    data.xfrc_applied[52][:] = np.concatenate((th44a,tq44a),axis=0).flatten()

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

error = np.array([0, 0, 0], ndmin=2).T; error2 = np.array([0, 0, 0], ndmin=2).T
error3 = np.array([0, 0, 0], ndmin=2).T; error4 = np.array([0, 0, 0], ndmin=2).T

while not glfw.window_should_close(window):
    res = np.zeros((9,1)); res2 = np.zeros((9,1)); res3 = np.zeros((9,1)); res4 = np.zeros((9,1))

    mj.mju_quat2Mat(res,data.qpos[3:7]); mj.mju_quat2Mat(res2,data.qpos[14:18]); mj.mju_quat2Mat(res3,data.qpos[25:29]); mj.mju_quat2Mat(res4,data.qpos[36:40])
    R = res.reshape((3,3)); R2 = res2.reshape((3,3)); R3 = res3.reshape((3,3)); R4 = res4.reshape((3,3))

    roll = np.arctan2(R[2][1],R[2][2]); roll2 = np.arctan2(R2[2][1],R2[2][2]); roll3 = np.arctan2(R3[2][1],R3[2][2]); roll4 = np.arctan2(R4[2][1],R4[2][2])
    pitch = np.arcsin(-R[2][0]); pitch2 = np.arcsin(-R2[2][0]); pitch3 = np.arcsin(-R3[2][0]); pitch4 = np.arcsin(-R4[2][0])
    yaw = np.arctan2(R[1][0],R[0][0]); yaw2 = np.arctan2(R2[1][0],R2[0][0]); yaw3 = np.arctan2(R3[1][0],R3[0][0]); yaw4 = np.arctan2(R4[1][0],R4[0][0])

    #drone1
    absangvel = data.qvel[3:6].reshape((3,1))
    bdyangvel = np.matmul(R.T,absangvel)
    Tmatinv = np.array([[1,np.sin(roll)*np.tan(pitch),np.cos(roll)*np.tan(pitch)],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll)/np.cos(pitch),np.cos(roll)/np.cos(pitch)]])
    rpyrates = np.matmul(Tmatinv,bdyangvel)
    rollrate, pitchrate, yawrate = rpyrates.flatten()

    bodyangvel1 = np.array([pitchrate, rollrate, yawrate], ndmin=2).T
    # bodyangvel1 = np.array([data.qvel[4], data.qvel[3], data.qvel[5]], ndmin=2).T
    

    #drone2
    absangvel2 = data.qvel[13:16].reshape((3,1))
    bdyangvel2 = np.matmul(R2.T,absangvel2)
    Tmatinv2 = np.array([[1,np.sin(roll2)*np.tan(pitch2),np.cos(roll2)*np.tan(pitch2)],[0,np.cos(roll2),-np.sin(roll2)],[0,np.sin(roll2)/np.cos(pitch2),np.cos(roll2)/np.cos(pitch2)]])
    rpyrates2 = np.matmul(Tmatinv2,bdyangvel2)
    rollrate2, pitchrate2, yawrate2 = rpyrates2.flatten()

    bodyangvel21 = np.array([pitchrate2, rollrate2, yawrate2], ndmin=2).T      

    #drone3
    absangvel3 = data.qvel[23:26].reshape((3,1))
    bdyangvel3 = np.matmul(R3.T,absangvel3)
    Tmatinv3 = np.array([[1,np.sin(roll3)*np.tan(pitch3),np.cos(roll3)*np.tan(pitch3)],[0,np.cos(roll3),-np.sin(roll3)],[0,np.sin(roll3)/np.cos(pitch3),np.cos(roll3)/np.cos(pitch3)]])
    rpyrates3 = np.matmul(Tmatinv3,bdyangvel3)
    rollrate3, pitchrate3, yawrate3 = rpyrates3.flatten()

    bodyangvel31 = np.array([pitchrate3, rollrate3, yawrate3], ndmin=2).T

    #drone4
    absangvel4 = data.qvel[33:36].reshape((3,1))
    bdyangvel4 = np.matmul(R4.T,absangvel4)
    Tmatinv4 = np.array([[1,np.sin(roll4)*np.tan(pitch4),np.cos(roll4)*np.tan(pitch4)],[0,np.cos(roll4),-np.sin(roll4)],[0,np.sin(roll4)/np.cos(pitch4),np.cos(roll4)/np.cos(pitch4)]])
    rpyrates4 = np.matmul(Tmatinv4,bdyangvel4)
    rollrate4, pitchrate4, yawrate4 = rpyrates4.flatten()

    bodyangvel41 = np.array([pitchrate4, rollrate4, yawrate4], ndmin=2).T

    t = data.time

    ##based on quadcopter matlab
    # currentpos1 = np.array(data.sensordata[24:27], ndmin=2).T; currentpos2 = np.array(data.sensordata[27:30], ndmin=2).T; currentpos3 = np.array(data.sensordata[30:33], ndmin=2).T; currentpos4 = np.array(data.sensordata[33:36], ndmin=2).T
    currentpos1 = np.array(data.qpos[0:3], ndmin=2).T; currentpos2 = np.array(data.qpos[11:14], ndmin=2).T; currentpos3 = np.array(data.qpos[22:25], ndmin=2).T; currentpos4 = np.array(data.qpos[33:36], ndmin=2).T
    currentvel1 = np.array(data.sensordata[0:3], ndmin=2).T; currentvel2 = np.array(data.sensordata[3:6], ndmin=2).T; currentvel3 = np.array(data.sensordata[6:9], ndmin=2).T; currentvel4 = np.array(data.sensordata[9:12], ndmin=2).T; currentppos = np.array(data.sensordata[12:15], ndmin=2).T
    currentpr1 = np.array([pitch, roll, yaw], ndmin=2).T; currentpr2 = np.array([pitch2, roll2, yaw2], ndmin=2).T; currentpr3 = np.array([pitch3, roll3, yaw3], ndmin=2).T; currentpr4 = np.array([pitch4, roll4, yaw4], ndmin=2).T
    # currentrp1vel = np.array(data.sensordata[0:3], ndmin=2); currentrp2vel = np.array(data.sensordata[3:6], ndmin=2); currentrp3vel = np.array(data.sensordata[6:9], ndmin=2); currentrp4vel = np.array(data.sensordata[9:12], ndmin=2)

    #integral error
    if data.time > 2:
        error = error + (target1-currentpos1)*(data.time-t_1)
        error2 = error2 + (target2-currentpos2)*(data.time-t_1)
        error3 = error3 + (target3-currentpos3)*(data.time-t_1)
        error4 = error4 + (target4-currentpos4)*(data.time-t_1)
        t_1 = data.time
    else:
        t_1 = data.time

    dpr1 = kp*(target1-currentpos1) + kd*(0-currentvel1) + ki*error
    dpr2 = kp*(target2-currentpos2) + kd*(0-currentvel2) + ki*error2
    dpr3 = kp*(target3-currentpos3) + kd*(0-currentvel3) + ki*error3
    dpr4 = kp*(target4-currentpos4) + kd*(0-currentvel4) + ki*error4
    
    #calculate desired rp angles in body frame
    # dpr1body = np.matmul(R.T, target1-currentpos1)
    # if target1[0]-currentpos1[0] > abs(4):
    #     dpr1body = (target1-currentpos1)*0.1 + 0.05*(0-currentvel1)
    # elif target1[0]-currentpos1[0] > abs(2):
    #     dpr1body = (target1-currentpos1)*0.01 + 0.001*(0 - currentvel1)
    # else:
    dpr1body = (target1-currentpos1)*0.1*50 + 0.05*50*(0-currentvel1)
    dpr2body = (target2-currentpos2)*0.1*50 + 0.05*50*(0-currentvel2)
    dpr3body = (target3-currentpos3)*0.1*50 + 0.05*50*(0-currentvel3)
    dpr4body = (target4-currentpos4)*0.1*50 + 0.05*50*(0-currentvel4)
    
    dpr1body[1] = dpr1body[1]*(-1)
    dpr2body[1] = dpr2body[1]*(-1)
    dpr3body[1] = dpr3body[1]*(-1)
    dpr4body[1] = dpr4body[1]*(-1)
    

    #saturation limit to 60 degrees
    dpr1body[0] = max(min(np.pi/4, dpr1body[0]), -np.pi/4); dpr1body[1] = max(min(np.pi/4, dpr1body[1]), -np.pi/4)
    dpr2body[0] = max(min(np.pi/4, dpr2body[0]), -np.pi/4); dpr2body[1] = max(min(np.pi/4, dpr2body[1]), -np.pi/4)
    dpr3body[0] = max(min(np.pi/4, dpr3body[0]), -np.pi/4); dpr3body[1] = max(min(np.pi/4, dpr3body[1]), -np.pi/4)
    dpr4body[0] = max(min(np.pi/4, dpr4body[0]), -np.pi/4); dpr4body[1] = max(min(np.pi/4, dpr4body[1]), -np.pi/4)


    dpr1error = dpr1body-currentpr1
    dpr2error = dpr2body-currentpr2
    dpr3error = dpr3body-currentpr3
    dpr4error = dpr4body-currentpr4

    if data.time > 1:
        breakpoint

    # if data.time > 5:
    #     dpr1error1 = dpr1error1 + dpr1error*(data.time-t_2)
    #     dpr2error1 = dpr2error1 + dpr2error*(data.time-t_2)
    #     # print(data.time-t_2)
    #     t_2 = data.time
    # else:
    #     dpr1error1 = dpr1error
    #     dpr2error1 = dpr2error
    #     t_2 = data.time



    prf1 = 5*2*dpr1error + 2*(0-bodyangvel1) #+ 1*dpr1error1
    prf2 = 5*2*dpr2error + 2*(0-bodyangvel21)  #+ 1*dpr2error1
    prf3 = 5*2*dpr3error + 2*(0-bodyangvel31) #+ 1*dpr3error1
    prf4 = 5*2*dpr4error + 2*(0-bodyangvel41) #+ 1*dpr4error1

    zf = 50 + dpr1[2] #+ prf1[1] + prf1[0]
    rf = prf1[1]
    pf = prf1[0] - 2*(currentpos1[0] - currentppos[0]  - 1)
    yf = 5*(0-yaw) + 3*(0-yawrate)
    # yf = kpr*(0 - yaw) - kdr*(bodyangvel1[2])

    zf2 = 50 + dpr2[2] #+ prf1[1] + prf1[0]
    rf2 = prf2[1]
    pf2 = prf2[0] + 2*(currentpos2[0] - currentppos[0]  - 1)
    yf2 = 5*(0-yaw2) + 3*(0-yawrate2)

    zf3 = 50 + dpr3[2] #+ prf1[1] + prf1[0]
    rf3 = prf3[1]
    pf3 = prf3[0] + 2*(currentpos3[0] - currentppos[0])
    yf3 = 5*(0-yaw3) + 3*(0-yawrate3)

    zf4 = 50 + dpr4[2] #+ prf1[1] + prf1[0]
    rf4 = prf4[1]
    pf4 = prf4[0] + 2*(currentpos4[0] - currentppos[0])
    yf4 = 5*(0-yaw4) + 3*(0-yawrate4)

    print(currentpos1[0], currentpos2[0], currentpos3[1], currentpos4[1], currentpos1[2])
    #quad1
    data.ctrl[0] = zf[0]-rf[0]-pf[0]-yf
    data.ctrl[1] = zf[0]+rf[0]+pf[0]-yf
    data.ctrl[2] = zf[0]+rf[0]-pf[0]+yf
    data.ctrl[3] = zf[0]-rf[0]+pf[0]+yf

    #quad2
    data.ctrl[4] = zf2[0]-rf2[0]-pf2[0]-yf2
    data.ctrl[5] = zf2[0]+rf2[0]+pf2[0]-yf2
    data.ctrl[6] = zf2[0]+rf2[0]-pf2[0]+yf2
    data.ctrl[7] = zf2[0]-rf2[0]+pf2[0]+yf2

    #quad3
    data.ctrl[8] = zf3[0]-rf3[0]-pf3[0]-yf3
    data.ctrl[9] = zf3[0]+rf3[0]+pf3[0]-yf3
    data.ctrl[10] = zf3[0]+rf3[0]-pf3[0]+yf3
    data.ctrl[11] = zf3[0]-rf3[0]+pf3[0]+yf3

    #quad4
    data.ctrl[12] = zf4[0]-rf4[0]-pf4[0]-yf4
    data.ctrl[13] = zf4[0]+rf4[0]+pf4[0]-yf4
    data.ctrl[14] = zf4[0]+rf4[0]-pf4[0]+yf4
    data.ctrl[15] = zf4[0]-rf4[0]+pf4[0]+yf4

    # pyplot.figure(1)
    # pyplot.plot(data.ctrl[0], data.time)
    # pyplot.show()

    for _ in range(2):
      mj.mj_step(model, data)


    if (data.time>=simend):
        break
      
    if data.time>=2:
        movement = 1
        target1 = target1 + np.array([0.01,0,0], ndmin=2).T
        target2 = target2 + np.array([0.01,0,0], ndmin=2).T
        target3 = target3 + np.array([0.01,0,0], ndmin=2).T
        target4 = target4 + np.array([0.01,0,0], ndmin=2).T
    
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
