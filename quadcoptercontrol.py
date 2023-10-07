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

xml_path = 'single_rope.xml'
simend = 20
print_camera_config = 0

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

#target for quad1
target1 = np.array([1.4, 0, 10], ndmin=2).T

#target for quad2
target2 = np.array([-1.4, 0, 10], ndmin=2).T

#target for quad3
target3 = np.array([0, 1.4, 10], ndmin=2).T

#target for quad4
target4 = np.array([0, -1.4, 10], ndmin=2).T

# kthr = 0.00005
# ktrq = kthr/5
kthr = 1.08E-05
ktrq = 1.62E-07

kp=1
kd=0.1
kpr=20
kdr=20


#arm positions
r1 = np.array([0.1591,-0.1591,0],ndmin=2).T
r2 =  np.array([-0.1591,0.1591,0],ndmin=2).T
r3 =  np.array([0.1591,0.1591,0],ndmin=2).T
r4 =  np.array([-0.1591,-0.1591,0],ndmin=2).T

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

    thrust = kthr*propvel**2; thrust2 = kthr*propvel2**2; thrust3 = kthr*propvel3**2; thrust4 = kthr*propvel4**2
    torque = ktrq*propvel**2; torque2 = ktrq*propvel2**2; torque3 = ktrq*propvel3**2; torque4 = ktrq*propvel4**2

    global th1, th2, th3, th4, tq1, tq2, tq3, tq4, th21, th22, th23, th24, tq21, tq22, tq23, tq24, th31, th32, th33, th34, tq31, tq32, tq33, tq34, th41, th42, th43, th44, tq41, tq42, tq43, tq44
    th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th21, th22, th23, th24, tq21, tq22, tq23, tq24 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th31, th32, th33, th34, tq31, tq32, tq33, tq34 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th41, th42, th43, th44, tq41, tq42, tq43, tq44 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))

    th1[2], th2[2], th3[2], th4[2] = thrust
    th21[2], th22[2], th23[2], th24[2] = thrust2
    th31[2], th32[2], th33[2], th34[2] = thrust3
    th41[2], th42[2], th43[2], th44[2] = thrust4
    tq1[2], tq2[2], tq3[2], tq4[2] = torque
    tq21[2], tq22[2], tq23[2], tq24[2] = torque2
    tq31[2], tq32[2], tq33[2], tq34[2] = torque3
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

while not glfw.window_should_close(window):
    res = np.zeros((9,1)); res2 = np.zeros((9,1)); res3 = np.zeros((9,1)); res4 = np.zeros((9,1))

    mj.mju_quat2Mat(res,data.qpos[3:7]); mj.mju_quat2Mat(res2,data.qpos[14:18]); mj.mju_quat2Mat(res3,data.qpos[25:29]); mj.mju_quat2Mat(res4,data.qpos[36:40])
    R = res.reshape((3,3)); R2 = res2.reshape((3,3)); R3 = res3.reshape((3,3)); R4 = res4.reshape((3,3))

    roll = np.arctan2(R[2][1],R[2][2]); roll2 = np.arctan2(R2[2][1],R2[2][2]); roll3 = np.arctan2(R3[2][1],R3[2][2]); roll4 = np.arctan2(R4[2][1],R4[2][2])
    pitch = np.arcsin(-R[2][0]); pitch2 = np.arcsin(-R2[2][0]); pitch3 = np.arcsin(-R3[2][0]); pitch4 = np.arcsin(-R4[2][0])
    yaw = np.arctan2(R[1][0],R[0][0]); yaw2 = np.arctan2(R2[1][0],R2[0][0]); yaw3 = np.arctan2(R3[1][0],R3[0][0]); yaw4 = np.arctan2(R4[1][0],R4[0][0])

    bodyangvel1 = np.array([data.qvel[4], data.qvel[3], data.qvel[5]], ndmin=2).T; bodyangvel2 = np.array([data.qvel[14], data.qvel[13], data.qvel[15]], ndmin=2).T; bodyangvel3 = np.array([data.qvel[24], data.qvel[23], data.qvel[25]], ndmin=2).T; bodyangvel4 = np.array([data.qvel[34], data.qvel[33], data.qvel[35]], ndmin=2).T
    
    t = data.time

    ##based on quadcopter matlab
    # currentpos1 = np.array(data.sensordata[24:27], ndmin=2).T; currentpos2 = np.array(data.sensordata[27:30], ndmin=2).T; currentpos3 = np.array(data.sensordata[30:33], ndmin=2).T; currentpos4 = np.array(data.sensordata[33:36], ndmin=2).T
    currentpos1 = np.array(data.qpos[0:3], ndmin=2).T; currentpos2 = np.array(data.qpos[11:14], ndmin=2).T; currentpos3 = np.array(data.qpos[22:25], ndmin=2).T; currentpos4 = np.array(data.qpos[33:36], ndmin=2).T
    currentvel1 = np.array(data.sensordata[36:39], ndmin=2).T; currentvel2 = np.array(data.sensordata[39:42], ndmin=2).T; currentvel3 = np.array(data.sensordata[42:45], ndmin=2).T; currentvel4 = np.array(data.sensordata[45:48], ndmin=2).T
    currentpr1 = np.array([pitch, roll, yaw], ndmin=2).T; currentpr2 = np.array([pitch2, roll2, yaw2], ndmin=2).T; currentpr3 = np.array([pitch3, roll3, yaw3], ndmin=2).T; currentpr4 = np.array([pitch4, roll4, yaw4], ndmin=2).T
    # currentrp1vel = np.array(data.sensordata[0:3], ndmin=2); currentrp2vel = np.array(data.sensordata[3:6], ndmin=2); currentrp3vel = np.array(data.sensordata[6:9], ndmin=2); currentrp4vel = np.array(data.sensordata[9:12], ndmin=2)
    poserror1 = target1-currentpos1; poserror2 = target2-currentpos2; poserror3 = target3-currentpos3; poserror4 = target4-currentpos4
    dpr1 = kp*poserror1 + kd*(0-currentvel1); dpr2 = kp*poserror2 + kd*(0-currentvel2); dpr3 = kp*poserror3 + kd*(0-currentvel3); dpr4 = kp*poserror4 + kd*(0-currentvel4)
    
    #calculate desired rp angles in body frame
    dpr1body = np.matmul(R.T, dpr1);  dpr2body = np.matmul(R2.T, dpr2); dpr3body = np.matmul(R3.T, dpr3); dpr4body = np.matmul(R4.T, dpr4)

    dpr1body[1] = dpr1body[1]*(-1); dpr2body[1] = dpr2body[1]*(-1); dpr3body[1] = dpr3body[1]*(-1); dpr4body[1] = dpr4body[1]*(-1)
    

    #saturation limit to 60 degrees
    dpr1body[0] = max(min(np.pi/3, dpr1body[0]), -np.pi/3); dpr1body[1] = max(min(np.pi/3, dpr1body[1]), -np.pi/3)
    dpr2body[0] = max(min(np.pi/3, dpr2body[0]), -np.pi/3); dpr2body[1] = max(min(np.pi/3, dpr2body[1]), -np.pi/3)
    dpr3body[0] = max(min(np.pi/3, dpr3body[0]), -np.pi/3); dpr3body[1] = max(min(np.pi/3, dpr3body[1]), -np.pi/3)
    dpr4body[0] = max(min(np.pi/3, dpr4body[0]), -np.pi/3); dpr4body[1] = max(min(np.pi/3, dpr4body[1]), -np.pi/3)

    dpr1error = dpr1body-currentpr1; dpr2error = dpr2body-currentpr2; dpr3error = dpr3body-currentpr3; dpr4error = dpr4body-currentpr4
    prf1 = kpr*dpr1error - kdr*(bodyangvel1); prf2 = kpr*dpr2error - kdr*(bodyangvel2); prf3 = kpr*dpr3error - kdr*(bodyangvel3); prf4 = kpr*dpr4error - kdr*(bodyangvel4)

    zf = 10*dpr1[2]; zf2 = 10*dpr2[2]; zf3 = 10*dpr3[2]; zf4 = 10*dpr4[2]
    rf = prf1[1]; rf2 = prf2[1]; rf3 = prf3[1]; rf4 = prf4[1]
    pf = prf1[0]; pf2 = prf2[0]; pf3 = prf3[0]; pf4 = prf4[0]
    # yf = 0; yf2 = 0; yf3 = 0; yf4 = 0
    yf = kpr*(0 - yaw) - kdr*(bodyangvel1[2]); yf2 = kpr*(0 - yaw2) - kdr*(bodyangvel2[2]); yf3 = kpr*(0 - yaw3) - kdr*(bodyangvel3[2]); yf4 = kpr*(0 - yaw4) - kdr*(bodyangvel4[2])

    # print(yf, yf2, yf3, yf4)
    #quad1
    data.ctrl[0] = zf-rf-pf-yf
    data.ctrl[1] = zf+rf+pf-yf
    data.ctrl[2] = zf+rf-pf+yf
    data.ctrl[3] = zf-rf+pf+yf

    #quad2
    data.ctrl[4] = zf2-rf2-pf2-yf2
    data.ctrl[5] = zf2+rf2+pf2-yf2
    data.ctrl[6] = zf2+rf2-pf2+yf2
    data.ctrl[7] = zf2-rf2+pf2+yf2

    #quad3
    data.ctrl[8] = zf3-rf3-pf3-yf3
    data.ctrl[9] = zf3+rf3+pf3-yf3
    data.ctrl[10] = zf3+rf3-pf3+yf3
    data.ctrl[11] = zf3-rf3+pf3+yf3

    #quad4
    data.ctrl[12] = zf4-rf4-pf4-yf4
    data.ctrl[13] = zf4+rf4+pf4-yf4
    data.ctrl[14] = zf4+rf4-pf4+yf4
    data.ctrl[15] = zf4-rf4+pf4+yf4

    for _ in range(5):
      mj.mj_step(model, data)

    # torques due to thrust forces
    t1 = np.cross(r1,th1,axis=0)
    t2 = np.cross(r2,th2,axis=0)
    t3 = np.cross(r3,th3,axis=0)
    t4 = np.cross(r4,th4,axis=0)

    netTorque = (t1+t2+t3+t4-tq1-tq2+tq3+tq4).reshape(3)

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
