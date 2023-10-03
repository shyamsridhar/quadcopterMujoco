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

# kthr = 0.00005
# ktrq = kthr/5
kthr = 1.08E-05
ktrq = 1.62E-07


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
    data.qpos[2] = 0
    data.qvel[:] = 0
    pass

def controller(model, data):
    res = np.zeros((9,1))
    mj.mju_quat2Mat(res,data.qpos[3:7])
    R = res.reshape((3,3))
    propvel = data.qvel[6:10]
    thrust = kthr*propvel**2
    torque = ktrq*propvel**2
    global th1, th2, th3, th4, tq1, tq2, tq3, tq4, th21, th22, th23, th24, tq21, tq22, tq23, tq24, th31, th32, th33, th34, tq31, tq32, tq33, tq34, th41, th42, th43, th44, tq41, tq42, tq43, tq44
    th1, th2, th3, th4, tq1, tq2, tq3, tq4 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th21, th22, th23, th24, tq21, tq22, tq23, tq24 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th31, th32, th33, th34, tq31, tq32, tq33, tq34 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    th41, th42, th43, th44, tq41, tq42, tq43, tq44 = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
    
    th1[2], th2[2], th3[2], th4[2] = thrust
    th21[2], th22[2], th23[2], th24[2] = thrust
    th31[2], th32[2], th33[2], th34[2] = thrust
    th41[2], th42[2], th43[2], th44[2] = thrust
    tq1[2], tq2[2], tq3[2], tq4[2] = torque
    tq21[2], tq22[2], tq23[2], tq24[2] = torque
    tq31[2], tq32[2], tq33[2], tq34[2] = torque
    tq41[2], tq42[2], tq43[2], tq44[2] = torque

    th1a = np.matmul(R,th1)
    th2a = np.matmul(R,th2)
    th3a = np.matmul(R,th3)
    th4a = np.matmul(R,th4)
    tq1a = np.matmul(R,tq1)
    tq2a = np.matmul(R,tq2)
    tq3a = np.matmul(R,tq3)
    tq4a = np.matmul(R,tq4)

    th21a = np.matmul(R,th21)
    th22a = np.matmul(R,th22)
    th23a = np.matmul(R,th23)
    th24a = np.matmul(R,th24)
    tq21a = np.matmul(R,tq21)
    tq22a = np.matmul(R,tq22)
    tq23a = np.matmul(R,tq23)
    tq24a = np.matmul(R,tq24)

    th31a = np.matmul(R,th31)
    th32a = np.matmul(R,th32)
    th33a = np.matmul(R,th33)
    th34a = np.matmul(R,th34)
    tq31a = np.matmul(R,tq31)
    tq32a = np.matmul(R,tq32)
    tq33a = np.matmul(R,tq33)
    tq34a = np.matmul(R,tq34)

    th41a = np.matmul(R,th41)
    th42a = np.matmul(R,th42)
    th43a = np.matmul(R,th43)
    th44a = np.matmul(R,th44)
    tq41a = np.matmul(R,tq41)
    tq42a = np.matmul(R,tq42)
    tq43a = np.matmul(R,tq43)
    tq44a = np.matmul(R,tq44)
    

    #xfrc 
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
    # dr, dp, dy = 20.0*np.pi/180*np.sin(10*t), 20.0*np.pi/180*np.sin(2*t), 20.0*np.pi/180*np.sin(2*t)
    dr, dp, dy = 0, 0, 0 
    zf = 70 + 10*(2-data.qpos[2])+6*(0-data.qvel[2])
    rf = 10*(dr-roll)+5*(0-rollrate)
    pf = 10*(dp-pitch)+5*(0-pitchrate)
    yf = 10*(dy-yaw)+5*(0-yawrate)

    #quad1
    data.ctrl[0] = zf-rf-pf-yf
    data.ctrl[1] = zf+rf+pf-yf
    data.ctrl[2] = zf+rf-pf+yf
    data.ctrl[3] = zf-rf+pf+yf

    #quad2
    data.ctrl[4] = zf-rf-pf-yf
    data.ctrl[5] = zf+rf+pf-yf
    data.ctrl[6] = zf+rf-pf+yf
    data.ctrl[7] = zf-rf+pf+yf

    #quad3
    data.ctrl[8] = zf-rf-pf-yf
    data.ctrl[9] = zf+rf+pf-yf
    data.ctrl[10] = zf+rf-pf+yf
    data.ctrl[11] = zf-rf+pf+yf

    #quad4
    data.ctrl[12] = zf-rf-pf-yf
    data.ctrl[13] = zf+rf+pf-yf
    data.ctrl[14] = zf+rf-pf+yf
    data.ctrl[15] = zf-rf+pf+yf

    for _ in range(5):
      mj.mj_step(model, data)
    
    #print("Time: {0:5.2f}, Zpos: {1:5.3f}, Roll: {2:5.2f}, Pitch: {3:5.2f}, Yaw: {4:5.2f}".format(t,data.qpos[2],roll*180/np.pi,pitch*180/np.pi,yaw*180/np.pi))
    print(data.sensordata[6],data.sensordata[7],data.sensordata[8])
    #print("Time: {0:5.2f}, Zpos: {1:5.3f}, Roll: {2:5.2f}, Rollrate: {3:5.2f}".format(t,data.qpos[2],roll*180/np.pi,data.sensordata[0]*180/np.pi))
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
