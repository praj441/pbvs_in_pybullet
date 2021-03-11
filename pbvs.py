import pybullet as p
import pybullet_data
import time
import numpy as np
from math import *

from camera import get_image
from sphere_fitting import sphereFit
from cam_ik import accurateIK



def relative_ee_pose_to_ee_world_pose1(robotId,eeTargetPos,eeTargetOrn):
    ee_link_state = p.getLinkState(robotId,linkIndex=7,computeForwardKinematics=1)
    ee_pos_W=ee_link_state[-2]
    ee_ort_W=ee_link_state[-1]
    return p.multiplyTransforms(ee_pos_W,ee_ort_W,eeTargetPos,eeTargetOrn)

def getJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques


def getMotorJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
  joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques


def setJointPosition(robot, position, kp=1.0, kv=0.3):
  num_joints = p.getNumJoints(robot)
  zero_vec = [0.0] * num_joints
  if len(position) == num_joints:
    p.setJointMotorControlArray(robot,
                                range(num_joints),
                                p.POSITION_CONTROL,
                                targetPositions=position,
                                targetVelocities=zero_vec,
                                positionGains=[kp] * num_joints,
                                velocityGains=[kv] * num_joints)
  else:
    print("Not setting torque. "
          "Expected torque vector of "
          "length {}, got {}".format(num_joints, len(torque)))


def multiplyJacobian(robot, jacobian, vector):
  result = [0.0, 0.0, 0.0]
  i = 0
  for c in range(len(vector)):
    if p.getJointInfo(robot, c)[3] > -1:
      for r in range(3):
        result[r] += jacobian[r][i] * vector[c]
      i += 1
  return result

def multiplyJacobian1(robot, jacobian, vector):
  result = [0.0, 0.0, 0.0,0.0,0.0,0.0]
  i = 0
  for c in range(len(vector)):
    print(c,p.getJointInfo(robot, c)[3])
    if p.getJointInfo(robot, c)[3] > -1:
      for r in range(6):
        result[r] += jacobian[r][i] * vector[c]
      i += 1
  return result


clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
  p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

time_step = 0.001
gravity_constant = -9.81
p.resetSimulation()
p.setTimeStep(time_step)
p.setGravity(0.0, 0.0, gravity_constant)

p.loadURDF("plane.urdf", [0, 0, -0.3])

# kukaId = p.loadURDF("TwoJointRobot_w_fixedJoints.urdf", useFixedBase=True)
#kukaId = p.loadURDF("TwoJointRobot_w_fixedJoints.urdf",[0,0,0])
#kukaId = p.loadURDF("kuka_iiwa/model.urdf",[0,0,0])
#kukaId = p.loadURDF("kuka_lwr/kuka.urdf",[0,0,0])
#kukaId = p.loadURDF("humanoid/nao.urdf",[0,0,0])
# kukaId = p.loadURDF("ur_object_reaching/ur_description/urdf/ur10_robot.urdf",[0,0,0])
# p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("table/table.urdf",cubeStartPos, cubeStartOrientation)
# appleId = p.loadURDF("apple.urdf",[0.0,0.0,0.625])
appleId = p.loadURDF("urdf/apple1/apple.urdf",[0.0,0.0,0.7],useFixedBase=1)
kukaId = p.loadURDF("ur_description/urdf/ur10_robot.urdf",[-0.6,0,0.625], cubeStartOrientation)
roboId = kukaId
required_joints = [0,-1.9,1.9,-1.57,-1.57,0,0]
for i in range(1,7):
    p.resetJointState(bodyUniqueId=roboId,
                            jointIndex=i,
                            targetValue=required_joints[i-1])
p.resetDebugVisualizerCamera( cameraDistance=2.2, cameraYaw=140, cameraPitch=-60, cameraTargetPosition=[0,0,0])

numJoints = p.getNumJoints(kukaId)
kukaEndEffectorIndex = numJoints - 1

# Set a joint target for the position control and step the sim.
# setJointPosition(kukaId, [0.1] * numJoints)
# p.stepSimulation()
fov = 50
aspect = 1
near = 0.01
far = 10
width = 200
height = 200
pi=3.14159265359
f_x = width/(2*tan(fov*pi/360))
f_y = height/(2*tan(fov*pi/360))

def pix_to_ee_frame(x_pix,y_pix,d):
  Zc = d
  # print('depth_pix,pix_x_bbox,pix_y_bbox = ',xin,pix_x_bbox,pix_x_bbox)
  Xc = (x_pix - (width/2))*(d/(f_x))    
  Yc = (y_pix - (height/2))*(d/(f_y))

  Xe = Zc 
  Ye = Xc
  Ze = Yc
  return(Xe,Ye,Ze)

def test_jecobian(kukaId):
  # Get the joint and link state directly from Bullet.
  pos, vel, torq = getJointStates(kukaId)
  mpos, mvel, mtorq = getMotorJointStates(kukaId)

  result = p.getLinkState(kukaId,
                          kukaEndEffectorIndex,
                          computeLinkVelocity=1,
                          computeForwardKinematics=1)
  link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
  # Get the Jacobians for the CoM of the end-effector link.
  # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
  # The localPosition is always defined in terms of the link frame coordinates.

  zero_vec = [0.0] * len(mpos)
  jac_t, jac_r = p.calculateJacobian(kukaId, kukaEndEffectorIndex, com_trn, mpos, zero_vec, zero_vec)
  
  J = jac_t + jac_r
  print(np.shape(J))

  # print("Link linear velocity of CoM from getLinkState:")
  # print(link_vt)
  # print("Link linear velocity of CoM from linearJacobian * q_dot:")
  # print(multiplyJacobian(kukaId, jac_t, vel))
  # print("Link angular velocity of CoM from getLinkState:")
  # print(link_vr)
  # print("Link angular velocity of CoM from angularJacobian * q_dot:")
  # print(multiplyJacobian(kukaId, jac_r, vel))

  # print(multiplyJacobian1(kukaId,J,vel))

  # Ja = np.array(J)
  # print(np.shape(vel))
  # print(np.matmul(Ja,vel[1:7]))


def run_ee_link(kukaId,eevel):
  pos, vel, torq = getJointStates(kukaId)
  mpos, mvel, mtorq = getMotorJointStates(kukaId)

  result = p.getLinkState(kukaId,
                          kukaEndEffectorIndex,
                          computeLinkVelocity=1,
                          computeForwardKinematics=1)
  link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
  # Get the Jacobians for the CoM of the end-effector link.
  # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
  # The localPosition is always defined in terms of the link frame coordinates.

  zero_vec = [0.0] * len(mpos)
  jac_t, jac_r = p.calculateJacobian(kukaId, kukaEndEffectorIndex, com_trn, mpos, zero_vec, zero_vec)
  import numpy as np
  J = jac_t + jac_r
  Ja = np.linalg.inv(np.array(J))
  Q = np.matmul(Ja,eevel)
  maxForce = 500
  for i in range(1,7):
    p.setJointMotorControl2(bodyUniqueId=kukaId, 
    jointIndex=i, 
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity = Q[i-1], #check if mapping is proper
    force = maxForce)

def mask_points(Sbuf,Dbuf,appleId):
    depthImg_buf = np.reshape(Dbuf, [width, height])
    D = far * near / (far - (far - near) * depthImg_buf)
    S = np.reshape(Sbuf,[width,height])
    Xl = []
    Yl = []
    Zl = []
    for i in range(width):
        for j in range(height):
            if S[i][j] == appleId:
                x,y,z = pix_to_ee_frame(i,j,D[i][j])
                Xl.append(x)
                Yl.append(y)
                Zl.append(z)
    return Xl,Yl,Zl



# accurateIK(roboId,7,pos2,ort2,useNullSpace=False)
dpos = np.array((0.0,0.0,0.0))
def visual_servo_control(pos,ort):
    lmd = 5
    # print(np.shape(pos),np.shape(ort))
    Vc = -lmd*((dpos-pos)+np.cross(pos,ort))
    Wc = -lmd*(ort)
    # temp = Vc.copy()
    # Vc[0] = temp[2]
    # Vc[1] = temp[1]
    # Vc[2] = -temp[0]
    # temp = Vc[2]
    # Vc[2] = -Vc[0]
    # Vc[0] = temp
    # temp = Wc[2]
    # Wc[2] = -Wc[0]
    # Wc[0] = temp

    # Qc = p.getQuaternionFromEuler(Wc)
    # Vc,c = relative_ee_pose_to_ee_world_pose1(roboId,Vc,Qc)
    V = np.concatenate((Vc,Wc))
    print(pos)
    return V

I,Dbuf,Sbuf = get_image(kukaId)
sx,sy,sz = mask_points(Sbuf,Dbuf,appleId)
r,cx,cy,cz = sphereFit(sx,sy,sz)
pos = np.array((cx,cy,cz))[:,0]
ort = np.array((0.0,0.0,0.0))
# pos2,ort2 = relative_ee_pose_to_ee_world_pose1(roboId,pos,cubeStartOrientation)
Ve = visual_servo_control(pos,ort)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
    # if i%20==0:
    I,Dbuf,Sbuf = get_image(kukaId)
    sx,sy,sz = mask_points(Sbuf,Dbuf,appleId)
    r,cx,cy,cz = sphereFit(sx,sy,sz)
    pos = np.array((cx,cy,cz))[:,0]
    ort = np.array((0.0,0.0,0.0))
    Ve = visual_servo_control(pos,ort)
    pos = 0.01*Ve[0:3]
    rot = 0.01*Ve[3:6]
    ort = p.getQuaternionFromEuler(rot)
    pos2,ort2 = relative_ee_pose_to_ee_world_pose1(roboId,pos,ort)
    accurateIK(roboId,7,pos2,ort2,useNullSpace=False)
    # run_ee_link(kukaId,Ve)
      # run_ee_link(kukaId,np.array((1.0,0.0,0.0,0.0,0.0,0.0)))
    # I,Dbuf,Sbuf = get_image(kukaId)
    # sx,sy,sz = mask_points(Sbuf,Dbuf,appleId)
    # r,cx,cy,cz = sphereFit(sx,sy,sz)
    # print(r,cx,cy,cz)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()

