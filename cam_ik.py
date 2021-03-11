import pybullet as p
import pybullet_data
import time
import numpy as np
from camera import get_image
# from camera import ur5_camera
# from pix_co_to_cartesian_co import pix_to_cartesian_pos

def accurateIK(bodyId, endEffectorId, targetPosition,targetOrientation, lowerLimits=0.0, upperLimits=0.0, jointRanges=0.0, restPoses=0.0, 
               useNullSpace=True, maxIter=100, threshold=1e-13):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
    lowerLimits : [float] 
    upperLimits : [float] 
    jointRanges : [float] 
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float
    Returns
    -------
    jointPoses : [float] * numDofs
    """
    closeEnough = False
    iter = 0
    dist2 = 1e30

    numJoints = p.getNumJoints(bodyId)

    while (not closeEnough and iter<maxIter):
        if useNullSpace:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,
                lowerLimits=lowerLimits, upperLimits=upperLimits, jointRanges=jointRanges, 
                restPoses=restPoses)
        else:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,targetOrientation)
            # Put targetOrienttion in the list of functional arguments in Quaternion (Refer the Quick guide for exact info)
    
        for i in range(numJoints):
            jointInfo = p.getJointInfo(bodyId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(bodyId,i,jointPoses[qIndex-7])
        # for _ in range(500):
        #     p.stepSimulation()
        ls = p.getLinkState(bodyId,endEffectorId)
        # Extract both postion and orientation i.e. ls[4] and ls[5] and then define appropriate "diff" and "dist"    
        newPos = ls[4]
        diff = [targetPosition[0]-newPos[0],targetPosition[1]-newPos[1],targetPosition[2]-newPos[2]]
        dist2 = np.sqrt((diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]))
        closeEnough = (dist2 < threshold)
        # print('Te',dist2)
        newOrt = ls[5]
        diff = [targetOrientation[0]-newOrt[0],targetOrientation[1]-newOrt[1],targetOrientation[2]-newOrt[2],targetOrientation[3]-newOrt[3]]
        dist2 = np.sqrt((diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]+diff[3]*diff[3]))
        closeEnough = (dist2 < threshold)
        # print('Re',dist2)
        # print('Itr',iter)
        iter=iter+1
        # time.sleep(0.01)
#    print("iter=",iter)
    return jointPoses

def move_eye_camera(robotID,x,y,z):
    cam_link_state = p.getLinkState(robotID,linkIndex=7,computeForwardKinematics=1)
    camera_pos_W=cam_link_state[-2]
    camera_ort_W=cam_link_state[-1]

    roll = 0.0 #this will cause end effector to rotate on its axis
    pitch= 0.0
    yaw = 0.0
    cameraTargetPos = [x,y,z]
    cameraTargetOrn = p.getQuaternionFromEuler([roll,pitch,yaw])
    new_camera_pos_W,new_camera_ort_W = p.multiplyTransforms(camera_pos_W,camera_ort_W,cameraTargetPos,cameraTargetOrn)
    # print('new_camera_ort_W',new_camera_ort_W)
    #following is the new function defined above in this file
    accurateIK(robotID,7,new_camera_pos_W,new_camera_ort_W,useNullSpace=False)
    return get_image(robotID)
    
