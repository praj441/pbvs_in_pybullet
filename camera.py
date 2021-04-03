import pybullet as p
import numpy as np
# import pybullet_data
import time
# import random
from PIL import Image
from math import*
import random

fov = 50
aspect = 1
near = 0.01
far = 10
width = 200
height = 200
pi=3.14159265359

default_orn = p.getQuaternionFromEuler([0,0,0])

def cam_to_ee_calibration(pos,orn):
	# print('cam',x,y,z,rx,ry,rz)
	eu = p.getEulerFromQuaternion(orn)
	#camera to ee calibration
	xe = pos[2] 
	ye = -pos[0]
	ze = -pos[1]
	rxe = eu[2]
	rye = -eu[0]
	rze = -eu[1]
	p2 = [xe,ye,ze]
	q2 = p.getQuaternionFromEuler([rxe,rye,rze])
	return p2,q2

def cam_pose_to_world_pose(camTargetPos,robotID,camTargetOrn=default_orn):
	epos,eorn = cam_to_ee_calibration(camTargetPos,camTargetOrn)
	cam_link_state = p.getLinkState(robotID,linkIndex=7,computeForwardKinematics=1)
	pos=cam_link_state[-2]
	ort=cam_link_state[-1]
	return p.multiplyTransforms(pos,ort,epos,eorn)

def get_image(robotID,width=200,height=200):
	cam_link_state = p.getLinkState(robotID,linkIndex=7,computeForwardKinematics=1)
	pos=cam_link_state[-2]
	ort=cam_link_state[-1]
	rot_mat=np.array(p.getMatrixFromQuaternion(ort))
	rot_mat=np.reshape(rot_mat,[3,3])
	dir=rot_mat.dot([1,0,0])
	up_vector=rot_mat.dot([0,0,1])
	s = 0.01
	view_matrix=p.computeViewMatrix(pos,pos+s*dir,up_vector)
	# p.addUserDebugText(text=".",textPosition=pos+s*dir,textColorRGB=[1,0,0],textSize=10)
	projection_matrix = p.computeProjectionMatrixFOV(fov,aspect,near,far)
	# f_len = projection_matrix[0]
	ld = [random.uniform(-20,20),random.uniform(-20,20),random.uniform(10,20)]
	lc = [random.random(),random.random(),random.random()]
	if random.random() > 0.0:
		_,_,rgbImg,depthImg_buffer,segImg=p.getCameraImage(width,height,view_matrix,projection_matrix,renderer=p.ER_BULLET_HARDWARE_OPENGL)
	else:
		_,_,rgbImg,depthImg_buffer,segImg=p.getCameraImage(width,height,view_matrix,projection_matrix,renderer=p.ER_TINY_RENDERER,
		lightColor=lc,lightAmbientCoeff=random.random(),lightDiffuseCoeff=random.random(),lightSpecularCoeff=random.random())
	A = np.reshape(rgbImg, (width,height, 4))[:, :, :3]
	return Image.fromarray(A),depthImg_buffer,segImg
