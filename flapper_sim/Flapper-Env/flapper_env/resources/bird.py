import os
import time
import pybullet as p
import pybullet_data
import numpy as np
import math
import vg


class Bird:
	def __init__(self, client):
		self.client = client
		f_name = os.path.join(os.path.dirname(__file__),
							  'spm-asm-v6-2.SLDASM/urdf/spm-asm-v6-2.SLDASM.urdf')
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		planeID = p.loadURDF("plane.urdf")
		self.bird = p.loadURDF(fileName=f_name,
							   basePosition=[0,0,1],
							   physicsClientId=client)

		self.stroke_plane_joint = [0]
		self.wing_joints = [1,2] # left, right

		self.joint_pos_lims = ([-1.3, 1.3],[-0.7854, 0.7854],[-0.7854, 0.7854]) # rad
		p.resetBaseVelocity(self.bird, np.array([0,-1,0]), np.array([0,0,0]), self.client)
		
	def get_ids(self):
		return self.client, self.bird

	def apply_action(self, action, c):
		# print(action)
		stroke_plane_angle, wing_torque = action

		# if c == 0:
		# 	p.resetBaseVelocity(se.bird, np.array([0,-1,0]), np.array([0,0,0]), self.client)
		# Set stroke plane angle (position control)
		p.setJointMotorControlArray(self.bird, self.stroke_plane_joint,
									controlMode=p.POSITION_CONTROL,
									targetPositions=[stroke_plane_angle],
									physicsClientId=self.client)



		# Using wing position for direct control, infer torques based on resulting kinematics
		wing_positions = [wing_torque, wing_torque]
		p.setJointMotorControlArray(self.bird, self.wing_joints,
									controlMode=p.POSITION_CONTROL,
									targetPositions=wing_positions,
									physicsClientId=self.client)		

		# Set wing torques (torque control)
		# wing_torques = [wing_torque, wing_torque] # left, right (same direction since motors face each other)
		# p.setJointMotorControlArray(self.bird, self.wing_joints,
		# 							controlMode=p.VELOCITY_CONTROL,
		# 							forces=[0,0],
		# 							physicsClientId=self.client)

		# p.setJointMotorControlArray(self.bird, self.wing_joints,
		# 							controlMode=p.TORQUE_CONTROL,
		# 							forces=wing_torques,
		# 							physicsClientId=self.client)

	def get_observation(self):
		'''
		The sim and the phys should only the IMU's readings
		as state observations
		Total state pack consists of 
		 - Base dynamics
		 	- IMU
		 		- ang vel in IMU local frame (x,y,z): 3
		 		- lin vel in IMU local frame (x,y,z): 3
		 	- Joints
		 		- stroke plane 
		 			- motor angle, ang vel: 2
		 		- left wing 
		 			- motor angle, ang vel: 2
		 			- torque
		 		- right wing
		 			- motor angle, ang vel: 2
		 			- torque


		Total action pack consists of 
			- Stroke plane motor angular position
			- Left wing motor torque
			- Right wing motor torque
		'''


		# This is where we would work in the 
		# Get pos and orien of bird in sim
		pos, ang = p.getBasePositionAndOrientation(self.bird, self.client)
		ang = p.getEulerFromQuaternion(ang)
		ori = (math.cos(ang[2]), math.sin(ang[2]))
		# Get velocity
		vel, angvel = p.getBaseVelocity(self.bird, self.client)



		# p.resetBaseVelocity(self.bird, np.array([0,-1,0]), np.array([0,0,0]), self.client)
		print("\n")
		# Joints states
		joint_state_pack, lim_broken_bool = self.get_joint_dynamics()

		# IMU state
		IMU_state_pack = self.get_imu_dynamics()
		# print(IMU_state_pack)

		# Concatenate pos, ori, and vel
		observation = np.concatenate((pos, np.array(vel), np.array(angvel)))

		pos_1, dT_1, num_nodes = self.BEMT2(1)
		pos_2, dT_2, num_nodes = self.BEMT2(2)

		for n in range(num_nodes):
			self.apply_external_force(1,pos_1[n,:], dT_1[n,:])
			self.apply_external_force(2,pos_2[n,:], dT_2[n,:])

		self.get_joint_dynamics()
		
		kill_bool = None
		# kill_bool = self.kill()
		# print(kill_bool)
		return observation, kill_bool

	def get_imu_dynamics(self):
		IMU_link_num = 3

		IMU_state = p.getLinkState(self.bird, IMU_link_num,
									computeLinkVelocity=True, 
									computeForwardKinematics=True)


		IMU_ori = np.array(IMU_state[1]) # linkWorldOrientation, quaternion [x,y,z,w]
		IMU_lin_vel_GLOBAL = np.array(IMU_state[6]) # worldLinkLinearVelocity, 3vec [x,y,z]
		IMU_ang_vel_GLOBAL = np.array(IMU_state[7]) # worldLinkAngularVelocity, 3vec [x,y,z]

		IMU_rot_mat_L2G = np.array(p.getMatrixFromQuaternion(IMU_ori)).reshape([3,3]) # converts local to global
		IMU_rot_mat_G2L = np.linalg.inv(IMU_rot_mat_L2G) # converts global to local

		IMU_lin_vel_LOCAL = np.dot(IMU_rot_mat_G2L, IMU_lin_vel_GLOBAL)
		IMU_ang_vel_LOCAL = np.dot(IMU_rot_mat_G2L, IMU_ang_vel_GLOBAL)

		IMU_state_pack = np.concatenate((IMU_lin_vel_LOCAL, IMU_ang_vel_LOCAL)).flatten()

		return IMU_state_pack

	def get_joint_dynamics(self):
		# stroke plane 
		spm_joint_pos, spm_joint_vel = self.get_joint_pos_vel(0) # single floats, since they just refer to the joint dynamics
		# print(spm_joint_vel)

		# left wing
		left_joint_pos, left_joint_vel = self.get_joint_pos_vel(1) # single floats, since they just refer to the joint dynamics

		# right wing
		right_joint_pos, right_joint_vel = self.get_joint_pos_vel(2) # single floats, since they just refer to the joint dynamics


		lim_broken_bool = False
		if self.is_joint_limit_broken(1) == True or self.is_joint_limit_broken(2) == True:
			lim_broken_bool = True

		joint_state_pack = np.array([spm_joint_pos, spm_joint_vel,
									left_joint_pos, left_joint_vel,
									right_joint_pos, right_joint_vel])

		return joint_state_pack, lim_broken_bool


	def get_link_state(self, link_num):
		# https://github.com/bulletphysics/bullet3/issues/2429 <-- see how to get vels and pos of links
		cur_link_state = p.getLinkState(self.bird, link_num,
											computeLinkVelocity=True, 
											computeForwardKinematics=True)

		cur_link_pos = np.array(cur_link_state[0]) # linkWorldPosition, 3vec [x,y,z]
		cur_link_ori = np.array(cur_link_state[1]) # linkWorldOrientation, quaternion [x,y,z,w]
		cur_link_lin_vel = np.array(cur_link_state[6]) # worldLinkLinearVelocity, 3vec [x,y,z]
		cur_link_ang_vel = np.array(cur_link_state[7]) # worldLinkAngularVelocity, 3vec [x,y,z]
		return cur_link_pos, cur_link_ori, cur_link_lin_vel, cur_link_ang_vel

	def get_joint_pos_vel(self, joint_num):
		cur_joint_states = p.getJointState(self.bird, joint_num)
		cur_joint_pos = cur_joint_states[0]
		cur_joint_vel = cur_joint_states[1]
		return cur_joint_pos, cur_joint_vel

	def is_joint_limit_broken(self, joint_num):
		joint_pos = self.get_joint_pos_vel(joint_num)[0]
		joint_lims = self.joint_pos_lims[joint_num]
		if joint_lims[0] <= joint_pos <= joint_lims[1]: # limit is not broken
			return False
		else: # limit is broken
			return True

	def is_height_limit_broken(self, height_lims=[0.5,1.5]):
		# Get the world height of the base
		global_height = np.array(p.getLinkState(self.bird, -1)[0])[2]

		if global_height > height_lims[0] and global_height < height_lims[1]:
			return False
		else:
			return True

		
	def apply_external_force(self, link_num, pos, force):
		p.applyExternalForce(self.bird, link_num, 
							force, 
							pos, 
							p.LINK_FRAME)
		pass

	def BEMT_wing_kinem(self, link_num):
		x_i = np.array([1,0,0])
		y_i = np.array([0,1,0])
		z_i = np.array([0,0,1])

		pos, ori, lin_vel, _ = self.get_link_state(link_num)
		q = ori # q(t)
		q_inv = np.array([-ori[0],-ori[1],-ori[2],ori[3]]).T # q^-1

		v_wing = lin_vel

		x_t = np.dot(q, np.outer(x_i, q_inv).T)
		y_t = np.dot(q, np.outer(y_i, q_inv).T)
		z_t = np.dot(q, np.outer(z_i, q_inv).T) #xt and zt vectors define new cross section plane in global frame

		v_flow_3D = -v_wing # linear velocity of flow relative to wing

		n = np.cross(x_t,z_t)
		proj_v_on_xz_plane_norm = (np.dot(v_wing, n)/(np.sqrt(sum(n**2))**2))*n
		v_flow_2D = v_flow_3D - proj_v_on_xz_plane_norm # 2d vel used in BEMT

		psi = np.arccos(np.dot(x_t, v_flow_2D)/(np.linalg.norm(x_t)*np.linalg.norm(v_flow_2D))) # angle between wing's local x_t in global frame and 2d flow vec in global frame

		Cl = 1
		Cd = 1

		rho = 1 # fluid density
		N = 10 # num of elements
		c = 1 # chord
		s = 10 # span
		# alpha = 0 # angle of attack
		# theta = 0 # alpha + psi
		dr = s/N

		v_theta = np.array([v_flow_2D[0],0])*np.cos(psi)
		v_ax = np.array([0,v_flow_2D[1]])*np.sin(psi)

		vf_norm = np.cross(y_t, v_flow_3D)
		dL = 0.5*rho*(np.linalg.norm(v_flow_3D)**2)*Cl*c*dr*vf_norm/np.linalg.norm(vf_norm) # lift is normal to local flow
		dD = 0.5*rho*(np.linalg.norm(v_flow_3D)**2)*Cd*c*dr*v_flow_3D/np.linalg.norm(v_flow_3D) # drag is parallel with local flow vector	

		dT = dL*np.cos(psi) - dD*np.sin(psi) # thrust force, 

		if np.isnan(np.sum(dT)):
			dT = N*np.array([0,0,0])
		else:
			pass

		print(dT, np.linalg.norm(v_flow_3D))
		return pos, dT

	def BEMT2(self, link_num):
		x_i = np.array([1,0,0]) 
		y_i = np.array([0,1,0])
		z_i = np.array([0,0,1])



		pos, ori, lin_vel, ang_vel = self.get_link_state(link_num)
		v_base = p.getBaseVelocity(self.bird, self.client)[0] # global vel of base CM

		q = ori # q(t)
		rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape([3,3])

		cg_node_offset = np.array([0,0,0]) # disp (link local) from link CG to 0th node
		s = 0.15 # span 
		N = 20 # num of nodes
		Cla = 0.6
		Cda = 0.3
		# Cl = 2.0
		Cd = 0.04
		rho = 1000 # fluid density
		c = 0.1 # chord
		dr = s/N

		# construct nodes in link frame
		R_nodes = np.zeros([N,3])
		R_nodes[:,0] = np.linspace(-s/2,s/2,N) # vector from link CG to node in local coordinates
		r_nodes = np.array([np.dot(rot_mat, R_nodes[i,:]) for i in range(N)]) # vector from link CG to node in global coordinates
		v_nodes_about_link_CG = np.array([np.cross(ang_vel,r_nodes[i,:]) for i in range(N)]) # velocity about link CG in global coordinates
		v_nodes = np.array([v_nodes_about_link_CG[i,:] + v_base for i in range(N)]) #

		# flow in global frame
		v_flow_3D_global = -v_nodes

		# convert 3d flow from global to local frame
		v_flow_3D_local = np.array([np.dot(rot_mat.T,v_flow_3D_global[i,:]) for i in range(N)])

		# project local 3d flow onto local YZ plane (cross section wing)
		v_flow_3D_local_proj_y = np.array([np.dot(v_flow_3D_local[i,:],y_i)*y_i for i in range(N)])
		v_flow_3D_local_proj_z = np.array([np.dot(v_flow_3D_local[i,:],z_i)*z_i for i in range(N)])
		v_flow_2D_local = v_flow_3D_local_proj_y + v_flow_3D_local_proj_z
		# print(v_flow_2D_local[10,:])

		psi = np.array([np.arctan2(v_flow_2D_local[i,2],v_flow_2D_local[i,1]) for i in range(N)])
		alpha = psi

		v_sq = np.power(np.linalg.norm(v_flow_2D_local,axis=1),2) # squared velocity of flow
		v_sq = np.tile(v_sq,(3,1)).T
		# print(psi[10])

		vf_norm = (-1)**(link_num+1)*np.cross(x_i, v_flow_2D_local) # normal to flow (span X flow)

		vf_norm_mag = np.linalg.norm(vf_norm,axis=1) # magnitude of each normal 
		# print(vf_norm[10,:])

		lift_unit_vector = np.divide(vf_norm.T, vf_norm_mag).T # lift is normal to local flow
		drag_unit_vector = np.divide(v_flow_2D_local.T, np.linalg.norm(v_flow_2D_local,axis=1)).T # drag is parallel with local flow vector

		print(np.dot(rot_mat,lift_unit_vector[10,:]))

		# dL = 0.5*rho*Cla*c*dr*np.multiply(v_sq, lift_unit_vector)
		# dD = 0.5*rho*Cd*c*dr*np.multiply(v_sq, drag_unit_vector)

		dL = Cla*0.5*rho*c*dr*np.multiply(v_sq, np.multiply(np.tile(alpha,[3,1]).T,lift_unit_vector))
		dD = -Cda*0.5*rho*c*dr*np.multiply(v_sq, np.multiply(np.tile(np.power(alpha,2)+0.008,[3,1]).T, drag_unit_vector))
		# print(dL)

		dT = np.array([dL[i,:]*np.cos(psi[i]) - dD[i,:]*np.sin(psi[i]) for i in range(N)])

		if np.isnan(np.sum(dT)):
			dT = np.zeros([N,3])
		else:
			pass

		return R_nodes, dT, N


	def reward(self):
		'''
		Reward is based on magnitude of torque action (disregard stroke plane)
		Forward speed of base 
			get world velocity of base and orientation of base
			convert ori quaternion into rot matrix
			dot rot matrix with base world velocity to get local velocity of base
			take x component of local velocity of base

		'''
		pass

	def kill(self):
		'''
		Should kill for actions that break limits
			If the wing limits are broken
			If the height limits are broken
		'''

		height_lim_bool = self.is_height_limit_broken()
		lim_broken_bool_left = self.is_joint_limit_broken(1)
		lim_broken_bool_right = self.is_joint_limit_broken(2)

		lim_bools = [height_lim_bool, lim_broken_bool_left, lim_broken_bool_right]

		if True in lim_bools: #then one of the limits is broke and you should kill
			return True # YES KILL
		else:
			return False # NO KILL, WE ARE FINE