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

		# Enable joint torque sensors
		for j in [0,1,2]:
			p.enableJointForceTorqueSensor(self.bird, j, 1, self.client)

		# p.resetBaseVelocity(self.bird, np.array([0,-.1,0]), np.array([0,0,0]), self.client)
		
		self.datalog = {"Stroke" : None, 
						"AoA" : np.zeros(2),
						"Reward" : np.zeros(1)}

		self.ClCd_model_made = False
		self.ClCd_model_vect = np.vectorize(self.ClCd_model)

	def get_ids(self):
		return self.client, self.bird

	def apply_action(self, action):
		# p.resetBaseVelocity(self.bird, np.array([0,-.01,0]), np.array([0,0,0]), self.client)

		'''
		Total action pack consists of 
			- Stroke plane motor angular position
			- Left wing motor angular position
			- Right wing motor angular position
		'''
		stroke_plane_angle, left_wing_pos, right_wing_pos = action

		# Set stroke plane angle (position control)
		p.setJointMotorControlArray(self.bird, np.array(self.stroke_plane_joint),
									controlMode=p.POSITION_CONTROL,
									targetPositions=[stroke_plane_angle],
									physicsClientId=self.client)

		# Using wing position for direct control, infer torques based on resulting kinematics
		wing_positions = [left_wing_pos, right_wing_pos]
		same_wing_positions = [left_wing_pos, left_wing_pos]
		p.setJointMotorControlArray(self.bird, self.wing_joints,
									controlMode=p.POSITION_CONTROL,
									targetPositions=wing_positions,
									physicsClientId=self.client)		

		# Aero forces generated from link motion
		pos_1, dT_1, num_nodes = self.BEMT2(1)
		pos_2, dT_2, num_nodes = self.BEMT2(2)

		for n in range(num_nodes):
			self.apply_external_force(1,pos_1[n,:], dT_1[n,:])
			self.apply_external_force(2,pos_2[n,:], dT_2[n,:])

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
		 			- motor angle, ang vel, torque: 3
		 		- right wing
		 			- motor angle, ang vel, torque: 3
		'''

		# p.resetBaseVelocity(self.bird, np.array([0,-1,0]), np.array([0,0,0]), self.client)
		# print("\n")

		# Observation contains IMU readings (lin and rot vels), joint readings (pos, vels, trqs)
		joint_state_pack = self.get_joint_dynamics()
		IMU_state_pack = self.get_imu_dynamics()
		observation = np.concatenate((IMU_state_pack, joint_state_pack))
		pos, ori = p.getBasePositionAndOrientation(self.bird, self.client)
		pos, ori = np.array(pos), np.array(ori)
		# print(observation)


		# Should we kill it? (are the lims broken)
		kill_bool = self.kill()
		# print(kill_bool)
		return observation, pos, ori, kill_bool

	def get_imu_dynamics(self):
		IMU_link_num = 3

		IMU_state = p.getLinkState(self.bird, IMU_link_num,
									computeLinkVelocity=True, 
									computeForwardKinematics=True,
									physicsClientId=self.client)


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
		# stroke plane pos and vel 
		spm_joint_pos, spm_joint_vel, spm_joint_trq = self.get_joint_pos_vel_trq(0) # single floats, since they just refer to the joint dynamics
		
		# left wing pos and vel
		left_joint_pos, left_joint_vel, left_joint_trq = self.get_joint_pos_vel_trq(1) # single floats, since they just refer to the joint dynamics

		# right wing pos and vel
		right_joint_pos, right_joint_vel, right_joint_trq = self.get_joint_pos_vel_trq(2) # single floats, since they just refer to the joint dynamics

		joint_state_pack = np.array([spm_joint_pos, spm_joint_vel, # we cant read nema torque, dont include
									left_joint_pos, left_joint_vel, left_joint_trq,
									right_joint_pos, right_joint_vel, right_joint_trq])

		self.datalog["Stroke"] = joint_state_pack

		return joint_state_pack


	def get_link_state(self, link_num):
		# https://github.com/bulletphysics/bullet3/issues/2429 <-- see how to get vels and pos of links
		cur_link_state = p.getLinkState(self.bird, link_num,
										computeLinkVelocity=True, 
										computeForwardKinematics=True,
									 	physicsClientId=self.client)

		cur_link_pos = np.array(cur_link_state[0]) # linkWorldPosition, 3vec [x,y,z]
		cur_link_ori = np.array(cur_link_state[1]) # linkWorldOrientation, quaternion [x,y,z,w]
		cur_link_lin_vel = np.array(cur_link_state[6]) # worldLinkLinearVelocity, 3vec [x,y,z]
		cur_link_ang_vel = np.array(cur_link_state[7]) # worldLinkAngularVelocity, 3vec [x,y,z], about CG in global coords
		return cur_link_pos, cur_link_ori, cur_link_lin_vel, cur_link_ang_vel

	def get_joint_pos_vel_trq(self, joint_num):
		cur_joint_states = p.getJointState(self.bird, joint_num,  physicsClientId=self.client)
		cur_joint_pos = cur_joint_states[0]
		cur_joint_vel = cur_joint_states[1]
		cur_joint_motor_trq = cur_joint_states[3]
		return cur_joint_pos, cur_joint_vel, cur_joint_motor_trq

	def is_joint_limit_broken(self, joint_num):
		joint_pos = self.get_joint_pos_vel_trq(joint_num)[0]
		joint_lims = self.joint_pos_lims[joint_num]
		if joint_lims[0] <= joint_pos <= joint_lims[1]: # limit is not broken
			return False
		else: # limit is broken
			return True

	def is_height_limit_broken(self, height_lims=[0.5,1.5]):
		# Get the world height of the base
		global_height = np.array(p.getBasePositionAndOrientation(self.bird, self.client)[0])[2]

		if global_height > height_lims[0] and global_height < height_lims[1]:
			return False
		else:
			return True

	def is_instability_limit_broken(self):
		# Instability - how much it is spinning around
		IMU_state = p.getLinkState(self.bird, 3,
									computeLinkVelocity=True, 
									computeForwardKinematics=True,
									physicsClientId=self.client)
		IMU_ang_vel_GLOBAL = np.array(IMU_state[7]) # worldLinkAngularVelocity, 3vec [x,y,z]
		instability = np.linalg.norm(IMU_ang_vel_GLOBAL)**2

		if instability > 50:
			return True
		else:
			return False

		
	def apply_external_force(self, link_num, pos, force):
		p.applyExternalForce(self.bird, link_num, 
							force, 
							pos, 
							p.LINK_FRAME)
		pass

	def ClCd_model(self, alpha):
		# Here you would init training for clcd model
		# or run (once) the fit on xfoil data
		# Once it is made, it returns cl and cd for alpha
		# Mock Cla/Cda data

		mock_data_num = 1000
		alpha_data = np.arange(-np.pi,np.pi,mock_data_num)
		Cla_data = np.sin(alpha_data) + (1e-3)*np.random.normal(size=mock_data_num)
		Cda_data = np.power(np.sin(alpha_data), 2) + (1e-6)*np.random.normal(size=mock_data_num)
		
		if not self.ClCd_model_made: # model has not been made yet, so make it
			# train/interpolate
			self.ClCd_model_made = True
			pass

		Cla_pred = 0.6*np.sin(alpha) # model prediction
		Cda_pred = 0.03*np.power(np.sin(alpha),2) # model prediction

		return Cla_pred, Cda_pred

	def BEMT2(self, link_num):

		x_i = np.array([1,0,0]) 
		y_i = np.array([0,1,0])
		z_i = np.array([0,0,1])

		pos, ori, lin_vel, ang_vel = self.get_link_state(link_num)
		# print(lin_vel)
		v_base = p.getBaseVelocity(self.bird, self.client)[0] # global vel of base CM

		q = ori # q(t)
		rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape([3,3])

		cg_node_offset = np.array([0,0,0]) # disp (link local) from link CG to 0th node
		s = 0.15 # span 
		N = 20 # num of nodes
		Cla = 0.6
		Cda = 0.03
		# Cl = 2.0
		Cd = 0.04
		rho = 1000 # fluid density
		c = 0.1 # chord
		dr = s/N

		# construct nodes in link frame
		R_nodes = np.zeros([N,3])
		R_nodes[:,0] = np.linspace(-s/2,s/2,N) # vector from link CG to node in local coordinates

		# print(np.dot(rot_mat,ang_vel))

		r_nodes = np.array([np.dot(rot_mat, R_nodes[i,:]) for i in range(N)]) # vector from link CG to node in global coordinates

		v_nodes_about_link_CG = np.array([np.cross(ang_vel,r_nodes[i,:]) for i in range(N)]) # velocity about link CG in global coordinates
		v_nodes = np.array([v_nodes_about_link_CG[i,:] + lin_vel for i in range(N)]) #
		# v_nodes = np.array([lin_vel for i in range(N)]) #

		# print(np.linalg.norm(v_nodes[10,:]))

		# print(np.linalg.norm(v_nodes,axis=1))

		# flow in global frame
		v_flow_3D_global = -v_nodes
		# print(v_flow_3D_global[10,:])

		# convert 3d flow from global to local frame
		v_flow_3D_local = np.array([np.dot(np.linalg.inv(rot_mat),v_flow_3D_global[i,:]) for i in range(N)])
		# print(v_flow_3D_local[10,:])

		# project local 3d flow onto local YZ plane (cross section wing)
		v_flow_3D_local_proj_y = np.array([np.dot(v_flow_3D_local[i,:],y_i)*y_i for i in range(N)])
		v_flow_3D_local_proj_z = np.array([np.dot(v_flow_3D_local[i,:],z_i)*z_i for i in range(N)])
		v_flow_2D_local = v_flow_3D_local_proj_y + v_flow_3D_local_proj_z
		# print(v_flow_2D_local[10,:])
		# print(np.dot(rot_mat,v_flow_2D_local[10,:]))

		psi = np.array([np.arctan2(v_flow_2D_local[i,2],(-1)**(link_num+1)*v_flow_2D_local[i,1]) for i in range(N)])
		# psi = np.array([np.arctan(v_flow_2D_local[i,2]/v_flow_2D_local[i,1]) for i in range(N)])
		# psi = np.array([np.arctan2((-1)**(link_num+1)*v_flow_2D_local[i,2],(-1)**(link_num+1)*v_flow_2D_local[i,1]) for i in range(N)])
		# alpha = np.array([np.arctan2(np.abs(v_flow_2D_local[i,2]),np.abs(v_flow_2D_local[i,1])) for i in range(N)])
		# print(psi)

		v_sq = np.power(np.linalg.norm(v_flow_2D_local,axis=1),2) # squared velocity of flow
		v_sq = np.tile(v_sq,(3,1)).T
		# print(psi)
		if link_num == 1:
			self.datalog["AoA"][0] = psi[10]
		else:
			self.datalog["AoA"][1] = psi[10]

		vf_norm = (-1)**(link_num+1)*np.cross(x_i, v_flow_2D_local) # normal to flow (span X flow)
		# print(vf_norm[10,:])
		# vf_norm = -np.cross(x_i, v_flow_2D_local) # normal to flow (span X flow)

		vf_norm_mag = np.linalg.norm(vf_norm,axis=1) # magnitude of each normal 

		lift_unit_vector = np.divide(vf_norm.T, vf_norm_mag).T # lift is normal to local flow
		drag_unit_vector = np.divide(v_flow_2D_local.T, np.linalg.norm(v_flow_2D_local,axis=1)).T # drag is parallel with local flow vector
		# print(drag_unit_vector[10,:])

		# print(np.dot(rot_mat,lift_unit_vector[10,:]))
		# print(np.dot(rot_mat,drag_unit_vector[10,:]))

		Cl, Cd = self.ClCd_model_vect(psi) # uses cl, cd model above

		dL = 0.5*rho*c*dr*np.multiply(v_sq, np.multiply(np.tile(Cl,[3,1]).T,lift_unit_vector))
		dD = 0.5*rho*c*dr*np.multiply(v_sq, np.multiply(np.tile(Cd,[3,1]).T, drag_unit_vector))
		# print(np.dot(rot_mat,dL[10,:]))
		# print(np.dot(rot_mat,dD[10,:]))
		# print(dL)

		dT = np.array([dL[i,:]*np.cos(psi[i]) + dD[i,:]*np.sin(psi[i]) for i in range(N)])

		dT_mag = np.linalg.norm(dT,axis=1)
		# print(dT_mag)

		# dT = np.array([dL[i,:] + dD[i,:] for i in range(N)])*0.1
		if np.isnan(np.sum(dT)):
			dT = np.zeros([N,3])
		else:
			# dT = np.zeros([N,3])
			# print("Force: ", np.dot(rot_mat,dT[0,:]))
			pass

		return R_nodes, dT, N

	def log_data(self, data_classes, t):
		for data_name, data_class in data_classes.items():
			data_class = data_classes[data_name]
			new_x = t
			new_y = self.datalog[data_name]

			if new_x not in data_class.XData:
				data_class.XData = np.append(data_class.XData, t)
				data_class.YData = np.append(data_class.YData, self.datalog[data_name]).reshape(
										   len(data_class.XData),len(new_y))
			else:
				data_class.XData[-1] = new_x
				data_class.YData[-1] = new_y


	def reward(self):
		'''
		Reward is based on magnitude of torque action (disregard stroke plane)
		Forward speed of base 
			get world velocity of base and orientation of base
			convert ori quaternion into rot matrix
			dot rot matrix with base world velocity to get local velocity of base
			take x component of local velocity of base

		'''

		# Velocity rewards
		v_base = p.getBaseVelocity(self.bird, self.client)[0] # velocity of base in global coords
		pos_base, ori_base = p.getBasePositionAndOrientation(self.bird, self.client)
		rot_mat_base = np.array(p.getMatrixFromQuaternion(ori_base)).reshape((3,3))
		local_forward_vec = np.array([0,-1,0]) # local base vector in the forward direction
		global_forward_vec = np.dot(rot_mat_base, local_forward_vec)
		# dot global forward vector with global forward velocity
		forward_dot = np.dot(v_base,global_forward_vec)

		# Torque rewards
		_, _, spm_joint_trq = self.get_joint_pos_vel_trq(0)
		_, _, left_joint_trq = self.get_joint_pos_vel_trq(1)
		_, _, right_joint_trq = self.get_joint_pos_vel_trq(2)
		torque_penalty = spm_joint_trq**2 + left_joint_trq**2 + right_joint_trq**2

		# Height rewards
		target_height = 1.0
		global_height = np.array(p.getBasePositionAndOrientation(self.bird, self.client)[0])[2]
		height_penalty = (global_height - target_height)**2

		total_reward = forward_dot - torque_penalty - height_penalty 
		self.datalog["Reward"] = np.array([total_reward])

		return total_reward

	def kill(self):
		'''
		Should kill for actions that break limits
			If the wing limits are broken
			If the height limits are broken
		'''

		height_lim_bool = self.is_height_limit_broken()
		lim_broken_bool_left = self.is_joint_limit_broken(1)
		lim_broken_bool_right = self.is_joint_limit_broken(2)
		instability_bool = self.is_instability_limit_broken()

		lim_bools = [height_lim_bool, 
					 lim_broken_bool_left, 
					 lim_broken_bool_right,
					 instability_bool]

		if True in lim_bools: #then one of the limits is broke and you should kill
			return True # YES KILL
		else:
			return False # NO KILL, WE ARE FINE