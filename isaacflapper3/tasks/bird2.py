import math
import numpy as np
import os
import torch
from pytorch3d.transforms import quaternion_to_matrix
import sys
import xml.etree.ElementTree as ET
import time

from isaacgymenvs.utils.torch_jit_utils import *
sys.path.append("/home/afarah/Downloads/IsaacGymEnvs/isaacgymenvs/tasks")
from base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi


class Bird(VecTask):

	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

		self.cfg = cfg

		self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
		self.debug_viz = self.cfg["env"]["enableDebugVis"]

		self.cfg["env"]["numObservations"] = 13
		self.cfg["env"]["numActions"] = 3

		super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

		self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
		self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
		self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
		
		self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
		self.root_states = gymtorch.wrap_tensor(self.actor_root_state)
		self.root_pos = self.root_states.view(self.num_envs, 1, 13)[..., 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
		self.root_rot = self.root_states.view(self.num_envs, 1, 13)[..., 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
		self.root_linvel = self.root_states.view(self.num_envs, 1, 13)[..., 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
		self.root_angvel = self.root_states.view(self.num_envs, 1, 13)[..., 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

		rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
		self.rb_states = gymtorch.wrap_tensor(rigid_body_tensor)

		self.rb_pos = self.rb_states.view(self.num_envs, self.num_bodies, 13)[..., :, 0:3] #num_envs, num_bodies, 13 (pos,ori,Lvel,Avel)
		self.rb_rot = self.rb_states.view(self.num_envs, self.num_bodies, 13)[..., :, 3:7] #num_envs, num_bodies, 13 (pos,ori,Lvel,Avel)
		self.rb_linvel = self.rb_states.view(self.num_envs, self.num_bodies, 13)[..., :, 7:10] #num_envs, num_bodies, 13 (pos,ori,Lvel,Avel)
		self.rb_angvel = self.rb_states.view(self.num_envs, self.num_bodies, 13)[..., :, 10:13] #num_envs, num_bodies, 13 (pos,ori,Lvel,Avel)

		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)

		# control tensors
		self.N = 5
		self.lforces = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		self.rforces = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		self.lnodes = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		self.rnodes = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		cam_pos = gymapi.Vec3(-2.0, -2.0, 1.0)
		# cam_pos = gymapi.Vec3(0.0, -1.0, 1.0)
		cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

		self.t = 0

		# if self.t == 0:
		# 	init_state = to_torch([[0,0,1,0,0,0,0,-.01,0,0,0,0,0],[0,0,1,0,0,0,0,-.01,0,0,0,0,0]], device=self.device)
		# 	self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(init_state))

	def create_sim(self):
		self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
		self.dt = self.sim_params.dt
		self._create_ground_plane()
		self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

	def _create_ground_plane(self):
		plane_params = gymapi.PlaneParams()
		# set the normal force to be z dimension
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)

	def _create_envs(self, num_envs, spacing, num_per_row):
		# define plane on which environments are initialized
		lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
		upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

		asset_root = "/home/afarah/Documents/UML/willis/rluff/assets"
		asset_file = "/spm-asm-v6-2/urdf/spm-asm-v6-2.urdf"

		asset_options = gymapi.AssetOptions()
		asset_options.fix_base_link = False
		asset_options.collapse_fixed_joints = False

		bird_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
		self.num_dof = self.gym.get_asset_dof_count(bird_asset)
		self.num_bodies = self.gym.get_asset_rigid_body_count(bird_asset)

		pose = gymapi.Transform()
		pose.p.z = 1.0
		# asset is rotated z-up by default, no additional rotations needed
		pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
		# pose.r = gymapi.Quat(0.7071, 0.7071, 0, 0)

		self.bird_handles = []
		# self.target_handles = []
		self.envs = []
		for i in range(self.num_envs):
			# create env instance
			env_ptr = self.gym.create_env(
				self.sim, lower, upper, num_per_row
			)
			bird_handle = self.gym.create_actor(env_ptr, bird_asset, pose, "bird", i, 1, 0)
			dof_props = self.gym.get_actor_dof_properties(env_ptr, bird_handle)

			self.gym.set_actor_dof_properties(env_ptr, bird_handle, dof_props)

			self.envs.append(env_ptr)
			self.bird_handles.append(bird_handle)

		self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, bird_handle)
		print(self.body_dict)
		for b in self.body_dict:
			print(b)

		self.lwing_handle = self.gym.find_actor_rigid_body_handle(env_ptr, bird_handle, "left")
		self.rwing_handle = self.gym.find_actor_rigid_body_handle(env_ptr, bird_handle, "right")

		self.init_data()

	def init_data(self):
		lwing = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bird_handles[0], "left")
		rwing = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bird_handles[0], "right")

		lwing_pose = self.gym.get_rigid_transform(self.envs[0], lwing)
		rwing_pose = self.gym.get_rigid_transform(self.envs[0], rwing)

		self.lwing_pose_inv = lwing_pose.inverse()
		self.rwing_pose_inv = rwing_pose.inverse()

		self.lwing_local_pose = self.lwing_pose_inv * lwing_pose
		self.rwing_local_pose = self.rwing_pose_inv * rwing_pose

		self.lwing_local_pos = to_torch([self.lwing_local_pose.p.x, self.lwing_local_pose.p.y,
												self.lwing_local_pose.p.z], device=self.device).repeat((self.num_envs, 1))
		self.lwing_local_rot = to_torch([self.lwing_local_pose.r.x, self.lwing_local_pose.r.y,
												self.lwing_local_pose.r.z, self.lwing_local_pose.r.w], device=self.device).repeat((self.num_envs, 1))
		self.rwing_local_pos = to_torch([self.rwing_local_pose.p.x, self.rwing_local_pose.p.y,
												self.rwing_local_pose.p.z], device=self.device).repeat((self.num_envs, 1))
		self.rwing_local_rot = to_torch([self.rwing_local_pose.r.x, self.rwing_local_pose.r.y,
												self.rwing_local_pose.r.z, self.rwing_local_pose.r.w], device=self.device).repeat((self.num_envs, 1))

		self.lwing_pos = torch.zeros_like(self.lwing_local_pos)
		self.lwing_rot = torch.zeros_like(self.lwing_local_rot)
		self.rwing_pos = torch.zeros_like(self.rwing_local_pos)
		self.rwing_rot = torch.zeros_like(self.rwing_local_rot)

	# def compute_observations(self, env_ids=None):


		# left_wing_pos_G = self.rb_pos[:,self.body_dict["left"],:]
		# right_wing_pos_G = self.rb_pos[:,self.body_dict["right"],:]

		# left_wing_ori = self.rb_ori[:,self.body_dict["left"],:]
		# right_wing_ori = self.rb_ori[:,self.body_dict["right"],:]

		# self.lwing_pos = self.rb_pos[:,self.lwing_handle,:]
		# self.rwing_pos = self.rb_pos[:,self.rwing_handle,:]

		# self.lwing_rot = self.rb_rot[:,self.rwing_handle,:]
		# self.rwing_rot = self.rb_rot[:,self.rwing_handle,:]

	def compute_observations(self):
		# if env_ids is None:
		# 	env_ids = np.arange(self.num_envs)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		self.obs_buf = self.root_states.view(self.num_envs, 1, 13)
		return self.obs_buf

	def compute_reward(self):
		self.rew_buf[:], self.reset_buf[:] = compute_bird_reward(self.obs_buf.view(self.num_envs, 13), 
																	  self.reset_buf,
																	  self.progress_buf,
																	  self.max_episode_length)
		# print(compute_bird_reward(self.obs_buf, self.reset_buf, self.progress_buf, self.max_episode_length))

	def ClCd_model(self, alpha):
		# Here you would init training for clcd model
		# or run (once) the fit on xfoil data
		# Once it is made, it returns cl and cd for alpha
		# Mock Cla/Cda data

		# mock_data_num = 1000
		# alpha_data = np.arange(-np.pi,np.pi,mock_data_num)
		# Cla_data = np.sin(alpha_data) + (1e-3)*np.random.normal(size=mock_data_num)
		# Cda_data = np.power(np.sin(alpha_data), 2) + (1e-6)*np.random.normal(size=mock_data_num)
		
		# if not self.ClCd_model_made: # model has not been made yet, so make it
		# 	# train/interpolate
		# 	self.ClCd_model_made = True
		# 	pass

		Cla_pred = 0.6*torch.sin(alpha) # model prediction
		Cda_pred = 0.03*torch.pow(torch.sin(alpha),2) # model prediction

		return Cla_pred, Cda_pred

	def bemt2(self, wing):

		s = 0.15 # span 
		# self.N = 5 # num of nodes #<----bug: force function throws error with stations neq than 5?
		rho = 1000 # fluid density
		c = 0.1 # chord
		dr = s/self.N

		wing_handle = {"left" : self.lwing_handle,
						"right" : self.rwing_handle}[wing]

		wing_pos = self.rb_pos[:,wing_handle,:]
		wing_rot = self.rb_rot[:,wing_handle,:] # xyzw
		wing_linvel = self.rb_linvel[:,wing_handle,:]
		wing_angvel = self.rb_angvel[:,wing_handle,:]

		# print(wing_pos[0,:])

		# unit vectors
		x_i = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		y_i = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		z_i = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		
		x_i[:,:,0] = 1.0
		y_i[:,:,1] = 1.0
		z_i[:,:,2] = 1.0

		# torch3d quaternion uses wxyz, must roll isaac xyzw quaternion
		wing_rot_wxyz = torch.roll(wing_rot, 1, 1)

		rot_mat = quaternion_to_matrix(wing_rot_wxyz)

		center_node_offset = s/2 + 0.1
		R_nodes = torch.zeros((self.num_envs,self.N,3), dtype=torch.float32, device=self.device, requires_grad=False)
		# R_nodes[...,1] = (-1)**(wing_handle)*0.02
		R_nodes[...,0] = torch.linspace(center_node_offset + -s/2,center_node_offset + s/2,self.N)

		r_nodes = torch.matmul(rot_mat, R_nodes.transpose(1,2)).transpose(1,2) \
					+ torch.tile(wing_pos, (1,self.N)).reshape(self.num_envs, self.N, 3)


		v_nodes_about_link_origin = torch.cross(torch.tile(wing_angvel, (1,self.N)).reshape(self.num_envs, self.N, 3), 
											r_nodes, dim=2)

		# wing_angvel_local = torch.bmm(torch.inverse(rot_mat), wing_angvel[..., None]) # in local frame
		# # get vel about local frame origin in LOCAL frame, then transform to global
		# V_nodes_about_link_origin = torch.cross(torch.tile(wing_angvel_local, (1,self.N)).reshape(self.num_envs, self.N, 3), 
		# 									R_nodes, dim=2)
		# v_nodes_about_link_origin = torch.matmul(rot_mat, V_nodes_about_link_origin.transpose(1,2)).transpose(1,2)


		v_nodes = torch.add(torch.tile(wing_linvel, (1,self.N)).reshape(self.num_envs, self.N, 3),
							v_nodes_about_link_origin)

		v_flow_3D_global = -v_nodes

		v_flow_3D_local = torch.matmul(torch.inverse(rot_mat), v_flow_3D_global.transpose(1,2)).transpose(1,2)

		v_flow_2D_local = v_flow_3D_local.detach().clone()
		v_flow_2D_local[:,:,0] = 0.0
		v_flow_2D_local_mag = torch.linalg.norm(v_flow_2D_local, axis=2).repeat_interleave(3,dim=1).reshape(self.num_envs, self.N, 3) 

		psi = torch.atan2(v_flow_2D_local[:,:,2],(-1)**(wing_handle+1)*v_flow_2D_local[:,:,1])
		psi = psi.repeat_interleave(3,dim=1).reshape(self.num_envs, self.N, 3)
		# print(psi[0,...])

		v_sq = torch.pow(torch.linalg.norm(v_flow_2D_local, axis=2),2)
		# repeat for element wise div for unit vectors
		v_sq = v_sq.repeat_interleave(3,dim=1).reshape(self.num_envs, self.N, 3) 

		vf_norm = (-1)**(wing_handle+1)*torch.cross(x_i, v_flow_2D_local, dim=2)
		vf_norm_mag = torch.linalg.norm(vf_norm,axis=2) # magnitude of each normal 
		# repeat for element wise div for unit vectors
		vf_norm_mag = vf_norm_mag.repeat_interleave(3,dim=1).reshape(self.num_envs, self.N, 3) 

		lift_unit_vector = torch.div(vf_norm,vf_norm_mag)
		drag_unit_vector = torch.div(v_flow_2D_local,v_flow_2D_local_mag)
		# print(drag_unit_vector)


		Cl, Cd = self.ClCd_model(psi)

		dL = 0.5*rho*c*dr*torch.mul(v_sq, torch.mul(Cl, lift_unit_vector))
		dD = 0.5*rho*c*dr*torch.mul(v_sq, torch.mul(Cd, drag_unit_vector))
		dT = torch.add(torch.mul(dL, torch.cos(psi)),torch.mul(dD, torch.sin(psi)))

		if torch.isnan(torch.sum(dT)):
			dT = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
			# dT = None

		dT_global = torch.matmul(rot_mat, dT.transpose(1,2)).transpose(1,2)

		# global vectors needed for viz
		v_flow_2D_global = torch.matmul(rot_mat, v_flow_2D_local.transpose(1,2)).transpose(1,2)
		lift_unit_vector_global = torch.matmul(rot_mat, lift_unit_vector.transpose(1,2)).transpose(1,2)
		drag_unit_vector_global = torch.matmul(rot_mat, drag_unit_vector.transpose(1,2)).transpose(1,2)
		
		aux_data_pack = {"v_nodes" : v_nodes, "dL" : dL, "dD" : dD, 
						 "lift" : lift_unit_vector_global, "drag" : drag_unit_vector_global,
						 "v2D" : v_flow_2D_global,
						 "nodes local" : R_nodes,
						 "forces local" : dT}

		# actual global pos of nodes						 
		rnodes_global = torch.matmul(rot_mat, R_nodes.transpose(1,2)).transpose(1,2) \
						+ torch.tile(wing_pos, (1,self.N)).reshape(self.num_envs, self.N, 3)

		return wing_handle, r_nodes, dT_global, aux_data_pack

	def pre_physics_step(self, actions):

		indices = to_torch([0,1], 
							dtype=torch.int32, 
							device=self.device).repeat((self.num_envs, 1))


		if self.t < 10000:
			# self.root_linvel[:,1] = -1. #<--- this is how you set HoG control!
			# self.root_linvel[:,0] = 0 #<--- this is how you set HoG control!
			# self.root_angvel[:,:] = 0
			self.root_angvel[:,1] = 0
			self.root_angvel[:,2] = 0
			self.gym.set_actor_root_state_tensor_indexed(self.sim, 
														 gymtorch.unwrap_tensor(self.root_states),
														 gymtorch.unwrap_tensor(indices), len(indices))

		_actions = [-1.*np.sin(5*self.t*0.01), 10*np.sin(5*self.t*0.01), 10*np.sin(5*self.t*0.01)]
		# _actions = [0, 1*np.sin(10*self.t*0.01), 1*np.sin(10*self.t*0.01)]
		# _actions = [0,-1,1]
		# _actions = [-0.5,0,0]
		# _actions = [-1.*np.sin(10*self.t*0.01), 0, 0]

		actions = to_torch(_actions, 
							dtype=torch.float, 
							device=self.device).repeat((self.num_envs, 1))
		# print(actions)
		self.actions = actions.clone().to(self.device)
		self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.actions))

		lwing_handle, self.lr_nodes, self.ldT, self.laux_data_pack = self.bemt2("left")
		rwing_handle, self.rr_nodes, self.rdT, self.raux_data_pack = self.bemt2("right")

		forces_rf = "env"
		forces_app = True # apply forces?

		if forces_rf == "local":
			self.lforces = self.dt*self.laux_data_pack["forces local"]
			self.rforces = self.dt*self.raux_data_pack["forces local"]
			self.lnodes = self.laux_data_pack["nodes local"]
			self.rnodes = self.raux_data_pack["nodes local"]
			frame = gymapi.LOCAL_SPACE

		if forces_rf == "env":
			self.lforces[...] = 0*self.dt*self.ldT
			self.rforces[...] = self.dt*self.rdT
			self.lnodes[...] = self.lr_nodes
			self.rnodes[...] = self.rr_nodes
			frame = gymapi.ENV_SPACE


		print(self.lforces, "\n\n", self.rforces)

		# print(gymtorch.wrap_tensor(gymtorch.unwrap_tensor(lforces.contiguous()))).view((self.num_envs, self.N, 3))

		# if self.t > 2:
		# 	exit()

		# lforces = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		# rforces = torch.zeros((self.num_envs, self.N, 3), dtype=torch.float32, device=self.device, requires_grad=False)

		# print(lnodes, "\n\n", rnodes)

		if self.t > 400 and forces_app:

			self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(self.lforces), 
															gymtorch.unwrap_tensor(self.lnodes), 
															frame)
			self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(self.rforces), 
															gymtorch.unwrap_tensor(self.rnodes), 
															frame)

		'''
		if force_rf == "env":
			forces = torch.cat((self.ldT, self.rdT), 1)
			nodes = torch.cat((self.lr_nodes, self.rr_nodes), 1)
			frame = gymapi.LOCAL_SPACE

		if force_rf == "local":
			forces = torch.cat((self.laux_data_pack["forces local"], 
								self.raux_data_pack["forces local"]), 1)
			nodes = torch.cat((self.laux_data_pack["nodes local"],
							   self.raux_data_pack["nodes local"]), 1)
			frame = gymapi.ENV_SPACE

		self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, 
													   gymtorch.unwrap_tensor(forces),
													   gymtorch.unwrap_tensor(nodes),
													   frame)
		'''

		# ldT[...] = 0
		# ldT[...,2] = 1
		# exit()

		# if self.t == 10:
		# 	print("Applying force")
		# 	time.sleep(1)
		# 	self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(ldT.contiguous()), 
		# 													gymtorch.unwrap_tensor(lr_nodes.contiguous()), 
		# 													gymapi.self.ENV_SPACE)

	def post_physics_step(self):
		self.progress_buf += 1
		self.t += 1
		env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		if len(env_ids) > 0:
			self.reset_idx(env_ids)

		if self.viewer:
			l_starts = self.lr_nodes 
			l_ends = l_starts + 100*self.dt*self.ldT
			# l_ends = l_starts + 1 * self.laux_data_pack["lift"] 
			
			r_starts = self.rr_nodes 
			r_ends = r_starts + 100*self.dt*self.rdT
			# r_ends = r_starts + 1 * self.raux_data_pack["lift"]

			# submit debug line geometry
			l_verts = torch.stack([l_starts, l_ends], dim=2).cpu().numpy()
			r_verts = torch.stack([r_starts, r_ends], dim=2).cpu().numpy()
			colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
			colors[..., 0] = 1.0
			self.gym.clear_lines(self.viewer)
			self.gym.add_lines(self.viewer, None, self.num_envs * 4, l_verts, colors)
			self.gym.add_lines(self.viewer, None, self.num_envs * 4, r_verts, colors)
		
		self.compute_observations()
		# self.compute_reward()

	def reset_idx(self, env_ids):
		# thank u @Tbarkin121 !!
		positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
		velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

		self.dof_pos[env_ids, :] = positions[:]
		self.dof_vel[env_ids, :] = velocities[:]

		env_ids_int32 = env_ids.to(dtype=torch.int32)
		self.gym.set_dof_state_tensor_indexed(self.sim,
											  gymtorch.unwrap_tensor(self.dof_state),
											  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
		
		root_pos_update = torch.zeros((len(env_ids), 3), device=self.device)
		root_pos_update[:,2] = 0.3

		rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

		root_rot_update = to_torch([0,0,0,1], device=self.device).repeat((self.num_envs, 1))

		root_linvel_update = torch.zeros((len(env_ids), 3), device=self.device)
		root_angvel_update = torch.zeros((len(env_ids), 3), device=self.device)
		self.root_pos[env_ids, :] = root_pos_update
		self.root_rot[env_ids, :] = root_rot_update
		self.root_linvel[env_ids, :] = root_linvel_update
		self.root_angvel[env_ids, :] = root_angvel_update
		self.gym.set_actor_root_state_tensor_indexed(self.sim,
											  gymtorch.unwrap_tensor(self.root_states),
											  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

		self.reset_buf[env_ids] = 0
		self.progress_buf[env_ids] = 0

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_bird_reward(obs_buf, reset_buf, progress_buf, max_episode_length):
	# type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

	# Velocity rewards
	base_rot = obs_buf[..., 3:7] 
	base_vel = obs_buf[..., 7:10] 
	local_forward_vec = torch.zeros_like(base_vel)
	local_forward_vec[:,1] = -1.
	base_rot_wxyz = torch.roll(base_rot, 1, 1)
	rot_mat = quaternion_to_matrix(base_rot_wxyz)
	global_forward_vec = torch.bmm(rot_mat, local_forward_vec[...,None])
	# forward_dot = torch.matmul(base_vel,global_forward_vec)
	forward_dot = torch.bmm(base_vel.view(base_vel.shape[0], 1, 3), global_forward_vec.view(base_vel.shape[0], 3, 1)) 
	print(forward_dot.flatten())
	reward = forward_dot.flatten()

	# # resets due to episode length
	reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

	return reward, reset
