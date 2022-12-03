import math
import numpy as np
import os
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_euler_angles
import sys
import xml.etree.ElementTree as ET
import time

from isaacgymenvs.utils.torch_jit_utils import *
# sys.path.append("/home/milo/Documents/isaacflapper3/tasks")
from tasks.base.vec_task import VecTask

from isaacgym import gymutil, gymtorch, gymapi


class Bird(VecTask):

	def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

		self.cfg = cfg
		self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
		self.debug_viz = self.cfg["env"]["enableDebugVis"]

		self.cfg["env"]["numObservations"] = 15
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
		self.N = 15

		self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		self.torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device, requires_grad=False)
		

		cam_pos = gymapi.Vec3(-2.0, -2.0, 1.0)
		# cam_pos = gymapi.Vec3(0.0, -1.0, 1.0)
		cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

		self.lifting_line_setup()
		self.reset_idx(torch.arange(self.num_envs, device=self.device))


	def create_sim(self):
		self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
		self.dt = self.sim_params.dt
		# self._create_ground_plane()
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

		asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
		asset_file = "urdf/spm-asm-v6-2/urdf/spm-asm-v6-2.urdf"


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

			rand_color = torch.rand((3), device=self.device)
			for i in range(5):
				self.gym.set_rigid_body_color(env_ptr, bird_handle, i, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
			

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



	def compute_observations(self):
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)

		base_quat = self.root_states[:, 3:7]
		lin_vel_scale = (1/10)
		ang_vel_scale = (1/10)
		base_lin_vel = quat_rotate_inverse(base_quat, self.root_linvel) * lin_vel_scale
		base_ang_vel = quat_rotate_inverse(base_quat, self.root_angvel) * ang_vel_scale
		
		#linvel
		#euler angles
		Q = torch.roll(base_quat,1,1)
		euler_angles = matrix_to_euler_angles( quaternion_to_matrix(Q), 'ZYX' )
		self.obs_buf = torch.cat((base_lin_vel, base_ang_vel, euler_angles/(torch.pi), self.dof_pos, self.dof_vel), dim=1)

		return self.obs_buf

	def compute_reward(self):
		
		self.rew_buf, self.reset_buf = compute_bird_reward(self.obs_buf, 
															self.psi,
															self.reset_buf,
															self.progress_buf,
															self.max_episode_length)

		# print(self.rew_buf)
		
	
	def make_trap_profile(self, c0, c1, s):

		y_n = torch.reshape((((torch.linspace(self.eps-1.0,1.0-self.eps,self.N, device=self.device))**2)**0.6)**0.5,[self.N,1]); # "station" locs
		idx = torch.tensor(torch.floor(self.N/2), dtype=torch.int32, device=self.device)
		y_n[0:idx,0] = -y_n[0:idx,0]
		y = torch.tensor(s*y_n, dtype=torch.float32,device=self.device)
		c = (y+s)*((c1-c0)/2.0/s)+c0

		return torch.unsqueeze(c, 0), torch.unsqueeze(y, 0)

	def lifting_line_setup(self):

		params = {"eps" : 1e-3,         # spacing from wing ends
				  "wings" : 2,           # number of wings
				  "cla" : 6.5,             # cla
				  "rho" : 1000,             # density
				  "S1" : 0.075,              # wing 1 semispan
				  "S2" : 0.075,              # wing 2 semispan                  
				  "C1" : [0.1, 0.1],              # wing 1 chord
				  "C2" : [0.1, 0.1]}             # wing 2 chord

		self.eps = torch.tensor(params["eps"], device=self.device)
		self.N = torch.tensor(self.N, device=self.device)
		self.W = torch.tensor(params["wings"], device=self.device)
		self.M = torch.tensor(self.num_envs, device=self.device)
		self.Cla = torch.tensor(params["cla"], device=self.device)
		self.S1 = torch.tensor(params["S1"], device=self.device)
		self.S2 = torch.tensor(params["S2"], device=self.device)
		self.C1 = torch.tensor(params["C1"], device=self.device)
		self.C2 = torch.tensor(params["C2"], device=self.device)

		self.s = torch.reshape(torch.tensor((params["S1"],params["S2"]), device=self.device), [2,1,1])

		C1, Y1 = self.make_trap_profile(params["C1"][0], params["C1"][1], self.s[0])
		C2, Y2 = self.make_trap_profile(params["C2"][0], params["C2"][1], self.s[1])

		self.C = torch.cat([C1, C2], dim=0)
		self.Y = torch.cat([Y1, Y2], dim=0) 

		self.theta = torch.acos(self.Y/self.s)
		self.vec1 = torch.sin(self.theta)*self.C*self.Cla/8.0/self.s

		self.n = torch.reshape(torch.linspace(1,self.N,self.N, dtype=torch.float32, device=self.device),[1,self.N])
		self.mat1 = (self.n*self.C*self.Cla/8.0/self.s + torch.sin(self.theta))*torch.sin(self.n*self.theta)
		self.mat2 = 4.0*self.s*torch.sin(self.n*self.theta)
		# Used in drag calculation 
		self.mat3 = torch.sin(self.n*self.theta)
		self.vec3 = torch.tensor(torch.reshape(torch.arange(1,self.N+1,device=self.device), (self.N,1))/torch.sin(self.theta), dtype=torch.float, device=self.device)

		self.force_scale = torch.squeeze(2*self.s/(self.N), dim=-1)

	def lifting_line(self):

		s = 0.15 # span 
		# self.N = 5 # num of nodes #<----bug: force function throws error with stations neq than 5?
		self.rho = 1000 # fluid density
		c = 0.1 # chord
		dr = s/self.N

		COM_wing_1 = None
		COM_wing_2 = None

		psi_wing1 = None
		psi_wing2 = None

		v_flow_2D_local_mag_wing_1 = None
		v_flow_2D_local_mag_wing_2 = None

		v_flow_3D_global_wing_1 = None
		v_flow_3D_global_wing_2 = None	

		global_x_i_wing_1 = None
		global_x_i_wing_2 = None

		nodes_wing_1 = None
		nodes_wing_2 = None

		for wing in ["left", "right"]:

			wing_handle = {"left" : self.lwing_handle, "right" : self.rwing_handle}[wing]

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
			global_x_i = torch.matmul(rot_mat, x_i.transpose(1,2)).transpose(1,2)

			center_node_offset = s/2 + 0.1
			R_nodes = torch.zeros((self.num_envs,self.N,3), dtype=torch.float32, device=self.device, requires_grad=False)
			R_nodes[...,0] = torch.linspace(center_node_offset + -s/2,center_node_offset + s/2,self.N)

			# the wing pos is not the COM, offset it a little bit along the span to correct it
			wing_pos_offset = torch.matmul(rot_mat, (s/2+.1)*x_i.transpose(1,2)).transpose(1,2)[:,0,:] 
			wing_pos += wing_pos_offset

			r_nodes = torch.matmul(rot_mat, R_nodes.transpose(1,2)).transpose(1,2) \
						+ torch.tile(wing_pos, (1,self.N)).reshape(self.num_envs, self.N, 3)



			v_nodes_about_link_origin = torch.cross(torch.tile(wing_angvel, (1,self.N)).reshape(self.num_envs, self.N, 3), 
												torch.matmul(rot_mat, R_nodes.transpose(1,2)).transpose(1,2), dim=2)

			v_nodes = torch.add(torch.tile(wing_linvel, (1,self.N)).reshape(self.num_envs, self.N, 3),
								v_nodes_about_link_origin)

			v_flow_3D_global_wing = -v_nodes


			v_flow_3D_local = torch.matmul(torch.inverse(rot_mat), v_flow_3D_global_wing.transpose(1,2)).transpose(1,2)

			v_flow_2D_local = v_flow_3D_local.detach().clone()
			v_flow_2D_local[:,:,0] = 0.0
			v_flow_2D_local_mag_wing = torch.linalg.norm(v_flow_2D_local, axis=2)

			psi_wing = torch.atan2(v_flow_2D_local[:,:,2],(-1)**(wing_handle+1)*v_flow_2D_local[:,:,1])

			if wing == "left":
				COM_wing_1 = wing_pos
				psi_wing1 = psi_wing
				v_flow_2D_local_mag_wing_1 = v_flow_2D_local_mag_wing
				v_flow_3D_global_wing_1 = v_flow_3D_global_wing
				global_x_i_wing_1 = global_x_i
				nodes_wing_1 = r_nodes

			if wing == "right":
				COM_wing_2 = wing_pos
				psi_wing2 = psi_wing
				v_flow_2D_local_mag_wing_2 = v_flow_2D_local_mag_wing
				v_flow_3D_global_wing_2 = v_flow_3D_global_wing
				global_x_i_wing_2 = global_x_i
				nodes_wing_2 = r_nodes


		# print(psi_wing1.shape, psi_wing2.shape)

		# stack psis, v 2d local mags
		psi = torch.stack([psi_wing1, psi_wing2])
		psi = torch.nan_to_num(psi)
		# print(psi)
		self.r_wing_COM = torch.stack([COM_wing_1, COM_wing_2]).transpose(0,1)
		self.vec2 = psi.transpose(1,2)
		v_flow_2D_local_mag = torch.stack([v_flow_2D_local_mag_wing_1, v_flow_2D_local_mag_wing_2]).transpose(1,2)
		v_flow_3D_global = torch.stack([v_flow_3D_global_wing_1, v_flow_3D_global_wing_2])
		# print(v_flow_3D_global)
		r_nodes = torch.stack([nodes_wing_1, nodes_wing_2]).transpose(0,1)

		###########################################################################################################
		###########################################################################################################

		# print(self.vec1.shape, self.vec2.shape)
		RHS = self.vec1*self.vec2

		self.RHS = RHS
		A = torch.linalg.solve(self.mat1,RHS)   
		# A = torch.matmul(mat1inv,RHS)  
		# each col in above will have the "A" coeffs for the mth wing 
		Gamma = torch.matmul(self.mat2,A)*v_flow_2D_local_mag
		LiftDist = Gamma*v_flow_2D_local_mag*self.rho


		# exec_time = time.perf_counter() - now; 
		Alpha_i = torch.matmul(self.mat3, A * self.vec3) 
		DragDist = v_flow_2D_local_mag*Gamma*Alpha_i*self.rho
		DragDist = DragDist.type(torch.float)

		lift_min=-0.1
		lift_max=0.1
		drag_min=-0.1
		drag_max=0.1
		LiftDist = torch.clamp(LiftDist, lift_min, lift_max)
		DragDist = torch.clamp(DragDist, drag_min, drag_max)


		########### LIFT AND DRAG UNIT VECTORS
		#  LIFT = global flow X global span 
		# stack x_i for 2 wings
		spanwise_vector = torch.stack([global_x_i_wing_1, global_x_i_wing_2])

		Direction_Of_Lift = torch.cross(spanwise_vector, v_flow_3D_global, dim=3)
		Direction_Of_Lift[1,:,:,2] *= -1
		# Direction_Of_Lift[1,:,:,2] *= -1


		Direction_Of_Lift = torch.div(Direction_Of_Lift, torch.linalg.norm(Direction_Of_Lift, axis=3).reshape(2, self.num_envs, self.N, 1))
		
		#  DRAG = UNIT vector of 2d global flow
		Direction_Of_Drag = torch.div(v_flow_3D_global, torch.linalg.norm(v_flow_3D_global, axis=3).reshape(2, self.num_envs, self.N, 1))
		###########

		# F_global = Direction_Of_Lift*LiftDist + Wind_apparent_global_normalized*DragDist
		LD1 = torch.permute(LiftDist, (2,0,1))*self.force_scale
		LiftDist_ = torch.reshape(LD1, (self.W*self.M,self.N,1))

		DD1 = torch.permute(DragDist, (2,0,1))*self.force_scale
		DragDist_ = torch.reshape(DD1, (self.W*self.M,self.N,1))

		DOL1 = torch.permute(Direction_Of_Lift, (0,2,1,3))
		DOD1 = torch.permute(Direction_Of_Drag, (0,2,1,3))

		Direction_Of_Lift_ = torch.reshape(DOL1, (self.W*self.M, self.N, 3))
		Direction_Of_Drag_ = torch.reshape(DOD1, (self.W*self.M, self.N, 3))

		self.DOL = Direction_Of_Lift
		self.Lift_global = LiftDist_*Direction_Of_Lift_	
		self.Drag_global = DragDist_*Direction_Of_Drag_
		self.F_global = self.Lift_global + self.Drag_global        
		self.F_global = torch.reshape(self.F_global, (self.M, 2, self.N, 3)) # global forces at each node
	

		#-----------------------------------------------------#
		# Turning force distributions (lift + drag) into single forces and torques that act on each wing
		# Sum forces, apply to COM
		forces_sum = torch.sum(self.F_global, dim=2)
		forces_sum = torch.cat((torch.zeros(self.num_envs, 3, 3, device=self.device), forces_sum), dim=1)
		forces_sum = torch.nan_to_num(forces_sum)
		# print("Sum forces: ", forces_sum)
		
		# Calculate torque contribution from each force on each node

		# Get vector from COM to node
		r_wing_COM = self.r_wing_COM.repeat_interleave(self.N,dim=1).reshape(self.num_envs, self.W, self.N, 3)
		r_COM_to_node = r_nodes - r_wing_COM

		# Take r_COM_to_node X F_global (torque cont from each node)
		torques = torch.cross(r_COM_to_node, self.F_global, dim=3)
		# Sum torques, apply to COM
		torques_sum = torch.sum(torques, dim=2)
		torques_sum = torch.cat((torch.zeros(self.num_envs, 3, 3, device=self.device), torques_sum), dim=1)
		torques_sum = torch.nan_to_num(torques_sum)

		# Assign these forces and torques to the selfforces/torques defined up top, in the 3rd and 4th rigid bodies (the wings)


		#-----------------------------------------------------#

		return forces_sum, torques_sum, self.r_wing_COM, psi
		###########################################################################################################
		###########################################################################################################

	def pre_physics_step(self, actions):
		self.actions = actions.clone().to(self.device)
		self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.actions))


		##################################### LIFTING LINE ###################################################
		self.forces, self.torques, self.wing_CoMs, self.psi = self.lifting_line()

		##############################################################################################
		# print("APPLYING FORCES")
		# print(self.forces)
		# print(self.psi)
		self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), 
															gymtorch.unwrap_tensor(self.torques), 
															gymapi.ENV_SPACE)

	

	def post_physics_step(self):
		self.progress_buf += 1

		env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
		if len(env_ids) > 0:
			self.reset_idx(env_ids)


		# if self.viewer:
		# 	self.starts = self.wing_CoMs
		# 	ends_test = torch.zeros_like(self.starts)
		# 	ends_test[:,:,2] = 1
		# 	self.ends = self.starts + self.forces[0,3:5,:]
		# 	# self.ends = self.starts + ends_test

		# 	# print(self.ends)
		# 	verts_all = torch.stack([self.starts, self.ends], dim=2)

		# 	verts = verts_all[0,:,:]
		# 	verts = torch.unsqueeze(verts,dim=0).cpu().numpy()			
		# 	colors = np.zeros((4, 3), dtype=np.float32)
		# 	colors[..., 0] = 1.0
		# 	self.gym.clear_lines(self.viewer)
		# 	self.gym.add_lines(self.viewer, None, 20, verts, colors)


		self.compute_observations()
		self.compute_reward()
		# print('REWARD')
		# print(self.rew_buf)

		# print('~~~~~~~~~~~~~~~~~~~~~~~')
		# print('Reset buff')	
		# print(self.reset_buf)
		# print('Obs')
		# print(self.obs_buf)
		# print('Actions')
		# print(self.actions)
		# print('DOF pos')
		# print(self.dof_pos)
		# print('DOF vels')
		# print(self.dof_vel)
		# print('Psi')
		# print(self.psi)
		# print('forces')
		# print(self.forces)
		# print('torques')
		# print(self.torques)
		# print('ARE YOU NAN!?!?!')
		# print('Reset buff')
		# print(torch.any(torch.isnan(self.reset_buf)))
		# print('Obs')
		# print(torch.any(torch.isnan(self.obs_buf)))
		# print('Actions')
		# print(torch.any(torch.isnan(self.actions)))
		# print('DOF pos')
		# print(torch.any(torch.isnan(self.dof_pos)))
		# print('DOF vels')
		# print(torch.any(torch.isnan(self.dof_vel)))
		# print('Psi')
		# print(torch.any(torch.isnan(self.psi)))
		# print('forces')
		# print(torch.any(torch.isnan(self.forces)))
		# print('torques')
		# print(torch.any(torch.isnan(self.torques)))
		

		

		

	def reset_idx(self, env_ids):
		# thank u @Tbarkin121 !!
		positions = torch.zeros((len(env_ids), self.num_dof), device=self.device)
		velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

		self.dof_pos[env_ids, :] = positions[:]
		self.dof_vel[env_ids, :] = velocities[:]

		env_ids_int32 = env_ids.to(dtype=torch.int32)
		# print('env ids')
		# print(env_ids_int32)
		self.gym.set_dof_state_tensor_indexed(self.sim,
											  gymtorch.unwrap_tensor(self.dof_state),
											  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
		
		root_pos_update = torch.zeros((len(env_ids), 3), device=self.device)
		root_pos_update[:,2] = 0.3

		root_rot_update = to_torch([0,0,0,1], device=self.device).repeat((len(env_ids), 1))

		root_linvel_update = torch.zeros((len(env_ids), 3), device=self.device)
		root_angvel_update = torch.zeros((len(env_ids), 3), device=self.device)

		# SET INIT HoG CONTROL HERE!!! #
		root_linvel_update[:,1] = -0.1	

		self.root_pos[env_ids, :] = root_pos_update
		self.root_rot[env_ids, :] = root_rot_update
		self.root_linvel[env_ids, :] = root_linvel_update
		self.root_angvel[env_ids, :] = root_angvel_update
		self.gym.set_actor_root_state_tensor_indexed(self.sim,
											  gymtorch.unwrap_tensor(self.root_states),
											  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

		self.reset_buf[env_ids] = 1
		self.progress_buf[env_ids] = 0

	
#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
def compute_bird_reward(obs_buf, psi, reset_buf, progress_buf, max_episode_length):
	# type: (Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

	# Velocity rewards
	base_vel = obs_buf[..., 0:3]/100
	base_rot = obs_buf[..., 3:6] 

	speed_reward = torch.norm(base_vel, dim=1)
	# speed_reward = base_vel[:,1]**2 -base_vel[:,0]**2 -base_vel[:,2]**2
	rot_reward = -base_rot[:,1]**2 -base_rot[:,0]**2 -base_rot[:,2]**2
	psi_reward =  -torch.sum(torch.mean(psi,dim=2)**2, dim=0)

	psi_fail = torch.abs(psi) >  1.0													#psi fail condition
	psi_reset = torch.where(psi_fail, torch.ones_like(psi_fail), torch.zeros_like(psi_fail))
	psi_reset = torch.squeeze(torch.sum(psi_reset, dim=2)) 				    			#looking for any fail conditions among the station points
	psi_reset = torch.squeeze(torch.sum(psi_reset, dim=0))								#looking for any fail conditions among the two wings		

	
	# reward = speed_reward + rot_reward + psi_reward
	alive_reward = torch.ones_like(speed_reward, dtype=torch.float)

	reward = speed_reward
	# reward = torch.where(psi_reset>0, torch.zeros_like(reward, dtype=torch.float), reward)
	# # resets due to episode length
	time_out = progress_buf >= max_episode_length - 1  # no terminal reward for time-outs
	
	# reset = time_out | psi_reset>0
	reset = time_out
	


	# reset = time_out

	# if(torch.any(time_out)):
	# 	print('timeout reset')
	# if(torch.any(psi_reset)):
	# 	print('psi_reset')

	return reward.detach(), reset