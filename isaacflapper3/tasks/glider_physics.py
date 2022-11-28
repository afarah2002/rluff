from hashlib import shake_128
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgym.torch_utils import *
# import pygame
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix, quaternion_apply
import matplotlib.pyplot as plt
from .csv_logger import CSVLogger

class LiftingLine:
    def __init__(self, glider_params, dt, debug_flags):
        # https://journals.sfu.ca/ts/index.php/ts/article/viewFile/42/38
        # https://www.researchgate.net/publication/318421320_Dynamic_Maneuver_Loads_Calculation_for_a_Sailplane_and_Comparison_with_Flight_Test/link/5d0ccb22a6fdcc2462982ede/download
        self.device = 'cuda'
        self.mass = 0.5
        Ixx = 0.2725
        Ixz = 0.007
        Iyy = 0.0917
        Izz = 0.3614
        self.inertia = torch.tensor([[Ixx, 0.0, -Ixz],
                                     [0.0, Iyy, 0.0],
                                     [-Ixz, 0.0, Izz]], device=self.device)
        self.dt = dt
        self.glider_params = glider_params
        self.setup(self.glider_params)

        self.alpha_log = CSVLogger('alphas.csv', fields=['Env','A1','A2','A3','A4'], test_name=debug_flags['log_name'])
        self.state_global_log = CSVLogger('state_global.csv', fields=['Env','px', 'py', 'pz',
                                                                      'q1', 'q2', 'q3', 'q4',
                                                                      'vx', 'vy', 'vz', 
                                                                      'wx', 'wy', 'wz'], test_name=debug_flags['log_name'])
        self.euler_angles_log = CSVLogger('rk4_euler_angs.csv', fields=['Env','roll', 'pitch', 'yaw'], test_name=debug_flags['log_name'])
        self.lv_loc_log = CSVLogger('rk4_lv_loc.csv', fields=['Env','x', 'y', 'z'], test_name=debug_flags['log_name'])
        self.av_loc_log = CSVLogger('rk4_av_loc.csv', fields=['Env','x', 'y', 'z'], test_name=debug_flags['log_name'])
        self.as_log = CSVLogger('as.csv', fields=['Env','air_speed'], test_name=debug_flags['log_name'])
        self.gs_log = CSVLogger('gs.csv', fields=['Env','ground_speed'], test_name=debug_flags['log_name'])

        
    def make_T_wing(self, dx, dy, dz):
        return torch.tensor((dx,dy,dz), device=self.device)

    def make_H_wing(self, roll, pitch):
        theta_x = roll*torch.pi/180.0
        theta_y = pitch*torch.pi/180.0


        H_wing_x = torch.tensor( [[1.0, 0.0, 0.0],
                                [0.0, torch.cos(theta_x), torch.sin(theta_x)],
                                [0.0, -torch.sin(theta_x), torch.cos(theta_x)]], device=self.device)

        # Check Sines on sines later
        H_wing_y = torch.tensor( [[torch.cos(theta_y), 0.0, -torch.sin(theta_y)],
                                [0.0, 1.0, 0.0],
                                [torch.sin(theta_y), 0.0, torch.cos(theta_y)]], device=self.device)

        H_wing = torch.matmul(H_wing_y, H_wing_x)

        return torch.unsqueeze(H_wing, dim=0)
    
    def make_trap_profile(self, c0, c1, s):
        y_n = torch.reshape((((torch.linspace(self.eps-1.0,1.0-self.eps,self.N, device=self.device))**2)**0.6)**0.5,[self.N,1]); # "station" locs
        idx = torch.tensor(torch.floor(self.N/2), dtype=torch.int32, device=self.device)
        y_n[0:idx,0] = -y_n[0:idx,0]
        y = torch.tensor(s*y_n, dtype=torch.float32,device=self.device)
        c = (y+s)*((c1-c0)/2.0/s)+c0

        return torch.unsqueeze(c, 0), torch.unsqueeze(y, 0)

    def setup(self, params):
        self.eps = torch.tensor(params["eps"], device=self.device)
        self.N = torch.tensor(params["station_pts"], device=self.device)
        self.W = torch.tensor(params["wings"], device=self.device)
        self.M = torch.tensor(params["envs"], device=self.device)
        self.Cla = torch.tensor(params["cla"], device=self.device)
        self.rho = torch.tensor(params["rho"], device=self.device)

        TW1 = self.make_T_wing(params["TW1"][0], params["TW1"][1], params["TW1"][2])
        TW2 = self.make_T_wing(params["TW2"][0], params["TW2"][1], params["TW2"][2])
        TW3 = self.make_T_wing(params["TW3"][0], params["TW3"][1], params["TW3"][2])
        TW4 = self.make_T_wing(params["TW4"][0], params["TW4"][1], params["TW4"][2])
        self.T_wing = torch.cat([TW1, TW2, TW3, TW4])
        self.T_wing = torch.reshape(self.T_wing, (4,1,3))

        HW1 = self.make_H_wing(params["HW1"][0], params["HW1"][1])
        HW2 = self.make_H_wing(params["HW2"][0], params["HW2"][1])
        HW3 = self.make_H_wing(params["HW3"][0], params["HW3"][1])
        HW4 = self.make_H_wing(params["HW4"][0], params["HW4"][1])

        self.H_wing = torch.cat([HW1, HW2, HW3, HW4], dim=0)
        self.H_wing = torch.reshape(self.H_wing, (1,self.W*3,3)) 
        # The plan will keep this matrix

        s1 = ( torch.sqrt(torch.pow(params["TW1"][1],2) + torch.pow(params["TW1"][2],2)) )
        s2 = ( torch.sqrt(torch.pow(params["TW2"][1],2) + torch.pow(params["TW2"][2],2)) )
        s3 = ( torch.sqrt(torch.pow(params["TW3"][1],2) + torch.pow(params["TW3"][2],2)) )
        s4 = ( torch.sqrt(torch.pow(params["TW4"][1],2) + torch.pow(params["TW4"][2],2)) )
        self.s = torch.reshape(torch.tensor((s1,s2,s3,s4), device=self.device), [4,1,1])

        C1, Y1 = self.make_trap_profile(params["C1"][0], params["C1"][1], self.s[0])
        C2, Y2 = self.make_trap_profile(params["C2"][0], params["C2"][1], self.s[1])
        C3, Y3 = self.make_trap_profile(params["C3"][0], params["C3"][1], self.s[2])
        C4, Y4 = self.make_trap_profile(params["C4"][0], params["C4"][1], self.s[3])
        self.C = torch.cat([C1, C2, C3, C4], dim=0)
        self.Y = torch.cat([Y1, Y2, Y3, Y4], dim=0) 

        H_ = self.H_wing[:, 1::3, :]
        H_ = torch.reshape(H_, (4,1,3))
        Y_ = torch.unsqueeze(self.Y[...,0], dim=-1)
        self.r_plane = H_*Y_ + self.T_wing


        self.theta = torch.acos(self.Y/self.s)
        self.vec1 = torch.sin(self.theta)*self.C*self.Cla/8.0/self.s


        self.n = torch.reshape(torch.linspace(1,self.N,self.N, dtype=torch.float32, device=self.device),[1,self.N])
        self.mat1 = (self.n*self.C*self.Cla/8.0/self.s +  torch.sin(self.theta))*torch.sin(self.n*self.theta)
        self.mat2 = 4.0*self.s*torch.sin(self.n*self.theta)
        # Used in drag calculation 
        self.mat3 = torch.sin(self.n*self.theta)
        self.vec3 = torch.tensor(torch.reshape(torch.arange(1,self.N+1,device=self.device), (self.N,1))/torch.sin(self.theta), dtype=torch.float, device=self.device)

        self.force_scale = torch.squeeze(2*self.s/(self.N), dim=-1)


    def wind_function(self, height):
        k999 = 2*6.907
        k99 = 2*4.595
        
        speed = 10.
        
        thickness = 10.
        center = thickness*0.5
        c = k999/thickness
        
        w_speed = speed /(1+torch.exp(-c*(height - center)))
        w_speed = torch.unsqueeze(w_speed,dim=-1)
        wind = torch.cat((w_speed, torch.zeros_like(w_speed, device=self.device), torch.zeros_like(w_speed, device=self.device)), dim=1)
        wind = torch.unsqueeze(wind,dim=-1)
        return wind


    def compute_force_moment(self, Y, actions, initial_root_states):
        glider_2_world_quat = quat_mul(Y[:,3:7], quat_conjugate(initial_root_states[:,3:7]))
        world_lin_vel = Y[:,7:10]
        world_ang_vel = Y[:,10:13]
        

        alpha_mod = torch.zeros([self.W,1,self.M], device=self.device)
        alpha_mod[0,0,:] = (actions[:,3])
        alpha_mod[1,0,:] = (-actions[:,3])
        alpha_mod[2,0,:] = -actions[:,1]
        alpha_mod[3,0,:] = -actions[:,1]
        
        #Roll Rate Compensation
        self.plane_lin_vel = quat_rotate_inverse(glider_2_world_quat, world_lin_vel.to('cuda'))
        self.plane_ang_vel = quat_rotate_inverse(glider_2_world_quat, world_ang_vel.to('cuda'))

        roll_rates_plane = self.plane_ang_vel[:,0]

        # Wind_global = torch.cat([0.0*torch.ones([self.M,1,1]), 0.0*torch.ones([self.M,1,1]),  0.0*torch.ones([self.M,1,1])],1)
        Wind_global = self.wind_function(Y[:,2])
        self.W_global = Wind_global
        # V_global = torch.cat([0.0*torch.ones([self.M,1,1]), 0.0*torch.ones([self.M,1,1]),  0.0*torch.ones([self.M,1,1])],1)
        V_global = torch.unsqueeze(Y[:,7:10], dim=-1)
        self.V_global = V_global
        Wind_apparent_global = V_global - Wind_global
        self.WAG = Wind_apparent_global
        Vinf = torch.reshape(torch.sqrt(torch.sum(Wind_apparent_global**2,1)),[1,self.M])

        Vinf_ = torch.unsqueeze(Vinf, dim=0)
        roll_rates_plane = torch.unsqueeze(torch.unsqueeze(roll_rates_plane, dim=0), dim=0)

        r_sign = torch.unsqueeze(torch.sign(self.r_plane[:,:,1]), dim=-1)
        r_norm = r_sign*torch.unsqueeze(torch.norm(self.r_plane[:,:,1:3], dim=2), dim=-1)
        alpha_roll = roll_rates_plane*r_norm/Vinf_


        Wind_apparent_global_normalized = Wind_apparent_global/torch.reshape(Vinf,[self.M,1,1])
       
        Quat = torch.unsqueeze(glider_2_world_quat, dim=1)
        Quat = torch.roll(Quat, 1, 2)
        WingInWorldFrame = quaternion_apply(Quat, self.H_wing)
        self.WIWF = WingInWorldFrame
        # self.WIWF = WingInWorldFrame 
        # self.WAGN = Wind_apparent_global_normalized
              
        Wind_apparent_wing_normalized = torch.matmul(WingInWorldFrame, -Wind_apparent_global_normalized)
        self.WAWN = Wind_apparent_wing_normalized
        alpha0_rad = torch.atan2(Wind_apparent_wing_normalized[:,2::3],Wind_apparent_wing_normalized[:,0::3])
        
        alpha = torch.permute(alpha0_rad,(1,2,0))
        self.ground_speed = torch.norm(Y[:, 7:10], dim=-1)
        self.ground_speed = torch.unsqueeze(self.ground_speed, dim=-1)
        self.V_inf = Vinf
        self.alpha = alpha
        self.alpha_log.write([self.alpha[:,0,0].to('cpu').numpy()])
        self.vec2 = alpha + alpha_mod - alpha_roll*1.0



        RHS = self.vec1*self.vec2
        self.RHS = RHS
        A = torch.linalg.solve(self.mat1,RHS)      
        # A = torch.matmul(mat1inv,RHS)  
        # each col in above will have the "A" coeffs for the mth wing 
        Gamma = torch.matmul(self.mat2,A)*Vinf*self.rho
        LiftDist = Gamma*Vinf*self.rho
        # exec_time = time.perf_counter() - now; 
        Alpha_i = torch.matmul(self.mat3, A * self.vec3) 
        DragDist = Vinf*Gamma*Alpha_i*self.rho
        DragDist = DragDist.type(torch.float)

        # WingYWorld = WingInWorldFrame[:, :, 1]
        # WingYWorld = torch.reshape(WingYWorld, (self.M, self.W, 3))
        WingYWorld = WingInWorldFrame[:, 1::3, :]
        WingYWorld = torch.permute(WingYWorld, (0, 2, 1))
        # Wind_apparent_global_normalized

        self.WIWF = WingInWorldFrame
        self.WYW = WingYWorld
        self.WAGN = Wind_apparent_global_normalized

        Direction_Of_Lift = torch.cross(WingYWorld, Wind_apparent_global_normalized, dim=1)

        # F_global = Direction_Of_Lift*LiftDist + Wind_apparent_global_normalized*DragDist
        LD1 = torch.permute(LiftDist, (2,0,1))*self.force_scale

        LiftDist_ = torch.reshape(LD1, (self.W*self.M,self.N,1))

        DD1 = torch.permute(DragDist, (2,0,1))*self.force_scale
        DragDist_ = torch.reshape(DD1, (self.W*self.M,self.N,1))

        DOL1 = torch.permute(Direction_Of_Lift, (0,2,1))
        Direction_Of_Lift_ = torch.reshape(DOL1, (self.W*self.M, 1, 3))
        # Direction_Of_Lift_ = torch.permute(Direction_Of_Lift_, (2,0,1))
        self.DOL = Direction_Of_Lift

        Wind_apparent_global_normalized_ = Wind_apparent_global_normalized.repeat(1,1,4)
        Wind_apparent_global_normalized_ = torch.permute(Wind_apparent_global_normalized_, (1,0,2))
        Wind_apparent_global_normalized_ = torch.reshape(Wind_apparent_global_normalized_, (1, 3, self.W*self.M))
        Wind_apparent_global_normalized_ = torch.permute(Wind_apparent_global_normalized_, (2,0,1))

        self.WAGN = Wind_apparent_global_normalized

        self.Lift_global = torch.matmul(LiftDist_, Direction_Of_Lift_)
        self.Drag_global = torch.matmul(DragDist_, -Wind_apparent_global_normalized_)
        self.F_global = self.Lift_global + self.Drag_global*1.0
        F_sum1 = torch.sum(self.F_global, dim=1)
        self.F_sum1 = F_sum1
        F_sum2 = torch.reshape(F_sum1, (self.M,self.W,3))
        self.Force_sum = torch.sum(F_sum2, dim=1)
        
        r_plane_ = torch.reshape(self.r_plane, (1,self.N*self.W,3))
        r_global_ = quaternion_apply(Quat, r_plane_)
        self.WIWF = WingInWorldFrame
        self.r_global = torch.reshape(r_global_, (self.M,self.W,self.N,3))

        # F_ = torch.reshape(self.F_global, (2,4,20,3))
        r_global_reshaped = torch.reshape(self.r_global, [self.M*self.W*self.N, 3])
        F_reshaped = torch.reshape(self.F_global, [self.M*self.W*self.N, 3])
        Torque_reshaped = torch.cross(r_global_reshaped, F_reshaped)
        Torque = torch.reshape(Torque_reshaped, [self.M, self.W, self.N, 3])

        Torque_sum = torch.sum(Torque, dim=1)
        self.Torque_sum = torch.sum(Torque_sum, dim=1)

        return self.Force_sum, self.Torque_sum

    def update(self, root_states, actions, initial_root_states, debug_flags):
        self.root_states = root_states

        t = 0.0 #We aren't really using t for anything right now anyways
        RK4_update = self.rk4(t, root_states, actions, initial_root_states)
        if(debug_flags['pos_xy_hold']):
            RK4_update[:, 0:2] = 0.0
        if(debug_flags['pos_z_hold']):
            RK4_update[:, 2] = 0.0

        root_states[:,0:3] = RK4_update[:, 0:3]         #Position
        root_states[:,3:7] = RK4_update[:, 3:7]         #Quaternion
        root_states[:,7:10] = RK4_update[:, 7:10]       #LinVel
        root_states[:,10:13] = RK4_update[:, 10:13]     #AngVel
        
        if(debug_flags['logging']):
            for m in debug_flags['log_envs']: 
                self.log_writer(m)
            
        return root_states, self.alpha, self.V_inf, self.ground_speed

    def rk4(self, t, root_states, actions, initial_root_states): #States = [Pos, Ori (Quaternion), LinVel, AngVel] 3+4+3+3 = 13
        # Y = torch.cat( (root_states[:,0:3], root_states[:,7:10], root_states[:,10:13]), dim=1) #(x, dx, da) 3+3+3
        Y = root_states.clone()

        forces_, moments_ = self.compute_force_moment( Y, actions, initial_root_states)
        k1 = self.step(t, Y, forces_, moments_)
        
        # Y_ = Y + self.dt * k1/2
        # forces_, moments_ = self.compute_force_moment( Y_, actions, root_quat, initial_root_states)
        # k2 = self.step(t + self.dt/2, Y_, forces_, moments_)

        # Y_ = Y + self.dt * k2/2
        # forces_, moments_ = self.compute_force_moment( Y_, actions, root_quat, initial_root_states)
        # k3 = self.step(t + self.dt/2, Y_, forces_, moments_)

        # Y_ = Y + self.dt * k3
        # forces_, moments_ = self.compute_force_moment( Y_, actions, root_quat, initial_root_states)
        # k4 = self.step(t + self.dt, Y_,   forces_, moments_)
        
        # Y += (1/6)*self.dt*(k1 + 2*k2 + 2*k3 + k4)
        Y += k1*self.dt
        # Make sure we have a unit quaternion
        Y[:,3:7] = quat_unit(Y[:,3:7])
        return Y
        
    def step(self, t, Y, forces, moments):
        
        world_lin_vel = Y[:,7:10]
        world_ang_vel = Y[:,10:13]
        # world_ang_vel[:,2] = 1.0
        world_ang_vel = torch.unsqueeze(world_ang_vel, dim=-1)


        accel = forces / self.mass
        accel[:,2] += -9.81
        
        Quat = torch.unsqueeze(Y[:, 3:7], dim=1)
        Quat = torch.roll(Quat, 1, 2)
        rotated_inertia = quaternion_apply(Quat, torch.unsqueeze(self.inertia,dim=0))
        rotated_inertia = torch.permute(rotated_inertia,(0,2,1))
        rotated_inertia = quaternion_apply(Quat, rotated_inertia)
        rotated_inertia = torch.permute(rotated_inertia,(0,2,1))



        tmp_var = torch.matmul(rotated_inertia,world_ang_vel)
        tmp_var2 = torch.squeeze(torch.cross(world_ang_vel, tmp_var))
        tmp_var3 = moments - tmp_var2
        alpha = torch.linalg.solve(rotated_inertia, tmp_var3) # Rotate moment of inertia tensor    

        omega_mat = torch.zeros((self.M, 4, 4), device=self.device)
        omega_mat[:, 0, 1:4] = -world_ang_vel[:,:,0]
        omega_mat[:, 1:4, 0] = world_ang_vel[:,:,0]
        omega_mat[:, 1, 2:4] = torch.cat( (-world_ang_vel[:,2,:], 
                                        world_ang_vel[:,1,:]), dim=-1)

        omega_mat[:, 2, 1:4] = torch.cat( (world_ang_vel[:,2,:], 
                                        torch.zeros_like(world_ang_vel[:,2,:]), 
                                        -world_ang_vel[:,0,:]), dim=-1)

        omega_mat[:, 3, 1:3] = torch.cat( (-world_ang_vel[:,1,:], 
                                        world_ang_vel[:,0,:]), dim=-1)

        
        Quat = torch.unsqueeze(Y[:, 3:7], dim=-1)
        Quat = torch.roll(Quat, 1, 1)
        dQuat = torch.squeeze(0.5 * torch.matmul(omega_mat, 
                                                 Quat))
        dQuat = torch.roll(dQuat, -1, 1)

        dYdt = torch.cat( (world_lin_vel, dQuat, accel, alpha), dim=1)
        return dYdt

    def rotationMatrixToEulerAngles(self, R):
        sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
        singular = sy < 1e-6

        x_not_singular = torch.atan2(R[:,2,1] , R[:,2,2])
        y_not_singular = torch.atan2(-R[:,2,0], sy)
        z_not_singular = torch.atan2(R[:,1,0], R[:,0,0])
        x_singular = torch.atan2(-R[:,1,2], R[:,1,1])
        y_singular = torch.atan2(-R[:,2,0], sy)
        z_singular = 0
        
        x = torch.unsqueeze(torch.where(singular, x_singular, x_not_singular), dim=-1)
        y = torch.unsqueeze(torch.where(singular, y_singular, y_not_singular), dim=-1)
        z = torch.unsqueeze(torch.where(singular, z_singular, z_not_singular), dim=-1)
        return torch.cat([x, y, z], dim=-1)

    def log_writer(self, env_num):
        # Root State
        en = torch.tensor((env_num,), device=self.device)
        self.state_global_log.write([torch.cat((en, self.root_states[env_num,...])).to('cpu').numpy()])
        #Euler Angles
        base_quat = self.root_states[:, 3:7]
        rot_mat = quaternion_to_matrix(torch.roll(base_quat, 1, 1))
        angles = self.rotationMatrixToEulerAngles(rot_mat)
        ang_scale = 1/(torch.pi)
        roll = torch.unsqueeze(angles[env_num,0], dim=-1) * ang_scale
        pitch = torch.unsqueeze(angles[env_num,1], dim=-1) * ang_scale
        yaw = torch.unsqueeze(angles[env_num,2], dim=-1) * ang_scale
        self.euler_angles_log.write([torch.cat((en, roll, pitch, yaw)).to('cpu').numpy()])
        #Linear Vel Local Frame
        self.lv_loc_log.write([torch.cat((en, self.plane_lin_vel[env_num,...])).to('cpu').numpy()])
        #Angular Vel Local Frame
        self.av_loc_log.write([torch.cat((en, self.plane_ang_vel[env_num,...])).to('cpu').numpy()])
        #Air Speed
        self.as_log.write([torch.cat((en, self.V_inf[...,env_num])).to('cpu').numpy()])
        #Ground Speed
        self.gs_log.write([torch.cat((en, self.ground_speed[env_num,...])).to('cpu').numpy()])
