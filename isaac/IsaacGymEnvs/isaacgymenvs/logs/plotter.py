import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

log_dir = 'TestName/'
df = pd.read_csv(log_dir + 'alphas.csv')
print(df.head())
raw_alpha = df.to_numpy()

plt.figure();
print(raw_alpha.shape)
plt.plot(raw_alpha[1:,1] * 180.0/np.pi)
plt.grid(True)
plt.ylim([-2.5, 2.5])

# plt.xlim([0,100])

#%%

df = pd.read_csv(log_dir+'state_global.csv')
raw_state_global = df.to_numpy()

df = pd.read_csv(log_dir+'as.csv')
raw_air_speed = df.to_numpy()

df = pd.read_csv(log_dir+'gs.csv')
raw_ground_speed = df.to_numpy()

df = pd.read_csv(log_dir+'rk4_euler_angs.csv')
raw_euler = df.to_numpy()

df = pd.read_csv(log_dir+'rk4_lv_loc.csv')
raw_loc_vel = df.to_numpy()

df = pd.read_csv(log_dir+'rk4_av_loc.csv')
raw_loc_avel = df.to_numpy()

df = pd.read_csv(log_dir+'force.csv')
raw_loc_force = df.to_numpy()


total_gliders = 1
curr_glider = 1

x_raw = raw_state_global[curr_glider-1::total_gliders,1]
y_raw = raw_state_global[curr_glider-1::total_gliders,2]
z_raw = raw_state_global[curr_glider-1::total_gliders,3]
vx_raw = raw_state_global[curr_glider-1::total_gliders,8]
vy_raw = raw_state_global[curr_glider-1::total_gliders,9]
vz_raw = raw_state_global[curr_glider-1::total_gliders,10]
avx_raw = raw_state_global[curr_glider-1::total_gliders,11]
avy_raw = raw_state_global[curr_glider-1::total_gliders,12]
avz_raw = raw_state_global[curr_glider-1::total_gliders,13]
gs_raw = raw_ground_speed[curr_glider-1::total_gliders,1]
as_raw = raw_air_speed[curr_glider-1::total_gliders,1]
force_x_raw = raw_loc_force[curr_glider-1::total_gliders,1]
force_y_raw = raw_loc_force[curr_glider-1::total_gliders,2]
force_z_raw = raw_loc_force[curr_glider-1::total_gliders,3]


#%%
data_start = 1
data_resets = np.where(np.abs(np.diff(raw_state_global[curr_glider-1::total_gliders,2])) > 1)[0]
num_flights = len(data_resets)
print("Number of Flights : {}".format(num_flights))
ax = plt.axes(projection='3d')
# Data for a three-dimensional line


dpi_setting = 80
data_end = len(force_x_raw)
print('Start')
print(data_start)
print('End')
print(data_end)

force_x_data = force_x_raw[data_start:data_end]
force_y_data = force_y_raw[data_start:data_end]
force_z_data = force_z_raw[data_start:data_end]




plt.figure("Force Plot", figsize=(10, 10), dpi=dpi_setting)
ax = plt.axes()
plt.plot(force_x_data)
plt.plot(force_y_data)
plt.plot(force_z_data)
plt.title('Local Force')
ax.set_xlabel('Samples')
ax.set_ylabel('Force (N)')
plt.grid(True)
ax.legend(['X', 'Y', 'Z'])
plt.show()

data_start = data_end + 1

    
#%%
data_start = 1
data_resets = np.where(np.abs(np.diff(raw_state_global[curr_glider-1::total_gliders,2])) > 1)[0]
num_flights = len(data_resets)
print("Number of Flights : {}".format(num_flights))
ax = plt.axes(projection='3d')
# Data for a three-dimensional line


dpi_setting = 80
for i in range(num_flights):
    print(i)
    data_end = data_resets[i]
    print('Start')
    print(data_start)
    print('End')
    print(data_end)
    
    x_data = x_raw[data_start:data_end]
    y_data = y_raw[data_start:data_end]
    z_data = z_raw[data_start:data_end]
    vx_data = vx_raw[data_start:data_end]
    vy_data = vy_raw[data_start:data_end]
    vz_data = vz_raw[data_start:data_end]
    avx_data = avx_raw[data_start:data_end]
    avy_data = avy_raw[data_start:data_end]
    avz_data = avz_raw[data_start:data_end]
    gs_data = gs_raw[data_start:data_end]
    as_data = as_raw[data_start:data_end]
    force_x_data = force_x_raw[data_start:data_end]
    force_y_data = force_y_raw[data_start:data_end]
    force_z_data = force_z_raw[data_start:data_end]
    
    data_pts = len(z_data)
    
    G = 9.81
    E = G*z_data + 0.5*gs_data**2

    
    plt.figure("Position Plot", figsize=(10, 10), dpi=dpi_setting)
    ax = plt.axes(projection='3d')
    ax.view_init(45, 45)
    # ax.set_xlim([-100, 100])
    # ax.set_ylim([-100, 100])
    # ax.set_zlim([0, 20])
    # ax.plot3D(x_data, y_data, z_data)
    ax.scatter(x_data, y_data, z_data, s=1, c = plt.cm.jet(E/max(E))) 
    ax.scatter(x_data[0], y_data[0], z_data[0], s=100, c = 'g') 
    ax.scatter(x_data[-1], y_data[-1], z_data[-1], s=100, c = 'r') 
    plt.title('Aircraft Position')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.show()
    data_start = data_end + 1
    
    # plt.figure("Force Plot", figsize=(10, 10), dpi=dpi_setting)
    # ax = plt.axes()
    # plt.plot(force_x_data)
    # plt.plot(force_y_data)
    # plt.plot(force_z_data)
    # plt.title('Local Force')
    # ax.set_xlabel('Samples')
    # ax.set_ylabel('Force (N)')
    # plt.grid(True)
    # ax.legend(['X', 'Y', 'Z'])
    # plt.show()
    
    # data_start = data_end + 1
    
    
    # plt.figure("Velocity Plot", figsize=(10, 10), dpi=dpi_setting)
    # ax = plt.axes(projection='3d')
    # ax.view_init(45, 45)
    # # ax.plot3D(x_data, y_data, z_data)
    # ax.scatter(vx_data, vy_data, vz_data, s=1, c = plt.cm.jet(E/max(E))) 
    # plt.title('Aircraft Velocity')
    # ax.set_xlabel('X (m/s)')
    # ax.set_ylabel('Y (m/s)')
    # ax.set_zlabel('Z (m/s)')
    # plt.show()
    # data_start = data_end + 1
    
    # plt.figure("Angular Plot", figsize=(10, 10), dpi=dpi_setting)
    # ax = plt.axes(projection='3d')
    # ax.view_init(45, 45)
    # # ax.plot3D(x_data, y_data, z_data)
    # ax.scatter(avx_data, avy_data, avz_data, s=1, c = plt.cm.jet(E/max(E))) 
    # plt.title('Aircraft Angular Velocity')
    # ax.set_xlabel('X (rad/s)')
    # ax.set_ylabel('Y (rad/s)')
    # ax.set_zlabel('Z (rad/s)')
    # plt.show()
    # data_start = data_end + 1
    
    # plt.figure('AirSpeed GroundSpeed', figsize=(10, 10), dpi=dpi_setting)
    # ax = plt.axes()
    # ax.scatter(gs_data, as_data, s=1, c = plt.cm.jet(E/max(E)))
    # plt.grid(True)
    # plt.title('Ground Speed vs. Air Speed')
    # plt.xlabel('ground_speed')
    # plt.ylabel('air_speed')

    # G = 9.81
    # E = G*z_data + 0.5*gs_data**2

    plt.figure('EnergyPlot', figsize=(10, 10), dpi=300)
    ax = plt.axes()
    ax.scatter(np.arange(0,len(E))/100, E, s=1, c = 'b')
    plt.grid(True)
    plt.title('Aircraft Energy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scaled Energy')
    plt.show()
    
    
    
#%% Data Slicing
# print(i)
# data_start=700
# data_end = data_start+11000
# print('Start')
# print(data_start)
# print('End')
# print(data_end)

# x_data = x_raw[data_start:data_end]
# y_data = y_raw[data_start:data_end]
# z_data = z_raw[data_start:data_end]
# vx_data = vx_raw[data_start:data_end]
# vy_data = vy_raw[data_start:data_end]
# vz_data = vz_raw[data_start:data_end]
# avx_data = avx_raw[data_start:data_end]
# avy_data = avy_raw[data_start:data_end]
# avz_data = avz_raw[data_start:data_end]
# gs_data = gs_raw[data_start:data_end]
# as_data = as_raw[data_start:data_end]

# data_pts = len(z_data)

# G = 9.81
# E = G*z_data + 0.5*gs_data**2


# plt.figure("Position Plot", figsize=(10, 10), dpi=dpi_setting)
# ax = plt.axes(projection='3d')
# ax.view_init(0, 135)
# # ax.set_xlim([-100, 100])
# # ax.set_ylim([-100, 100])
# # ax.set_zlim([0, 20])
# # ax.plot3D(x_data, y_data, z_data)
# ax.scatter(x_data, y_data, z_data, s=1, c = plt.cm.jet(E/max(E))) 
# ax.scatter(x_data[0], y_data[0], z_data[0], s=100, c = 'g') 
# ax.scatter(x_data[-1], y_data[-1], z_data[-1], s=100, c = 'r') 
# plt.title('Aircraft Position')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# plt.show()
# data_start = data_end + 1

# # plt.figure("Velocity Plot", figsize=(10, 10), dpi=dpi_setting)
# # ax = plt.axes(projection='3d')
# # ax.view_init(45, 45)
# # # ax.plot3D(x_data, y_data, z_data)
# # ax.scatter(vx_data, vy_data, vz_data, s=1, c = plt.cm.jet(E/max(E))) 
# # plt.title('Aircraft Velocity')
# # ax.set_xlabel('X (m/s)')
# # ax.set_ylabel('Y (m/s)')
# # ax.set_zlabel('Z (m/s)')
# # plt.show()
# # data_start = data_end + 1

# # plt.figure("Angular Plot", figsize=(10, 10), dpi=dpi_setting)
# # ax = plt.axes(projection='3d')
# # ax.view_init(45, 45)
# # # ax.plot3D(x_data, y_data, z_data)
# # ax.scatter(avx_data, avy_data, avz_data, s=1, c = plt.cm.jet(E/max(E))) 
# # plt.title('Aircraft Angular Velocity')
# # ax.set_xlabel('X (rad/s)')
# # ax.set_ylabel('Y (rad/s)')
# # ax.set_zlabel('Z (rad/s)')
# # plt.show()
# # data_start = data_end + 1

# # plt.figure('AirSpeed GroundSpeed', figsize=(10, 10), dpi=dpi_setting)
# # ax = plt.axes()
# # ax.scatter(gs_data, as_data, s=1, c = plt.cm.jet(E/max(E)))
# # plt.grid(True)
# # plt.title('Ground Speed vs. Air Speed')
# # plt.xlabel('ground_speed')
# # plt.ylabel('air_speed')

# # G = 9.81
# # E = G*z_data + 0.5*gs_data**2

# # plt.figure('EnergyPlot', figsize=(10, 10), dpi=300)
# # ax = plt.axes()
# # ax.scatter(np.arange(0,len(E))/100, E, s=1, c = 'b')
# # plt.grid(True)
# # plt.title('Aircraft Energy')
# # ax.set_xlabel('Time (s)')
# # ax.set_ylabel('Scaled Energy')
# # plt.show()




# #%%
# plt.figure()
# for curr_glider in range(1,total_gliders+1):
#     pass
#     # plt.figure()
#     # plt.plot(raw_state_global[curr_glider-1::total_gliderplt.figure('EnergyPlot', figsize=(10, 10), dpi=300)
# ax = plt.axes()
# ax.scatter(np.arange(0,len(E))/100, E, s=1, c = plt.cm.jet(E/max(E)))
# plt.grid(True)
# plt.title('Aircraft Energy')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Scaled Energy')
# plt.show()
# #%%



# #%%

# ax = plt.axes(projection='3d')
# # Data for a three-dimensional line


# data_start = 1
# data_end = 1000
# ax.plot3D(gs_data, as_data, z_data, 'gray')
# ax.view_init(35, -90)


# for i in range(data_pts-1):
#     ax.plot(x_data[i:i+2], y_data[i:i+2], z_data[i:i+2], color=plt.cm.jet(z_data[i]/max(z_data)))

# #%%
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)

# #1 colored by value of `z`
# ax.scatter(x, y, z, c = plt.cm.jet(z/max(z))) 

# #2 colored by index (same in this example since z is a linspace too)

# data_pts = len(z)
# ax.scatter(x, y, z, c = plt.cm.jet(np.linspace(0,1,data_pts)))

# for i in range(data_pts-1):
#     ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.jet(z[i]/max(z)))
# #%%
