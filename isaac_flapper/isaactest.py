from pathlib import Path

import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
sim_params.use_gpu_pipeline = True
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)


asset_root = '/home/nasa01/Documents/UML/willis/rluff/isaac_flapper/Flapper-Env/resources/spm-asm-v6-2'
asset_file = "urdf/spm-asm-v6-2.urdf"

object_asset_options = gymapi.AssetOptions()
object_asset_options.disable_gravity = False
object_asset = gym.load_asset(sim, asset_root, asset_file, object_asset_options)
# Set up the env grid
num_envs = 50
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

num_per_row = 6
envs = []
object_start_pose = gymapi.Transform()
object_start_pose.p = gymapi.Vec3()
object_start_pose.p.x = 0.2
object_start_pose.p.y = 0.2
object_start_pose.p.z = 0.5
object_handles = []

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    object_handle = gym.create_actor(env, object_asset, object_start_pose, "foam", i, 2)
    object_handles.append(object_handle)

objpos = gymapi.Vec3(0., -0., 0.)
bp = gym.get_actor_rigid_body_properties(envs[0], object_handles[0])
object_mass = bp[0].mass
print(object_mass)
force = gymapi.Vec3(0, 0, object_mass * 9.8)
force = gymapi.Vec3(0, 0, 0)
gym.prepare_sim(sim)
while not gym.query_viewer_has_closed(viewer):
    # Every 0.01 seconds the pose of the attactor is updated
    t = gym.get_sim_time(sim)
    for i in range(len(envs)):
        s = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS)
        objpos = gymapi.Vec3(*s['pose']['p'][0])
        # objpos = gymapi.Vec3(0, 0, 0)
        gym.apply_body_force(envs[i], object_handles[i], force, objpos)

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)
    # time.sleep(0.1)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)