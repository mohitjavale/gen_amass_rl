from pathlib import Path
import numpy as np
import torch
import genesis as gs

from amp_rsl_rl.utils import AMPLoader

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_h1 import MotionLibH1
from tkinter_nonBlocking_WidgetApp import NonBlockingTkinterApp


# * load motion
# h1_xml = Path(__file__, '../../robots/h1/xml/h1.xml').resolve()
h1_xml = '/home/mohitjavale/gen_amass_rl/robots/h1/xml/h1.xml'

sk_tree = SkeletonTree.from_mjcf(h1_xml)
# motion_file = Path(__file__, '../../data/amass_phc_filtered.pkl').resolve()
motion_file = '/home/mohitjavale/gen_amass_rl/data/amass_phc_filtered.pkl'
# device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))
device = torch.device("cpu")
motion_lib = MotionLibH1(motion_file=motion_file, device=device,masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=h1_xml)
num_motions = 100 # 8277 motions
motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
# motion_keys = motion_lib.curr_motion_keys
motion_id = 9
num_envs = 1

# * genesis stuff
dt = 1/50
gs.init(backend=gs.gpu)
# gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=False,  sim_options=gs.options.SimOptions(dt=dt))
plane = scene.add_entity(gs.morphs.Plane())
h1 = scene.add_entity(gs.morphs.MJCF(file=str(h1_xml)))
scene.build(n_envs=num_envs)
jnt_names = [
    'left_hip_yaw_joint',
    'left_hip_roll_joint',
    'left_hip_pitch_joint',
    'left_knee_joint',
    'left_ankle_joint',
    'right_hip_yaw_joint',
    'right_hip_roll_joint',
    'right_hip_pitch_joint',
    'right_knee_joint',
    'right_ankle_joint',
    'torso_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
]
dofs_idx = [h1.get_joint(name).dof_idx_local for name in jnt_names]



t = 1
amp_dict = {}
amp_dict['joints_list'] = jnt_names
amp_dict['joint_positions'] = []
amp_dict['root_position'] = []
amp_dict['root_quaternion'] = []
amp_dict['fps'] = 50.0
motion_len = motion_lib.get_motion_length(motion_id).item()
while t<=motion_len:

    # * calc motion time
    motion_len = motion_lib.get_motion_length(motion_id).item()

    # * get motion data
    motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(device), torch.tensor([t]).to(device))

    root_pos = motion_res['root_pos']
    root_rot = torch.index_select(motion_res['root_rot'], 1, torch.tensor([3,0,1,2]).to(device))
    root_vel = motion_res['root_vel']
    root_ang_vel = motion_res['root_ang_vel']

    dof_pos = motion_res['dof_pos']
    dof_vel = motion_res['dof_vel']

    rb_reindex_order = torch.tensor([0,1,6,11,2,7,12,16,3,8,13,17,4,9,14,18,5,10,15,19], dtype=torch.int).to(device)
    rb_pos = motion_res['rg_pos']
    rb_pos = torch.index_select(rb_pos, 1, rb_reindex_order)

    rb_rot = torch.index_select(motion_res['rb_rot'], 2, torch.tensor([3,0,1,2]).to(device))
    rb_rot = torch.index_select(rb_rot, 1, rb_reindex_order)

    rb_vel = motion_res['body_vel']
    rb_vel = torch.index_select(rb_vel, 1, rb_reindex_order)

    rb_ang_vel = motion_res['body_ang_vel']
    rb_ang_vel = torch.index_select(rb_ang_vel, 1, rb_reindex_order)


    h1.set_pos(root_pos)
    h1.set_quat(root_rot) # need to reorder quaternions
    h1.set_dofs_position(dof_pos, dofs_idx, zero_velocity=True)

    amp_dict['joint_positions'].append(dof_pos.cpu().numpy().reshape(-1,))
    amp_dict['root_position'].append(root_pos.cpu().numpy().reshape(-1,))
    amp_dict['root_quaternion'].append(root_rot.cpu().numpy().reshape(-1,))


    
    scene.clear_debug_objects()
    
    # draw rb_pos from motion
    # import ipdb; ipdb.set_trace()
    scene.draw_debug_spheres(rb_pos[0], radius=0.05)

    # draw rb_pos from robot
    # scene.draw_debug_spheres(h1.get_links_pos()[0], radius=0.05, color=(0.0, 1.0, 0.0, 0.5))

    # draw motion root_vel
    # scene.draw_debug_arrow(root_pos, root_vel, radius=0.04)

    
    scene.step()
    t += dt

np.save(Path(__file__, f'../amp_data/amass_{motion_id}.npy').resolve(), amp_dict)


dataset_names = [f'amass_{motion_id}']
loader = AMPLoader(
        device="cpu",  # Use CPU for processing (change to "cuda" for GPU)
        dataset_path_root=Path(__file__, f'../amp_data/').resolve(),  # Path to downloaded datasets
        dataset_names=dataset_names,  # Names of the loaded datasets
        dataset_weights=[1.0] * len(dataset_names),  # Equal weights for all motions
        simulation_dt=1 / 50.0,  # Simulation timestep (60Hz)
        slow_down_factor=1,  # Don't slow down the motions
        expected_joint_names=None,  # Use default joint ordering
    )

# import ipdb; ipdb.set_trace()
motion = loader.motion_data[0] # first motion
print("Loaded dataset with", len(motion), "frames.")
sample_obs = motion.get_amp_dataset_obs(torch.tensor([0]))  # Get frame 0
print("Sample AMP observation:", sample_obs)