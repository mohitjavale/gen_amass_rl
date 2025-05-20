from pathlib import Path
import numpy as np
import torch
import genesis as gs

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_h1 import MotionLibH1
from tkinter_nonBlocking_WidgetApp import NonBlockingTkinterApp


# * load motion
# h1_xml = Path(__file__, '../../robots/h1/xml/h1.xml').resolve()
h1_xml = '/home/mohitjavale/gen_amass_rl/robots/h1/xml/h1.xml'

sk_tree = SkeletonTree.from_mjcf(h1_xml)
# motion_file = Path(__file__, '../../data/amass_phc_filtered.pkl').resolve()
motion_file = '/home/mohitjavale/gen_amass_rl/data/amass_phc_filtered.pkl'
device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))
motion_lib = MotionLibH1(motion_file=motion_file, device=device,masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=h1_xml)
num_motions = 100 # 8277 motions
motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
# motion_keys = motion_lib.curr_motion_keys
motion_id = 8 
num_envs = 1

# * genesis stuff
dt = 1/50
gs.init(backend=gs.gpu)
# gs.init(backend=gs.cpu)
# h1_xml = str(Path(__file__, '../../robots/h1/xml/h1.xml').resolve())
h1_xml = '/home/mohitjavale/gen_amass_rl/robots/h1/xml/h1.xml'
scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=dt))
plane = scene.add_entity(gs.morphs.Plane())
h1 = scene.add_entity(gs.morphs.MJCF(file=h1_xml))
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

# * app stuff
app = NonBlockingTkinterApp()
def prev_callback(): 
    global motion_id
    motion_id -= 1
    motion_id = motion_id % num_motions
def next_callback(): 
    global motion_id
    motion_id += 1
    motion_id = motion_id % num_motions
app.bind_button("Prev", prev_callback, row=0, column=0)
app.bind_button("Next", next_callback, row=0, column=1)
app_label = app.bind_label(f'{motion_id=}', row=1, column=0)


t = 0
while True:

    # * calc motion time
    motion_len = motion_lib.get_motion_length(motion_id).item()
    motion_time = t % motion_len

    # * get motion data
    motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).cuda(), torch.tensor([motion_time]).cuda())

    root_pos = motion_res['root_pos']
    root_rot = torch.index_select(motion_res['root_rot'], 1, torch.tensor([3,0,1,2]).cuda())
    root_vel = motion_res['root_vel']
    root_ang_vel = motion_res['root_ang_vel']

    dof_pos = motion_res['dof_pos']
    dof_vel = motion_res['dof_vel']

    rb_reindex_order = torch.tensor([0,1,6,11,2,7,12,16,3,8,13,17,4,9,14,18,5,10,15,19], dtype=torch.int).cuda()
    rb_pos = motion_res['rg_pos']
    rb_pos = torch.index_select(rb_pos, 1, rb_reindex_order)
    rb_rot = torch.index_select(motion_res['rb_rot'], 2, torch.tensor([3,0,1,2]).cuda())
    rb_rot = torch.index_select(rb_rot, 1, rb_reindex_order)
    rb_vel = motion_res['body_vel']
    rb_vel = torch.index_select(rb_vel, 1, rb_reindex_order)
    rb_ang_vel = motion_res['body_ang_vel']
    rb_ang_vel = torch.index_select(rb_ang_vel, 1, rb_reindex_order)

    h1.set_pos(root_pos)
    h1.set_quat(root_rot) # need to reorder quaternions
    h1.set_dofs_position(dof_pos, dofs_idx, zero_velocity=True)
    
    # scene.clear_debug_objects()
    
    # draw rb_pos from motion
    # scene.draw_debug_spheres(rb_pos[0], radius=0.05)

    # draw rb_pos from robot
    # scene.draw_debug_spheres(h1.get_links_pos()[0], radius=0.05, color=(0.0, 1.0, 0.0, 0.5))

    # if t!= 0:
    #     for i in range(20):
    #         scene.draw_debug_spheres(rb_pos[0, i, :], radius=0.05, color=(0.0, 1.0, 0.0, 0.5))
    #         scene.draw_debug_spheres(h1.get_links_pos()[0, i, :], radius=0.05, color=(0.0, 0.0, 1.0, 1))

    # sim_rb_pos = h1.get_links_pos()
    # motion_rb_pos = rb_pos
    # motion_rb_pos = torch.index_select(rb_pos, 1, rb_reindex_order)
    

    # import ipdb; ipdb.set_trace()
    # if t != 0:
    #     for i in range(20):
    #         for j in range(20):
    #             if torch.norm(sim_rb_pos[0,i,:] - motion_rb_pos[0,j,:]) <=0.001:
    #             # if torch.norm(sim_rb_pos[0,j,:] - motion_rb_pos[0,i,:]) <=0.001:
    #                 break
    #         print(f'Sim {i} = {h1.links[i].name} => Motion {j}')
            # print(f'Sim {j} = {h1.links[j].name} => Motion {i}'
        


        


    # draw motion root_vel
    # scene.draw_debug_arrow(root_pos, root_vel, radius=0.04)

    
    scene.step()
    print(device)


    t += dt
    app_label.config(text=f'{motion_id=}')
    app.update()
