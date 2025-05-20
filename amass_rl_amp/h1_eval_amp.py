import argparse
import os
import pickle
from pathlib import Path

import torch
from h1_env_amp import H1Env
from rsl_rl.runners import OnPolicyRunner
from amp_rsl_rl.runners import AMPOnPolicyRunner

from tkinter_nonBlocking_WidgetApp import NonBlockingTkinterApp

import genesis as gs
import pyqtgraph as pg
import numpy as np


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1")
    parser.add_argument("--ckpt", type=int, default=4999)
    args = parser.parse_args()

    # init genesis
    # gs.init(logging_level="warning", backend=gs.cpu)
    gs.init(backend=gs.cuda)

    # load configs
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    train_cfg['policy']['class_name'] = "ActorCritic"
    # train_cfg['algorithm']['class_name'] = 'PPO'
    train_cfg['algorithm']['class_name'] = 'AMP_PPO'
    train_cfg['amp_data_path'] = Path(__file__, f'../amp_data/').resolve()
    # reward_cfg["reward_scales"] = {}
    # reward_cfg["tracking_sigma"] = {          
    #         # tracking
    #         "track_rb_pos": 1/1,
    #         "track_rb_quat": 1/1,
    #         "track_rb_lin_vel": 1/10,
    #         "track_rb_ang_vel": 1/100,
    #         "track_dof_pos": 1,
    #         "track_dof_vel": 1/10,
    #         # termination
    #     }
    # reward_cfg["reward_scales"] = {
    #         # regularization
    #         "action_rate": -1e-2,
    #         "dof_acc": -1e-5, 
    #         "dof_vel": -1e-2,
    #         # tracking
    #         "track_rb_pos": 1,
    #         "track_rb_quat": 1,
    #         "track_rb_lin_vel": 1,
    #         "track_rb_ang_vel": 1,
    #         "track_dof_pos": 1,
    #         "track_dof_vel": 1,
    #         # termination
    #         "termination": -100.0,
    #     }
    
    # babel-local compatibility path stuff
    # use babel paths (even if policy trained on local)
    # env_cfg['urdf_path'] = str(Path(__file__, "../../robots/h1/urdf/h1.urdf").resolve())
    # env_cfg["xml_path"] =  str(Path(__file__, '../../robots/h1/xml/h1.xml').resolve())
    # env_cfg["motion_data_path"] = str(Path(__file__, '../../data/amass_phc_filtered.pkl').resolve())
    # use local (for faster loading)
    env_cfg['urdf_path'] = '/home/mohitjavale/gen_amass_rl/robots/h1/urdf/h1.urdf'
    env_cfg["xml_path"] =  '/home/mohitjavale/gen_amass_rl/robots/h1/xml/h1.xml'
    env_cfg["motion_data_path"] = '/home/mohitjavale/gen_amass_rl/data/amass_phc_filtered.pkl'
    env_cfg["randomize_friction"] = False
    env_cfg["randomize_base_mass"] = False
    env_cfg["randomize_com_displacement"] = False
    env_cfg["continuous_push"] = False
    env_cfg["randomize_motor_strength"] = False
    env_cfg["randomize_motor_offset"] = False
    env_cfg["randomize_kp_scale"] = False
    env_cfg["randomize_kd_scale"] = False
    env_cfg["num_motions"] = 20


    # load env
    env = H1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        # device='cpu'
        device='cuda'
    )

    # load runner
    if train_cfg['algorithm']['class_name'] == 'PPO':
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
    elif train_cfg['algorithm']['class_name'] == 'AMP_PPO':
        runner = AMPOnPolicyRunner(env, train_cfg, log_dir, device=env.device)

    # load policy
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    # policy = runner.get_inference_policy(device="cuda:0")
    # policy = runner.get_inference_policy(device="cpu")
    policy = runner.get_inference_policy(device=env.device)

    # app stuff
    app = NonBlockingTkinterApp()
    app.motion_id = 5
    app.mape_log = []
    def prev_callback(app, env): 
        app.motion_id -= 1
        app.motion_id = app.motion_id % env_cfg['num_motions']
        env.motion_ids[:] = torch.ones_like(env.motion_ids[:], device=env.device)*app.motion_id
        env.reset_eval_without_resampling()
    def next_callback(app, env): 
        app.motion_id += 1
        app.motion_id = app.motion_id % env_cfg['num_motions']
        env.motion_ids[:] = torch.ones_like(env.motion_ids[:], device=env.device)*app.motion_id
        env.reset_eval_without_resampling()

    def reset_callback(app, env): 
        app.motion_id = app.motion_id % env_cfg['num_motions']
        env.motion_ids[:] = torch.ones_like(env.motion_ids[:], device=env.device)*app.motion_id
        env.reset_eval_without_resampling()
        app.mape_log = []

    app.bind_button("Prev", prev_callback, app, env, row=0, column=0)
    app.bind_button("Reset", reset_callback, app, env, row=0, column=1)
    app.bind_button("Next", next_callback, app, env, row=0, column=2)
    app_label = app.bind_label(f'{app.motion_id=}', row=2, column=0)
    

    # sim loop
    obs, _ = env.reset()
    last_actions = env.last_actions
    last_dof_vel = env.last_dof_vel

    # plot
    # plot_app = pg.mkQApp()
    # plot_win = pg.GraphicsLayoutWidget(show=True, title="Multiple Real-time Plots")
    # plot1 = plot_win.addPlot(title="Mean Absolute Keypoint Position Error")
    # plot1.setXRange(0,500)
    # plot1.setYRange(0,0.5)
    # # plot1.setAspectLocked(1.0)
    # curve1 = plot1.plot(pen='r')
    # curve1_buffer = torch.zeros(500)

    logs = {name:0 for name in env.reward_functions.keys()}
    with torch.no_grad():
        while True:
            
            env.motion_ids[:] = torch.ones_like(env.motion_ids[:], device=env.device)*app.motion_id
            app_label.config(text=f'{app.motion_id=}')
            app.update()

            # draw motion spheres
            env.scene.clear_debug_objects()
            motion_rb_pos = env.motion_res['rg_pos'][0].reshape(-1,3).reshape(20,3)
            motion_rb_pos = torch.index_select(motion_rb_pos, 0,env.rb_motion_reindex_order)
            env.scene.draw_debug_spheres(motion_rb_pos, radius=0.06, color=(0.0, 1.0, 0.0, 0.5))

            actions = policy(obs)

            obs, rews, dones, infos = env.step(actions)

            # calc rewards for debugging
            # env.last_actions = last_actions.clone()
            # env.last_dof_vel = last_dof_vel.clone()
            # for name, reward_func in env.reward_functions.items():
            #     rew = reward_func() * env.reward_scales[name]

            # calc mape errors
            rb_pos_tracking_error = torch.mean(torch.abs((env.rb_pos - env.motion_rb_pos)))
            # rb_quat_tracking_error = torch.sum(torch.sum(torch.square((env.rb_quat - env.motion_rb_quat)), dim=2), dim=1)
            # rb_lin_vel_tracking_error = torch.sum(torch.sum(torch.square((env.rb_lin_vel - env.motion_rb_lin_vel)), dim=2), dim=1)
            # rb_ang_vel_tracking_error = torch.sum(torch.sum(torch.square((env.rb_ang_vel - env.motion_rb_ang_vel)), dim=2), dim=1)
            # dof_pos_tracking_error = torch.sum(torch.square((env.dof_pos - env.motion_dof_pos)), dim=1)
            # dof_vel_tracking_error = torch.sum(torch.square((env.dof_vel - env.motion_dof_vel)), dim=1)
            
            # display mape error
            # print(rb_pos_tracking_error)
            app.mape_log.append(rb_pos_tracking_error)
            print(np.array(app.mape_log).mean())
            print('======')

            # plotting
            # curve1_buffer = torch.cat((curve1_buffer[1:], torch.tensor([rb_pos_tracking_error])))
            # curve1.setData(curve1_buffer)
            # plot_app.processEvents()

            # last_actions = env.actions.clone()
            # last_dof_vel = env.dof_vel.clone()

if __name__ == "__main__":
    main()

