import argparse
import os
import pickle

import torch
from h1_env import H1Env
from rsl_rl.runners import OnPolicyRunner

from tkinter_nonBlocking_WidgetApp import NonBlockingTkinterApp

import genesis as gs


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1")
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()

    # init genesis
    gs.init(logging_level="warning", backend=gs.cpu)

    # load configs
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # reward_cfg["reward_scales"] = {}

    # load env
    env = H1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device='cpu'
    )

    # load runner
    # runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")

    # load policy
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    # policy = runner.get_inference_policy(device="cuda:0")
    policy = runner.get_inference_policy(device="cpu")

    # app stuff
    app = NonBlockingTkinterApp()
    app.motion_id = 0
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

    app.bind_button("Prev", prev_callback, app, env, row=0, column=0)
    app.bind_button("Reset", reset_callback, app, env, row=0, column=1)
    app.bind_button("Next", next_callback, app, env, row=0, column=2)
    app_label = app.bind_label(f'{app.motion_id=}', row=2, column=0)
    

    # sim loop
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            
            env.motion_ids[:] = torch.ones_like(env.motion_ids[:], device=env.device)*app.motion_id
            app_label.config(text=f'{app.motion_id=}')
            app.update()


            env.scene.clear_debug_objects()
            # env.scene.draw_debug_spheres(obs[0, 6:6+60].reshape(20,3), radius=0.05)
            env.scene.draw_debug_spheres(env.motion_res['rg_pos'][0].reshape(-1,3).reshape(20,3), radius=0.05)

            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            # for name, reward_func in env.reward_functions.items():
            #     rew = reward_func() * env.reward_scales[name]
            #     print(f'{name}: {rew}')
            # print('======')

if __name__ == "__main__":
    main()

