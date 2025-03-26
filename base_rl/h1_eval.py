import argparse
import os
import pickle

import torch
from h1_env import H1Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1")
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()

    # init genesis
    gs.init(logging_level="warning")

    # load configs
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    # load env
    env = H1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # load runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # load policy
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # sim loop
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

