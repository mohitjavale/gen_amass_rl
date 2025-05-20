import argparse
import os
import pickle
import shutil
from pathlib import Path

from h1_env_amp import H1Env
from rsl_rl.runners import OnPolicyRunner
from amp_rsl_rl.runners import AMPOnPolicyRunner


import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "class_name": "AMP_PPO",
            # "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            # "init_noise_std": 1.0,
            # "fixed_std": False,
            "init_noise_std": 0.05,
            "fixed_std": True,
        },
        "runner": {
            # "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            # "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
        "discriminator":{
            "hidden_dims":[512, 256, 128],
            "reward_scale":1,
        },
        "amp_data_path":Path(__file__, f'../amp_data/').resolve(),
        # "dataset_names": ['amass_5'],
        "dataset_names": ['amass_0', 'amass_1', 'amass_2', 'amass_3', 'amass_4', 'amass_5', 'amass_6', 'amass_7', 'amass_8', 'amass_9'],
        "dataset_weights": [1.0],
        "slow_down_factor":1,
        "num_steps_per_env":24,
        "save_interval": 100,
        "empirical_normalization":True,
        "logger":"tensorboard",
        "wandb_project":"gen_amass_rl",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        # sim stuff
        "control_freq": 50, # Hz
        "decimation": 4,

        # times
        "episode_length_s": 20.0,
        # "resampling_time_s": 4.0,

        # modify actions
        "num_actions": 19,
        "action_scale": 0.1,
        "clip_actions": 100.0,

        # robot init
        # "urdf_path": str(Path(__file__, "../../robots/h1/urdf/h1.urdf").resolve()),
        "urdf_path": str(Path(__file__, "../../robots/h1/urdf/h1.urdf").resolve()),
        "base_init_pos": [0.0, 0.0, 1.1],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "default_joint_angles": {  # = target angles [rad] when action = 0.0
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.4,
            "left_knee_joint": 0.8,
            "left_ankle_joint": -0.4,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0,
            "right_hip_pitch_joint": -0.4,
            "right_knee_joint": 0.8,
            "right_ankle_joint": -0.4,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
        "dof_names": [
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ],
        "kp": {
            "hip_yaw": 200.0,
            "hip_roll": 200.0,
            "hip_pitch": 200.0,
            "knee": 300.0,
            "ankle": 40.0,
            "torso": 300.0,
            "shoulder": 100.0,
            "elbow": 100.0,
        },
        "kd": {
            "hip_yaw": 5.0,
            "hip_roll": 5.0,
            "hip_pitch": 5.0,
            "knee": 6.0,
            "ankle": 2.0,
            "torso": 6.0,
            "shoulder": 2.0,
            "elbow": 2.0,
        },
        # robot controller 
        "use_sim_PD_controller": False,
        "simulate_action_latency": False, # 1 step action latency (ie use last_actions)

        # termination
        # "termination_if_roll_greater_than": 45,  # degree
        # "termination_if_pitch_greater_than": 45,
        "termination_track_rb_threshold": 0.25,

        # domain_randomization
        "randomize_friction": True,
        "randomize_base_mass": True,
        # "randomize_com_displacement": True,
        "continuous_push": False,
        # "randomize_motor_strength": True,
        # "randomize_motor_offset": True,
        "randomize_kp_scale": True,
        "randomize_kd_scale": True,

        "friction_range":[0.3,0.8],
        "added_mass_range":[-1,+1],
        "continuous_push_force_range":[-1,1],
        "kp_scale_range":[0.8, 1.2],
        "kd_scale_range":[0.8, 1.2],

        # amass
        "xml_path": str(Path(__file__, '../../robots/h1/xml/h1.xml').resolve()),
        "motion_data_path": str(Path(__file__, '../../data/amass_phc_filtered.pkl').resolve()),
        "num_motions": 10, # max 8277

    }
    obs_cfg = {
        "num_obs": 3+3+3+19+19+60+80+60+60 +60+80+60+60+19+19+60+80 +19,
        "obs_history_length":1,
        "num_privileged_obs": 3+3+3+19+19+60+80+60+60 +60+80+60+60+19+19+60+80 +19,
        "privileged_obs_history_length":1,
    
        # "num_privileged_obs": None,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": {
            "track_rb_pos": 0.25,
            "track_rb_quat": 0.25,
            "track_rb_lin_vel": 0.25,
            "track_rb_ang_vel": 0.25,
            "track_dof_pos": 0.25,
            "track_dof_vel": 0.25,
            
        },
        "reward_scales": {
            # regularization
            "action_rate": -1e-2,
            "dof_acc": -1e-5, 
            "dof_vel": -1e-2,
            # tracking
            "track_rb_pos": 60,
            "track_rb_quat": 30,
            "track_rb_lin_vel": 30,
            "track_rb_ang_vel": 30,
            "track_dof_pos": 60,
            "track_dof_vel": 60,
            # termination
            "termination": -100.0,
            
        },
    }
    command_cfg = {
        # "num_commands": 60+19+19,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-r", "--resume_path", type=str, default=None)
    args = parser.parse_args()

    # init genesis
    gs.init(logging_level="warning")

    # load configs
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # delete/make direction if exists/doesn't exist
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # load env
    env = H1Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    # load runner
    if train_cfg['algorithm']['class_name'] == 'AMP_PPO':
        runner = AMPOnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    elif train_cfg['algorithm']['class_name'] == 'PPO':
        runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    if args.resume_path != None:
        runner.load(args.resume_path)

    # save configs
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # train
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()


