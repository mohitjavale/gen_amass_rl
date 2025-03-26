import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, inv_transform_by_trans_quat

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_h1 import MotionLibH1
import numpy as np


def gs_rand_float(lower, upper, shape, device):
	return (upper - lower) * torch.rand(size=shape, device=device) + lower


class H1Env:
	def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
		
		self.device = torch.device(device)

		self.num_envs = num_envs
		self.num_obs = obs_cfg["num_obs"]
		# self.num_privileged_obs = obs_cfg["num_privileged_obs"]
		self.num_privileged_obs = None
		self.num_actions = env_cfg["num_actions"]
		# self.num_commands = command_cfg["num_commands"]
		# self.num_privileged_commands = command_cfg["num_privileged_commands"]

		self.simulate_action_latency = env_cfg["simulate_action_latency"]  # there is a 1 step latency on real robot
		self.dt = 1 / env_cfg["control_freq"]  # control frequency on real robot is 50hz
		self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
		if env_cfg['use_sim_PD_controller']==True:
			self.sim_dt = self.dt
			self.sim_substeps = env_cfg["decimation"]
		else:
			self.sim_dt = self.dt / env_cfg["decimation"]
			self.sim_substeps = 1

		self.env_cfg = env_cfg
		self.obs_cfg = obs_cfg
		self.reward_cfg = reward_cfg
		self.command_cfg = command_cfg

		self.obs_scales = obs_cfg["obs_scales"]

		# create scene
		self.scene = gs.Scene(
			sim_options=gs.options.SimOptions(dt=self.sim_dt, substeps=self.sim_substeps),
			viewer_options=gs.options.ViewerOptions(
				max_FPS=60,
				camera_pos=(2.0, 0.0, 2.5),
				camera_lookat=(0.0, 0.0, 0.5),
				camera_fov=40,
			),
			vis_options=gs.options.VisOptions(n_rendered_envs=1),
			rigid_options=gs.options.RigidOptions(
				dt=self.dt,
				constraint_solver=gs.constraint_solver.Newton,
				enable_collision=True,
				enable_joint_limit=True,
			),
			show_viewer=show_viewer,
		)

		# add plain
		self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

		# add robot
		self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
		self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
		self.inv_base_init_quat = inv_quat(self.base_init_quat)
		self.robot = self.scene.add_entity(
			gs.morphs.URDF(
				file=self.env_cfg['urdf_path'],
				pos=self.base_init_pos.cpu().numpy(),
				quat=self.base_init_quat.cpu().numpy(),
			),
		)

		# add camera
		self.cam = self.scene.add_camera(
			pos=np.array((2.0, 0.0, 2.5)),
			lookat=np.array([0, 0, 0.5]),
			# res=(720, 480),
			fov=40,
			GUI=True,
		)


		# build
		self.scene.build(n_envs=num_envs)

		# names to indices
		self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

		# PD control parameters
		kp = self.env_cfg['kp']
		kd = self.env_cfg['kd']
		self.p_gains, self.d_gains = [], []
		for dof_name in self.env_cfg["dof_names"]:
			for key in kp.keys():
				if key in dof_name:
					self.p_gains.append(kp[key])
					self.d_gains.append(kd[key])
		self.p_gains = torch.tensor(self.p_gains, device=self.device)
		self.d_gains = torch.tensor(self.d_gains, device=self.device)
		self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1) # for implemented PD controller
		self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1) # for implemented PD controller
		self.robot.set_dofs_kp(self.p_gains, self.motor_dofs) # for implicit controller
		self.robot.set_dofs_kv(self.d_gains, self.motor_dofs) # for implicit controller

		# prepare reward functions and multiply reward scales by dt
		self.reward_scales = self.reward_cfg["reward_scales"]
		self.reward_functions, self.episode_sums = dict(), dict()
		for name in self.reward_scales.keys():
			self.reward_scales[name] *= self.dt
			self.reward_functions[name] = getattr(self, "_reward_" + name)
			self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)


		# initialize buffers
		self.init_buffers()

	def init_buffers(self):
		# self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
		# if self.num_privileged_commands != None:
		# 	self.privileged_commands = torch.zeros((self.num_envs, self.num_privileged_commands), device=self.device, dtype=gs.tc_float)
		# self.commands_scale = torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=self.device, dtype=gs.tc_float)
		self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
		self.last_actions = torch.zeros_like(self.actions)

		# robot state
		self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
		self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.dof_pos = torch.zeros_like(self.actions)
		self.dof_vel = torch.zeros_like(self.actions)
		self.last_dof_vel = torch.zeros_like(self.actions)
		self.default_dof_pos = torch.tensor([self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]], device=self.device, dtype=gs.tc_float)
		self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)
		self.rb_pos = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device)
		self.rb_quat = torch.zeros((self.num_envs, self.robot.n_links, 4), device=self.device)
		self.rb_lin_vel = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device)
		self.rb_ang_vel = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device)

		self.extras = dict()  # extra information for logging

		self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
		if self.num_privileged_obs != None:
			self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
		self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
		self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
		self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

		# init moCap stuff
		sk_tree = SkeletonTree.from_mjcf(self.env_cfg['xml_path'])
		self.motion_lib = MotionLibH1(motion_file=self.env_cfg['motion_data_path'], device=self.device,masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=self.env_cfg['xml_path'])
		self.num_motions = self.env_cfg['num_motions'] # 8277 motions
		self.motion_lib.load_motions(skeleton_trees=[sk_tree] * self.num_motions, gender_betas=[torch.zeros(17)] * self.num_motions, limb_weights=[np.zeros(10)] * self.num_motions, random_sample=False)
		# motion_keys = self.motion_lib.curr_motion_keys
		self.motion_ids = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
		self.motion_lengths = torch.zeros((self.num_envs,), device=self.device)
		self.motion_times = torch.zeros((self.num_envs,), device=self.device)

		self.motion_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.motion_base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
		self.motion_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.motion_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
		self.motion_dof_pos = torch.zeros_like(self.actions)
		self.motion_dof_vel = torch.zeros_like(self.actions)
		self.motion_rb_pos = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device)
		self.motion_rb_quat = torch.zeros((self.num_envs, self.robot.n_links, 4), device=self.device)
		self.motion_rb_lin_vel = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device)
		self.motion_rb_ang_vel = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device)



	def resample_commands(self, envs_idx):
		self.motion_ids[envs_idx] = torch.ones_like(self.motion_ids[envs_idx], device=self.device)*5
		# self.motion_ids[envs_idx] = torch.randint_like(self.motion_ids[envs_idx], low=0, high=self.num_motions, device=self.device)
		self.motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
		# self.motion_times[envs_idx] = torch.zeros_like(self.motion_times[envs_idx], device=self.device)
		self.motion_times[envs_idx] = torch.rand_like(self.motion_times[envs_idx], device=self.device) * self.motion_lengths[envs_idx]
		self.motion_res = self.motion_lib.get_motion_state(self.motion_ids, self.motion_times)

	def update_buffers(self):
		self.episode_length_buf += 1
		self.base_pos[:] = self.robot.get_pos()
		self.base_quat[:] = self.robot.get_quat()
		self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
		self.inv_base_quat = inv_quat(self.base_quat)
		self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), self.inv_base_quat)
		self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), self.inv_base_quat)
		self.projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_quat)
		self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
		self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
		# self.rb_pos[:] = self.robot.get_links_pos()
		# self.rb_lin_vel[:] = self.robot.get_links_vel() 
		self.rb_pos[:] = inv_transform_by_trans_quat(self.robot.get_links_pos().reshape(-1,3), self.base_pos.repeat_interleave(self.robot.n_links, dim=0), self.base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)
		self.rb_quat[:] = transform_quat_by_quat(self.robot.get_links_quat().reshape(-1,4), self.inv_base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)
		self.rb_lin_vel[:] = transform_by_quat(self.robot.get_links_vel().reshape(-1,3), self.inv_base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)
		self.rb_ang_vel[:] = transform_by_quat(self.robot.get_links_ang().reshape(-1,3), self.inv_base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)

		self.motion_times += self.dt
		self.motion_res = self.motion_lib.get_motion_state(self.motion_ids, self.motion_times)

		self.motion_base_pos = self.motion_res['root_pos']
		self.motion_base_quat = torch.index_select(self.motion_res['root_rot'], 1, torch.tensor([3,0,1,2], device=self.device))
		self.motion_inv_base_quat = inv_quat(self.motion_base_quat)
		self.motion_base_lin_vel[:] = transform_by_quat(self.motion_res['root_vel'], self.inv_base_quat)
		self.motion_base_ang_vel[:] = transform_by_quat(self.motion_res['root_ang_vel'], self.inv_base_quat)
		self.motion_dof_pos[:] = self.motion_res['dof_pos']
		self.motion_dof_vel[:] = self.motion_res['dof_vel']
		self.motion_rb_pos[:] = inv_transform_by_trans_quat(self.motion_res['rg_pos'].reshape(-1,3), self.base_pos.repeat_interleave(self.robot.n_links, dim=0), self.base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)
		self.motion_rb_quat[:] = transform_quat_by_quat(torch.index_select(self.motion_res['rb_rot'], 2, torch.tensor([3,0,1,2], device=self.device)).reshape(-1,4), self.inv_base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)
		self.motion_rb_lin_vel[:] = transform_by_quat(self.motion_res['body_vel'].reshape(-1,3), self.inv_base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)
		self.motion_rb_ang_vel[:] = transform_by_quat(self.motion_res['body_ang_vel'].reshape(-1,3), self.inv_base_quat.repeat_interleave(self.robot.n_links, dim=0)).reshape(self.num_envs, self.robot.n_links, -1)



	def check_termination(self):

		# termination if max episode length reached
		# self.reset_buf = self.episode_length_buf > self.max_episode_length

		# termination of motion_time exceeded 
		self.reset_buf = self.motion_times >= self.motion_lengths
		
		# termination if pitch/roll greater than limits
		# self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
		# self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

		# termination if mocap rb tracking exceeds threshold
		rb_pos_tracking_error = torch.mean(torch.sum(torch.square(self.rb_pos - self.motion_rb_pos), dim=2), dim=1)
		self.reset_buf |= rb_pos_tracking_error > self.env_cfg["termination_track_rb_threshold"]
		
		# time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
		# self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
		# self.extras["time_outs"][time_out_idx] = 1.0

	def compute_reward(self):
		self.rew_buf[:] = 0.0
		for name, reward_func in self.reward_functions.items():
			rew = reward_func() * self.reward_scales[name]
			self.rew_buf += rew
			self.episode_sums[name] += rew

	def compute_observations(self):

		self.obs_buf = torch.cat(
			[
				self.projected_gravity,                                                    # 3
				self.base_ang_vel * self.obs_scales["ang_vel"],                            # 3
				(self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],        # 19
				self.dof_vel * self.obs_scales["dof_vel"],                                 # 19

				self.motion_rb_pos.reshape(self.num_envs, -1),							   # 60
				self.motion_rb_quat.reshape(self.num_envs, -1),							   # 80
				self.motion_rb_lin_vel.reshape(self.num_envs, -1),						   # 60
				self.motion_rb_ang_vel.reshape(self.num_envs, -1),						   # 60
				self.motion_dof_pos.reshape(self.num_envs, -1),							   # 19
				self.motion_dof_vel.reshape(self.num_envs, -1),							   # 19

				self.actions,                                                              # 19
			],
			axis=-1,
		)
		self.last_actions[:] = self.actions[:]
		self.last_dof_vel[:] = self.dof_vel[:]

		if self.num_privileged_obs != None:
			self.privileged_obs_buf = torch.cat(
			[	
				self.projected_gravity,                                                    # 3
				self.base_lin_vel * self.obs_scales['lin_vel'],							   # 3
				self.base_ang_vel * self.obs_scales["ang_vel"],                            # 3
				(self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],        # 19
				self.dof_vel * self.obs_scales["dof_vel"],                                 # 19

				self.motion_rb_pos.reshape(self.num_envs, -1),							   # 60
				self.motion_rb_quat.reshape(self.num_envs, -1),							   # 80
				self.motion_rb_lin_vel.reshape(self.num_envs, -1),						   # 60
				self.motion_rb_ang_vel.reshape(self.num_envs, -1),						   # 60
				self.motion_dof_pos.reshape(self.num_envs, -1),							   # 19
				self.motion_dof_vel.reshape(self.num_envs, -1),							   # 19

				self.actions,                                                              # 19
			],
			axis=-1,
		)
	def compute_torques(self, actions):
		actions_scaled = actions * self.env_cfg["action_scale"]
		torques = self.batched_p_gains*((actions_scaled + self.default_dof_pos) - self.dof_pos) - self.batched_d_gains * self.dof_vel
		return torques

	def step(self, actions):

		# cam render
		# self.cam.render()

		# motion key points render
		# self.scene.clear_debug_objects()
		# self.scene.draw_debug_spheres(self.motion_res['rg_pos'][0], radius=0.05)
		# self.scene.draw_debug_spheres(self.motion_rb_pos[0], radius=0.05, color=(0.0, 1.0, 0.0, 0.5))
		
				

		# modify actions
		self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
		exec_actions = self.last_actions if self.simulate_action_latency else self.actions

		# simulation step
		if self.env_cfg["use_sim_PD_controller"]:
			target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
			self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
			self.scene.step()
		else:
			for _ in range(self.env_cfg["decimation"]):
				self.torques = self.compute_torques(exec_actions)
				self.robot.control_dofs_force(self.torques, self.motor_dofs)
				self.scene.step()
				self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
				self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
				

		# update buffers
		self.update_buffers()        

		# resample commands
		# envs_idx = (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten()
		# self.resample_commands(envs_idx)

		# check termination and reset
		self.check_termination()
		self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

		# compute reward
		self.compute_reward()

		# compute observations
		self.compute_observations()

		return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

	def get_observations(self):
		return self.obs_buf

	def get_privileged_observations(self):
		if self.num_privileged_obs != None:
			return self.privileged_obs_buf
		else:
			return None

	def reset_idx(self, envs_idx):
		if len(envs_idx) == 0:
			return

		# reset dofs
		# self.dof_pos[envs_idx] = self.default_dof_pos
		# self.dof_vel[envs_idx] = 0.0
		# self.robot.set_dofs_position(position=self.dof_pos[envs_idx], dofs_idx_local=self.motor_dofs, zero_velocity=True, envs_idx=envs_idx)
		# self.robot.zero_all_dofs_velocity(envs_idx)

		# # reset base
		# self.base_pos[envs_idx] = self.base_init_pos
		# self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
		# self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
		# self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
		# self.base_lin_vel[envs_idx] = 0
		# self.base_ang_vel[envs_idx] = 0
		# self.robot.zero_all_dofs_velocity(envs_idx)

		# reset buffers
		self.last_actions[envs_idx] = 0.0
		self.last_dof_vel[envs_idx] = 0.0
		self.episode_length_buf[envs_idx] = 0
		self.reset_buf[envs_idx] = True

		# fill extras
		self.extras["episode"] = {}
		for key in self.episode_sums.keys():
			self.extras["episode"]["rew_" + key] = (
				torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
			)
			self.episode_sums[key][envs_idx] = 0.0

		self.resample_commands(envs_idx)

		# intialize to motion data poses
		self.robot.set_pos(self.motion_res['root_pos'][envs_idx], zero_velocity=True, envs_idx=envs_idx)
		self.robot.set_quat(torch.index_select(self.motion_res['root_rot'], 1, torch.tensor([3,0,1,2], device=self.device))[envs_idx], zero_velocity=True, envs_idx=envs_idx)
		self.base_lin_vel[envs_idx] = 0
		self.base_ang_vel[envs_idx] = 0
		self.robot.set_dofs_position(position=self.motion_res['dof_pos'][envs_idx], dofs_idx_local=self.motor_dofs, zero_velocity=False, envs_idx=envs_idx)
		self.robot.set_dofs_velocity(velocity=self.motion_res['dof_vel'][envs_idx], dofs_idx_local=self.motor_dofs,envs_idx=envs_idx)
		# self.robot.zero_all_dofs_velocity(envs_idx)

	def reset_eval_without_resampling(self):       

		envs_idx = torch.arange(self.num_envs, device=self.device)

		self.last_actions[envs_idx] = 0.0
		self.last_dof_vel[envs_idx] = 0.0
		self.episode_length_buf[envs_idx] = 0
		self.reset_buf[envs_idx] = True

		self.motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
		self.motion_times[envs_idx] = torch.zeros_like(self.motion_times[envs_idx], device=self.device)
		self.motion_res = self.motion_lib.get_motion_state(self.motion_ids, self.motion_times)


		# intialize to motion data poses
		self.robot.set_pos(self.motion_res['root_pos'][envs_idx], zero_velocity=True, envs_idx=envs_idx)
		self.robot.set_quat(torch.index_select(self.motion_res['root_rot'], 1, torch.tensor([3,0,1,2], device=self.device))[envs_idx], zero_velocity=True, envs_idx=envs_idx)
		self.base_lin_vel[envs_idx] = 0
		self.base_ang_vel[envs_idx] = 0
		self.robot.set_dofs_position(position=self.motion_res['dof_pos'][envs_idx], dofs_idx_local=self.motor_dofs, zero_velocity=False, envs_idx=envs_idx)
		self.robot.set_dofs_velocity(velocity=self.motion_res['dof_vel'][envs_idx], dofs_idx_local=self.motor_dofs,envs_idx=envs_idx)
		# self.robot.zero_all_dofs_velocity(envs_idx)

	def reset(self):
		self.reset_buf[:] = True
		self.reset_idx(torch.arange(self.num_envs, device=self.device))
		return self.obs_buf, None

	# ------------ reward functions----------------

	def _reward_action_rate(self):
		# Penalize changes in actions
		return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

	def _reward_dof_acc(self):
		# Penalize dof accelerations
		return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

	def _reward_dof_vel(self):
		# Penalize dof velocities
		return torch.sum(torch.square(self.dof_vel), dim=1)
	
	def _reward_track_rb_pos(self):
		rb_pos_tracking_error = torch.sum(torch.sum(torch.square((self.rb_pos - self.motion_rb_pos)), dim=2), dim=1)
		return torch.exp(-rb_pos_tracking_error * self.reward_cfg["tracking_sigma"])

	def _reward_track_rb_quat(self):
		rb_quat_tracking_error = torch.sum(torch.sum(torch.square((self.rb_quat - self.motion_rb_quat)), dim=2), dim=1)
		return torch.exp(-rb_quat_tracking_error * self.reward_cfg["tracking_sigma"])

	def _reward_track_rb_lin_vel(self):
		rb_lin_vel_tracking_error = torch.sum(torch.sum(torch.square((self.rb_lin_vel - self.motion_rb_lin_vel)), dim=2), dim=1)
		return torch.exp(-rb_lin_vel_tracking_error * self.reward_cfg["tracking_sigma"])

	def _reward_track_rb_ang_vel(self):
		rb_ang_vel_tracking_error = torch.sum(torch.sum(torch.square((self.rb_ang_vel - self.motion_rb_ang_vel)), dim=2), dim=1)
		return torch.exp(-rb_ang_vel_tracking_error * self.reward_cfg["tracking_sigma"])

	def _reward_track_dof_pos(self):
		dof_pos_tracking_error = torch.sum(torch.square((self.dof_pos - self.motion_dof_pos)), dim=1)
		return torch.exp(-dof_pos_tracking_error * self.reward_cfg["tracking_sigma"])
	
	def _reward_track_dof_vel(self):
		dof_vel_tracking_error = torch.sum(torch.square((self.dof_vel - self.motion_dof_vel)), dim=1)
		return torch.exp(-dof_vel_tracking_error * self.reward_cfg["tracking_sigma"])

	def _reward_termination(self):
		return self.reset_buf