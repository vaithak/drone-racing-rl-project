# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

    def get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on progress, gate passing, and penalties."""
        # Check to change setpoint
        dist_to_gate = torch.linalg.norm(self.env._pose_drone_wrt_gate, dim=1)
        gate_passed = (dist_to_gate < 1.0) & \
                      (self.env._pose_drone_wrt_gate[:, 0] < 0.0) & \
                      (self.env._prev_x_drone_wrt_gate > 0.0) & \
                      (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < self.env._gate_model_cfg_data.gate_side / 2) & \
                      (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < self.env._gate_model_cfg_data.gate_side / 2)
        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()
        ids_gate_passed = torch.where(gate_passed)[0]

        self.env._n_gates_passed[ids_gate_passed] += 1

        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

        lap_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        lap_completed[ids_gate_passed] = (self.env._n_gates_passed[ids_gate_passed] > self.env._waypoints.shape[0]) & \
                                        ((self.env._n_gates_passed[ids_gate_passed] % self.env._waypoints.shape[0]) == 1)

        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, :2] += self.env._terrain.env_origins[ids_gate_passed, :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        distance_to_goal = torch.linalg.norm(self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        self.env._last_distance_to_goal[ids_gate_passed] = 1.05 * distance_to_goal[ids_gate_passed].clone()

        drone_pos = self.env._robot.data.root_link_pos_w

        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask

        if self.cfg.is_train:
            progress = self.env._last_distance_to_goal - distance_to_goal
            self.env._last_distance_to_goal = distance_to_goal.clone()

            roll_pitch_rates = torch.sum(torch.square(self.env._actions[:, 1:3]), dim=1)
            yaw_rate = torch.square(self.env._actions[:, 3])

            yaw_des = torch.atan2(self.env._desired_pos_w[:, 1] - drone_pos[:, 1],
                                   self.env._desired_pos_w[:, 0] - drone_pos[:, 0])
            delta_cam = (self.env.unwrapped_yaw - yaw_des + torch.pi) % (2 * torch.pi) - torch.pi
            perception = torch.exp(-4 * delta_cam**4)

            attitude_mat = matrix_from_quat(self.env._robot.data.root_quat_w)
            cos_tilt = attitude_mat[:, 2, 2]
            tilt_angle = torch.acos(cos_tilt)
            tilt_arg = self.env.rew["max_tilt_reward_scale"] * (torch.exp(1 * (tilt_angle - self.cfg.max_tilt_thresh) / self.cfg.max_tilt_thresh) - 1.0)
            tilt = -torch.where(tilt_arg > 0, tilt_arg, 0)

            rewards = {
                "gate_passed": gate_passed * self.env.rew['gate_passed_reward_scale'],
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "roll_pitch_rates": roll_pitch_rates * self.env.rew['roll_pitch_rates_reward_scale'] * self.env.step_dt,
                "yaw_rate": yaw_rate * self.env.rew["yaw_rate_reward_scale"] * self.env.step_dt,
                "lap_completed": lap_completed * 100.0 * self.env.rew['lap_completed_reward_scale'],
                "perception": perception * self.env.rew['perception_reward_scale'] * self.env.step_dt,
                "crash": crashed * self.env.rew['crash_reward_scale'],
                "max_tilt": tilt * self.env.step_dt,
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations including waypoint positions and drone state."""
        curr_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        wp_curr_pos = self.env._waypoints[curr_idx, :3]
        wp_next_pos = self.env._waypoints[next_idx, :3]
        quat_curr = self.env._waypoints_quat[curr_idx]
        quat_next = self.env._waypoints_quat[next_idx]

        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        verts_curr = torch.bmm(self.env._local_square, rot_curr.transpose(1, 2)) + wp_curr_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)
        verts_next = torch.bmm(self.env._local_square, rot_next.transpose(1, 2)) + wp_next_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)

        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3)
        )
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3)
        )

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                *( [self.env._robot.data.root_link_pos_w - self.env._terrain.env_origins[:, :3],] if self.cfg.use_wall else [] ),
                self.env._robot.data.root_com_lin_vel_b,
                attitude_mat.view(attitude_mat.shape[0], -1),
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # Update yaw tracking
        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.env._previous_yaw
        self.env._previous_yaw = yaw_w
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.env.unwrapped_yaw = yaw_w + 2 * np.pi * self.env._yaw_n_laps

        self.env._previous_actions = self.env._actions.clone()

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # Choose reset positions
        waypoint_indices = torch.randint(0, self.env._waypoints.shape[0], (n_reset,),
                                        device=self.device, dtype=self.env._idx_wp.dtype)

        # Get random starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = torch.empty(n_reset, device=self.device).uniform_(-2.0, -0.5)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z
        default_root_state[:, :3] += self.env._terrain.env_origins[env_ids]

        initial_yaw = torch.atan2(-initial_y, -initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat

        # Add ground starts after iteration 800
        if self.env.iteration > 800:
            percent_ground = 0.1
            ground_mask = torch.rand(n_reset, device=self.device) < percent_ground
            ground_local_ids = torch.nonzero(ground_mask, as_tuple=False).squeeze(-1)

            if ground_local_ids.numel() > 0:
                x_local = torch.empty(len(ground_local_ids), device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(len(ground_local_ids), device=self.device).uniform_(-1.0, 1.0)

                x0_wp = self.env._waypoints[self.env._initial_wp, 0]
                y0_wp = self.env._waypoints[self.env._initial_wp, 1]
                theta = self.env._waypoints[self.env._initial_wp, -1]

                cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                x_rot = cos_theta * x_local - sin_theta * y_local
                y_rot = sin_theta * x_local + cos_theta * y_local

                x0 = x0_wp - x_rot
                y0 = y0_wp - y_rot
                z0 = torch.full((len(ground_local_ids),), 0.05, device=self.device)

                yaw0 = torch.atan2(-y0, -x0) + torch.empty(len(ground_local_ids), device=self.device).uniform_(-0.15, 0.15)
                quat0 = quat_from_euler_xyz(
                    torch.zeros(len(ground_local_ids), device=self.device),
                    torch.zeros(len(ground_local_ids), device=self.device),
                    yaw0,
                )

                default_root_state[ground_local_ids, 0] = x0 + self.env._terrain.env_origins[env_ids[ground_local_ids], 0]
                default_root_state[ground_local_ids, 1] = y0 + self.env._terrain.env_origins[env_ids[ground_local_ids], 1]
                default_root_state[ground_local_ids, 2] = z0 + self.env._terrain.env_origins[env_ids[ground_local_ids], 2]
                default_root_state[ground_local_ids, 3:7] = quat0

                waypoint_indices[ground_local_ids] = self.env._initial_wp

        # Handle play mode initial position
        if not self.cfg.is_train:
            x0 = None
            y0 = None
            z0 = None
            yaw0 = None

            if x0 == None:
                x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

                x0_wp = self.env._waypoints[self.env._initial_wp, 0]
                y0_wp = self.env._waypoints[self.env._initial_wp, 1]
                theta = self.env._waypoints[self.env._initial_wp, -1]

                cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                x_rot = cos_theta * x_local - sin_theta * y_local
                y_rot = sin_theta * x_local + cos_theta * y_local
                x0 = x0_wp - x_rot
                y0 = y0_wp - y_rot
                z0 = 0.05
                yaw0 = torch.atan2(-y0, -x0) + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
            else:
                x0 = torch.tensor(x0, device=self.device)
                y0 = torch.tensor(y0, device=self.device)
                z0 = torch.tensor(z0, device=self.device)
                yaw0 = torch.tensor(yaw0, device=self.device)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            self.env._n_run += 1
            print(f'Run #{self.env._n_run}: {x0.item()}, {y0.item()}, {yaw0.item()}')

            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, :2] += self.env._terrain.env_origins[env_ids, :2]
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3] + self.env._terrain.env_origins[env_ids, :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        # Randomize parameters
        self.env._K_aero[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(
            self.env._k_aero_xy_min, self.env._k_aero_xy_max
        )
        self.env._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
            self.env._k_aero_z_min, self.env._k_aero_z_max
        )

        kp_omega_rp = torch.empty(n_reset, device=self.device).uniform_(
            self.env._kp_omega_rp_min, self.env._kp_omega_rp_max
        )
        ki_omega_rp = torch.empty(n_reset, device=self.device).uniform_(
            self.env._ki_omega_rp_min, self.env._ki_omega_rp_max
        )
        kd_omega_rp = torch.empty(n_reset, device=self.device).uniform_(
            self.env._kd_omega_rp_min, self.env._kd_omega_rp_max
        )

        kp_omega_y = torch.empty(n_reset, device=self.device).uniform_(
            self.env._kp_omega_y_min, self.env._kp_omega_y_max
        )
        ki_omega_y = torch.empty(n_reset, device=self.device).uniform_(
            self.env._ki_omega_y_min, self.env._ki_omega_y_max
        )
        kd_omega_y = torch.empty(n_reset, device=self.device).uniform_(
            self.env._kd_omega_y_min, self.env._kd_omega_y_max
        )

        self.env._kp_omega[env_ids] = torch.stack([kp_omega_rp, kp_omega_rp, kp_omega_y], dim=1)
        self.env._ki_omega[env_ids] = torch.stack([ki_omega_rp, ki_omega_rp, ki_omega_y], dim=1)
        self.env._kd_omega[env_ids] = torch.stack([kd_omega_rp, kd_omega_rp, kd_omega_y], dim=1)

        tau_m = torch.empty(n_reset, device=self.device).uniform_(self.env._tau_m_min, self.env._tau_m_max)
        self.env._tau_m[env_ids] = tau_m.unsqueeze(1).repeat(1, 4)

        self.env._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(
            self.env._twr_min, self.env._twr_max
        )

        self.env._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)

        self.env._crashed[env_ids] = 0