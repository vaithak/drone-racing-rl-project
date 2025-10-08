# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import pandas as pd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi, euler_xyz_from_quat

from matplotlib import pyplot as plt
from collections import deque

from typing import List

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.
        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 100.0             # episode_length = episode_length_s / dt / decimation
    action_space = 4
    observation_space = (
         3 +     # linear velocity
         9 +     # attitude matrix
        12 +     # relative desired position vertices waypoint 1
        12 +     # relative desired position vertices waypoint 2
         2       # conditioning
    )
    state_space = 0
    debug_vis = True

    sim_rate_hz = 100
    policy_rate_hz = 50
    pd_loop_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    moment_scale = 0.01

    # motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    # CTBR Parameters
    kp_omega = 120.0      # default taken from RotorPy, needs to be checked on hardware
    kd_omega = 0.75        # default taken from RotorPy, needs to be checked on hardware

    k_aero_xy = 9.1785e-7
    k_aero_z = 10.311e-7

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        self._all_target_models_paths: List[List[str]] = []
        self._models_paths_initialized: bool = False
        self.target_models_prim_base_name: str | None = None
        
        super().__init__(cfg, render_mode, **kwargs)

        self.iteration = 0

        # Initialize tensors
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)

        # Motor dynamics
        self.cfg.thrust_to_weight = 1.8
        r2o2 = np.sqrt(2.0) / 2.0
        self._rotor_positions = torch.cat(
            [
                self.cfg.arm_length * torch.FloatTensor([[ r2o2,  r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[ r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[-r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[-r2o2,  r2o2, 0]]),
            ],
            dim=0).to(self.device)
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device)
        self.k = self.cfg.k_m / self.cfg.k_eta

        self.f_to_TM = torch.cat(
            [
                torch.tensor([[1, 1, 1, 1]], device=self.device),
                torch.cat(
                    [
                        torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
                    ],
                    dim=1,
                ).to(self.device),
                self.k * self._rotor_directions.view(1, -1),
            ],
            dim=0
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        self.max_len_deque = 100
        self.roll_rate_history = deque(maxlen=self.max_len_deque)
        self.pitch_rate_history = deque(maxlen=self.max_len_deque)
        self.yaw_rate_history = deque(maxlen=self.max_len_deque)

        # self.z_history = deque(maxlen=self.max_len_deque)
        # self.v_history = deque(maxlen=self.max_len_deque)
        self.n_steps = 0
        self.data_fig, self.data_axes = plt.subplots(3, 1, figsize=(10, 10))
        self.roll_rate_lines = [self.data_axes[0].plot([], [], label=f"{legend}")[0] for legend in ["Roll rate sim", "Roll rate RW", "Roll rate des"]]
        self.pitch_rate_lines = [self.data_axes[1].plot([], [], label=f"{legend}")[0] for legend in ["Pitch rate sim", "Pitch rate RW", "Pitch rate des"]]
        self.yaw_rate_lines = [self.data_axes[2].plot([], [], label=f"{legend}")[0] for legend in ["Yaw rate sim", "Yaw rate RW", "Yaw rate des"]]


        # self.z_line, = self.data_axes[2].plot([], [], 'y', label="Z")
        # self.v_line, = self.data_axes[3].plot([], [], 'r', label="v")

        # Configure subplots
        for ax, title in zip(self.data_axes, ["Roll", "Pitch", "Yaw"]):
            # ax.set_title(title)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Ang. vel. (deg/s)")
            ax.legend(loc="upper left")
            ax.grid(True)

        plt.tight_layout()
        plt.ion()  # interactive mode

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)

        self.set_debug_vis(self.cfg.debug_vis)

        self._K_aero = torch.zeros(self.num_envs, 3, device=self.device)
        self._K_aero[:, 0] = self.cfg.k_aero_xy
        self._K_aero[:, 1] = self.cfg.k_aero_xy
        self._K_aero[:, 2] = self.cfg.k_aero_z
        self._kp_omega = self.cfg.kp_omega
        self._kd_omega = self.cfg.kd_omega

        self._thrust_to_weight = self.cfg.thrust_to_weight * torch.ones(self.num_envs, device=self.device)

        df = pd.read_csv("/home/lorenzo/Github/University/ros_quad_sim2real/logs/csv/20250612_test2_0.csv")
        self._data_csv = {col: df[col].to_numpy() for col in df.columns}

        self._time_idx = 0

    def update_iteration(self, iter):
        self.iteration = iter

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_motor_speeds(self, wrench_des):
        f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        motor_speed_squared = f_des / self.cfg.k_eta
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

        return motor_speeds_des

    def _get_moment_from_ctbr(self):
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        self._data_csv['roll_rate'][self._time_idx] = self._data_csv['roll_rate'][self._time_idx]
        self._data_csv['pitch_rate'][self._time_idx] = self._data_csv['pitch_rate'][self._time_idx]
        self._data_csv['yaw_rate'][self._time_idx] = self._data_csv['yaw_rate'][self._time_idx]
        omega_des[:, 0] = self._data_csv['roll_rate'][self._time_idx] * np.pi / 180.0
        omega_des[:, 1] = self._data_csv['pitch_rate'][self._time_idx] * np.pi / 180.0
        omega_des[:, 2] = self._data_csv['yaw_rate'][self._time_idx] * np.pi / 180.0

        omega_err = omega_des - self._robot.data.root_ang_vel_b         # FIXME
        self._previous_omega_err = torch.where(torch.abs(self._previous_omega_err) < 0.0001, omega_err, self._previous_omega_err)
        omega_dot_err = (omega_err - self._previous_omega_err) * self.cfg.pd_loop_rate_hz
        self._previous_omega_err = omega_err.clone()
        omega_dot = self._kp_omega * omega_err + self._kd_omega * (-omega_dot_err)

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    ##########################################################
    ### Functions called in direct_rl_env.py in this order ###
    ##########################################################

    def _pre_physics_step(self, _):
        self._wrench_des[:, 0] = self._data_csv['thrust_newton'][self._time_idx]
        print(self._time_idx)
        self.pd_loop_counter = 0

    def _apply_action(self):
        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        if episode_time > self._data_csv['time'][self._time_idx + 1]:
            self._time_idx += 1

        if self.pd_loop_counter % self.cfg.pd_loop_decimation == 0:
            self._wrench_des[:, 1:] = self._get_moment_from_ctbr()
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        self.pd_loop_counter += 1

        motor_accel = (self._motor_speeds_des - self._motor_speeds) / self.cfg.tau_m
        self._motor_speeds += motor_accel * self.physics_dt

        self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
        motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
        wrench = torch.matmul(motor_forces, self.f_to_TM.t())

        # Compute drag
        lin_vel_b = self._robot.data.root_com_lin_vel_b

        theta_dot = torch.sum(self._motor_speeds, dim=1, keepdim=True)
        drag = -theta_dot * self._K_aero.unsqueeze(0) * lin_vel_b

        self._thrust[:, 0, :] = drag
        self._thrust[:, 0, 2] += wrench[:, 0]
        self._moment[:, 0, :] = wrench[:, 1:]

        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cond_max_h = self._robot.data.root_link_pos_w[:, 2] > 5.0
        died = cond_max_h

        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        time_out = episode_time >= self._data_csv['time'][-1]

        if died or time_out:
            input()

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._motor_speeds[env_ids] = 0.0
        self._previous_omega_err[env_ids] = 0.0

        # Reset joints state
        joint_pos = self._robot.data.default_joint_pos[env_ids]     # not important
        joint_vel = self._robot.data.default_joint_vel[env_ids]     #
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self._robot.data.default_root_state[env_ids]   # [pos, quat, lin_vel, ang_vel] in local environment frame. Shape is (num_instances, 13)

        # Reset robots state
        default_root_state[:, 0] = 0.0
        default_root_state[:, 1] = 0.0
        default_root_state[:, 2] = 0.0
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )
        default_root_state[:, 3:7] = quat

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._time_idx = 0

    def _get_observations(self) -> dict:
        observations = {"policy": torch.zeros(self.num_envs, self.cfg.observation_space)}

        ang_vel = self._robot.data.root_ang_vel_b
        print(ang_vel)
        roll_rate_sim = ang_vel[0, 0].cpu().numpy() * 180.0 / np.pi
        pitch_rate_sim = ang_vel[0, 1].cpu().numpy() * 180.0 / np.pi
        yaw_rate_sim = ang_vel[0, 2].cpu().numpy() * 180.0 / np.pi

        roll_rate_bag = self._data_csv['wx'][self._time_idx]
        pitch_rate_bag = self._data_csv['wy'][self._time_idx]
        yaw_rate_bag = self._data_csv['wz'][self._time_idx]

        roll_rate_des = self._data_csv['roll_rate'][self._time_idx]
        pitch_rate_des = self._data_csv['pitch_rate'][self._time_idx]
        yaw_rate_des = self._data_csv['yaw_rate'][self._time_idx]

        self.roll_rate_history.append(np.array([roll_rate_sim, roll_rate_bag, roll_rate_des]))
        self.pitch_rate_history.append(np.array([pitch_rate_sim, pitch_rate_bag, pitch_rate_des]))
        self.yaw_rate_history.append(np.array([yaw_rate_sim, yaw_rate_bag, yaw_rate_des]))

        self.n_steps += 1
        if self.n_steps >= self.max_len_deque:
            steps = np.arange(self.n_steps - self.max_len_deque, self.n_steps)
        else:
            steps = np.arange(self.n_steps)

        for i in range(3):
            self.roll_rate_lines[i].set_data(steps, np.array(self.roll_rate_history)[:, i])
            self.pitch_rate_lines[i].set_data(steps, np.array(self.pitch_rate_history)[:, i])
            self.yaw_rate_lines[i].set_data(steps, np.array(self.yaw_rate_history)[:, i])

        for ax in self.data_axes:
            ax.relim()
            ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)

        return observations
