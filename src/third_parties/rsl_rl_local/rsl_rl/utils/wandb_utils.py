# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        # Try to get entity from environment variables
        # First check for WANDB_ENTITY (team/organization), then WANDB_USERNAME (personal)
        entity = os.environ.get("WANDB_ENTITY", os.environ.get("WANDB_USERNAME", None))

        if entity is None:
            # If no entity is specified, let wandb use the default from login
            print("[INFO] No WANDB_ENTITY or WANDB_USERNAME found. Using default entity from wandb login.")
            entity = None  # wandb.init will use default entity
        else:
            print(f"[INFO] Using wandb entity: {entity}")

        # Initialize wandb run
        try:
            wandb.init(project=project, entity=entity)
            print(f"[INFO] Successfully initialized wandb run in {entity}/{project}")
        except wandb.errors.CommError as e:
            if "permission denied" in str(e).lower():
                print(f"[ERROR] Permission denied for entity '{entity}' in project '{project}'")
                print(f"[INFO] Trying with personal account instead...")
                # Try with no entity (uses default from wandb login)
                wandb.init(project=project)
                print(f"[INFO] Successfully initialized wandb run with default entity")
            else:
                raise

        # Change generated name to project-number format
        wandb.run.name = project + wandb.run.name.split("-")[-1]

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

        # Video logging
        self.saved_video_files = {}

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": asdict(env_cfg)})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def update_video_files(self, log_name: str, fps: int):
        # Check if there are new video files
        log_dir = pathlib.Path(self.log_dir)
        video_files = list(log_dir.rglob("*.mp4"))
        for video_file in video_files:
            file_size_kb = os.stat(str(video_file)).st_size / 1024
            # If it is new file
            if str(video_file) not in self.saved_video_files:
                self.saved_video_files[str(video_file)] = {"size": file_size_kb, "added": False, "count": 0}
            else:
                # Only upload if the file size is not changing anymore to avoid uploading non-ready video.
                video_info = self.saved_video_files[str(video_file)]
                if video_info["added"] is False and video_info["size"] == file_size_kb and file_size_kb > 100:
                    if video_info["count"] > 10:
                        print(f"[Wandb] Uploading {os.path.basename(str(video_file))}.")
                        wandb.log({log_name: wandb.Video(str(video_file), fps=fps)})
                        self.saved_video_files[str(video_file)]["added"] = True
                    else:
                        video_info["count"] += 1
                else:
                    self.saved_video_files[str(video_file)]["size"] = file_size_kb
                    video_info["count"] = 0

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
