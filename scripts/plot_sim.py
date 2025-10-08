import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load CSV
df = pd.read_csv("logs/csv/log_20250624_170446.csv")

# Parse list strings into actual lists
df["lin_vel"] = df["lin_vel"].apply(ast.literal_eval)
df["ang_vel"] = df["ang_vel"].apply(lambda x: ast.literal_eval(x)[0])  # unwrap outer list
df["attitude"] = df["attitude"].apply(lambda x: np.array(ast.literal_eval(x)[0]).reshape(3, 3))
df["actions"] = df["actions"].apply(lambda x: ast.literal_eval(x)[0])  # unwrap outer list
df["curr_gate"] = df["curr_gate"].apply(lambda x: ast.literal_eval(x)[0])  # unwrap outer list
df["next_gate"] = df["next_gate"].apply(lambda x: ast.literal_eval(x)[0])  # unwrap outer list

# Convert to numpy arrays
time = df["time"].astype(float).to_numpy()
lin_vel = np.array(df["lin_vel"].to_list())
ang_vel = np.array(df["ang_vel"].to_list())
quaternions = []
for mat in df["attitude"]:
    rot = R.from_matrix(mat)
    q = rot.as_quat()  # [x, y, z, w]
    quaternions.append(q)
quaternions = np.array(quaternions)
actions = np.array(df["actions"].to_list())
curr_gate = np.array(df["curr_gate"].to_list())  # shape (n, 12)
corners = curr_gate.reshape(-1, 4, 3)  # shape (n, 4, 3)
next_gate = np.array(df["next_gate"].to_list())  # shape (n, 12)
next_corners = next_gate.reshape(-1, 4, 3)  # shape (n, 4, 3)

# Find where time resets
reset_indices = np.where(np.diff(time) < 0)[0] + 1

# Compute cumulative time to continue after resets
cumulative_time = time.copy()
for idx in reset_indices:
    cumulative_time[idx:] += cumulative_time[idx - 1]

# Plot linear velocities
plt.figure(figsize=(10, 4))
plt.plot(cumulative_time, lin_vel[:, 0], label="lin_vel_x")
plt.plot(cumulative_time, lin_vel[:, 1], label="lin_vel_y")
plt.plot(cumulative_time, lin_vel[:, 2], label="lin_vel_z")
for idx in reset_indices:
    plt.axvline(cumulative_time[idx], color="red", linestyle="--", label="reset" if idx == reset_indices[0] else "")
plt.xlabel("Time [s]")
plt.ylabel("Linear Velocity [m/s]")
plt.title("Linear Velocities with Reset Markers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Plot angular velocities
plt.figure(figsize=(10, 4))
plt.plot(cumulative_time, ang_vel[:, 0], label="ang_vel_x")
plt.plot(cumulative_time, ang_vel[:, 1], label="ang_vel_y")
plt.plot(cumulative_time, ang_vel[:, 2], label="ang_vel_z")
for idx in reset_indices:
    plt.axvline(cumulative_time[idx], color="red", linestyle="--", label="reset" if idx == reset_indices[0] else "")
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity [rad/s]")
plt.title("Angular Velocities with Reset Markers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Plot actions
plt.figure(figsize=(10, 4))
plt.plot(cumulative_time, actions[:, 0], label="roll cmd")
plt.plot(cumulative_time, actions[:, 1], label="pitch cmd")
plt.plot(cumulative_time, actions[:, 2], label="yaw_rate cmd")
plt.plot(cumulative_time, actions[:, 3], label="thrust cmd")
for idx in reset_indices:
    plt.axvline(cumulative_time[idx], color="red", linestyle="--", label="reset" if idx == reset_indices[0] else "")
plt.xlabel("Time [s]")
plt.ylabel("Action Value")
plt.title("Control Actions with Reset Markers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Plot quaternion components
plt.figure(figsize=(10, 4))
plt.plot(cumulative_time, quaternions[:, 0], label="quat_x")
plt.plot(cumulative_time, quaternions[:, 1], label="quat_y")
plt.plot(cumulative_time, quaternions[:, 2], label="quat_z")
plt.plot(cumulative_time, quaternions[:, 3], label="quat_w")
for idx in reset_indices:
    plt.axvline(cumulative_time[idx], color="red", linestyle="--", label="reset" if idx == reset_indices[0] else "")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion Component")
plt.title("Orientation as Quaternion with Reset Markers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# Plot gate corners
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
labels = ["corner 1 [m]", "corner 2 [m]", "corner 3 [m]", "corner 4 [m]"]
coords = ["x", "y", "z"]

for i in range(4):
    for j in range(3):
        axs[i].plot(cumulative_time, corners[:, i, j], label=f"{coords[j]}")
    for idx in reset_indices:
        axs[i].axvline(cumulative_time[idx], color="red", linestyle="--", label="reset" if idx == reset_indices[0] else "")
    axs[i].set_ylabel(labels[i])
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time [s]")
fig.suptitle("Current Gate Corners (XYZ)")
plt.tight_layout()
plt.show(block=False)

# Plot next gate corners
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
labels = ["corner 1 [m]", "corner 2 [m]", "corner 3 [m]", "corner 4 [m]"]
coords = ["x", "y", "z"]

for i in range(4):
    for j in range(3):
        axs[i].plot(cumulative_time, next_corners[:, i, j], label=f"{coords[j]}")
    for idx in reset_indices:
        axs[i].axvline(cumulative_time[idx], color="red", linestyle="--", label="reset" if idx == reset_indices[0] else "")
    axs[i].set_ylabel(labels[i])
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Time [s]")
fig.suptitle("Next Gate Corners (XYZ)")
plt.tight_layout()
plt.show()