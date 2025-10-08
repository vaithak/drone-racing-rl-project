from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from jirl_interfaces.msg import Observations, CommandCTBR
from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib.pyplot as plt

# Percorso alla bag
bag_path = "/home/lorenzo/Github/University/ros_quad_sim2real/logs/20250624_test4"

# Setup reader
reader = SequentialReader()
reader.open(
    StorageOptions(uri=bag_path, storage_id='mcap'),
    ConverterOptions('', '')
)

# Nomi dei topic
topic_obs = "/crazy_jirl_03/observations"
topic_cmd = "/ctbr_cmd"

# Dati
timestamps, lin_vels, ang_vels, quats = [], [], [], []
curr_gates, next_gates, actions = [], [], []

# Lettura messaggi
while reader.has_next():
    topic, data, _ = reader.read_next()
    if topic == topic_obs:
        msg = deserialize_message(data, Observations)
        t = msg.stamp.sec + 1e-9 * msg.stamp.nanosec
        timestamps.append(t)
        lin_vels.append([msg.lin_vel.x, msg.lin_vel.y, msg.lin_vel.z])
        ang_vels.append([msg.ang_vel.x, msg.ang_vel.y, msg.ang_vel.z])
        quats.append([msg.attitude.x, msg.attitude.y, msg.attitude.z, msg.attitude.w])
        curr_gates.append([[p.x, p.y, p.z] for p in msg.curr_gate])
        next_gates.append([[p.x, p.y, p.z] for p in msg.next_gate])
    elif topic == topic_cmd:
        msg = deserialize_message(data, CommandCTBR)
        actions.append([msg.roll, msg.pitch, msg.yaw_rate, msg.thrust])

# Converti in numpy array
timestamps = np.array(timestamps)
lin_vels = np.array(lin_vels)
ang_vels = np.array(ang_vels)
quats = np.array(quats)
curr_gates = np.array(curr_gates)
next_gates = np.array(next_gates)
actions = np.array(actions)

# Plot actions
plt.figure()
plt.plot(timestamps[:len(actions)], actions[:, 0], label="roll")
plt.plot(timestamps[:len(actions)], actions[:, 1], label="pitch")
plt.plot(timestamps[:len(actions)], actions[:, 2], label="yaw_rate")
plt.plot(timestamps[:len(actions)], actions[:, 3], label="thrust")
plt.xlabel("Time [s]")
plt.ylabel("Command")
plt.title("Control Actions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot current gate
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for i in range(4):
    axs[i].plot(timestamps, curr_gates[:, i, 0], label="x")
    axs[i].plot(timestamps, curr_gates[:, i, 1], label="y")
    axs[i].plot(timestamps, curr_gates[:, i, 2], label="z")
    axs[i].set_ylabel(f"corner {i+1}")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time [s]")
fig.suptitle("Current Gate Corners")
plt.tight_layout()
plt.show()

# Plot next gate
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for i in range(4):
    axs[i].plot(timestamps, next_gates[:, i, 0], label="x")
    axs[i].plot(timestamps, next_gates[:, i, 1], label="y")
    axs[i].plot(timestamps, next_gates[:, i, 2], label="z")
    axs[i].set_ylabel(f"corner {i+1}")
    axs[i].legend()
    axs[i].grid(True)
axs[-1].set_xlabel("Time [s]")
fig.suptitle("Next Gate Corners")
plt.tight_layout()
plt.show()
