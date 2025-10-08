import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

def find_file(base_path, keyword):
    """
    Finds the first CSV file in base_path that contains the given keyword (case-insensitive).
    """
    files = glob.glob(os.path.join(base_path, "*.csv"))
    for f in files:
        if keyword.lower() in f.lower():
            return f
    return None

def load_and_plot_logdata(folder_name, t_start=None, t_end=None):
    """
    Loads and plots log data for Pitch, Roll, and Yaw axes.
    Each file must contain 3 columns: timestamp, commanded rate, gyro measurement.
    Gyro values are converted from millirad/s to deg/s before plotting.
    The y-axis is limited to [-100, 100] deg/s for all subplots.
    """
    base_path = os.path.expanduser(f"~/.config/cfclient/logdata/{folder_name}")
    axes = ['Pitch', 'Roll', 'Yaw']
    _, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i, axis in enumerate(axes):
        keyword = f"{axis}_rate"
        filepath = find_file(base_path, keyword)
        print(filepath)

        if not filepath:
            print(f"❌ File not found for axis {axis}: keyword '{keyword}'")
            continue

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()  # remove whitespace from column names

        timestamp = df.iloc[:, 0]
        controller = df.iloc[:, 1]
        gyro = df.iloc[:, 2]

        # Apply time filtering
        mask = pd.Series([True] * len(timestamp))
        if t_start is not None:
            mask &= timestamp >= t_start
        if t_end is not None:
            mask &= timestamp <= t_end

        # Filtered data
        timestamp_filtered = timestamp[mask]
        controller_filtered = controller[mask]
        gyro_filtered = gyro[mask]

        if i == 0:  # flip pitch
            controller_filtered *= -1

        # Convert gyro from millirad/s to deg/s
        gyro_deg = gyro_filtered

        # Plot
        axs[i].plot(timestamp_filtered, controller_filtered, label=f'{axis} Rate Commanded [deg/s]')
        axs[i].plot(timestamp_filtered, gyro_deg, label=f'{axis} Gyro Measurement [deg/s]')
        axs[i].set_ylabel(f'{axis} Rate')
        axs[i].set_ylim(-100, 100)  # y-axis limit
        axs[i].legend()
        axs[i].grid(True)
        axs[i].set_title(f"{axis} Axis")

    axs[-1].set_xlabel('Timestamp')
    plt.suptitle('Controller Rate vs Gyro Measurement')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage: python plot_cfclient_logs.py <folder_name> [t_start] [t_end]")
        sys.exit(1)

    folder = sys.argv[1]
    t_start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t_end = int(sys.argv[3]) if len(sys.argv) > 3 else None

    load_and_plot_logdata(folder, t_start, t_end)
