import pandas as pd
import numpy as np
import json
import argparse
import re

# parser
parser = argparse.ArgumentParser(description="Read a CSV log file")
parser.add_argument("csv_file", type=str, help="Path to the CSV file")
args = parser.parse_args()

# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load CSV
df = pd.read_csv(args.csv_file, sep=',')

match = re.search(r'\d{8}_\d{6}', args.csv_file)
if match:
    filename = match.group()

# Compute the norm of linear velocity for each row
df['lin_vel'] = np.sqrt(df['lin_vel_x']**2 + df['lin_vel_y']**2 + df['lin_vel_z']**2)

# Remove the first two rows
df = df.iloc[2:].reset_index(drop=True)

# Create 'n_test' column
# Increment counter whenever time is 0
df['n_test'] = (df['time'] < 0.0001).cumsum()

df['n_laps'] = ((df['gates_passed'] - 1).clip(lower=0)) // df['n_gates']

df['n_laps_shift'] = df['n_laps'].shift(1)  # shift previous row
lap_times = df.loc[df['n_laps'] > df['n_laps_shift'], 'time'].tolist()

# print(df[['time', 'gates_passed', 'n_gates', 'n_laps', 'n_test']].head(1000))

# print(lap_times)

laps_by_test = {}

for test_id, df_test in df.groupby('n_test'):
    # detect where n_laps changes within this test
    lap_changes = df_test.index[df_test['n_laps'].diff() != 0].tolist()
    lap_starts = [df_test.index[0]] + [i for i in lap_changes if i != df_test.index[0]]
    lap_ends = lap_starts[1:] + [df_test.index[-1]]  # end of each lap
    laps_by_test[test_id] = list(zip(lap_starts, lap_ends))

print(laps_by_test)

laps_info_by_test = {}

num_tests = 1
for _, laps in laps_by_test.items():
    if len(laps_info_by_test) == 50:
        break

    laps_info = []
    for start, end in laps:
        t0 = float(df.loc[start, 'time'])
        tf = float(df.loc[end, 'time'])
        n_lap_start = int(df.loc[start, 'n_laps'])
        n_lap_end = int(df.loc[end, 'n_laps'])
        collisions = int(df.loc[end, 'n_crashes'])

        if t0 == tf:
            continue

        # Check if the lap is complete
        lap_time = round(tf - t0, 2) if df.loc[start, 'n_laps'] != n_lap_end else None

        # Compute average speed
        vel_data = df.loc[start:end, ['lin_vel']].to_numpy()
        avg_vel = round(float(np.mean(vel_data)), 3)

        lap_dict = {
            'n_lap': n_lap_start + 1,
            't0': t0,
            'tf': tf,
            'lap_time': lap_time,
            'avg_vel': avg_vel,
            'collisions': collisions
        }
        laps_info.append(lap_dict)

    if len(laps_info) == 1 and laps_info[0]['tf'] < 2.0:
        continue

    laps_info_by_test[num_tests] = laps_info
    num_tests += 1

print(json.dumps(laps_info_by_test, indent=4))

input()

crashes = 0

for key, laps in laps_info_by_test.items():
    if len(laps) < 3 or laps[2]['lap_time'] is None:
        crashes += 1

collisions = 0
for key, laps in laps_info_by_test.items():
    collisions += laps[-1]['collisions']
print(f'collisions = {collisions}')

success = 1.0 - crashes / len(laps_info_by_test)

print(f'crashes = {crashes}')
print(f'success = {round(success*100, 2)}%')

# Convert results to a DataFrame and save to CSV
all_laps = []
for test_id, laps in laps_info_by_test.items():
    for lap in laps:
        row = {"test_id": test_id, **lap}
        all_laps.append(row)

df_laps = pd.DataFrame(all_laps)

# Compute average speed and average time for valid laps
valid_laps = df_laps.dropna(subset=['lap_time'])
mean_speed = valid_laps['avg_vel'].mean()
mean_lap_time = valid_laps['lap_time'].mean()

print(f"Average speed across laps = {mean_speed:.3f} m/s")
print(f"Average lap time = {mean_lap_time:.2f} s")

# Save in CSV
df_laps.to_csv(f'logs/csv/icra_tests/laps_results_{filename}.csv', index=False)