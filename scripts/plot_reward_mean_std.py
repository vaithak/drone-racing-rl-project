import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib as mpl

# --- Use real LaTeX font ---
mpl.rcParams.update({
    "text.usetex": True,       # True per usare LaTeX
    "font.family": "serif",
    "mathtext.fontset": "cm"
})

####################
### Single-Agent ###
####################

# # --- Load the CSV ---
# df = pd.read_csv("logs/csv/wandb_export_2025-09-12T17_01_31.710+02_00.csv")

# # --- Calculate mean and standard deviation across all runs ---
# mean = df.iloc[:, 1:].mean(axis=1)
# std = df.iloc[:, 1:].std(axis=1)

# # --- Set Seaborn style ---
# sns.set_style("whitegrid")
# sns.set_context("talk")

# # --- Plot ---
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.set_facecolor("#ebebf3")

# # Mean line
# num_envs = 24576
# num_steps_per_env = 24
# agent_steps = df["Step"] * num_envs * num_steps_per_env
# ax.plot(agent_steps, mean, label="Mean", color="darkorange", linewidth=2.5)

# # ± standard deviation shaded area
# ax.fill_between(agent_steps, mean - std, mean + std, color="#FFD8A8", label="Std Dev")

# # Grid
# ax.grid(True, color='white', linestyle='-', linewidth=1.0)

# # Axes
# ax.set_xlim(0, 3e9)
# ax.set_ylim(0, 7000)
# ax.tick_params(axis='both', labelsize=20)
# ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

# # Labels and title
# ax.set_xlabel("Agent Steps", fontsize=21)
# ax.set_ylabel("Cumulative Reward", fontsize=24)
# ax.set_title("Dense, Single-Agent", fontsize=26)

# # Legend
# ax.legend(frameon=True, facecolor='white', fontsize=24)

# # Save figure in PDF (EPS will crash senza TeX completo)
# plt.savefig("logs/plots/single_agent_cumulative_reward_mean_std.pdf", bbox_inches='tight', dpi=300)
# plt.show()

###################
### Multi-Agent ###
###################

# --- Set Seaborn style ---
sns.set_style("whitegrid")
sns.set_context("talk")

# --- Create a single figure ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor("#ebebf3")

num_envs = 10240

for name in ["ego", "adv"]:
    # --- Load the CSV ---
    df = pd.read_csv(f"logs/csv/{name}_mean_total_reward.csv")

    # --- Calculate mean and standard deviation across all runs ---
    mean = df.iloc[:, 2:].mean(axis=1)
    std = df.iloc[:, 2:].std(axis=1)

    # Mean line
    agent_steps = df["global_step"] * num_envs
    color_line = "crimson" if name == "ego" else "royalblue"
    color_fill = "#FFA8A8" if name == "ego" else "#A8EEFF"

    ax.plot(agent_steps, mean, label=f"{name.capitalize()} Mean", color=color_line, linewidth=2.5)
    ax.fill_between(agent_steps, mean - std, mean + std, color=color_fill, alpha=0.6, label=f"{name.capitalize()} Std Dev")

# --- Grid ---
ax.grid(True, color='white', linestyle='-', linewidth=1.0)

# --- Axes ---
ax.set_xlim(0, agent_steps.iloc[-1])
ax.set_ylim(0, 500)
ax.tick_params(axis='both', labelsize=20)
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)

# --- Labels and title ---
ax.set_xlabel("Agent Steps", fontsize=21)
ax.set_ylabel("Cumulative Reward", fontsize=24)
ax.set_title("Sparse, Multi-Agent", fontsize=26)

# --- Legend ---
ax.legend(
    frameon=True,
    facecolor='white',
    fontsize=18,
    ncol=2,
    columnspacing=0.5
)

# --- Save and show ---
plt.savefig("logs/plots/multi_agent_cumulative_reward_mean_std.pdf", bbox_inches='tight', dpi=300)
plt.show()