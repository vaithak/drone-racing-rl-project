import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# --- Use LaTeX font ---
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm"
})

num_tests = 50
labels = ["DS", "SS", "DM", "Ours"]
color_min = (0.274, 0.409, 0.705)
color_max = (1.0, 1.0, 0.4)
cmap = LinearSegmentedColormap.from_list("custom", [color_min, color_max])

def plot_table(ax, table, title=""):
    for i in range(4):
        for j in range(i):
            table[i,j] = [table[j,i][1], table[j,i][0]]

    results = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            results[i,j] = int(round(table[i,j,1] / num_tests * 100))
    # Convertiamo in stringhe con il simbolo %
    annot = np.array([[f"{val}\%" for val in row] for row in results])

    sns.heatmap(results, annot=annot, fmt="", cmap=cmap, annot_kws={"size": 24},
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=False, square=True)

    ax.set_title(title, fontsize=20)
    ax.set_xticklabels(labels, rotation=0, fontsize=20)
    ax.set_yticklabels(labels, rotation=0, fontsize=20)

if __name__ == '__main__':
    # LT
    table_LT = np.zeros((4, 4, 2))
    table_LT[0,0] = [19, 30]
    table_LT[0,1] = [0, 50]
    table_LT[0,2] = [2, 46]
    table_LT[0,3] = [50, 0]

    table_LT[1,1] = [26, 24]
    table_LT[1,2] = [6, 36]
    table_LT[1,3] = [50, 0]

    table_LT[2,2] = [25, 23]
    table_LT[2,3] = [49, 0]

    table_LT[3,3] = [23, 24]

    # LT with obstacles
    table_LT_obs = np.zeros((4, 4, 2))
    table_LT_obs[0,0] = [0, 0]
    table_LT_obs[0,1] = [11, 0]
    table_LT_obs[0,2] = [0, 0]
    table_LT_obs[0,3] = [48, 0]

    table_LT_obs[1,1] = [12, 10]
    table_LT_obs[1,2] = [0, 17]
    table_LT_obs[1,3] = [31, 11]

    table_LT_obs[2,2] = [0, 0]
    table_LT_obs[2,3] = [33, 0]

    table_LT_obs[3,3] = [17, 12]

    # CT with obstacles
    table_CT = np.zeros((4, 4, 2))
    table_CT[0,0] = [23, 20]
    table_CT[0,1] = [6, 44]
    table_CT[0,2] = [2, 48]
    table_CT[0,3] = [8, 41]

    table_CT[1,1] = [24, 26]
    table_CT[1,2] = [29, 17]
    table_CT[1,3] = [50, 0]

    table_CT[2,2] = [33, 17]
    table_CT[2,3] = [49, 0]

    table_CT[3,3] = [22, 22]

    # CT with obstacles
    table_CT_obs = np.zeros((4, 4, 2))
    table_CT_obs[0,0] = [0, 0]
    table_CT_obs[0,1] = [0, 0]
    table_CT_obs[0,2] = [0, 0]
    table_CT_obs[0,3] = [0, 0]

    table_CT_obs[1,1] = [0, 0]
    table_CT_obs[1,2] = [0, 0]
    table_CT_obs[1,3] = [0, 0]

    table_CT_obs[2,2] = [0, 0]
    table_CT_obs[2,3] = [0, 0]

    table_CT_obs[3,3] = [0, 0]

    fig, axes = plt.subplots(1, 4, figsize=(18,5))
    plot_table(axes[0], table_LT.copy(), title="LT")
    plot_table(axes[1], table_LT_obs.copy(), title="LT with obstacles")
    plot_table(axes[2], table_CT.copy(), title="CT")
    plot_table(axes[3], table_CT_obs.copy(), title="CT with obstacles")

    cbar_ax = fig.add_axes([0.955, 0.15, 0.01, 0.7])  # [left, bottom, width, height]

    # Colorbar
    cmap = plt.get_cmap(cmap)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)  # salva l'oggetto colorbar

    # aumenta font ticks e label
    cbar.ax.tick_params(labelsize=18)     # tick della colorbar

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # lascia spazio a destra per la colorbar

    plt.savefig("logs/plots/h2h_results.eps", bbox_inches='tight', format='eps', dpi=300)

    plt.show()
