# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:29:34 2025
@author: trott
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import preprocessing
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import itertools
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# %% PATHS
project_root = r"C:\Users\trott\Documents\Paper-2026-Test-to-Failure-Characterization"
file_path = os.path.join(project_root, "Data")                # contains Board 1,2,3
save_path = os.path.join(project_root, "Analysis", "Figures") # where plots go
os.makedirs(save_path, exist_ok=True)

# map renamed boards back to their true identities for labeling
board_map = {
    "Board 1": "Board 4.3",
    "Board 2": "Board 4.4",
    "Board 3": "Board 4.6"
}
board_save_map = {
    "Board 1": "Board 1",
    "Board 2": "Board 2",
    "Board 3": "Board 3"
}

# %% SETTINGS
MAX_VALID_RESISTANCE = 2
only_boards = [0, 1, 2]   # only keep Boards 1,2,3

# adjust impacts_to_remove for your new dataset if needed
impacts_to_remove = [
    [0], [0], [0]  # placeholder — tune for your boards
]

feature_names = [
    "Maximum", "Absolute Mean", "RMS", "Skewness", "Kurtosis",
    "Crest Factor", "Shape Factor", "Impulse Factor"
]

# %% DATA COLLECTION
data_list = sorted([d for d in os.listdir(file_path) if d.startswith("Board")])
impact_numbers_list, resistance_numbers_list = [], []
board_names, board_names_trendline = [], []

for i, board in enumerate(data_list):
    if i in only_boards:
        new_file_address = os.path.join(file_path, board)
        board_names.append(board_map.get(board, board))
        board_names_trendline.append(board_map.get(board, board) + " Trendline")

        lvm_files = sorted([f for f in os.listdir(new_file_address) if f.endswith(".lvm")])

        resistances, impacts = [], []
        for j, filename in enumerate(lvm_files):
            with open(os.path.join(new_file_address, filename), 'r') as file:
                first_line = file.readline().strip()
                columns = first_line.split('\t')
                try:
                    resistance = float(columns[-1])
                    if 0 < resistance < MAX_VALID_RESISTANCE:
                        resistances.append(resistance)
                        impacts.append(j)
                except (IndexError, ValueError):
                    continue

        impact_numbers_list.append(impacts)
        resistance_numbers_list.append(resistances)

# %% CLEAN / NORMALIZE
resistance_numbers_removed = copy.deepcopy(resistance_numbers_list)
impact_numbers_cleaned = copy.deepcopy(impact_numbers_list)

# ---- Remove flagged impacts ----
for i in range(len(impact_numbers_cleaned)):
    pop_total = 0
    for j in range(len(impact_numbers_list[i])):
        if impact_numbers_list[i][j] in impacts_to_remove[i]:
            impact_numbers_cleaned[i].pop(j - pop_total)
            resistance_numbers_removed[i].pop(j - pop_total)
            pop_total += 1

# ---- Normalize X and Y so first = 0, last = 100 ----
impact_numbers_percents, resistance_numbers_percents = [], []
for i in range(len(impact_numbers_cleaned)):
    if not impact_numbers_cleaned[i]:
        continue

    first_impact, last_impact = impact_numbers_cleaned[i][0], impact_numbers_cleaned[i][-1]
    first_res, last_res = resistance_numbers_removed[i][0], resistance_numbers_removed[i][-1]

    # normalize impacts
    x_norm = [((x - first_impact) / (last_impact - first_impact)) * 100
              for x in impact_numbers_cleaned[i]]
    # normalize resistances
    y_norm = [((y - first_res) / (last_res - first_res)) * 100
              for y in resistance_numbers_removed[i]]

    impact_numbers_percents.append(x_norm)
    resistance_numbers_percents.append(y_norm)

# %% TRENDLINE FITTING (anchored at 0,0 and 100,100)
def anchored_eq(x, b):
    return 100 * (1 - np.exp(-b * x)) / (1 - np.exp(-100 * b))

class AnchoredExpRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, b_init=0.05):
        self.b_init = b_init
    def fit(self, X, y):
        X, y = np.asarray(X).ravel(), np.asarray(y)
        params, _ = curve_fit(anchored_eq, X, y, p0=[self.b_init], maxfev=30000)
        self.b_ = params[0]
        return self
    def predict(self, X):
        return anchored_eq(np.asarray(X).ravel(), self.b_)

models, fit_line_x_values, fit_line_y_values = [], [], []
for i in range(len(impact_numbers_percents)):
    model = AnchoredExpRegressor()
    model.fit(impact_numbers_percents[i], resistance_numbers_percents[i])
    models.append(model)
    fit_line_x_values.append(np.linspace(0, 100, 1000))
    fit_line_y_values.append(model.predict(fit_line_x_values[i]))

# %% RESISTANCE PLOTS (data + fits)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

colors = itertools.cycle(plt.cm.tab10.colors)
plt.figure(figsize=(6.5, 2.5))
legend_handles, legend_labels = [], []

for i in range(len(board_names)):
    color = next(colors)
    plt.plot(impact_numbers_percents[i], resistance_numbers_percents[i],
             marker='.', linestyle='', linewidth=1, color=color)
    plt.plot(fit_line_x_values[i], fit_line_y_values[i],
             linewidth=1.2, color=color)
    handle = Line2D([0], [0], marker='.', color=color, linestyle='-', markersize=6, linewidth=1.2)
    legend_handles.append(handle)
    legend_labels.append(board_names[i])

plt.xlabel("impact percent")
plt.ylabel("resistance percent")
plt.grid(True)
plt.legend(legend_handles, legend_labels, loc="lower right",
           facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "selected_boards_resist.png"), dpi=300, bbox_inches="tight")
plt.show()

# %% FEATURE EXTRACTION
for i, board in enumerate(data_list):
    if i in only_boards:
        board_folder = os.path.join(file_path, board)
        lvm_files = sorted([f for f in os.listdir(board_folder) if f.endswith(".lvm")])

        impacts, features = [], [[] for _ in feature_names]
        for j, filename in enumerate(lvm_files):
            data = np.loadtxt(os.path.join(board_folder, filename), usecols=(2)).tolist()
            data_zeroed = [val - data[0] for val in data]
            current_max_index = np.argmax([abs(x) for x in data_zeroed])
            current_max = abs(data_zeroed[current_max_index])
            if current_max_index < 1000:
                data_small = data_zeroed[:(current_max_index + 11000)]
            else:
                data_small = data_zeroed[(current_max_index - 1000):(current_max_index + 11000)]
            abs_mean = np.mean(np.abs(data_small))
            rms_val = np.sqrt(np.mean(np.square(data_small)))
            features[0].append(current_max)
            features[1].append(abs_mean)
            features[2].append(rms_val)
            features[3].append(skew(data_small))
            features[4].append(kurtosis(data_small))
            features[5].append(current_max / rms_val if rms_val != 0 else 0)
            features[6].append(rms_val / abs_mean if abs_mean != 0 else 0)
            features[7].append(current_max / abs_mean if abs_mean != 0 else 0)
            impacts.append(j)

        features_normalized = [preprocessing.normalize([f])[0] for f in features]

        fig, axs = plt.subplots(2, 1, figsize=(6.5, 3), sharex=True)
        colors = itertools.cycle(cm.tab10.colors)
        feature_colors = [next(colors) for _ in feature_names]

        for idx in range(4):
            axs[0].plot(impacts, features_normalized[idx], marker='.', linestyle='-', linewidth=1,
                        label=feature_names[idx], color=feature_colors[idx])
        axs[0].grid(True)

        for idx in range(4, 8):
            axs[1].plot(impacts, features_normalized[idx], marker='.', linestyle='-', linewidth=1,
                        label=feature_names[idx], color=feature_colors[idx])
        axs[1].set_xlabel("impact number")
        axs[1].grid(True)

        fig.text(-0.03, 0.6, "normalized feature value", va="center", rotation="vertical")
        for ax in axs:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h); labels.extend(l)
        labels = [lbl.lower() for lbl in labels]

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=4,
                   facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        axs[0].set_ylim(0, 0.2)
        axs[1].set_ylim(0, 0.2)
        
        save_name = board_save_map[board]  # Board 1,2,3
        plt.savefig(os.path.join(save_path, f"{save_name}_feature_subplots.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

# %% Test Profile (example from Board 1 / true 4.3)
board_folder = os.path.join(file_path, "Board 1")
file_name = "HM4_b03_100mO_#8_12in_16thflt_wR2841_06.lvm"
full_file_path = os.path.join(board_folder, file_name)

data = np.loadtxt(full_file_path, skiprows=22)
time, excitation, response = data[:, 0], data[:, 1], data[:, 2]
excitation -= excitation[0]; response -= response[0]
time_ms = time * 1000  

plt.figure(figsize=(6.5, 2.5))
plt.plot(time_ms, response, label="response", linewidth=1)
plt.plot(time_ms, excitation, label="excitation", linewidth=1)
plt.xlim(0, 40)   # since 0.06 s = 60 ms
plt.xlabel("time (ms)"); plt.ylabel("acceleration (g)")
plt.legend(loc="upper right", facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(save_path, "excitation_response.png"), dpi=300, bbox_inches="tight")
plt.show()
