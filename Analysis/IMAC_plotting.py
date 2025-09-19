# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:29:34 2025

@author: trott
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:33:41 2025

@author: hugol
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
current_directory = os.getcwd()
file_path = current_directory.replace("\Analysis", "")
save_path = r"C:\Users\trott\Dropbox\Research Papers\In Progress\Roberts2026_IMAC\LaTeX\Figures"
os.makedirs(save_path, exist_ok=True)   # ensure folder exists

dataset_root = os.path.dirname(current_directory)
board_path = os.path.join(dataset_root, "Board 4.3")

# %% SETTINGS
MAX_VALID_RESISTANCE = 2
only_boards = [-1, -1, -1, 3, 4, -1, 6, -1, -1, -1]
impacts_to_remove = [[0], [5], [0], [0], [27, 33, 34, 53, 54],
                     [6, 9, 46, 79], [21], [22, 23, 24, 25, 26], [0], [23, 24]]

feature_names = [
    "Maximum", "Absolute Mean", "RMS", "Skewness", "Kurtosis",
    "Crest Factor", "Shape Factor", "Impulse Factor"
]

# %% DATA COLLECTION
data_list = os.listdir(file_path)
impact_numbers_list, resistance_numbers_list = [], []
board_names, board_names_trendline, board_save_paths = [], [], []

for i in data_list:
    if "Board" in i and i != "Board 0.0":
        new_file_address = os.path.join(file_path, i)
        new_save_address = os.path.join(new_file_address, "figures")
        board_names.append(i)
        board_names_trendline.append(i + " Trendline")
        board_save_paths.append(new_save_address)

        lvm_files = sorted([f for f in os.listdir(new_file_address) if f.endswith(".lvm")])

        resistance_baseline = 0
        impacts, resistances = [], []
        for j, filename in enumerate(lvm_files):
            with open(os.path.join(new_file_address, filename), 'r') as file:
                first_line = file.readline().strip()
                columns = first_line.split('\t')
                try:
                    resistance = float(columns[-1])
                    if 0 < resistance < MAX_VALID_RESISTANCE:
                        resistances.append(resistance - resistance_baseline)
                        impacts.append(j)
                except (IndexError, ValueError):
                    continue

        impact_numbers_list.append(impacts)
        resistance_numbers_list.append(resistances)

# %% CLEAN / NORMALIZE
resistance_numbers_removed = copy.deepcopy(resistance_numbers_list)
impact_numbers_percents = copy.deepcopy(impact_numbers_list)

for i in range(len(impact_numbers_list)):
    for j in range(len(impact_numbers_list[i])):
        impact_numbers_percents[i][j] = (impact_numbers_list[i][j] / impact_numbers_list[i][-1]) * 100

for i in range(len(impact_numbers_list)):
    pop_total = 0
    for j in range(len(impact_numbers_list[i])):
        for k in range(len(impacts_to_remove[i])):
            if (impact_numbers_list[i][j] == impacts_to_remove[i][k] or
                impact_numbers_list[i][j] in [0, 1, len(impact_numbers_list[i]) - 1]):
                impact_numbers_percents[i].pop(j - pop_total)
                resistance_numbers_removed[i].pop(j - pop_total)
                pop_total += 1
                break

resistance_numbers_percents = copy.deepcopy(resistance_numbers_removed)
for i in range(len(resistance_numbers_removed)):
    current_max, current_min = max(resistance_numbers_removed[i]), min(resistance_numbers_removed[i])
    for j in range(len(resistance_numbers_removed[i])):
        resistance_numbers_percents[i][j] = ((resistance_numbers_removed[i][j] - current_min) /
                                             (current_max - current_min)) * 100

# %% TRENDLINE FITTING
def custom_eq(x, a, b):
    return a * (1 - np.exp(-b * x))

class CustomExpRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, a_init=100, b_init=0.08):
        self.a_init, self.b_init = a_init, b_init
    def fit(self, X, y):
        X, y = np.asarray(X).ravel(), np.asarray(y)
        params, _ = curve_fit(custom_eq, X, y, p0=[self.a_init, self.b_init], maxfev=30000)
        self.a_, self.b_ = params
        return self
    def predict(self, X):
        return custom_eq(np.asarray(X).ravel(), self.a_, self.b_)

models, fit_line_x_values, fit_line_y_values = [], [], []
for i in range(len(impact_numbers_percents)):
    model = CustomExpRegressor()
    model.fit(impact_numbers_percents[i], resistance_numbers_percents[i])
    models.append(model)
    fit_line_x_values.append(np.linspace(0, 100, 1000))
    fit_line_y_values.append(model.predict(fit_line_x_values[i]))

# %% RESISTANCE PLOTS
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# Data only
plt.figure(figsize=(6, 2.5))
for i in range(len(board_names)):
    if i in only_boards and i != -1:   # <- filter correctly
        plt.plot(impact_numbers_percents[i], resistance_numbers_percents[i],
                 marker='.', linestyle='', linewidth=1, label=board_names[i].lower())
plt.legend(loc="lower right", facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.xlabel("impact percent"); plt.ylabel("resistance percent")
plt.grid(True); plt.tight_layout()
plt.savefig(f"{save_path}\\selected_boards_data_only.png", dpi=300, bbox_inches='tight')
plt.show()

# Trendlines only
plt.figure(figsize=(6, 2.5))
for i in range(len(board_names)):
    if i in only_boards and i != -1:
        plt.plot(fit_line_x_values[i], fit_line_y_values[i],
                 linewidth=1, label=board_names_trendline[i].lower())
plt.legend(loc="lower right", facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.xlabel("impact percent"); plt.ylabel("resistance percent")
plt.grid(True); plt.tight_layout()
plt.savefig(f"{save_path}\\selected_boards_trendlines_only.png", dpi=300, bbox_inches='tight')
plt.show()

# %% RESISTANCE PLOT (data + trendlines superimposed, matching colors)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

colors = itertools.cycle(plt.cm.tab10.colors)
plt.figure(figsize=(6, 2.5))

legend_handles = []
legend_labels = []

for i in range(len(board_names)):
    if i in only_boards and i != -1:
        color = next(colors)

        # scatter = resistance data
        plt.plot(
            impact_numbers_percents[i], resistance_numbers_percents[i],
            marker='.', linestyle='', linewidth=1, color=color
        )

        # line = exponential fit
        plt.plot(
            fit_line_x_values[i], fit_line_y_values[i],
            linewidth=1.2, color=color
        )

        # one combined legend entry per board
        combined_handle = Line2D(
            [0], [0],
            marker='.', color=color, linestyle='-',
            markersize=6, linewidth=1.2
        )
        legend_handles.append(combined_handle)
        legend_labels.append(board_names[i].lower())

plt.xlabel("impact percent")
plt.ylabel("resistance percent")
plt.grid(True)

# final legend: one row per board
plt.legend(
    legend_handles, legend_labels,
    loc="lower right",
    facecolor="white", edgecolor="lightgray",
    framealpha=1, frameon=True
)

plt.tight_layout()
plt.savefig(f"{save_path}\\selected_boards_resist.png", dpi=300, bbox_inches='tight')
plt.show()

# %% FEATURE EXTRACTION
for i, board in enumerate(board_names):
    if i in only_boards and i != -1:   # ✅ correct filter
        board_path = os.path.join(file_path, board)
        save_board_path = os.path.join(board_path, "figures")
        lvm_files = sorted([f for f in os.listdir(board_path) if f.endswith(".lvm")])

        impacts, features = [], [[] for _ in feature_names]
        for j, filename in enumerate(lvm_files):
            data = np.loadtxt(os.path.join(board_path, filename), usecols=(2)).tolist()
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

        # ============ CREATE SUBPLOTS ============
        fig, axs = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

        # unique colors for all 8 features
        colors = itertools.cycle(cm.tab10.colors)
        feature_colors = [next(colors) for _ in feature_names]

        # --- First 4 features ---
        for idx in range(4):
            axs[0].plot(
                impacts[:len(features_normalized[idx])],
                features_normalized[idx],
                marker='.', linestyle='-', linewidth=1,
                label=feature_names[idx],
                color=feature_colors[idx]
            )
        axs[0].grid(True)

        # --- Last 4 features ---
        for idx in range(4, 8):
            axs[1].plot(
                impacts[:len(features_normalized[idx])],
                features_normalized[idx],
                marker='.', linestyle='-', linewidth=1,
                label=feature_names[idx],
                color=feature_colors[idx]
            )
        axs[1].set_xlabel("impact number")
        axs[1].grid(True)

        # Shared y-axis label (centered across both subplots)
        fig.text(-0.03, 0.6, "normalized feature value", va="center", rotation="vertical")
        
        for ax in axs:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        
        # ============ ONE LEGEND FOR BOTH ============
        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        
        # convert to lowercase
        labels = [lbl.lower() for lbl in labels]
        
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=4,
            facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room for legend
        plt.savefig(os.path.join(save_path, f"{board}_feature_subplots.png"), dpi=300, bbox_inches="tight")
        plt.show()

# %% Test Profile
board_folder = os.path.join(file_path, "Board 4.3")
file_name = "HM4_b03_100mO_#8_12in_16thflt_wR2841_06.lvm"
full_file_path = os.path.join(board_folder, file_name)

# load data
data = np.loadtxt(full_file_path, skiprows=22)   # adjust skiprows for header lines
time = data[:, 0]        # time column
excitation = data[:, 1]  # excitation
response   = data[:, 2]  # response

# normalize both signals to zero (remove offset)
excitation = excitation - excitation[0]   # or excitation.mean()
response   = response - response[0]       # or response.mean()

# plot
plt.figure(figsize=(6, 2.5))
plt.plot(time, response, label="response", linewidth=1)
plt.plot(time, excitation, label="excitation", linewidth=1)
plt.xlim(0,0.06)
plt.xlabel("time (s)")
plt.ylabel("acceleration (g)")
plt.legend(loc="upper right", facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "excitation_response.png"), dpi=300, bbox_inches="tight")
plt.show()

# create subplots
fig, axs = plt.subplots(2, 1, figsize=(6, 2.5), sharex=True)

# --- Excitation subplot
axs[0].plot(time, excitation, linewidth=1, color="tab:blue")
axs[0].set_ylabel("excitation (g)")
axs[0].set_xlim(0, 0.06)
axs[0].grid(True)

# --- Response subplot
axs[1].plot(time, response, linewidth=1, color="tab:orange")
axs[1].set_xlabel("time (s)")
axs[1].set_ylabel("response (g)")
axs[1].set_xlim(0, 0.06)
axs[1].grid(True)

# align y-axis labels
fig.align_ylabels(axs)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "excitation_response_subplots.png"), dpi=300, bbox_inches="tight")
plt.show()