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
    [-1], [-1], [-1]  # placeholder — tune for your boards
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
             marker='.', linestyle='', linewidth=0.5, color=color)
    plt.plot(fit_line_x_values[i], fit_line_y_values[i],
             linewidth=0.5, color=color)
    handle = Line2D([0], [0], marker='.', color=color, linestyle='-', markersize=6, linewidth=0.5)
    legend_handles.append(handle)
    save_name = board_save_map.get(data_list[i], board_names[i])
    legend_labels.append(save_name.lower())

plt.xlabel("impact percent")
plt.ylabel("resistance percent")
plt.grid(True)
plt.legend(legend_handles, legend_labels, loc="lower right",
           facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "selected_boards_resist.png"), dpi=300, bbox_inches="tight")
plt.show()

# %% FEATURE EXTRACTION
y_ranges = {
    "Board 1": {"top": (-55, 55), "bottom": (-15, 15)},
    "Board 2": {"top": (-15, 15), "bottom": (-15, 15)},
    "Board 3": {"top": (-55, 55), "bottom": (-15, 50)}
}

# manually set x-axis ranges (example values – adjust as needed)
x_ranges = {
    "Board 1": (0, 50),
    "Board 2": (0, 50),
    "Board 3": (0, 80)
}

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

        # --- Convert to % deviation from baseline (first 5 points) ---
        features_percent = []
        for f in features:
            f = np.array(f)
            baseline = np.mean(f[:5]) if np.mean(f[:5]) != 0 else 1.0
            f_percent = 100 * (f - baseline) / baseline
            features_percent.append(f_percent)

        # # --- Plot ---
        # fig, axs = plt.subplots(2, 1, figsize=(6.5, 3), sharex=True)
        # colors = itertools.cycle(cm.tab10.colors)
        # feature_colors = [next(colors) for _ in feature_names]

        # # First 4 features
        # for idx in range(4):
        #     axs[0].plot(impacts, features_percent[idx], marker='.', linestyle='-', linewidth=1,
        #                 label=feature_names[idx], color=feature_colors[idx])
        # axs[0].grid(True)

        # # Last 4 features
        # for idx in range(4, 8):
        #     axs[1].plot(impacts, features_percent[idx], marker='.', linestyle='-', linewidth=1,
        #                 label=feature_names[idx], color=feature_colors[idx])
        # axs[1].set_xlabel("impact number")
        # axs[1].grid(True)

        # # Apply custom ranges (per board)
        # save_name = board_save_map[board]
        # board_ranges = y_ranges.get(save_name, {"top": (-50, 50), "bottom": (-50, 50)})
        # axs[0].set_ylim(*board_ranges["top"])
        # axs[1].set_ylim(*board_ranges["bottom"])

        # # Apply custom x-range
        # if save_name in x_ranges:
        #     axs[0].set_xlim(*x_ranges[save_name])
        #     axs[1].set_xlim(*x_ranges[save_name])

        # fig.text(-0.03, 0.6, "feature deviation from baseline (%)", va="center", rotation="vertical")

        # for ax in axs:
        #     ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        # # Legend
        # handles, labels = [], []
        # for ax in axs:
        #     h, l = ax.get_legend_handles_labels()
        #     handles.extend(h); labels.extend(l)
        # labels = [lbl.lower() for lbl in labels]

        # fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=4,
        #             facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)

        # plt.tight_layout(rect=[0, 0.05, 1, 1])
        # plt.savefig(os.path.join(save_path, f"{save_name}_feature_subplots.png"),
        #             dpi=300, bbox_inches="tight")
        # plt.show()

# %% FEATURE PLOTTING (SPIE-style stacked traces with manual outlier removal)

import numpy.ma as ma

# vertical spacing between traces
y_offset = 1.0  

# dictionary of outlier indices per board/feature (keys must be lowercase)
outliers = {
    "board 1": {
        "impulse factor": [8], "shape factor": [8], "crest factor": [8],
        "kurtosis": [8], "rms": [8], "absolute mean": [8], "maximum": [8]
    },
    # "board 2": {},
    "board 3": { 
        "impulse factor": [58], "shape factor": [58], "crest factor": [58],
        "kurtosis": [58], "rms": [58], "absolute mean": [58], "maximum": [58]}
}

x_ranges = {
    "board 1": (-2, 48),
    "board 2": (-2, 49),
    "board 3": (-2, 75)
}

for i, board in enumerate(data_list):
    if i in only_boards:
        board_folder = os.path.join(file_path, board)
        lvm_files = sorted([f for f in os.listdir(board_folder) if f.endswith(".lvm")])

        # --- Extract features ---
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

        save_name = board_save_map[board].lower()

        # --- Interpolate over outliers instead of dropping ---
        board_outliers = outliers.get(save_name, {})
        for f_idx, f_name in enumerate(feature_names):
            f_key = f_name.lower()
            if f_key in board_outliers:
                bad_idx = board_outliers[f_key]
                for idx in bad_idx:
                    if idx < len(features[f_idx]):
                        if 0 < idx < len(features[f_idx]) - 1:
                            # linear interpolation between neighbors
                            left = features[f_idx][idx - 1]
                            right = features[f_idx][idx + 1]
                            if not np.isnan(left) and not np.isnan(right):
                                features[f_idx][idx] = (left + right) / 2.0
                            else:
                                features[f_idx][idx] = np.nan
                        else:
                            # fallback: copy nearest neighbor
                            features[f_idx][idx] = features[f_idx][idx - 1] if idx > 0 else features[f_idx][idx + 1]

        # --- Normalize each feature relative to baseline ---
        norms = []
        for f in features:
            f = np.array(f, dtype=float)
            baseline = np.nanmean(f[:5]) if np.nanmean(f[:5]) != 0 else 1.0
            f_percent = 100 * (f - baseline) / baseline
            f_percent = np.clip(f_percent, -50, 50)
            f_min, f_max = np.nanmin(f_percent), np.nanmax(f_percent)
            f_norm = (f_percent - f_min) / (f_max - f_min + 1e-9)
            norms.append(f_norm)

        # # --- Plot stacked traces ---
        # fig, ax = plt.subplots(figsize=(6.5, 3.5))
        # j = len(norms) - 1
        # for trace, fname in zip(norms[::-1], feature_names[::-1]):  # reverse for correct order
        #     offset_trace = trace + j * y_offset
        #     trace_masked = ma.masked_invalid(offset_trace)
        #     ax.plot(np.arange(len(trace)), trace_masked,
        #             label=fname.lower() if fname.lower() != "rms" else "RMS",
        #             linewidth=0.5)
        #     j -= 1
            
        # --- Plot stacked traces (scatter + line) ---
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        j = len(norms) - 1
        for trace, fname in zip(norms[::-1], feature_names[::-1]):  # reverse for correct order
            offset_trace = trace + j * y_offset - 0.5  # shift upward so centered in its lane
            trace_masked = ma.masked_invalid(offset_trace)
        
            xvals = np.arange(len(trace))
            # scatter
            ax.scatter(xvals, trace_masked,
                       label=fname.lower() if fname.lower() != "rms" else "RMS",
                       s=5, alpha=0.8, zorder=2)
            # line through points
            ax.plot(xvals, trace_masked, linewidth=0.5, color="black", alpha=0.4, zorder=2)
        
            j -= 1
        
        # y-ticks with lowercase except RMS
        ytick_labels = [f.lower() if f.lower() != "rms" else "RMS" for f in feature_names]
        ax.set_yticks(np.arange(0, len(feature_names), 1))
        ax.set_yticklabels(ytick_labels)
        
        ax.set_xlabel("impact number")
        ax.set_ylabel("normalized feature traces")
        ax.grid(True, zorder=0)  # grid drawn underneath everything
        
        # --- Apply custom x-range per board ---
        save_name = board_save_map[board].lower()
        if save_name in x_ranges:
            ax.set_xlim(*x_ranges[save_name])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{save_name}_features_stacked.png"),
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
plt.plot(time_ms, response, label="response", linewidth=0.5)
plt.plot(time_ms, excitation, label="excitation", linewidth=0.5)
plt.xlim(0, 40)   # since 0.06 s = 60 ms
plt.xlabel("time (ms)"); plt.ylabel("acceleration (g)")
plt.legend(loc="upper right", facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(save_path, "excitation_response.png"), dpi=300, bbox_inches="tight")
plt.show()

# %% STATISTICAL ANALYSIS (Pearson and Spearman correlations between resistance and features)
from scipy.stats import pearsonr, spearmanr
import numpy as np

results = []

for i, board in enumerate(data_list):
    if i in only_boards:
        save_name = board_save_map[board]

        # --- Resistance fit interpolated to same length as features ---
        res_fit_x = fit_line_x_values[i]
        res_fit_y = fit_line_y_values[i]

        # --- Extract features for this board (recompute just like in plotting) ---
        board_folder = os.path.join(file_path, board)
        lvm_files = sorted([f for f in os.listdir(board_folder) if f.endswith(".lvm")])

        impacts, board_features = [], [[] for _ in feature_names]
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

            board_features[0].append(current_max)
            board_features[1].append(abs_mean)
            board_features[2].append(rms_val)
            board_features[3].append(skew(data_small))
            board_features[4].append(kurtosis(data_small))
            board_features[5].append(current_max / rms_val if rms_val != 0 else 0)
            board_features[6].append(rms_val / abs_mean if abs_mean != 0 else 0)
            board_features[7].append(current_max / abs_mean if abs_mean != 0 else 0)
            impacts.append(j)

        # --- Normalize features relative to baseline (first 5 points) ---
        board_features_percent = []
        for f in board_features:
            f = np.array(f, dtype=float)
            baseline = np.nanmean(f[:5]) if np.nanmean(f[:5]) != 0 else 1.0
            f_percent = 100 * (f - baseline) / baseline
            board_features_percent.append(f_percent)

        # --- Interpolate resistance to match feature length ---
        feature_length = len(board_features_percent[0])
        feature_impacts = np.linspace(0, 100, feature_length)
        res_fit_interp = np.interp(feature_impacts, res_fit_x, res_fit_y)

        # --- Correlation analysis (Pearson + Spearman) ---
        board_corrs = {}
        for f_idx, f_vals in enumerate(board_features_percent):
            f_array = np.array(f_vals, dtype=float)
            mask = ~np.isnan(f_array)
            if np.sum(mask) > 2:
                pear_r, pear_p = pearsonr(res_fit_interp[mask], f_array[mask])
                spear_r, spear_p = spearmanr(res_fit_interp[mask], f_array[mask])
                board_corrs[feature_names[f_idx].lower()] = {
                    "pearson": (pear_r, pear_p),
                    "spearman": (spear_r, spear_p),
                }
            else:
                board_corrs[feature_names[f_idx].lower()] = {
                    "pearson": (np.nan, np.nan),
                    "spearman": (np.nan, np.nan),
                }

        results.append((save_name, board_corrs))

# --- Print nicely ---
for board_name, corrs in results:
    print(f"\nBoard: {board_name}")
    for fname, vals in corrs.items():
        print(f"  {fname:15s} | "
              f"Pearson r = {vals['pearson'][0]:.3f}, p={vals['pearson'][1]:.2e} | "
              f"Spearman rho = {vals['spearman'][0]:.3f}, p={vals['spearman'][1]:.2e}")

import pandas as pd
import numpy as np

# --- Build DataFrames for Pearson and Spearman ---
pearson_df = pd.DataFrame({
    board: {f: vals["pearson"][0] for f, vals in corrs.items()}
    for board, corrs in results
}).T

spearman_df = pd.DataFrame({
    board: {f: vals["spearman"][0] for f, vals in corrs.items()}
    for board, corrs in results
}).T

# --- Compute mean ± std ---
pearson_mean = pearson_df.mean(axis=0)
pearson_std  = pearson_df.std(axis=0)
spearman_mean = spearman_df.mean(axis=0)
spearman_std  = spearman_df.std(axis=0)

# --- Print nicely ---
print("\n=== Pearson mean ± std across boards ===")
for feature in pearson_df.columns:
    print(f"{feature:15s}: {pearson_mean[feature]:.3f} ± {pearson_std[feature]:.3f}")

print("\n=== Spearman mean ± std across boards ===")
for feature in spearman_df.columns:
    print(f"{feature:15s}: {spearman_mean[feature]:.3f} ± {spearman_std[feature]:.3f}")



# %% Feature Importance Bar Plot
# import pandas as pd

# # --- Build DataFrame of r values ---
# df_r = pd.DataFrame(
#     {board: {f: vals[0] for f, vals in corrs.items()} for board, corrs in results}
# ).T  # boards as rows

# # --- Compute mean ± std across boards ---
# mean_r = df_r.mean(axis=0)
# std_r = df_r.std(axis=0)

# # --- Use absolute correlation as importance ---
# importance = mean_r.abs()

# # --- Sort by importance ---
# importance_sorted = importance.sort_values(ascending=False)

# # --- Fix RMS capitalization ---
# importance_sorted.index = [("RMS" if f.lower() == "rms" else f) for f in importance_sorted.index]

# # --- Plot feature importance bar chart ---
# plt.figure(figsize=(6.5, 3))
# plt.bar(importance_sorted.index, importance_sorted.values, zorder=2)

# plt.ylabel("feature importance") #(|mean r|)
# # plt.xlabel("feature")
# # plt.title("feature importance based on correlation magnitude")
# plt.xticks(rotation=45, ha="right")

# # Grid behind bars
# plt.grid(axis="y", alpha=0.6, zorder=1)

# plt.tight_layout()
# plt.savefig(os.path.join(save_path, "feature_importance.png"), dpi=300, bbox_inches="tight")
# plt.show()

# %% Feature Importance Bar Plots (Pearson and Spearman)
# --- Build DataFrames for Pearson and Spearman ---
df_pearson = pd.DataFrame(
    {board: {f: vals["pearson"][0] for f, vals in corrs.items()} for board, corrs in results}
).T
df_spearman = pd.DataFrame(
    {board: {f: vals["spearman"][0] for f, vals in corrs.items()} for board, corrs in results}
).T

# --- Absolute values as importance ---
df_pearson = df_pearson.abs()
df_spearman = df_spearman.abs()

# --- Compute means ---
pearson_mean = df_pearson.mean(axis=0)
spearman_mean = df_spearman.mean(axis=0)

# --- Fix RMS capitalization and lowercase everything else ---
def fix_labels(cols):
    return [("RMS" if f.lower() == "rms" else f.lower()) for f in cols]

df_pearson.columns = fix_labels(df_pearson.columns)
df_spearman.columns = fix_labels(df_spearman.columns)
pearson_mean.index = df_pearson.columns
spearman_mean.index = df_spearman.columns

# --- Common plotting function ---
def plot_importance(df, mean_vals, ylabel, filename):
    # Sort features by mean importance (descending)
    features_sorted = mean_vals.sort_values(ascending=False).index
    x = np.arange(len(features_sorted))
    width = 0.2  # bar width

    fig, ax = plt.subplots(figsize=(6.5, 3))

    # hidden mean bar (gray background)
    ax.bar(x, mean_vals[features_sorted], color="gray", alpha=0.2, width=0.8, zorder=1, label="mean")

    # each board as a separate bar
    for j, board in enumerate(df.index):
        ax.bar(x + (j - 1) * width, df.loc[board, features_sorted],
               width=width, label=board.lower(), zorder=2)

    # format
    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.6, zorder=0)

    # legend close to plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.show()

# --- Pearson plot ---
plot_importance(df_pearson, pearson_mean, ylabel="feature importance (|r|)", filename="feature_importance_pearson.png")

# --- Spearman plot ---
plot_importance(df_spearman, spearman_mean, ylabel="feature importance (|ρ|)", filename="feature_importance_spearman.png")

# %% Combined Feature Importance Plot (Pearson above Spearman, both x-axes visible, sorted)
# --- Build DataFrames for Pearson and Spearman ---
df_pearson = pd.DataFrame(
    {board: {f: vals["pearson"][0] for f, vals in corrs.items()} for board, corrs in results}
).T
df_spearman = pd.DataFrame(
    {board: {f: vals["spearman"][0] for f, vals in corrs.items()} for board, corrs in results}
).T

# --- Absolute values as importance ---
df_pearson = df_pearson.abs()
df_spearman = df_spearman.abs()

# --- Compute means ---
pearson_mean = df_pearson.mean(axis=0)
spearman_mean = df_spearman.mean(axis=0)

# --- Fix RMS capitalization and lowercase everything else ---
def fix_labels(cols):
    return [("RMS" if f.lower() == "rms" else f.lower()) for f in cols]

df_pearson.columns = fix_labels(df_pearson.columns)
df_spearman.columns = fix_labels(df_spearman.columns)
pearson_mean.index = df_pearson.columns
spearman_mean.index = df_spearman.columns

# --- Sort features independently ---
features_sorted_pearson = pearson_mean.sort_values(ascending=False).index
features_sorted_spearman = spearman_mean.sort_values(ascending=False).index

width = 0.2  # bar width

# --- Create subplots ---
fig, axs = plt.subplots(2, 1, figsize=(6.5, 6))

# --- Pearson subplot (top) ---
x = np.arange(len(features_sorted_pearson))
axs[0].bar(x, pearson_mean[features_sorted_pearson], color="gray", alpha=0.2,
           width=0.8, zorder=1, label="mean")
for j, board in enumerate(df_pearson.index):
    axs[0].bar(x + (j - 1) * width, df_pearson.loc[board, features_sorted_pearson],
               width=width, label=board.lower(), zorder=2)
axs[0].set_xticks(x)
axs[0].set_xticklabels(features_sorted_pearson, rotation=45, ha="right")
axs[0].set_ylabel("feature importance (|r|)")
axs[0].grid(axis="y", alpha=0.6, zorder=0)
axs[0].set_ylim(0, 1.0)  # <-- set Pearson y-limits manually

# --- Spearman subplot (bottom) ---
x = np.arange(len(features_sorted_spearman))
axs[1].bar(x, spearman_mean[features_sorted_spearman], color="gray", alpha=0.2,
           width=0.8, zorder=1, label="mean")
for j, board in enumerate(df_spearman.index):
    axs[1].bar(x + (j - 1) * width, df_spearman.loc[board, features_sorted_spearman],
               width=width, label=board.lower(), zorder=2)
axs[1].set_xticks(x)
axs[1].set_xticklabels(features_sorted_spearman, rotation=45, ha="right")
axs[1].set_ylabel("feature importance (|ρ|)")
axs[1].grid(axis="y", alpha=0.6, zorder=0)
axs[1].set_ylim(0, 1.0)  # <-- set Spearman y-limits manually

# --- One legend at top ---
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(save_path, "feature_importance_combined.png"),
            dpi=300, bbox_inches="tight")
plt.show()