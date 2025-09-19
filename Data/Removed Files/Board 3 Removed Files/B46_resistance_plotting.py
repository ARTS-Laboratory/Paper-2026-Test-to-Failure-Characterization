# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:44:20 2025

@author: trott
"""

import os
import matplotlib.pyplot as plt

# Folder containing your .lvm files
folder_path = r"C:\Users\trott\Documents\Dataset-11-rectangular-electronic-assembly-under-repeated-5-kg-impacts\Data\Board 4.6"
save_path = r"C:\Users\trott\Documents\Dataset-11-rectangular-electronic-assembly-under-repeated-5-kg-impacts\Data\Board 4.6\Figures"
MAX_VALID_RESISTANCE = 100000  # Define your threshold here
MIN_VALID_RESISTANCE = 0     # Avoid log(0) or negative

lvm_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".lvm")])

resistances = []
impact_numbers = []

for i, filename in enumerate(lvm_files):
    with open(os.path.join(folder_path, filename), 'r') as file:
        first_line = file.readline().strip()
        columns = first_line.split('\t')

        try:
            resistance = float(columns[-1])
            if MIN_VALID_RESISTANCE <= resistance <= MAX_VALID_RESISTANCE:
                resistances.append(resistance)
                impact_numbers.append(i + 1)
            else:
                print(f"Skipping {filename}: resistance {resistance} exceeds threshold")
        except (IndexError, ValueError):
            print(f"Skipping {filename}: invalid format")

# Plotting
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.figure(figsize=(6, 4))
plt.plot(impact_numbers, resistances, marker='.', linestyle='')
plt.xlabel('impact number')
plt.ylabel('resistance (Ohms)')
plt.title('Filtered Resistance vs. Impact Number')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_path}\\4_6_metric_plot.png", dpi=300)
plt.show()