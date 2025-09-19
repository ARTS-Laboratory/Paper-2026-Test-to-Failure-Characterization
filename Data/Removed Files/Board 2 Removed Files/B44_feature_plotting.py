# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 12:24:35 2025

@author: hugol
"""

import os
import math
import copy
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn import preprocessing

current_directory = os.getcwd()
file_path = current_directory
save_path = current_directory.replace("\Board 4.4", "\Board 4.4\\figures")

impacts = []
feature_names = ["Maximum", "Absolute Mean", "RMS", "Skewness", "Kurtosis", "Crest Factor", "Shape Factor", "Impulse Factor"]
features = [[] for i in range(len(feature_names))]

lvm_files = sorted([f for f in os.listdir(file_path) if f.endswith(".lvm")])
            
for j, filename in enumerate(lvm_files):
    with open(os.path.join(file_path, filename), 'r') as file:
        data = np.loadtxt(os.path.join(file_path, lvm_files[j]), usecols=(2))
        data = data.tolist()
        data_zeroed = data.copy()
        current_max = -1
        absolute_mean = 0
        rms_value = 0
        current_max_index = -1
        for k in range(len(data)):
            data_zeroed[k]-=data[0]
            if abs(data_zeroed[k]) > current_max:
                current_max = abs(data_zeroed[k])
                current_max_index = k
        if (current_max_index < 1000):
            data_zeroed_smaller = data_zeroed[:(current_max_index+11000)]
        else:     
            try:
                data_zeroed_smaller = data_zeroed[(current_max_index-1000):(current_max_index+11000)]
            except:
                data_zeroed_smaller = data_zeroed[(current_max_index-1000):]
        
        for k in range(len(data_zeroed_smaller)):
            absolute_mean += (abs(data_zeroed_smaller[k])/len(data_zeroed_smaller))
            rms_value += pow(data_zeroed_smaller[k], 2)
            
        rms_value = (math.sqrt(rms_value))/len(data_zeroed_smaller)
        
        features[0].append(current_max)
        features[1].append(absolute_mean)
        features[2].append(rms_value)
        features[3].append(skew(data_zeroed_smaller))
        features[4].append(kurtosis(data_zeroed_smaller))
        features[5].append((current_max/rms_value))
        features[6].append((rms_value/absolute_mean))
        features[7].append((current_max/absolute_mean))
        
        impacts.append(j)

#print(features[0])

for i in range(len(features[2])):
    features[2][i] = math.sqrt(((features[2][i]/len(features[2]))))

unfiltered_mean = 0
unfiltered_standard_deviation = 0
for i in range(len(features[0])):
    unfiltered_mean += (abs(features[0][i])/len(features[0]))
for i in range(len(features[0])):
    unfiltered_standard_deviation += pow((features[0][i]-unfiltered_mean), 2)
unfiltered_standard_deviation = math.sqrt((unfiltered_standard_deviation/(len(features[0])-1)))
outliers_fixed = False
while(not outliers_fixed):
    for i in range(len(features[0])):
        if ((features[0][i] > (unfiltered_mean + (4*unfiltered_standard_deviation))) or (features[0][i] < (unfiltered_mean - (4*unfiltered_standard_deviation)))):
            for j in range(len(features)):
                if j == 0:
                    impacts.pop(i)
                features[j].pop(i)
            break
    else:
        outliers_fixed = True

#print(np.isnan(features))

nan_fixed = False
while(not nan_fixed):
    for i in range(len(features[0])):
        for j in range(len(features)):
            if np.isnan(features[j][i]):
                for k in range(len(features)):
                    if k == 0:
                        impacts.pop(i)
                    features[k].pop(i)
                    nan_fixed = True
                break
        if nan_fixed:
            nan_fixed = False
            break
    else:
        nan_fixed = True
#print(features[0][0])
#print('\n')
#print(np.isnan(features))

features_zeroed = copy.deepcopy(features)

#for i in range(len(features[0])):
#    for j in range(len(features)):
#        features_zeroed[j][i] -= features[j][0]
        
features_normalized = copy.deepcopy(features_zeroed)

for i in range(len(features)):
   features_normalized[i] = preprocessing.normalize([features_zeroed[i]])

#print(features[1])

plt.figure(figsize=(10, 6))
for i in range(len(feature_names)):
    plt.plot(impacts, features_normalized[i][0],  marker='o', linestyle='-', label=feature_names[i])
#plt.plot(impacts, features_normalized[0][0],  marker='o', linestyle='', label=feature_names[0])
#plt.plot(impacts, features_normalized[1][0],  marker='o', linestyle='', label=feature_names[1])
#plt.plot(impacts, features_normalized[2][0],  marker='o', linestyle='', label=feature_names[2])
#plt.plot(impacts, features_normalized[3][0],  marker='o', linestyle='', label=feature_names[3])
#plt.plot(impacts, features_normalized[4][0],  marker='o', linestyle='', label=feature_names[4])
#plt.plot(impacts, features_normalized[4][0],  marker='o', linestyle='', label=feature_names[4])
plt.legend()
plt.xlabel('Impact Number')
plt.ylabel('Feature Values')
plt.title('Feature Plot of Board Acceleration')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_path}\\4_4_feature_plot", dpi=300)
plt.show()