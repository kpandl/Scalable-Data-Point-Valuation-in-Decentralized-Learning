print("Hallo")

import numpy as np
import os
from pathlib import Path
import random
from DataSet import *
import pickle
import os
import sys
import torch
import gc
import time
from numpy import load
import matplotlib.pyplot as plt


index_condition = 2

data = load('deep_knn_values_balanced.npz')

SVs = data['knn']

print("sum", sum(SVs))
print("scores_mean", np.mean(SVs))

labels = load('labels_dataset_train.npz')["arr_0"]

labels_only_one_condition = []

SVs_0 = []
SVs_1 = []
SVs_1_positive = []

for label in labels:
    labels_only_one_condition.append(label[index_condition])

for i in range(len(labels_only_one_condition)):
    label = labels_only_one_condition[i]
    if(label==0.0):
        SVs_0.append(SVs[i])
    if(label==1.0):
        SVs_1.append(SVs[i])
    if(label==1.0 and SVs[i] > 0):
        SVs_1_positive.append(SVs[i])

a = 0

plt.hist(SVs, density=True, bins=30)  # density=False would make counts
plt.ylabel('# of training data instances (logarithmic scale)')
plt.xlabel('Shapley Value')
plt.yscale('log')
plt.show()