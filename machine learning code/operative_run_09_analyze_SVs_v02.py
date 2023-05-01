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


index_condition = 0

data = load('deep_knn_values_unbalanced.npz')

SVs = data['knn']

print("sum", sum(SVs))
print("scores_mean", np.mean(SVs))

labels = load('labels_dataset_train.npz')["arr_0"]

labels_only_one_condition = []

SVs_0 = []
SVs_1 = []
SVs_1_positive = []
SVs_0_positive = []

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
    if(label==0.0 and SVs[i] > 0):
        SVs_0_positive.append(SVs[i])

a = 0

print("Total training data instances", len(SVs))
print("Training data instances label 1", len(SVs_1))
print("Training data instances label 0", len(SVs_0))
print("Sum of all SVs", sum(SVs))
print("Mean of all SVs of data instances with label 1", np.mean(SVs_1))
print("Mean of all SVs of data instances with label 0", np.mean(SVs_0))
print("Share of positive SVs of data instances with label 1 [%]", 100*len(SVs_1_positive)/len(SVs_1))
print("Share of positive SVs of data instances with label 0 [%]", 100*len(SVs_0_positive)/len(SVs_0))

SVs_all = np.array([SVs_0, SVs_1])#np.random.randn(1000, 3)

n_bins=30
labels = ['Training data instances with label 0', 'Training data instances with label 1']
plt.hist(SVs_all, n_bins, histtype='step', stacked=False, fill=False, label=labels)
plt.ylabel('Number of training data instances')
plt.xlabel('Shapley Value')
plt.yscale('log')
plt.legend(loc="upper right")
#plt.title('Stack Step (unfilled)')
plt.show()