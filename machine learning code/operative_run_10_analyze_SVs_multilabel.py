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

SV_list = []

for index_condition in range(8):
    data = load('deep_knn_values_balanced_label_' + str(index_condition) + '.npz')
    SV_list.append(data['knn'])

    print("sum", sum(SV_list[-1]))
    print("scores_mean", np.mean(SV_list[-1]))

SVs_all_labels = sum(SV_list)/8


labels = load('labels_dataset_train.npz')["arr_0"]

label_SV_dict_no_finding_true = dict() 
label_SV_dict_no_finding_false = dict() 

for i, label in enumerate(labels):
    label_no_finding = label[4]
    label_diseases = list(label[0:4]) + list(label[5:])
    label_count_diseases = sum(label_diseases)
    if(label_no_finding==1):
        label_SV_dict_no_finding_true.setdefault(label_count_diseases, []).append(SVs_all_labels[i])
    if(label_no_finding==0):
        label_SV_dict_no_finding_false.setdefault(label_count_diseases, []).append(SVs_all_labels[i])

list_of_keys = list(label_SV_dict_no_finding_true.keys())
list_of_keys.sort()

for key in list_of_keys:
    print("SVs of", len(label_SV_dict_no_finding_true[key]), "scans with no finding and", key, "disease labels, mean:", np.mean(label_SV_dict_no_finding_true[key]))

list_of_keys = list(label_SV_dict_no_finding_false.keys())
list_of_keys.sort()

for key in list_of_keys:
    print("SVs of", len(label_SV_dict_no_finding_false[key]), "scans with a finding and", key, "disease labels, mean:", np.mean(label_SV_dict_no_finding_false[key]))


