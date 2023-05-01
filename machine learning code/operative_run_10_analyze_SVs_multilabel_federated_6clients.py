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
import csv
import argparse
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--folder_path', default='', type=str)
parser.add_argument('--largest_coalition_name', default='', type=str)
parser.add_argument('--first_client_name', default='', type=str)

args=parser.parse_args()

testing_path_1 = os.path.join(os.getcwd(), "results", args.folder_path, args.largest_coalition_name, "overall_testing.csv")
if(os.path.exists(testing_path_1)):
    testing_path = testing_path_1
else:
    testing_path = os.path.join(os.getcwd(), "results", args.folder_path, args.largest_coalition_name, "testing.csv")
df_testing = pd.read_csv(testing_path)

df_AUC = df_testing[(df_testing["Testing criteron"] == "AUC") & (df_testing["Operating point"] == "-")]

mean_average_AUC = df_AUC["Average"].mean()

utility = mean_average_AUC - 0.5

print("utility", utility)


SV_list = []

for index_condition in range(8):
    path = os.path.join(os.getcwd(), "results", args.folder_path, args.largest_coalition_name, args.first_client_name, "deep_knn_values_balanced_federated_label_" + str(index_condition) + ".npz")
    data = load(path)
    SV_list.append(data['knn'])

    print("sum", sum(SV_list[-1]))
    print("scores_mean", np.mean(SV_list[-1]))
    print("length of SV_list element", len(SV_list[-1]))

SVs_all_labels = sum(SV_list)/8

print("length of SVs_all_labels", len(SVs_all_labels))
SVs_1 = SVs_all_labels[0:27999]
SVs_2 = SVs_all_labels[28000:55999]
SVs_3 = SVs_all_labels[56000:83999]
SVs_4 = SVs_all_labels[84000:111999]
SVs_5 = SVs_all_labels[112000:139999]
SVs_6 = SVs_all_labels[140000:167999]

print(sum(SVs_1))
print(sum(SVs_2))
print(sum(SVs_3))
print(sum(SVs_4))
print(sum(SVs_5))
print(sum(SVs_6))

sum_total = sum(SVs_1) + sum(SVs_2) + sum(SVs_3) + sum(SVs_4) + sum(SVs_5) + sum(SVs_6)
print(sum_total)

SVs_1 = sum(SVs_1)/sum_total*utility
SVs_2 = sum(SVs_2)/sum_total*utility
SVs_3 = sum(SVs_3)/sum_total*utility
SVs_4 = sum(SVs_4)/sum_total*utility
SVs_5 = sum(SVs_5)/sum_total*utility
SVs_6 = sum(SVs_6)/sum_total*utility

print("SVs")
print(SVs_1)
print(SVs_2)
print(SVs_3)
print(SVs_4)
print(SVs_5)
print(SVs_6)
print("end of SVs")

path_Shapley_results_file = os.path.join(os.getcwd(), "results", args.folder_path, "Shapley_federated_performance.csv")
with open(path_Shapley_results_file, 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["dataset", "testing critereon", "operating point", "Shapley value"])
    spamwriter.writerow(["client 1", "AUC", "-", SVs_1])
    spamwriter.writerow(["client 2", "AUC", "-", SVs_2])
    spamwriter.writerow(["client 3", "AUC", "-", SVs_3])
    spamwriter.writerow(["client 4", "AUC", "-", SVs_4])
    spamwriter.writerow(["client 5", "AUC", "-", SVs_5])
    spamwriter.writerow(["client 6", "AUC", "-", SVs_6])

labels = load(os.path.join(os.getcwd(), "results", args.folder_path, args.largest_coalition_name, args.first_client_name, 'labels_dataset_train.npz'))["arr_0"]

label_SV_dict_no_finding_true = dict() 
label_SV_dict_no_finding_false = dict() 

sum_SVs = sum(SVs_all_labels)
for i, label in enumerate(labels):
    ind = i # +28000 for chexpert, +2*28000 for mimic
    label_no_finding = label[4]
    label_diseases = list(label[0:4]) + list(label[5:])
    label_count_diseases = sum(label_diseases)
    if(label_no_finding==1):
        label_SV_dict_no_finding_true.setdefault(label_count_diseases, []).append(SVs_all_labels[ind]*utility/sum_SVs)
    if(label_no_finding==0):
        label_SV_dict_no_finding_false.setdefault(label_count_diseases, []).append(SVs_all_labels[ind]*utility/sum_SVs)

list_of_keys = list(label_SV_dict_no_finding_true.keys())
list_of_keys.sort()

path_Shapley_results_file = os.path.join(os.getcwd(), "results", args.folder_path, "Shapley_federated_performance_per_datapoint.csv")


for key in list_of_keys:
    print("SVs of", len(label_SV_dict_no_finding_true[key]), "scans with no finding and", key, "disease labels, mean:", np.mean(label_SV_dict_no_finding_true[key]))
    with open(path_Shapley_results_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["Finding existing", "disease labels", "mean"])
        spamwriter.writerow(["0", key, np.mean(label_SV_dict_no_finding_true[key])])

list_of_keys = list(label_SV_dict_no_finding_false.keys())
list_of_keys.sort()

for key in list_of_keys:
    print("SVs of", len(label_SV_dict_no_finding_false[key]), "scans with a finding and", key, "disease labels, mean:", np.mean(label_SV_dict_no_finding_false[key]))
    with open(path_Shapley_results_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        if(key <= 5):
            spamwriter.writerow(["1", key, np.mean(label_SV_dict_no_finding_false[key])])