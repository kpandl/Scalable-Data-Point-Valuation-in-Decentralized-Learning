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

parser=argparse.ArgumentParser()
parser.add_argument('--folder_path', default='default0_reduced_maxtranslation_0.1_labelflip_40', type=str)
parser.add_argument('--largest_coalition_name', default='mimic_0_mimic_1_mimic_2_mimic_3_mimic_4_mimic_5', type=str)
parser.add_argument('--first_client_name', default='mimic_0', type=str)
args=parser.parse_args()

print("folder_path", args.folder_path)
print("largest_coalition_name", args.largest_coalition_name)

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
    data_path = os.path.join(os.getcwd(), "results", args.folder_path, args.largest_coalition_name, args.first_client_name, "deep_knn_values_balanced_federated_label_" + str(index_condition) + ".npz")
    data = load(data_path)
    #data = load('deep_knn_values_balanced_federated_label_' + str(index_condition) + '.npz')
    SV_list.append(data['knn'])

    print("sum", sum(SV_list[-1]))
    print("scores_mean", np.mean(SV_list[-1]))

SVs_all_labels = sum(SV_list)/8

SVs_all_labels = SVs_all_labels * utility / sum(SVs_all_labels)

SVs_of_clients = []

scans_per_client = 28000

for i in range(6):
    print("SV client", str(i), sum(SVs_all_labels[i*scans_per_client:(i+1)*scans_per_client]))
    SVs_of_clients.append(sum(SVs_all_labels[i*scans_per_client:(i+1)*scans_per_client]))

#plot = plt.bar(range(6), SVs_of_clients)
##plt.xticks(range(6), ["client 1", "client 2", "client 3", "client 4", "client 5", "client 6"])
#plt.xticks(range(6), [str(i*5) for i in range(6)])
#plt.ylabel("Shapley value")
#plt.xlabel("client label flip probability [%]")
#plt.title("Shapley values of the clients")
#plt.show()

flipped_label_path = os.path.join(os.getcwd(), "results", args.folder_path, "flipped_label_indices.pkl")

with open(flipped_label_path, 'rb') as f:
    flipped_label_indices = pickle.load(f)

SVs_not_changed_labels = []
SVs_changed_labels = []

for i in range(len(SVs_all_labels)):
    if i in flipped_label_indices.keys():
        SVs_changed_labels.append(SVs_all_labels[i])
    else:
        SVs_not_changed_labels.append(SVs_all_labels[i])

print("mean SVs_not_changed_labels", np.mean(SVs_not_changed_labels))
print("mean SVs_changed_labels", np.mean(SVs_changed_labels))

print("---------------------------------")

SVs_not_changed_labels = []
SVs_changed_labels_1 = []
SVs_changed_labels_2 = []
SVs_changed_labels_3 = []
SVs_changed_labels_4 = []
SVs_changed_labels_5 = []
SVs_changed_labels_6 = []
SVs_changed_labels_7 = []
SVs_changed_labels_8 = []

keys_0 = []
keys_1 = []
keys_2 = []
keys_3 = []


path_Shapley_results_file = os.path.join(os.getcwd(), "results", args.folder_path, "Shapley_federated_performance_per_datapoint_flipped_consideration.csv")



for i in range(len(SVs_all_labels)):
    if i in flipped_label_indices.keys() and flipped_label_indices[i] == 1:
        SVs_changed_labels_1.append(SVs_all_labels[i])
        keys_1.append(i)
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 2:
        SVs_changed_labels_2.append(SVs_all_labels[i])
        keys_2.append(i)
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 3:
        SVs_changed_labels_3.append(SVs_all_labels[i])
        keys_3.append(i)
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 4:
        SVs_changed_labels_4.append(SVs_all_labels[i])
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 5:
        SVs_changed_labels_5.append(SVs_all_labels[i])
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 6:
        SVs_changed_labels_6.append(SVs_all_labels[i])
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 7:
        SVs_changed_labels_7.append(SVs_all_labels[i])
    elif i in flipped_label_indices.keys() and flipped_label_indices[i] == 8:
        SVs_changed_labels_8.append(SVs_all_labels[i])
    else:
        SVs_not_changed_labels.append(SVs_all_labels[i])
        keys_0.append(i)

print("mean SVs_not_changed_labels", np.mean(SVs_not_changed_labels))
print("mean SVs_changed_labels_1", np.mean(SVs_changed_labels_1))
print("mean SVs_changed_labels_2", np.mean(SVs_changed_labels_2))
print("mean SVs_changed_labels_3", np.mean(SVs_changed_labels_3))
print("mean SVs_changed_labels_4", np.mean(SVs_changed_labels_4))
print("mean SVs_changed_labels_5", np.mean(SVs_changed_labels_5))
if(len(SVs_changed_labels_6) > 0):
    print("mean SVs_changed_labels_6", np.mean(SVs_changed_labels_6))
if(len(SVs_changed_labels_7) > 0):
    print("mean SVs_changed_labels_7", np.mean(SVs_changed_labels_7))



with open(path_Shapley_results_file, 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["Changed labels", "mean SV"])
    spamwriter.writerow(["0", np.mean(SVs_not_changed_labels)])
    spamwriter.writerow(["1", np.mean(SVs_changed_labels_1)])
    spamwriter.writerow(["2", np.mean(SVs_changed_labels_2)])
    spamwriter.writerow(["3", np.mean(SVs_changed_labels_3)])
    spamwriter.writerow(["4", np.mean(SVs_changed_labels_4)])
    spamwriter.writerow(["5", np.mean(SVs_changed_labels_5)])
    if(len(SVs_changed_labels_6) > 0):
        spamwriter.writerow(["6", np.mean(SVs_changed_labels_6)])
    if(len(SVs_changed_labels_7) > 0):
        spamwriter.writerow(["7", np.mean(SVs_changed_labels_7)])