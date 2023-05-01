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

SV_list_male = []
SV_list_female = []

for index_condition in range(8):
    data = load('deep_knn_values_balanced_federated_male_label_' + str(index_condition) + '.npz')
    SV_list_male.append(data['knn'])
    data = load('deep_knn_values_balanced_federated_female_label_' + str(index_condition) + '.npz')
    SV_list_female.append(data['knn'])



SVs_all_labels_male = sum(SV_list_male)/8
SVs_all_labels_female = sum(SV_list_female)/8

SVs_1_male = SVs_all_labels_male[0:27999]
SVs_2_male = SVs_all_labels_male[28000:55999]
SVs_3_male = SVs_all_labels_male[56000:83999]

SVs_1_female = SVs_all_labels_female[0:27999]
SVs_2_female = SVs_all_labels_female[28000:55999]
SVs_3_female = SVs_all_labels_female[56000:83999]

SV_1 = sum(SVs_1_female) - sum(SVs_1_male)
SV_2 = sum(SVs_2_female) - sum(SVs_2_male)
SV_3 = sum(SVs_3_female) - sum(SVs_3_male)

sum_total = SV_1 + SV_2 + SV_3
print(sum_total)

SVs_1 = SV_1/sum_total*0.007598442781
SVs_2 = SV_2/sum_total*0.007598442781
SVs_3 = SV_3/sum_total*0.007598442781

print(SVs_1)
print(SVs_2)
print(SVs_3)

a = 0