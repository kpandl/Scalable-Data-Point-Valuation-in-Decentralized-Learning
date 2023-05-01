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

data = load('genders_dataset_test.npz')

arr = data["arr_0"][0:600]

f_counter = 0
m_counter = 0

for i in range(len(arr)):
    if(arr[i]=="F"):
        f_counter+=1
    if(arr[i]=="M"):
        m_counter+=1


a = 0