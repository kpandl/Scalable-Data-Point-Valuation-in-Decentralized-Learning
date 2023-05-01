import numpy as np
import os
from pathlib import Path
import csv
import pandas as pd
import argparse, sys

from torch import cosine_similarity
import globals
from scipy import spatial

parser=argparse.ArgumentParser()
parser.add_argument('--name_of_test_folder', nargs='?', default="default100_reduced_maxtranslation_0.1_sex_100_0", type=str, help='Specify the name of the test folder')
args=parser.parse_args()

def compute_cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

def compute_cosine_similarity_for_file(file, name_method, type):
    path_base = os.path.join(os.getcwd(), "results", args.name_of_test_folder)

    if(type=="performance"):
        path_original = os.path.join(path_base, "Shapley_original_method_predictive_performance.csv")
        path_original = os.path.join(path_base, "Shapley_original_method_predictive_performance_newResults.csv")
    if(type=="bias_gender"):
        path_original = os.path.join(path_base, "Shapley_original_method_bias_gender.csv")
        path_original = os.path.join(path_base, "Shapley_original_method_bias_gender_newResults.csv")

    path_other_method = os.path.join(path_base, file)

    df_original = pd.read_csv(path_original, sep=',', header=0)
    df_other_method = pd.read_csv(path_other_method, sep=',', header=0)

    cosine_similarity = compute_cosine_similarity(df_original["Shapley value"], df_other_method["Shapley value"])
    path_results_cosinesimilarity = os.path.join(os.getcwd(), "results", args.name_of_test_folder, "cosine_similarity.csv")
    with open(path_results_cosinesimilarity, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([name_method, type, cosine_similarity])
    a = 0


path_results_cosinesimilarity = os.path.join(os.getcwd(), "results", args.name_of_test_folder, "cosine_similarity.csv")
with open(path_results_cosinesimilarity, 'a') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(["method", "type", "cosine_similarity"])

#compute_cosine_similarity_for_file("Shapley_performance.csv", "OR", "performance")
#compute_cosine_similarity_for_file("Shapley.csv", "OR", "bias_gender")

#compute_cosine_similarity_for_file("Shapley_LR_method_predictive_performance.csv", "LR", "performance")
#compute_cosine_similarity_for_file("Shapley_LR_method_bias_gender.csv", "LR", "bias_gender")

compute_cosine_similarity_for_file("Shapley_LR_method_predictive_performance_newResults.csv", "LR", "performance")
compute_cosine_similarity_for_file("Shapley_LR_method_bias_gender_newResults.csv", "LR", "bias_gender")

compute_cosine_similarity_for_file("Shapley_OR_method_predictive_performance_newResults.csv", "OR", "performance")
compute_cosine_similarity_for_file("Shapley_OR_method_bias_gender_newResults.csv", "OR", "bias_gender")

#compute_cosine_similarity_for_file("Shapley_federated_performance.csv", "FederatedKNN", "performance")

a = 0