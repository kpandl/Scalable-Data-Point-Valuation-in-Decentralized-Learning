import string
import os
import time
import math

seeds_start = 0

client_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18]
dsc_modes_full = [48, 50, 52, 54, 56, 58, 60, 62, 64]
dsc_modes_lean = [49, 51, 53, 55, 57, 59,61, 63, 65]

num_jobs = math.ceil(len(client_counts) / 4)

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(4):
        if(j+4*i >= len(client_counts)):
            break
        print(j+4*i)
        line_list.append("python -u operative_run_01_compute_shapley_values_3c.py --seed=0 --start_environment=" + str(2**client_counts[j+4*i]-2) +  " --end_environment=" + str(2**client_counts[j+4*i]-1) + " --maximum_translation=0.1 --combine_weighted_and_unweighted_aggregation --folder_name_extension=_knn_scalability_" + str(client_counts[j+4*i]) + "_clients_b --dscmode=" + str(dsc_modes_full[j+4*i]) + " --use_specific_gpu="+str(j)+" &\n")
    line_list.append("wait\n")

    for j in range(4):
        if(j+4*i >= len(client_counts)):
            break
        line_list.append("python -u operative_run_02_test_models_3c.py --seed=0 --start_environment=" + str(2**client_counts[j+4*i]-2) + " --end_environment=" + str(2**client_counts[j+4*i]-1) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default0_reduced_maxtranslation_0.1_knn_scalability_" + str(client_counts[j+4*i]) + "_clients_b --dscmode=" + str(dsc_modes_full[j+4*i]) + " --use_specific_gpu="+str(j)+" --copy_only_nih_test &\n")
    line_list.append("wait\n")

    with open("job_knn_scalability_exp1_1_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --cpus-per-gpu=16 --time=48:00:00 --mem=236000 --gres=gpu:4 " + "job_knn_scalability_exp1_1_"+str(seeds_start+i*4)+".sh")