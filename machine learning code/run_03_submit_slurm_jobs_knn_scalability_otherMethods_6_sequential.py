import string
import os
import time

seeds_start = 0

client_counts = [2, 4, 6, 8, 10, 12, 14, 16, 18]
dsc_modes_full = [48, 50, 52, 54, 56, 58, 60, 62, 64]
dsc_modes_lean = [49, 51, 53, 55, 57, 59,61, 63, 65]

num_jobs = 1

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(len(client_counts)):
        line_list.append("python -u operative_run_07_compute_dataset_array_one_inst_3c.py --seed=0 --start_environment=" + str(2**client_counts[j+4*i]-2) + " --end_environment=" + str(2**client_counts[j+4*i]-1) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default0_reduced_maxtranslation_0.1_knn_scalability_" + str(client_counts[j+4*i]) + "_clients_b --dscmode=" + str(dsc_modes_full[j+4*i]) + "\n")


    with open("job_knn_scalability_exp1_6_"+str(seeds_start)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    #os.system("sbatch -p single --cpus-per-gpu=16 --time=48:00:00 --mem=236000 --gres=gpu:1 " + "job_knn_scalability_exp1_6_"+str(seeds_start)+".sh")