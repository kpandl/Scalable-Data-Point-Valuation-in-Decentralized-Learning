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
        line_list.append("python -u operative_run_03_create_approximated_models_optimized_3c.py --seed=0 --combine_weighted_and_unweighted_aggregation --start_environment=0 --end_environment=" + str(2**client_counts[j+4*i]-1) + " --name_of_test_folder=default0_reduced_maxtranslation_0.1_knn_scalability_" + str(client_counts[j+4*i]) + "_clients_b --dscmode=" + str(dsc_modes_full[j+4*i]) + "\n")

    with open("job_knn_scalability_exp1_2_"+str(seeds_start)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    #os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_knn_scalability_exp1_2_"+str(seeds_start)+".sh")