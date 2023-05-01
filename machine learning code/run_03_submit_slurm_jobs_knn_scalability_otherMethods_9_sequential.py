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
        #line_list.append("python3 -u operative_run_09_compute_SV_LR_coalitions_federated_ensembling_3c.py --seed=0 --start_environment=0 --end_environment=" + str(2**client_counts[j+4*i]-1) + " --name_of_test_folder=default0_reduced_maxtranslation_0.1_knn_scalability_" + str(client_counts[j+4*i]) + "_clients --dscmode=" + str(dsc_modes_full[j+4*i]) + " --skip_bias_gender\n")
        line_list.append("python3 -u operative_run_10_LR_compute_SV_3c.py --name_of_test_folder=default0_reduced_maxtranslation_0.1_knn_scalability_" + str(client_counts[j+4*i]) + "_clients_b --measure=predictive_performance --num_clients=" + str(client_counts[j]) + "\n")

    with open("job_knn_scalability_exp1_9_"+str(seeds_start)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_knn_scalability_exp1_9_"+str(seeds_start)+".sh")