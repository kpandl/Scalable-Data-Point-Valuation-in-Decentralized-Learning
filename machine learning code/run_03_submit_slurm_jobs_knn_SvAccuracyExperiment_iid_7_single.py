import string
import os
import time

seeds_start = 0
num_jobs = 3

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(4):
        line_list.append("python3 -u operative_run_09_compute_SV_LR_coalitions_federated_ensembling_3c.py --seed=" + str(seeds_start+i*4+j) + " --start_environment=0 --end_environment=7 --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c_iid --dscmode=46\n")

    with open("job_knn_exp1_iid_7_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_knn_exp1_iid_7_"+str(seeds_start+i*4)+".sh")