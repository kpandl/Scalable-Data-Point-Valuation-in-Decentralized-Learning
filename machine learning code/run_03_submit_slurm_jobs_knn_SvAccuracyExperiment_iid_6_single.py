import string
import os
import time

seeds_start = 0
num_jobs = 12

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    line_list.append("python -u operative_run_08_compute_SVs_federated_3c.py --seed=" + str(i) + " --start_environment=" + str(0) + " --end_environment=" + str(7) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(i) + "_reduced_maxtranslation_0.1_3c_iid --dscmode=47\n")
    
    with open("job_knn_exp1_iid_6_"+str(i)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    #if(i != 0 and i!= 4 and i!= 8):
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_knn_exp1_iid_6_"+str(i)+".sh")