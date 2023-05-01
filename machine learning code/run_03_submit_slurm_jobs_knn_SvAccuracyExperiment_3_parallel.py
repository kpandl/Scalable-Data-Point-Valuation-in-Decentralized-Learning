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

    #for j in range(4):
    #    line_list.append("python -u operative_run_05_compute_bias_metrics_sex_3c.py --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c &\n")
    #line_list.append("wait\n")

    for j in range(4):
        line_list.append("python -u operative_run_06_compute_shapley_predictive_performance_3c.py --seed=" + str(seeds_start+i*4+j) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c --dscmode=8 &\n")
    line_list.append("wait\n")

    with open("job_knn_exp1_3_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=248gb " + "job_knn_exp1_3_"+str(seeds_start+i*4)+".sh")