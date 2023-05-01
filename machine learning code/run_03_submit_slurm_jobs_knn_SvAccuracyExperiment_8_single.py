import string
import os
import time

seeds_start = 0
num_jobs = 1

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(12):
        line_list.append("python3 -u operative_run_10_LR_compute_SV_3c.py --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c --measure=predictive_performance\n")
        #line_list.append("python3 -u operative_run_10_analyze_SVs_multilabel_federated.py --largest_coalition_name=chexpert_m_mimic_m_nih_m --folder_path=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c --largest_coalition_name=chexpert_m_mimic_m_nih_m --first_client_name=nih_m\n")
        #line_list.append("python3 -u operative_run_09_compute_SV_original.py --seed=" + str(seeds_start+i*4+j) + " --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c\n")
        line_list.append("python3 -u operative_run_11_compute_cosine_similarities_3c.py --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_3c\n")


    with open("job_knn_exp1_8_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=248gb " + "job_knn_exp1_8_"+str(seeds_start+i*4)+".sh")