# Scalable Data Point Valuation in Decentralized Learning - code for the experiments
 
This repository contains the Python code for the research article entitled "Scalable Data Point Valuation in Decentralized Learning" currently under review. The experiments consist of machine learning experiments and token economy experiments. For the machine learning experiments, we conducted within a Python 3.7.11 environment on a high performance computing cluster, running on a Unix operating systems and running with the Slurm Workload Manager. For the token economy experiments, we used a set of Ubuntu servers.

## Preparing to run the machine learning experiments

For the experiments, you need to download the three datasets NIH ChestX-ray8, CheXpert, and MIMIC-CXR. Then, you need to specify the storage location of these datasets in the "config.json" file, and run the "run_01_index_data.py" file to index the datasets.

The medical image scans have a higher resolution than the convolutional neural network input requires (256x256 pixels). To prevent data loading being a bottleneck, we re-scaled and re-stored the images using the "run_02_resize_scans.py" file. To do this, you first should specify the storage location of the rescaled images in the "config_resized.json" file before. After running the Python file, you need to replace the the "config.json" content file with the content of the "config_resized.json" file and re-run the "run_01_index_data.py" file to re-index the datasets with the resized storage paths.

## Running the experiments on a high performance computing cluster

To run the experiments efficiently on a SLURM high performance computing cluster, we created Python programs that automatically submit multiple compute jobs. We used the "bwForCluster Helix" cluster, but the following files can be easily adapted for other SLURM and non-SLURM high performance computing clusters.

We have different files, and the jobs each file submits need to be finished executing, before the next file can be run.

## Running the experiments on a local computer

Alternatively, you can also run these files on your local computer, instead of a high performance computing cluster. For this, execute the files starting with 'operative_run' in the same order as stated in the code of the above mentioned files.

## Token economy experiments

In the correspondig folder, you can find the smart contracts and the python scripts to interact with the blockchains and IPFS.