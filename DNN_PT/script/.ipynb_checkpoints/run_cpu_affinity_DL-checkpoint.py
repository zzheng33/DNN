import subprocess
import time
import csv
import os


def count_cores(cores_string):
    core_ranges = cores_string.split(',')
    num_cores = 0
    for core_range in core_ranges:
        if '-' in core_range:
            start, end = map(int, core_range.split('-'))
            num_cores += end - start + 1
        else:
            num_cores += 1
    return num_cores


dl_models = ["AlexNet", "ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "LeNet"]
dl_models = ["Inception-V3", "LeNet"]
dl_epochs = 4
dl_gpus = "0,1"
dl_cores_pack = "16-23,48-55,24-31,56-63"
dl_cores_spread = "0-31"
cores_dist = [dl_cores_pack, dl_cores_spread]
dl_output = "../result/cpu_affinity_DL/test_DL.csv"
batch_size = 8

for dl_model in dl_models:
    for dl_cores in cores_dist:
        num_dl_cores = count_cores(dl_cores)

        # Run the Deep Learning job
        print(f"Running {dl_model} with {num_dl_cores} CPU cores and 2 GPUs for {dl_epochs} epochs")
        dl_command = f"taskset -c {dl_cores} python ../runNet.py --model_name {dl_model} --batch_size {batch_size} --number_worker {num_dl_cores} --GPU_selection {dl_gpus} --output {dl_output} --epoch {dl_epochs}"
        subprocess.run(dl_command, shell=True, check=True)
