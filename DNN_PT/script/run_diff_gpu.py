import os
import subprocess

batch_size = 256
models = ["LeNet", "ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "AlexNet"]
num_gpus_range = [1, 2, 3, 4]
num_workers = 4
share = 1

for model in models:
    for num_gpus in num_gpus_range:
        print(f"Running training with model: {model}, batch size: {batch_size}, using {num_gpus} GPUs, using {num_workers} workers")
        GPU_selection = ",".join(str(i) for i in range(num_gpus))
        output_file = f"/diff_gpu_result/diff_gpu_{batch_size}_{num_workers}.csv"
        
        command = f"python ../runNet.py --model_name {model} --batch_size {batch_size} --number_worker {num_workers} --GPU_selection {GPU_selection} --output {output_file} --share 1"
        
        subprocess.run(command, shell=True, check=True)
