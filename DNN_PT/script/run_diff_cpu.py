import subprocess

models = ["LeNet", "ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "AlexNet"]

batch_size = 512
gpu_selection = "0,1"
cores_list = [1, 2, 4, 8, 16, 32, 64]

for model in models:
    for cores in cores_list:
        output_file = f"diff_cpu_result/diff_cpu_{batch_size}.csv"
        print(f"Running training with model: {model}, batch size: {batch_size}, using {cores} CPU cores and 2 GPUs (GPU {gpu_selection})")
        
        command = f"taskset -c 0-{cores - 1} python ../runNet.py --model_name {model} --batch_size {batch_size} --number_worker {cores} --GPU_selection {gpu_selection} --output {output_file}"
        
        subprocess.run(command, shell=True, check=True)
