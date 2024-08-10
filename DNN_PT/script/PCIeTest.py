import subprocess

num_gpus = [0, 1, 2, 3]

for gpu in num_gpus:
    output_file = f"../nsys_res/nsys_output_gpu_{gpu}"
    command = f"nsys profile -o {output_file} -t cuda,nvtx --stats=true --force-overwrite=true --gpu-metrics-device=all python ../runNet.py --GPU_selection {gpu}"
    
    subprocess.run(command, shell=True, check=True)
