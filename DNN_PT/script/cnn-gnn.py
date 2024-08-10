import subprocess

fixed_model = "ResNet-101"
batch_size = 512
num_workers = 16
GPU_selection_fixed = "0,1"
other_models = ["LeNet", "ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "AlexNet"]
gnn_dir = "/home/john/pytorch_geometric/benchmark/training/"
share = 1
cores = 16

# Run the fixed model alone on 2 GPUs
output = f"corun_shareCPU_res/corun_gnn_{fixed_model}_{batch_size}_{num_workers}_2_2.csv"
print(f"Running fixed model: {fixed_model}, batch size: {batch_size}, using GPUs: {GPU_selection_fixed}")

# command_fixed = f"taskset -c 0-{cores} python ../corunTest.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {cores} --GPU_selection {GPU_selection_fixed} --output {output} --share {share}"

command_fixed = f"python ../runNet.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {num_workers} --GPU_selection {GPU_selection_fixed} --output {output} --share {share}"

subprocess.run(command_fixed, shell=True, check=True)

# Co-run the fixed model with other models
# for other_model in other_models:
#     print(f"Co-running fixed model: {fixed_model} and other model: {other_model}, each with batch size: {batch_size}, using 2 GPUs each")
    
GPU_selection_other = "2,3"

# command_fixed = f"taskset -c 0-32 python ../runNet.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {cores} --GPU_selection {GPU_selection_fixed} --output {output} --share {share}"
# command_other = f"taskset -c 32-64 python ../gat.py --GPU_selection {GPU_selection_other} --number_worker {cores}"

command_fixed = f" python ../runNet.py --model_name {fixed_model} --batch_size {batch_size} --GPU_selection {GPU_selection_fixed} --output {output} --share {share} --number_worker {num_workers}"
# command_other = f" python ../gat.py --GPU_selection {GPU_selection_other} --number_worker {num_workers}"
command_other = f" python /home/john/pytorch_geometric/benchmark/training/training_benchmark.py --models=gcn --datasets=Reddit --num-workers={num_workers} --batch-sizes=512 --num-layers=8 --num-hidden-channels=64 --num-steps=50 --num-epochs=20"


process_fixed = subprocess.Popen(command_fixed, shell=True)
process_other = subprocess.Popen(command_other, shell=True)

process_fixed.wait()
process_other.wait()

