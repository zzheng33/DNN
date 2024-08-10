import subprocess

fixed_model = "ResNet-101"
batch_size = 512
num_workers = 4
GPU_selection_fixed = "0,1"
other_models = ["LeNet", "ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "AlexNet"]

share = 1
cores = 32

# Run the fixed model alone on 2 GPUs
output = f"corun_shareCPU_res/corun_diffData_{fixed_model}_{batch_size}_{cores}_2_2.csv"
print(f"Running fixed model: {fixed_model}, batch size: {batch_size}, using GPUs: {GPU_selection_fixed}")

command_fixed = f"taskset -c 0-{cores} python ../runNet.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {cores} --GPU_selection {GPU_selection_fixed} --output {output} --share {share} --epoch 5"
subprocess.run(command_fixed, shell=True, check=True)

# Co-run the fixed model with other models
for other_model in other_models:
    print(f"Co-running fixed model: {fixed_model} and other model: {other_model}, each with batch size: {batch_size}, using 2 GPUs each")
    
    GPU_selection_other = "2,3"
    
    command_fixed = f"taskset -c 0-{cores} python ../corunTest.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {cores} --GPU_selection {GPU_selection_fixed} --output {output} --share {share} --epoch 20"
    command_other = f"taskset -c 0-{cores} python ../corunTest.py --model_name {other_model} --batch_size {batch_size} --number_worker {cores} --GPU_selection {GPU_selection_other} --output {output} --share {share}"
    
    process_fixed = subprocess.Popen(command_fixed, shell=True)
    process_other = subprocess.Popen(command_other, shell=True)
    
    process_fixed.wait()
    process_other.wait()
