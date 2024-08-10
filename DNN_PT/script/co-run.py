import subprocess

output = "./corun_res/corun_ResNet50_512_4_2_2.csv"
fixed_model = "ResNet-50"
batch_size = 512
num_workers = 4
GPU_selection_fixed = "0,1"
other_models=["LeNet", "ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "AlexNet"]
share = 1

# Run the fixed model alone on 2 GPUs
print(f"Running fixed model: {fixed_model}, batch size: {batch_size}, using GPUs: {GPU_selection_fixed}")

command_fixed = f"python ../corunTest.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {num_workers} --GPU_selection {GPU_selection_fixed} --output {output} --share {share}"
subprocess.run(command_fixed, shell=True, check=True)

# Co-run the fixed model with other models
for other_model in other_models:
    print(f"Co-running fixed model: {fixed_model} and other model: {other_model}, each with batch size: {batch_size}, using 2 GPUs each")
    
    GPU_selection_other = "2,3"
    
    command_fixed = f"python ../runNet.py --model_name {fixed_model} --batch_size {batch_size} --number_worker {num_workers} --GPU_selection {GPU_selection_fixed} --output {output} --share {share}"
    command_other = f"python ../runNet.py --model_name {other_model} --batch_size {batch_size} --number_worker {num_workers} --GPU_selection {GPU_selection_other} --output {output} --share {share}"
    
    process_fixed = subprocess.Popen(command_fixed, shell=True)
    process_other = subprocess.Popen(command_other, shell=True)
    
    process_fixed.wait()
    process_other.wait()
