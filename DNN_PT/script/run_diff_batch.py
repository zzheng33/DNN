import subprocess

batch_sizes = [16, 32, 64, 128, 256, 512]
models = ["ResNet-50", "ResNet-101", "VGG-16", "Inception-V3", "AlexNet.py", "LeNet.py"]

for model in models:
    for batch_size in batch_sizes:
        print(f"Running training with model: {model}, batch size: {batch_size}")
        
        command = f"python ../runNet.py --model-name {model} --batch_size {batch_size} --number_worker 4 --GPU_selection 0,1,2"
        
        subprocess.run(command, shell=True, check=True)
