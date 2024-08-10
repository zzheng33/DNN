#!/bin/bash

batch_size=256
models=("LeNet" "ResNet-50" "ResNet-101" "VGG-16" "Inception-V3" "AlexNet")


num_gpus_range=(1 2 3 4)
num_workers=4
share=1

for model in "${models[@]}"
do
    for num_gpus in "${num_gpus_range[@]}"
    do
        echo "Running training with model: $model, batch size: $batch_size, using $num_gpus GPUs, using $num_workers workers"
        GPU_selection=$(seq -s, 0 $((num_gpus-1)))
        output_file="diff_gpu_${batch_size}_${num_workers}.csv"
        python ../runNet.py --model_name "$model" --batch_size "$batch_size" --number_worker "$num_workers" --GPU_selection "$GPU_selection" --output "$output_file" --share 1
    done
done


