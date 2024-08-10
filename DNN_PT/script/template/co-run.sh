#!/bin/bash

output="./corun_res/corun_ResNet50_256_4_2_2.csv"
fixed_model="ResNet-152"
batch_size=350
num_workers=64
GPU_selection_fixed="0,1"
other_models=("ResNet-50" "LeNet" "ResNet-101" "VGG-16" "Inception-V3" "AlexNet")
# other_models=("ResNet-101")
share=1


# Run the fixed model alone on 2 GPUs

echo "Running fixed model: $fixed_model, batch size: $batch_size, using GPUs: $GPU_selection_fixed"
python ../corunTest.py --model_name "$fixed_model" --batch_size "$batch_size" --number_worker "$num_workers" --GPU_selection "$GPU_selection_fixed" --output "$output" --share $share

# Co-run the fixed model with other models
for other_model in "${other_models[@]}"
do
    echo "Co-running fixed model: $fixed_model and other model: $other_model, each with batch size: $batch_size, using 2 GPUs each"
    
    GPU_selection_other="2,3"
    python ../corunTest.py --model_name "$fixed_model" --batch_size "$batch_size" --number_worker "$num_workers" --GPU_selection "$GPU_selection_fixed" --output "$output"  --share $share &
    python ../corunTest.py --model_name "$other_model" --batch_size "$batch_size" --number_worker "$num_workers" --GPU_selection "$GPU_selection_other" --output "$output" --share $share &

    wait
done
