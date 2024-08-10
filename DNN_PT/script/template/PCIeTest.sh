#!/bin/bash

num_gpus=(0 1 2 3)


for gpu in "${num_gpus[@]}"
do
  nsys profile -o ../nsys_res/nsys_output_gpu_${gpu} -t cuda,nvtx --stats=true --force-overwrite=true --gpu-metrics-device=all python ../runNet.py --GPU_selection ${gpu}  
 
done


