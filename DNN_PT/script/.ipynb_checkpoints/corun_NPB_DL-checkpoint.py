import subprocess
import time
import csv
import os


def count_cores(cores_string):
    core_ranges = cores_string.split(',')
    num_cores = 0
    for core_range in core_ranges:
        if '-' in core_range:
            start, end = map(int, core_range.split('-'))
            num_cores += end - start + 1
        else:
            num_cores += 1
    return num_cores

dl_model = "ResNet-50"
dl_epochs = 20
dl_gpus = "0,1"
dl_cores = "0-7,32-39,8-15,40-47"
dl_output = "../result/test_DL.csv"
batch_size=512

npb_app = "is.D.x"
# npb_cores = "16-23,48-55,24-31,56-63"
npb_cores = "16-23,48-55,24-31,56-63"
npb_output = "../result/test_OMP.csv"


num_dl_cores = count_cores(dl_cores)
num_npb_cores = count_cores(npb_cores)

# Run the Deep Learning job
print(f"Running {dl_model} with {dl_cores} CPU cores and 2 GPUs for {dl_epochs} epochs")
dl_command = f"taskset -c {dl_cores} python ../runNet.py --model_name {dl_model} --batch_size {batch_size} --number_worker {num_dl_cores} --GPU_selection {dl_gpus} --epoch {dl_epochs} --output {dl_output} &"
subprocess.run(dl_command, shell=True, check=True)
time.sleep(60)

# Run the NPB application
print(f"Running {npb_app} with {num_npb_cores} cores")
os.environ['OMP_NUM_THREADS'] = str(num_npb_cores)
npb_command = f"taskset -c {npb_cores} ../NPB3.4.2/NPB3.4-OMP/bin/{npb_app}"


start_time = time.time()
subprocess.run(npb_command, shell=True, check=True)
end_time = time.time()

running_time = end_time - start_time

# Record the NPB application running time to the CSV file
file_exists = os.path.isfile(npb_output)

with open(npb_output, "a", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    if not file_exists:
        csv_writer.writerow(["Application", "Cores", "Running Time"])

    csv_writer.writerow([npb_app, num_npb_cores, running_time])
