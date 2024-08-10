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

# npb_apps = ["cg.W","cg.S","cg.B","cg.C","ep.W","ep.S","ep.B","ep.C","mg.W","mg.S","mg.B","mg.C"]
npb_apps = ["is.S.x","is.W.x","is.A.x","is.B.x","is.C.x","is.D.x"]
npb_apps = ["dc.A.x"]
npb_cores_pack = "16-23,48-55,24-31,56-63"
npb_cores_spread = "0-31"
cores_dist = [npb_cores_pack,npb_cores_spread]

npb_output = "../result/NPB_result/affinity.csv"

num_npb_cores = count_cores(npb_cores_spread)

for npb_app in npb_apps:
    # Run the NPB application
    for i in range(2):
        npb_cores = cores_dist[i]
        print(f"Running {npb_app} with {num_npb_cores} cores")
        os.environ['OMP_NUM_THREADS'] = str(num_npb_cores)
        
        npb_command = f"taskset -c {npb_cores} ../NPB3.4.2/NPB3.4-OMP/bin/{npb_app}"
        start_time = time.time()
        try:
            subprocess.run(npb_command, shell=True, check=True)
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            running_time = end_time - start_time
             # Record the NPB application running time to the CSV file
            file_exists = os.path.isfile(npb_output)

            with open(npb_output, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)

                if not file_exists:
                    csv_writer.writerow(["Application", "Cores", "Running Time", "Spread"])

                csv_writer.writerow([npb_app, num_npb_cores, running_time, i])

            npb_command = f"rm *.0"
            subprocess.run(npb_command, shell=True, check=True)
            print(f"Error while running command '{npb_command}': {e}")
            continue

        

       