import subprocess
import time
import csv
import os

applications = ["cg.W.x", "ep.W.x", "mg.W.x"]
applications = ["is.D.x"]

cores_list = [64,60,56,52,48,44,40,36,32,16,8,4]
csv_file = "../result/NPB_result/npb_omp_is_D.csv"

file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    
    if not file_exists:
        csv_writer.writerow(["Application", "Cores", "Running Time"])

    for app in applications:
        for cores in cores_list:
            print(f"Running {app} with {cores} cores")
            
            os.environ['OMP_NUM_THREADS'] = str(cores)
            # command = f"taskset -c 0-{cores - 1} /home/john/NPB-CPP-master/NPB-OMP/bin/{app}"
            command = f"taskset -c 0-{cores - 1} ../NPB3.4.2/NPB3.4-OMP/bin/{app}"
            start_time = time.time()
            subprocess.run(command, shell=True, check=True)
            end_time = time.time()
            
            running_time = end_time - start_time
            print(f"{app} completed in {running_time} seconds with {cores} cores")
            
            csv_writer.writerow([app, cores, running_time])
            csvfile.flush()  # Flush the data to the file after each row is written
