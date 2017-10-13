import glob
import os
import subprocess
import time
# basepath = "Q:\\EIN Group\\MartinHolub\\1mth-day1\\"
basepath = ""
file_name = "*.m4v"

for i, fname in enumerate(list(glob.glob(basepath + file_name))):
    base_call = "python motrack.py"
    base_call = base_call + " -v " + fname
    
    if i == 0:
        continue
        call = base_call + " -p " + "params_init"
    if i > 0:
        call = base_call
       
    start_time = time.time()
    completed = subprocess.run(call)    
    check_status = "returncode: {},".format(completed)
    timing = " time: {:.2f}s".format(time.time() - start_time)
    
    print(fname)
    print(check_status + timing)
    
    if i>0: break;