import glob
import os
import subprocess
import time
import pandas as pd
import utils 
# basepath = "Q:\\EIN Group\\MartinHolub\\1mth-day1\\"
basepath = ""
file_name = "*.avi"
frame_ranges = pd.read_csv("start_stop.csv", sep = ";", skiprows = 2)

def adjust_filename(fname, ext_out):
    
    (head, tail) = os.path.split(fname)
    (base, ext) = os.path.splitext(tail)
    
    tail_out = base + ext_out
    return tail_out   

for i, fname in enumerate(list(glob.glob(basepath + file_name))):

    
    #base_call = "python motrack.py"
    #base_call = base_call + " -v " + fname
    
    base_call = ["python", "motrack.py"]
    base_call.extend(["-v", fname])
    
    base_lookup = adjust_filename(fname, ".MP4")
    frame_range = frame_ranges  [frame_ranges["movie"]==base_lookup]\
                                [["start [frame]", "stop [frame]"]].values[0]
    frame_range = utils.convert_list_str(frame_range)
    #base_call = base_call + " -fr " + frame_range
    
    base_call.extend(["-fr", frame_range])
    
    if i == 0:
        #call = base_call + " -p " + "params_init"
        base_call.extend(["-p", "params_init"])
        call = base_call
    if i > 0:
        call = base_call
       
    start_time = time.time()
    completed = subprocess.run(call, shell = True)    
    check_status = "returncode: {},".format(completed)
    timing = " time: {:.2f}s".format(time.time() - start_time)
    
    print(fname)
    print(check_status + timing)