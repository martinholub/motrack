#!/usr/bin/env python
import glob
import os
import subprocess
import time
import pandas as pd
import utils
import argparse

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reproduce", type = str,
                help= "reproduce process without user interaction", 
                default = 0)
try:
    args = ap.parse_args()
    skip_initialization = args.reproduce 
except Exception as ex: 
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
    skip_initialization = 0
    
basepath = "vids\\"
file_name = "*.avi"
csv_name = "vids\\start_stop.csv"

try:
    # must keep consistent separator and skiprows
    frame_ranges = pd.read_csv(csv_name, sep = ";", skiprows = 2)
except FileNotFoundError:
    print("ERROR:FileNotFoundError:{} file does ot exist".format(csv_name))
   
for i, fname in enumerate(list(glob.glob(basepath + file_name))):
 
    base_call = ["python", "motrack.py"]
    base_call.extend(["-v", fname])
    if frame_ranges is not None:
        base_lookup = utils.adjust_filename(fname, ".MP4")
        try:
            frame_range = frame_ranges  [frame_ranges["movie"]==base_lookup]\
                                        [["start [frame]", "stop [frame]"]].values[0]
        except IndexError: # video not annotated in csv
            print("Video `{}` not defined in CSV file.".format(fname))
            if i == 0:
                break
            else:
                continue
        frame_range = utils.convert_list_str(frame_range)        
        base_call.extend(["-fr", frame_range])
    
    if i == 0:
        if skip_initialization:
            continue; 
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