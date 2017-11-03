#!/usr/bin/env python
import glob
import os
import utils
import datetime
import re

basepath = "res/"
fnames_in = "*.txt"
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fname_out = "res_" + time_stamp + ".txt"

# distance_re = re.compile("Total dist in mm:")
# time_re = re.compile("Total time in sec:")
# warn_re = re.compile("")
numeric_re = re.compile(r"[0-9]+\.[0-9]+$")

with open(fname_out, "w") as f_out:
    f_out.write("fname,dist[mm],time[s],#warns\n")
    
    for i, fn in enumerate(list(glob.glob(basepath + fnames_in))):
        base = utils.adjust_filename(fn, "")
        warnings = []
        with open(fn) as f_in:
            f_out.write("{},".format(base))
                        
            for line in f_in:
                if line.startswith("Total dist"):
                    dist = numeric_re.search(line).group(0)
                    f_out.write("{},".format(dist))
                if line.startswith("Total time"):
                    time = numeric_re.search(line).group(0)
                    f_out.write("{},".format(time))
                if line.startswith("WARNING"):
                    warnings.append(line)
                
            if warnings:
                num_warnings = len(warnings)
                f_out.write("{:d}\n".format(num_warnings))
            else:
                f_out.write("{:d}\n".format(0)) 
        # f.close is implicit
    