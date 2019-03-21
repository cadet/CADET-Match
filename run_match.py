# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:22:44 2018

@author: kosh_000
"""

import sys
import subprocess
import time
from pathlib import Path


#Simulation Values
json_file = Path(r"C:\Users\kosh_000\Documents\Visual Studio 2017\Projects\CADETMatch\Examples\Example1\Dextran\NSGA3_dextran.json")
cadet_match_location = Path(r"C:\Users\kosh_000\Documents\Visual Studio 2017\Projects\CADETMatch\CADETMatch\CADETMatch.py")
python_path = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Anaconda3_64\python.exe")
default_cpus = 6

#Default Values
enable_parallelization = True

try:
    import psutil
    cpus = psutil.cpu_count(logical=False)
except ImportError:
    cpus = default_cpus

if python_path.stem == 'pythonw':
    python_path.stem = 'python'

def main():
    command = [str(python_path),]
    if enable_parallelization:
        command.extend(['-m', 'scoop', '-n', str(cpus)])
    command.append(str(cadet_match_location))
    command.append(str(json_file))
    print(command)
    rc = subprocess.run(command)
    print("Retrun Code %s" % rc)
    
if __name__ == "__main__":
    main()
    a = input()
