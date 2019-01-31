import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Compile the library (won't run if nothing has changed)
import os
cwd = os.getcwd()
os.chdir('../python/')
import subprocess
out = subprocess.run(["python", "setup.py", "build_ext", "--inplace"], stdout=subprocess.PIPE)
os.chdir(cwd)

import sys
sys.path.append('../python/')
import gropt
from helper_utils import *

import hdf5storage
from tqdm import tqdm


def compare_waveforms(G0, G1):
    if not (G0.size == G1.size):
        print('ERROR: output waveforms are not the same size')
    
    res0 = np.linalg.norm(G0-G1) 
    res1 = np.linalg.norm(G0+G1)
    min_res = min(res0, res1)
    
    rel_err = min_res/np.linalg.norm(G0)

    if (rel_err > 1e-3):
        print('ERROR: output waveforms are different')


def run_testcase(casefile):

    data = hdf5storage.read(filename=casefile)
    dt = data['params_in']['dt']
    N = data['params_in']['N']
    gmax = data['params_in']['gmax']
    smax = data['params_in']['smax']
    MMT = data['params_in']['MMT']
    TE = data['params_in']['TE']
    T_readout = data['params_in']['T_readout']
    T_90 = data['params_in']['T_90']
    T_180 = data['params_in']['T_180']
    diffmode = data['params_in']['diffmode']

    if (dt < 0) and (N < 0):
        print('ERROR: dt or N needs to be set for test case')
    elif (dt > 0):
        start = timer()
        G, dd = gropt.run_diffkernel_fixdt(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, dt=dt, verbose=0)
        end = timer()
    elif (N > 0):
        start = timer()
        G, dd = gropt.run_diffkernel_fixN(gmax, smax, MMT, TE, T_readout, T_90, T_180, diffmode, N0=N, verbose=0)
        end = timer()

    compare_waveforms(data['G'], G)


import os

all_cases = []

for root, dirs, files in os.walk('./cases/'):
    for f in files:
        if f.endswith('.h5'):
             all_cases.append(os.path.join(root, f))

for i in tqdm(range(len(all_cases)), desc='Test Progress', ncols=110):
    run_testcase(all_cases[i])