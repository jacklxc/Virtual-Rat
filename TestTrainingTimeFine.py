#!/usr/bin/env python
"""
This code tests the performance of the agent trained before.
For each set of weights, load them and test the performance and it save to .pkl file.
"""
import argparse

import numpy as npp
import minpy.numpy as np
import cPickle
import matplotlib.pyplot as plt
import minpy
minpy.set_global_policy('only_numpy')

from VirtualRatModel import VirtualRatModel
from VirtualRatSolver import VirtualRatSolver
from VirtualRatBox import VirtualRatBox
from VirtualRat import VirtualRat
from dataProcessFunctions import *

# To run in parallel on HPC 
parser = argparse.ArgumentParser()

parser.add_argument("index", help="job_array_index",
                    type=int)
args = parser.parse_args()

load_directory = "/scratch/xl1066/VirtualRat/publication/TrainingTime/"

epoch_per_loop = 2
num_loop = 250
rats = []

for j in [args.index]:
    ratname = 'VirtualRat'+str(j)
    print ratname
    np.random.seed(j)
    npp.random.seed(j)
    box = VirtualRatBox(mode="alternative",length=1000,block_size=30)
    test_X, test_y = box.X,box.y
    model = VirtualRatModel()
    rat = VirtualRat(model)
    solver = VirtualRatSolver(model, box)
    for i in range(num_loop+1):
        print i*epoch_per_loop
        try:
            loaded_weights = load_weights(load_directory+"trainedTrainingTime"+"-"+str(j)+"-"+str(3000+i*epoch_per_loop)+".pkl")
        except IOError:
            continue
        np.random.seed(j)
        npp.random.seed(j)
        solver.init()
        solver.load_params(loaded_weights)
        probs = rat.predict(test_X, test_y)
        rat.add_prediction_history()
        trial_window = 3

    rats.append(rat)
# Save to pkl file.
pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = accuracy_vs_time_make_matrix(rats,num_loop+1,exclude = False)
matrices = pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix
save_weights("/gpfsnyu/home/xl1066/VirtualRat/publication/TrainingTime/TrainingTimeFine-"+str(j)+".pkl",matrices)