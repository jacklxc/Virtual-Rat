#!/usr/bin/env python
"""
This script tests the performance of the weights trained by TrainingTimeRatio.py and save the performance in to pkl files.
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
from VirtualRatFunctions import *

parser = argparse.ArgumentParser()
parser.add_argument("index", help="job_array_index",
                    type=int)
args = parser.parse_args()

start = args.index/11 * 10 
switch_index = args.index%11

epoch_per_loop = 100
num_loop = 100
rats = []
switch_ratio = switch_index/10.0
end = start + 10

load_directory = "/scratch/xl1066/VirtualRat/publication/RatioTime/"

for j in range(start, end):
    ratname = 'VirtualRat'+str(j)
    np.random.seed(j)
    npp.random.seed(j)
    box = VirtualRatBox(mode="alternative",length=1000,block_size=30)
    test_X, test_y = box.X,box.y
    print ratname
    try:
        loaded_weights = load_weights(load_directory+"trainedTrainingTimeRatio-"+ str(switch_ratio)+"-"+str(j)+"-"+str(0)+".pkl")
        print ratname
    except IOError:
        continue
    model = VirtualRatModel()
    rat = VirtualRat(model)
    solver = VirtualRatSolver(model, box)
    for i in range(num_loop+1):
        print i*epoch_per_loop
        loaded_weights = load_weights(load_directory+"trainedTrainingTimeRatio-"+ str(switch_ratio)+"-"+str(j)+"-"+str(i*epoch_per_loop)+".pkl")
        np.random.seed(j)
        npp.random.seed(j)
        solver.init()
        solver.load_params(loaded_weights)
        probs = rat.predict(test_X, test_y)
        rat.add_prediction_history()
        
    rats.append(rat)
pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = accuracy_vs_time_make_matrix(rats,num_loop+1,exclude = False)
matrices = pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix
save_weights("/gpfsnyu/home/xl1066/VirtualRat/publication/RatioTime/RatioTime-"+str(switch_ratio)+"-"+str(end)+".pkl",matrices)