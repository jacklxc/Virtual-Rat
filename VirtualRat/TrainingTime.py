#!/usr/bin/env python
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

parser = argparse.ArgumentParser()
parser.add_argument("index", help="job_array_index",
                    type=int)
args = parser.parse_args()

epoch_per_loop = 100
num_loop = 100
pre_train = 30
for j in [args.index]:
    ratname = 'VirtualRat'+str(j)
    print ratname
    np.random.seed(j)
    npp.random.seed(j)
    model = VirtualRatModel()
    rat = VirtualRat(model)
    box = VirtualRatBox(mode="no_rule",length=500000,block_size=30,
                trial_per_episode=30, repeat = False, p2a = 0.5,
                block_correction = True, left_right_correction = True)
    solver = VirtualRatSolver(model, box,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': 5e-3,
                            'decay_rate': 1
                        },
                        init_rule='xavier',
                        num_episodes=epoch_per_loop,
                        verbose=False,
                        supervised = True,
                        stop = False,
                        print_every=50)

    solver.init()
    solver.train()
    params = solver.save_params()
    save_weights("/scratch/xl1066/VirtualRat/poster/TrainingTime/trainedTrainingTime-"+str(j)+'-'+str(0)+".pkl",params)
    for i in range(1,pre_train+1):
        solver.train()
        params = solver.save_params()
        save_weights("/scratch/xl1066/VirtualRat/poster/TrainingTime/trainedTrainingTime-"+str(j)+'-'+str(i*epoch_per_loop)+".pkl",params)
        
    box.change_mode("alternative")
    for i in range(1,num_loop - pre_train+1):
        solver.train()
        params = solver.save_params()
        save_weights("/scratch/xl1066/VirtualRat/poster/TrainingTime/trainedTrainingTime-"+str(j)+'-'+str((i+pre_train)*epoch_per_loop)+".pkl",params)
