#!/usr/bin/env python
"""
This file trains RNN agents on target-only pre-training and Pro-Anti Orienting task.
It saves the weights of all agents per epoch_per_loop = 100.
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

# This part is used to run this code in parallel with HPC
parser = argparse.ArgumentParser()
parser.add_argument("index", help="job_array_index",
                    type=int)
args = parser.parse_args()

save_directory = "/scratch/xl1066/VirtualRat/publication/TrainingTime/"

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
    save_weights(save_directory+"trainedTrainingTime-"+str(j)+'-'+str(0)+".pkl",params)
    for i in range(1,pre_train+1):
        solver.train()
        params = solver.save_params()
        save_weights(save_directory+"trainedTrainingTime-"+str(j)+'-'+str(i*epoch_per_loop)+".pkl",params)
        
    box.change_mode("alternative")
    for i in range(1,num_loop - pre_train+1):
        if i>=1 and i<=5: # Specially record every epoch between 3000 and 3500 epochs.
            solver.change_settings(num_episodes = 2) # num_episodes must be even number, to include pro and anti blocks.
            for ii in range(epoch_per_loop/2):
                solver.train()
                params = solver.save_params()
                save_weights(save_directory+"trainedTrainingTime-"+str(j)+'-'+str((i+pre_train-1)*epoch_per_loop+(ii+1)*2)+".pkl",params)
        else:
            solver.change_settings(num_episodes = epoch_per_loop)
            solver.train()
            params = solver.save_params()
            save_weights(save_directory+"trainedTrainingTime-"+str(j)+'-'+str((i+pre_train)*epoch_per_loop)+".pkl",params)
