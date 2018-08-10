"""
This script trains (overfit) logistic regression classifier to predict each trials' target 
encoding score and save them in to an array which will be converted to MATLAB table for fitting GLME.
"""

import numpy as np
from ratEphysFunctions import *
import scipy.io as sio

SpikeCountsPerSession = sio.loadmat('mats/SpikeCountsPerSession.mat')['SpikeCountsPerSession']
SessionInfo = sio.loadmat('mats/SessionInfo.mat')['SessionInfo'] # Pro, Right, switch, hit
ratindex = sio.loadmat('mats/RatIndexPerSession.mat')['RatIndexPerSession'].T[0,:]
CellIndexPerSession = sio.loadmat('mats/CellIndexPerSession.mat')['CellIndexPerSession'][0,:]

# Normalize spike counts
all_normalized_spike_count = normalize_spike_count(SpikeCountsPerSession)

# Train over-fitted logistic regression classifiers
accuracies, clfs = train_SGD(all_normalized_spike_count, SessionInfo, verbose=True, target=True)

good_SGD_indices = select_good_sessions(SessionInfo)

normalized_table, session_table = make_tables(all_normalized_spike_count, SessionInfo,ratindex, clfs, 
	good_SGD_indices, normalize = True, verbose=True, target = True, time_steps=5)

sio.savemat('mats/SGD_table_target_normalized.mat',{'SGD_table':normalized_table})
sio.savemat('mats/SGD_session_table_target.mat',{'SGD_table':session_table})