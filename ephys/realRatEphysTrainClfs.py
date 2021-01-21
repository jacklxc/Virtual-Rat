"""
This script trains (overfit) logistic regression classifier to predict each trials' rule 
encoding score and save them in to an array which will be converted to MATLAB table for fitting GLME.
"""
import numpy as np
from ratEphysFunctions import *
import scipy.io as sio
np.random.seed(seed=12345)
# Load data and filter out Duan or Erlich's session sessions
experimentor = "Erlich"
if experimentor == "Duan":
    time_steps = 5
    brain_areas = ["mPFC","SC","all"]
    session_var_name = 'Duan_sessions'
    session_file_name = '../mats/'+experimentor+'_sessions.mat'
    these_sessions = sio.loadmat(session_file_name)[session_var_name][0,:] - 1 # Difference between Python and MATLAB
elif experimentor == "Erlich":
    time_steps = 4
    brain_areas = ["mPFC","SC","FOF","all"]
    session_var_name = 'Erlich_sessions'
    session_file_name = '../mats/'+experimentor+'_sessions.mat'
    these_sessions = sio.loadmat(session_file_name)[session_var_name][0,:] - 1 # Difference between Python and MATLAB
else:
    time_steps = 5
    brain_areas = ["mPFC","SC","FOF","all"]
    session_var_name = 'Duan_sessions'
    these_sessions = np.ones(364).astype(bool)

SpikeCountsPerSession = sio.loadmat('../mats/SpikeCountsPerSession'+experimentor+'.mat')['SpikeCountsPerSession'][0,these_sessions]
SessionInfo = sio.loadmat('../mats/SessionInfo.mat')['SessionInfo'][0,these_sessions] # Pro, Right, switch, hit
ratindex = sio.loadmat('../mats/RatIndexPerSession.mat')['RatIndexPerSession'].T[0,these_sessions]
BrainArea = sio.loadmat('../mats/BrainArea.mat')['BrainArea'][0,these_sessions]

# Exclude sessions with bad behavior performance
good_session_indices = select_good_sessions(SessionInfo,threshold = 0.67,time_steps=time_steps)
goodSpikeCountsPerSession = SpikeCountsPerSession[good_session_indices]
good_ratindex = ratindex[good_session_indices]
good_BrainArea = BrainArea[good_session_indices]
goodSessionInfo = SessionInfo[good_session_indices]

# Normalize spike counts
good_all_normalized_spike_count = normalize_spike_count(goodSpikeCountsPerSession,time_steps=time_steps)

# Split spike counts by brain area
spike_count_by_area = byBrainArea(good_all_normalized_spike_count,good_BrainArea)

for brain_area in brain_areas:
    print "Computing: "+brain_area 
    # Train over-fitted logistic regression classifiers
    spike_count = spike_count_by_area[brain_area]
    print "Training SGD clfs"
    accuracies, clfs = train_SGD(spike_count, goodSessionInfo,time_steps=time_steps)
    print "Making tables"
    normalized_table, session_table = make_tables(spike_count, goodSessionInfo,good_ratindex, clfs, 
        normalize = True, time_steps=time_steps)
    SGD_table_name = '../mats/'+experimentor+'SGD_table_'+"_".join(brain_area.split())+'.mat'
    session_table_name = '../mats/'+experimentor+'SGD_session_table_'+"_".join(brain_area.split())+'.mat'
    sio.savemat(SGD_table_name,{'SGD_table':normalized_table})
    sio.savemat(session_table_name,{'SGD_table':session_table})