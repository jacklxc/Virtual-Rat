import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
from tqdm import *

from scipy.stats import zscore, binned_statistic, ttest_rel
from sklearn.linear_model import LinearRegression, SGDClassifier
from scipy.stats import ttest_ind as t_test
from sklearn.metrics import roc_auc_score
import scipy.io as sio
from scipy.optimize import curve_fit

FONT_SIZE = 15

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
plt.rc('legend',fontsize=FONT_SIZE) # using a size in points

def normalize_spike_count(SpikeCountsPerSession, time_steps=5):
    """
    Normalize the spike counts by z-scoring across trials, so each cell is independent so far.
    SpikeCountsPerSession, all_normalized_spike_count: list of sessions. 
    Each session is a numpy ndarray of shape (time_step, num_cell, num_trial)
    """
    num_session = SpikeCountsPerSession.size
    all_normalized_spike_count = []
    for session in range(num_session):
        normalized_spike_count = np.zeros(SpikeCountsPerSession[session].shape)
        for t in range(time_steps):
            trial_num = SpikeCountsPerSession[session][t,:,:].shape[-1]
            means = np.mean(SpikeCountsPerSession[session][t,:,:],axis=1)
            stds = np.std(SpikeCountsPerSession[session][t,:,:],axis=1)
            tiled_mean = np.tile(means,(trial_num,1)).T
            tiled_std = np.tile(stds,(trial_num,1)).T
            normalized_spike_count[t,:,:] = (SpikeCountsPerSession[session][t,:,:] - tiled_mean) / tiled_std
        normalized_spike_count[np.isnan(normalized_spike_count)] = 0 
        all_normalized_spike_count.append(normalized_spike_count)
    return all_normalized_spike_count

def byBrainArea(spike_count, BrainArea):
    """
    Split spike_count according to different brain area.
    Index: brain area
    0: left mPFC
    1: right mPFC
    2: left SC
    3: right SC
    4: left FOF
    5: right FOF
    """
    #brain_area_side = ["left mPFC","right mPFC","left SC","right SC","left FOF","right FOF"]
    brain_area = ["mPFC","SC","FOF"]
    spike_count_by_area = {}
    spike_count_by_area["all"] = spike_count
    num_session = len(spike_count)
    #for area_index in range(len(brain_area_side)): # 6 brain areas, including left and right side
    #    area_spike_count = []
    #    for session in range(num_session):
    #        this_area = BrainArea[session][0,:]==area_index
    #        area_spike_count.append(spike_count[session][:][:,this_area,:])
    #    spike_count_by_area[brain_area_side[area_index]] = area_spike_count

    for area_index in range(len(brain_area)): # 3 brain areas
        area_spike_count = []
        for session in range(num_session):
            this_area = np.logical_or(BrainArea[session][0,:]==2*area_index,BrainArea[session][0,:]==2*area_index+1)
            area_spike_count.append(spike_count[session][:,this_area,:])
        spike_count_by_area[brain_area[area_index]] = area_spike_count

    return spike_count_by_area


def train_SGD(all_normalized_spike_count, SessionInfo, time_steps=5, target = False):
    """
    Train (overfit) SGDClassifier (logistic regression classifier) using correctly responded block trials.
    all_normalized_spike_count: list of sessions. Each session is a numpy ndarray of shape (time_step, num_cell, num_trial)
    """
    num_session = len(all_normalized_spike_count)
    accuracies = []
    clfs = []
    for session in tqdm(range(num_session)):
        for t in range(time_steps):
            if all_normalized_spike_count[session].size>0:
                trial_num = all_normalized_spike_count[session].shape[-1]
                X = all_normalized_spike_count[session][t,:,:].T
                if np.product(X==0)==1:
                    # Delay time step from Erlich's dataset, which should not be considered.
                    accuracies.append(None)
                    clfs.append(None)
                else:
                    if target:
                        pro = SessionInfo[session][:,1] # This pro actually means right!
                    else:
                        pro = SessionInfo[session][:,0]
                    hit = SessionInfo[session][:,-1]==1
                    accs = []
                    best_acc = 0
                    repeat = 100
                    for r in range(repeat):
                        clf = SGDClassifier(loss = "log", fit_intercept = False, penalty = "elasticnet", learning_rate = "optimal")
                        clf.fit(X[hit,:], pro[hit])
                        y_pred = clf.predict(X)
                        acc = np.mean(y_pred == pro)
                        accs.append(acc)
                        if acc > best_acc:
                            best_acc = acc
                            best_clf = clf
                    accuracies.append(np.mean(accs))
                    clfs.append(clf)
            else: # This brain area is not recorded for this session
                # But still put a placeholder
                accuracies.append(None)
                clfs.append(None)
    return accuracies, clfs

def select_good_sessions(SessionInfo, time_steps=5, threshold = 0.7):
    """
    Select good session indices by picking sessions with both pro and anti block perfornmance >= threshold.
    """
    num_session = SessionInfo.size
    good_sessions = []
    for session in range(num_session):
        pro_acc = np.sum(np.logical_and(SessionInfo[session][:,-1] ==1,  SessionInfo[session][:,0]==1))/\
                    float(np.sum(SessionInfo[session][:,0]==1))
        anti_acc = np.sum(np.logical_and(SessionInfo[session][:,-1] ==1,  SessionInfo[session][:,0]==0))/\
                float(np.sum(SessionInfo[session][:,0]==0))
        good_sessions.append(np.logical_and(pro_acc>=threshold,anti_acc>=threshold))
    good_session_indices = np.where(good_sessions)[0]
    return good_session_indices

def test_clf(clf, X_test, y_test):
    """
    Test the performance of the Classifiers.
    """
    if clf is not None:
        y_pred = clf.predict(X_test)
        acc = np.mean(y_pred == y_test)
        score = clf.decision_function(X_test)
    else: # For cases that the classifier should be ignored
        acc = 0
        score = np.zeros(y_test.shape)
    pro_score = score[y_test>0]
    anti_score = score[y_test==0]
    return acc, score, pro_score, anti_score

def make_tables(all_normalized_spike_count, SessionInfo, ratindex, clfs, 
    normalize = False, target=False, time_steps=5):
    """
    Make an array (later converted into MATLAB table) to store SGDClassifier's predictions and rule/target AUC.
    normalized_table has each trial as each row.
    session_table has each session * each time_step as each row.
    """
    categories = 21 # session_index, pro, right, switch, hit, rat_index, (index 0~5)
                    # score0, score1, score2, score3, score4, (index 6~10)
                    # accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, (index 11~15)
                    # encoding0, encoding1, encoding2, encoding3, encoding4 (index 16~20)
    normalized_table = np.zeros((0,categories))
    session_categories = 6
    session_table = np.zeros((0,session_categories)) # session_index, time_step, accuracy, auc, p, rat_index
    for session in tqdm(range(len(all_normalized_spike_count))):
        if all_normalized_spike_count[session].size>0:
            trial_num = all_normalized_spike_count[session].shape[-1]
            sub_table = np.zeros((trial_num,categories))
            sub_table[:,0] = session # session_index

            for t in range(time_steps):
                SGD_clf = clfs[session*time_steps + t]
                sub_session_table = np.zeros((1,session_categories))
                sub_session_table[:,0] = session
                sub_session_table[:,1] = t
                X = all_normalized_spike_count[session][t,:,:].T
                if target:
                    pro = SessionInfo[session][:,1]==1 # This pro actually means right!
                else:
                    pro = SessionInfo[session][:,0]==1
                anti = np.logical_not(pro)

                SGD_acc, SGD_score, SGD_pro_score, SGD_anti_score = test_clf(SGD_clf,X,pro)
                auc, p, _ = bootroc(SGD_pro_score,SGD_anti_score)


                # Normalize encoding score here
                if normalize:
                    #SGD_score[pro] = (SGD_score[pro] - np.mean(SGD_pro_score)) / np.std(SGD_pro_score)
                    #SGD_score[anti] = (SGD_score[anti] - np.mean(SGD_anti_score)) / np.std(SGD_anti_score)
                    SGD_score = (SGD_score - np.mean(SGD_score)) / np.std(SGD_score)
                
                
                sub_table[:,1:5] = SessionInfo[session] # Pro, Right, switch, hit
                sub_table[:,6+t] = SGD_score
                sub_table[:,11+t] = SGD_acc
                sub_session_table[:,2] = SGD_acc
                sub_session_table[:,-3] = auc
                sub_session_table[:,-2] = p
                sub_table[:,16+t] = pro * SGD_score + (pro-1) * SGD_score # encoding score, aka. same_score
                sub_table[:,5] = ratindex[session]
                sub_session_table[:,-1] = ratindex[session]
                session_table = np.append(session_table,sub_session_table,axis=0)
            normalized_table = np.append(normalized_table,sub_table,axis=0)
    return normalized_table, session_table

def bootroc(A,B, BOOTS=1000, CI=99, ttest = False):
    """
    Does bootstrapping to compute the significance of ROC. Translated from Erlich lab elutils bootroc.m 

    Inputs:
    - A, B: series of inputs with different labels.
    - BOOTS: Number of repetition.
    - CI: confidence level percentage.
    - ttest: boolean, whether to use ttest to replace permutation test.

    Returns:
    All float.
    - sd: Area under curve of the certain hidden unit at a certain time step.
    - sd_p: p-value from bootstrapping.
    - confidence_interval: confidence interval.
    """
    sd = sklearn_auc(A,B)
    sA = A.size
    if ttest:
        _, sd_p = t_test(A,B)
        confidence_interval = [0,0]
    else:
        all_data = np.append(np.reshape(A, A.size), np.reshape(B, B.size))
        boot_score = 0.5 + np.zeros(BOOTS)
        for bx in range(BOOTS):
            shuff_d = np.random.permutation(all_data)
            A = shuff_d[:sA]
            B = shuff_d[sA:]
            boot_score[bx] = sklearn_auc(A,B)

        sd_p = get_p(sd,boot_score)

        half = (100 - CI)/2
        confidence_interval = np.percentile(boot_score,[(100-CI-half),CI+half])
    return sd, sd_p, confidence_interval


def sklearn_auc(stim,nostim):
    """
    Use module from scikit learn to compute area under curve.

    Inputs:
    - stim: numpy array in shape (a,n)
    - nostim: numpy array in shape (b,n)
    """
    labels = np.append(np.ones((stim.size)), np.zeros((nostim.size)))
    values = np.append(np.reshape(stim, stim.size), np.reshape(nostim, nostim.size))
    return roc_auc_score(labels, values)
    
def get_p(datum, dist, tails = 2, high=1, precision = 100):
    """
    Calculate p-value that datum is from dist.
    If tails == 1 , then p is the prob that datum
    is higher (if high==True) or lower (if high==False) than dist.

    Inputs:
    - datum: a int or float
    - dist: numpy array of shape (m,)
    (n can be 1)
    """
    half_precision = precision / 2
    dist = np.reshape(dist,dist.size)
    ps = np.linspace(0,precision, dist.size, endpoint = True)
    sd_ps = np.percentile(dist, ps)
    closest = np.searchsorted(sd_ps,datum)
    if tails==2:
        if closest <= 0 or closest>=sd_ps.size:
            others = np.array([0])
        else:
            others = np.where(sd_ps == sd_ps[closest])[0]

        if ps[others[0]] <half_precision and ps[others[-1]]>half_precision:
            sd_p = 1
        elif datum < sd_ps[0] or datum > sd_ps[-1]:
            sd_p = 2 / dist.size
        elif ps[others[0]] > half_precision:
            sd_p = ps[others[0]] / precision
            sd_p = np.max([2*(1-sd_p), 2/dist.size])
        else:
            sd_p = ps[others[-1]] / precision
            sd_p = 2 * sd_p
    else:
        # if tail==1:
        if (closest <=0 or closest>=sd_ps.size) and high:
            if high:
                sd_p = 1
            else:
                sd_p = 1 / dist.size
        else:
            others = np.where(sd_ps == sd_ps[closest])
            sd_p = np.absolute(high - ps[others[0]]/precision)

    return sd_p

def AUC_histogram(auc, p_values, bins=40):
    """
    Plot the histogram of AUC distribution. White: all. Gray: p<=0.05. Black: p<=0.01
    """
    fig, ax = plt.subplots(figsize=(6,4.5))
    sig1 = p_values <= 0.01
    sig5 = p_values <= 0.05
    width = 1.0/bins * 0.8
    gray = [0.7,0.7,0.7]
    all_hist, _ = np.histogram(auc, bins=np.arange(bins+1)/float(bins))
    plt.bar(np.arange(bins)/float(bins),all_hist,width,color = 'w')
    sig5_hist, _ = np.histogram(auc[sig5], bins=np.arange(bins+1)/float(bins))
    plt.bar(np.arange(bins)/float(bins),sig5_hist,width,color = gray)
    sig1_hist, _ = np.histogram(auc[sig1], bins=np.arange(bins+1)/float(bins))
    plt.bar(np.arange(bins)/float(bins),sig1_hist,width,color = 'k')
    n = bins/10
    plt.xticks(np.arange(bins//n+1)/float(bins//n))
    plt.title('Session level task encoding')
    plt.ylabel('Session/time step count')
    plt.xlabel('AUC')
    plt.show()

def run_AUC_scatter_histogram(session_table,t, target = False):
    time_steps = ['ITI','Rule','Delay','Target','Choice']
    Duan_rats = np.array([2,3,4,5,6,7,13,14,15])
    p_values = session_table[:,-2]
    auc = session_table[:,-3]
    accuracy = session_table[:,-4]
    rat_indices = session_table[:,-1]
    # Unknown bug: don't use loop to generate pdf! Otherwise the plots will overlap.
    this_step = session_table[:,1]==t
    if t==2: # Exclude Erlich's delay step
        n_rec = session_table.shape[0]
        Duan_sessions = np.sum(np.tile(rat_indices,(Duan_rats.size,1)).T == np.tile(Duan_rats,(n_rec,1)),axis=1)
        this_step = np.logical_and(Duan_sessions,this_step)
    encoding_type = "target" if target else "rule"
    filename = "figures/AUC_scatter_histogram_"+encoding_type+"_"+time_steps[t]+".pdf"
    AUC_scatter_histogram(accuracy[this_step],auc[this_step],p_values[this_step],time_steps[t], 
                          filename = filename)    

def AUC_scatter_histogram(accuracy, auc, p_values, time_step, filename = None):
    """
    Plot the scatter plot of prediction accuracy and AUC value. As well as the histogram of AUC distribution
    and accuracy distribution. White: all. Gray: p-value of AUC results, p<=0.05. Black: p<=0.01
    """

    x = accuracy
    y = auc
    x_median = np.median(x)
    y_median = np.median(y)
    sig1 = p_values <= 0.01
    sig5 = p_values <= 0.05

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left + width + 0.08
    left_h = left + width + 0.12

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    fig = plt.figure(1,figsize=(4,4))
    plt.suptitle(time_step,y=1.07)
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axScatter.scatter(x, y, s=10, color='k', alpha=0.2)

    binwidth = 0.02
    width = binwidth * 0.75
    lim_min = 0.4
    lim_max = 1.0
    axScatter.set_xlim((lim_min, lim_max))
    axScatter.set_ylim((lim_min, lim_max))

    bins = np.arange(lim_min, lim_max + binwidth, binwidth)
    xCount,_,_ = axHistx.hist(x, width = width, bins=bins, weights=np.zeros_like(x) + 1. / x.size, color = 'w')
    yCount,_,_ = axHisty.hist(y, height = width, bins=bins, weights=np.zeros_like(y) + 1. / y.size, orientation='horizontal', color = 'w')

    gray = [0.7,0.7,0.7]
    axHistx.hist(x[sig5], width=width, bins=bins, weights=np.zeros_like(x[sig5]) + 1. / x.size, color = gray)
    axHisty.hist(y[sig5], height = width, bins=bins, weights=np.zeros_like(y[sig5]) + 1. / y.size, orientation='horizontal', color = gray)

    axHistx.hist(x[sig1], width=width, bins=bins, weights=np.zeros_like(x[sig1]) + 1. / x.size, color = 'k')
    axHisty.hist(y[sig1], height = width, bins=bins,weights=np.zeros_like(y[sig1]) + 1. / y.size, orientation='horizontal', color = 'k')

    lim_min_hist = 0.4
    lim_max_hist = 1.0

    axHistx.set_xlim([lim_min_hist,lim_max_hist])
    axHisty.set_ylim([lim_min_hist,lim_max_hist])

    xCountMax = np.max(xCount)
    yCountMax = np.max(yCount)

    axHistx.plot([x_median,x_median],[0,xCountMax],'k--')
    axHisty.plot([0,yCountMax],[y_median,y_median],'k--')

    #axHistx.yaxis.set_ticks(np.arange(0, xCountMax, step=0.1))
    #axHisty.xaxis.set_ticks(np.arange(0, yCountMax, step=0.1))
    axHistx.locator_params(axis = 'y', nbins = 2)
    axHisty.locator_params(axis = 'x', nbins = 2)
    axHistx.xaxis.set_ticks([])
    axHisty.yaxis.set_ticks([])
    axScatter.locator_params(axis = 'x', nbins = 4)
    axScatter.locator_params(axis = 'y', nbins = 4)


    axHistx.spines['top'].set_visible(False)
    axHistx.spines['right'].set_visible(False)
    axHistx.xaxis.set_ticks_position('bottom')
    axHistx.yaxis.set_ticks_position('left')

    axHisty.spines['top'].set_visible(False)
    axHisty.spines['right'].set_visible(False)
    axHisty.xaxis.set_ticks_position('bottom')
    axHisty.yaxis.set_ticks_position('left')

    axScatter.spines['top'].set_visible(False)
    axScatter.spines['right'].set_visible(False)
    axScatter.xaxis.set_ticks_position('bottom')
    axScatter.yaxis.set_ticks_position('left')

    axScatter.set_xlabel('Accuracy')
    axScatter.set_ylabel('AUC')
    axHistx.set_ylabel('Accuracy')
    axHisty.set_xlabel('AUC')

    if filename:
        fig.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def single_neuron_AUC(single_AUC_p, target=False, save = False, ymax=0.4):
    """
    Plot the histogram of AUC of rat's single neuron rule/target encoding.
    White: all. Gray: p-value of AUC results, p<=0.05. Black: p<=0.01
    """
    bins=20
    step_names = ["ITI", "Rule", "Delay", "Target", "Choice"]
    fig, axes = plt.subplots(nrows=1, ncols=len(step_names), figsize=(15,2.5))
    for t in range(len(step_names)):
        ax = axes[t]
        p_values = single_AUC_p[:,2*t+1]
        auc = single_AUC_p[:,2*t]
        p_values = p_values[np.logical_not(np.isnan(p_values))]
        auc = auc[np.logical_not(np.isnan(auc))]
        sig1 = p_values <= 0.01
        sig5 = p_values <= 0.05
        width = 0.04
        gray = [0.7,0.7,0.7]
        all_hist, _ = np.histogram(auc, weights=np.zeros_like(auc) + 1. / auc.size, bins=np.arange(bins+1)/float(bins))
        ax.bar(np.arange(bins)/float(bins),all_hist,width,color = 'w')
        sig5_hist, _ = np.histogram(auc[sig5], weights=np.zeros_like(auc[sig5]) + 1. / auc.size, bins=np.arange(bins+1)/float(bins))
        ax.bar(np.arange(bins)/float(bins),sig5_hist,width,color = gray)
        sig1_hist, _ = np.histogram(auc[sig1], weights=np.zeros_like(auc[sig1]) + 1. / auc.size, bins=np.arange(bins+1)/float(bins))
        ax.bar(np.arange(bins)/float(bins),sig1_hist,width,color = 'k')
        ax.set_xticks(np.arange(1,bins//4)/float(bins//4))
        ax.set_title(step_names[t])
        ax.set_ylim([0,ymax])
        if t>0:
            ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    if target:
        neg = "Left"
        pos = "Right"
    else:
        neg = "Anti"
        pos = "Pro"
    fig.text(0.36, -0.05, neg+' selective <-- AUC --> '+pos+' selective', va='center')
    fig.text(0.07, 0.5, "Fraction", va='center', rotation='vertical')
    if save:
        if target:
            TYPE = "target"
        else:
            TYPE = "rule"
        plt.savefig("figures/single_neuron_AUC_"+TYPE+".pdf",dpi=600,bbox_inches='tight')
    else:
        plt.show()


def flip_neurons(single_AUC_p, CellIndexPerSession, num_session, threshold = 0.01):
    """
    Calculate the fraction of significantly flipping rule/target encoding neurons 
    among significantly encoding rule/target neurons.
    """
    session_num = CellIndexPerSession.size
    auc_indices = np.arange(0,10,2)
    p_indices = auc_indices+1
    auc = single_AUC_p[:,auc_indices]
    p_values = single_AUC_p[:,p_indices]
    auc[p_values>threshold] = 0.5
    auc_flip_matrix = ((auc[:,0:-1]-0.5) * (auc[:,1:]-0.5)<0).astype(float)
    significant_matrix = np.logical_and(p_values[:,0:-1]<=threshold,p_values[:,1:]<=threshold)

    fraction_flip_per_session = []
    for session in range(num_session):
        cellIndex = CellIndexPerSession[session][0]-1
        flip_matrix = auc_flip_matrix[cellIndex,:]
        flip_count = np.sum(np.sum(flip_matrix,axis=1)>0)
        sig_matrix = significant_matrix[cellIndex,:]
        sig_count = np.sum(np.sum(sig_matrix,axis=1)>0)
        if sig_count>0:
            fraction = float(flip_count)/sig_count
            fraction_flip_per_session.append(fraction) 
    return np.array(fraction_flip_per_session)

def flip_neurons_RNN(AUC, significant):
    """
    Calculate the fraction of significantly flipping rule/target encoding neurons 
    among significantly encoding rule/target neurons for RNN.
    """

    AUC[np.logical_not(significant)] = 0.5
    flips = (AUC-0.5)[:,:,np.arange(4)] * (AUC-0.5)[:,:,np.arange(1,5)] < 0
    consecutive_sig = np.logical_and(significant[:,:,np.arange(4)], significant[:,:,np.arange(1,5)])
    total_flips = np.sum(np.sum(flips,axis=2) > 0,axis=1)
    total_sigs =  np.sum(np.sum(consecutive_sig,axis=2) > 0,axis=1)
    flips_fraction = total_flips / total_sigs.astype(float)
    return flips_fraction
    
def AUC_flip_histogram(flip_fraction_RNN, fraction_flip_per_session, filename = None, hidden_dim = 20.0):
    """
    Plot RNN's and rat's histogram of rule/target preference flips of each neuron
    """
    fig, ax = plt.subplots(figsize=(6,4.5))
    
    plt.hist(flip_fraction_RNN,bins = np.arange(hidden_dim+1)/hidden_dim, \
        weights=np.zeros_like(flip_fraction_RNN) + 1. / flip_fraction_RNN.size, color="black")

    plt.hist(fraction_flip_per_session,bins = np.arange(hidden_dim+1)/hidden_dim, \
        weights=np.zeros_like(fraction_flip_per_session) + 1. / fraction_flip_per_session.size, color="white")

    #plt.xlim([0.6,1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title("Histogram of preference flips of each hidden unit")
    plt.xlabel("Fraction of flipping neurons per RNN / per session")
    plt.ylabel("Frequency")
    plt.yticks(np.arange(2,11,2)/10.0)
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()


def sig4(x,y0,a,x0,b):
    """
    4-parameter sigmoid function to illustrate the fitted result of (Generalized) Linear Mixed Effect model.
    """
    return y0 + a/(1+np.exp(-(x-x0)/b))

def StandardError(x):
    return np.std(x)/np.sqrt(x.size)

def bin_center(x,bins):
    """
    Find the bin center by percentile.
    """
    percentiles = np.linspace(0,100,bins+1)
    centers = []
    for p in percentiles:
        centers.append(np.percentile(x,p))
    return np.array(centers)

def binned(x,y,bins):
    """
    Bin y for each x.
    """
    notnan = np.logical_not(np.isnan(y))
    mu, binc, n = binned_statistic(x[notnan],y[notnan], bins=bin_center(x[notnan],bins))
    SE, _, _ = binned_statistic(x[notnan],y[notnan],statistic=StandardError,bins=bin_center(x[notnan],bins))
    return mu, SE, binc, n

def plot_bins(fitted_tables, plot_steps, bins=15, bins_fit=50, target = False, filename = None):
    """
    Plot the binned actual (normalized) hit rate and fitted curve by  (Generalized) Linear Mixed Effect model for rats.
    """
    # session_index, pro, right, switch, hit, rat_index, (index 0~5)
    # score0, score1, score2, score3, score4, (index 6~10)
    # accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, (index 11~15)
    # encoding0, encoding1, encoding2, encoding3, encoding4 (index 16~20)
    # good0, good1, good2, good3, good4, encoding, fit, confidence_interval (index 21~27)
    time_step_names = ["ITI","Rule","Delay","Target","Choice"]
    rows = 1
    cols = 5
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,2.5))
    #fig.delaxes(axes[-1,-1]) 
    for t in range(len(fitted_tables)):
        hit = fitted_tables[t][:,4]
        if target:
            pro = fitted_tables[t][:,2]==1
        else:
            pro = fitted_tables[t][:,1]==1
        anti = np.logical_not(pro)
        encoding = fitted_tables[t][:,26]

        if target:
            pro_fit = fitted_tables[t][:,2]==1
        else:
            pro_fit = fitted_tables[t][:,1]==1
        anti_fit = np.logical_not(pro_fit)
        encoding_fit = fitted_tables[t][:,16+t]
        fitted = fitted_tables[t][:,27]

        binned_pro_fit = binned(encoding_fit[pro_fit],fitted[pro_fit],bins_fit)
        binned_pro = binned(encoding[pro],hit[pro],bins)

        binned_anti_fit = binned(encoding_fit[anti_fit],fitted[anti_fit],bins_fit)
        binned_anti = binned(encoding[anti],hit[anti],bins)

        mu_pro, SE_pro, binc_pro, _ = binned_pro
        mu_pro_fit, SE_pro_fit, binc_pro_fit, _ = binned_pro_fit
        mu_anti, SE_anti, binc_anti, _ = binned_anti
        mu_anti_fit, SE_anti_fit, binc_anti_fit, _ = binned_anti_fit


        if target:
            color1 = "r"
            color2 = "b"
        else:
            color1 = "green"
            color2 = (1,0.35,0)
        ax = axes[t]
        ax.errorbar(binc_pro[1:-1],mu_pro[1:],yerr=SE_pro[1:],linestyle='None',
            marker='o', capsize = 0,color=color1,label='binned_pro_hit')
        ax.errorbar(binc_anti[1:-1],mu_anti[1:],yerr=SE_anti[1:],linestyle='None',
            marker='o',capsize = 0,color=color2,label='binned anti hit')
        if t in plot_steps:
            ax.plot(binc_pro_fit[1:-1],mu_pro_fit[1:], linewidth=2,
                color=color1, label='fittted pro')
            ax.plot(binc_anti_fit[1:-1],mu_anti_fit[1:], linewidth=2,
                color=color2, label='fitted anti')
        ax.set_title(time_step_names[t])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if target and t==2:
            ax.set_xlim([-0.4,0.4])
            ax.set_xticks([-0.3,0,0.3])
        elif target:
            ax.set_xlim([-1.5,1.5])
            ax.set_xticks([-1,0,1])
        elif t==2:
            ax.set_xlim([-1,1])
            ax.set_xticks([-0.5,0,0.5])
        else:
            ax.set_xlim([-2,2])
            ax.set_xticks([-1,0,1])
        if target:
            ax.set_ylim([0.55,1.05])
            ax.set_yticks([0.6,0.7,0.8,0.9,1])
        else:
            ax.set_ylim([0.6,0.95])
            ax.set_yticks([0.6,0.7,0.8,0.9])


        if t>0:
            ax.set_yticklabels([])

    if target:
        fig.text(0.5, -0.04, "Normalized target encoding", ha='center')
    else:
        fig.text(0.5, -0.04, "Normalized rule encoding", ha='center')

    fig.text(0.07, 0.5, "% Correct", va='center', rotation='vertical')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()


def plot_bins_RNN(fitted_table, mode, plot_steps=[0,1,2,3,4], bins=15, bins_fit=30, target = False,filename = None):
    """
    Plot the binned actual (normalized) hit rate and fitted curve by  (Generalized) Linear Mixed Effect model for RNNs.
    Dashed lines are for switch trials.
    """
    if mode == "all":
        plot_block = True
        plot_switch = True
    elif mode == "block":
        plot_block = True
        plot_switch = False
    elif mode == "switch":
        plot_block = False
        plot_switch = True
    else:
        print "Wrong plot mode: should be 'all', 'block' or 'switch'"
        assert(0)
    time_step_names = ["ITI","Rule","Delay","Target","Choice"]
    fig, axes = plt.subplots(nrows=1, ncols=len(time_step_names), figsize=(15,2.5))
    for t in range(len(time_step_names)):
        same_score = fitted_table[:,10+t]
        fitted = fitted_table[:,15+t]
        hit = fitted_table[:,4]
        #these_steps = fitted_table[:,1]!=0
        if target:
            pro = fitted_table[:,2]>0 # Here pro is actually right, anti is left
            anti = fitted_table[:,2]==0
        else:
            pro = fitted_table[:,1]>0
            anti = fitted_table[:,1]==0
        switch = fitted_table[:,3]>0
        block = fitted_table[:,3]==0
        pro_block = np.logical_and(pro, block)
        anti_block = np.logical_and(anti, block)
        pro_switch = np.logical_and(pro, switch)
        anti_switch = np.logical_and(anti, switch)

        if plot_block:
            binned_pro_block_fit = binned(same_score[pro_block],fitted[pro_block],bins_fit)
            binned_pro_block = binned(same_score[pro_block],hit[pro_block],bins)
            binned_anti_block_fit = binned(same_score[anti_block],fitted[anti_block],bins_fit)
            binned_anti_block = binned(same_score[anti_block],hit[anti_block],bins)

        if plot_switch:
            binned_pro_switch_fit = binned(same_score[pro_switch],fitted[pro_switch],bins_fit)
            binned_pro_switch = binned(same_score[pro_switch],hit[pro_switch],bins)
            binned_anti_switch_fit = binned(same_score[anti_switch],fitted[anti_switch],bins_fit)
            binned_anti_switch = binned(same_score[anti_switch],hit[anti_switch],bins)

        if plot_switch:
            mu_pro_switch, SE_pro_switch, binc_pro_switch, _ = binned_pro_switch
            mu_pro_switch_fit, SE_pro_switch_fit, binc_pro_switch_fit, _ = binned_pro_switch_fit
            mu_anti_switch, SE_anti_switch, binc_anti_switch, _ = binned_anti_switch
            mu_anti_switch_fit, SE_anti_switch_fit, binc_anti_switch_fit, _ = binned_anti_switch_fit

        if plot_block:
            mu_pro_block, SE_pro_block, binc_pro_block, _ = binned_pro_block
            mu_pro_block_fit, SE_pro_block_fit, binc_pro_block_fit, _ = binned_pro_block_fit
            mu_anti_block, SE_anti_block, binc_anti_block, _ = binned_anti_block
            mu_anti_block_fit, SE_anti_block_fit, binc_anti_block_fit, _ = binned_anti_block_fit

        if target:
            color1 = "r"
            color2 = "b"
        else:
            color1 = "green"
            color2 = (1,0.35,0)
        ax = axes[t]

        if t in plot_steps:
            if plot_block:
                ax.plot(binc_pro_block_fit[:-1],mu_pro_block_fit, color=color1, linewidth=3)
                ax.plot(binc_anti_block_fit[:-1],mu_anti_block_fit, color=color2, linewidth=3)
            if plot_switch:
                ax.plot(binc_pro_switch_fit[:-1],mu_pro_switch_fit, color=color1, linestyle='--', linewidth=3)
                ax.plot(binc_anti_switch_fit[:-1],mu_anti_switch_fit, color=color2,linestyle='--', linewidth=3)

        if plot_block:
            ax.errorbar(binc_pro_block[1:-1],mu_pro_block[1:],yerr=SE_pro_block[1:],linestyle='None',
                marker='o', color=color1)
            ax.errorbar(binc_anti_block[1:-1],mu_anti_block[1:],yerr=SE_anti_block[1:],linestyle='None',
                marker='o',color=color2)
        if plot_switch:
            ax.errorbar(binc_pro_switch[1:-1],mu_pro_switch[1:],yerr=SE_pro_switch[1:],linestyle='None',
                marker='o', color=color1, fillstyle = 'none')
            ax.errorbar(binc_anti_switch[1:-1],mu_anti_switch[1:],yerr=SE_anti_switch[1:],linestyle='None',
                marker='o',color=color2,fillstyle = 'none')
        if plot_switch and plot_block and t==0:
            ax.set_xlim([-1.8,1.5])
            ax.set_xticks([-1,0,1])
        elif plot_switch and t==0:
            ax.set_xlim([-1.8,0])
            ax.set_xticks([-1.5,-1,-0.5,0])
        else:
            ax.set_xlim([0.3,1.5])
            ax.set_xticks([0.5,1,1.5])
        if target and plot_switch and not plot_block:
            ax.set_ylim([0.8,0.95])
            ax.set_yticks([0.8,0.85,0.9,0.95])
        elif target and not plot_switch and plot_block:
            ax.set_ylim([0.88,0.96])
            ax.set_yticks([0.9,0.95])
        elif target and plot_switch and plot_block:
            ax.set_ylim([0.8,1])
            ax.set_yticks([0.8,0.85,0.9,0.95,1])
        elif plot_switch:
            ax.set_ylim([0.76,1])
            ax.set_yticks([0.8,0.85,0.9,0.95,1])
        else:
            ax.set_ylim([0.84,1])
            ax.set_yticks([0.85,0.9,0.95,1])
        ax.set_title(time_step_names[t])
        if t>0:
            ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    if target:
        fig.text(0.5, -0.04, "Normalized target encoding", ha='center')
    else:
        fig.text(0.5, -0.04, "Normalized rule encoding", ha='center')

    fig.text(0.07, 0.5, "% Correct", va='center', rotation='vertical')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def encoding_score_parallel_plot(fitted_tables, trial_type, time_steps=5, minimum_trial = 3, filename = None, message=False):
    """
    Parallel plot of the mean of each RNN/session's rule/target score for block and switch trials.
    """
    step_names = ["ITI", "Rule", "Delay", "Target", "Choice"]
    fig, axes = plt.subplots(nrows=1, ncols=time_steps, figsize=(15,2.5))

    for t in range(time_steps):
        ax = axes[t]
        fitted_table = fitted_tables[t]
        sessid = fitted_table[:,0]
        u = np.unique(sessid)
        all_means = np.zeros((len(u),2))
        trial_count = np.zeros((len(u),2))
        for i in range(len(u)):
            this_session = sessid==u[i]
            this_table = fitted_table[this_session,:]
            switch = this_table[:,3]==1
            block = np.logical_not(switch)
            if trial_type == "pro":
                TYPE = this_table[:,1]==1
                color = "green" 
            elif trial_type == "anti":
                TYPE = this_table[:,1]==0
                color = (1,0.35,0) 
            elif trial_type == "left":
                TYPE = this_table[:,2]==0
                color = "b"
            elif trial_type == "right":
                TYPE = this_table[:,2]==1
                color = "r"
            else:
                raise ValueError("trial_type must be either pro, anti, left or right")
            TYPE_switch = np.logical_and(TYPE,switch)
            TYPE_block = np.logical_and(TYPE,block)
            same_score = this_table[:,26]
            all_means[i,0] = np.mean(same_score[TYPE_switch])
            all_means[i,1] = np.mean(same_score[TYPE_block])

            trial_count[i,0] = np.sum(TYPE_switch>0)
            trial_count[i,1] = np.sum(TYPE_block>0)

        # Paired t-test
        valid_types = trial_count>minimum_trial
        valid_sessions = np.logical_and(valid_types[:,0],valid_types[:,1])
        _,p_value = ttest_rel(all_means[valid_sessions,0],all_means[valid_sessions,1])
        mean_switch = np.mean(all_means[valid_sessions,0])
        mean_block = np.mean(all_means[valid_sessions,1])
        if message:
            print "time step %d" % (t,)
            print "%d sessions considered." % (np.sum(valid_sessions>0),)
            print "The mean normalized encoding score of switch trials is %f, block trials is %f" % (mean_switch,mean_block)
            print "The p value between switch and block trials is %f" % (p_value,)
        # Plot
        plot_count = 0
        ax.set_title(step_names[t])
        for i in range(len(u)):
            if np.product(trial_count[i,:]>minimum_trial):
                plot_count +=1
                ax.plot([0,1],all_means[i,:2],marker='o', color = color, alpha=0.5)
        if (trial_type=="left" or trial_type=="right") and plot_count<=1:
            fig.delaxes(ax)
            continue
        ax.set_xlim([-0.4,1.4])
        ax.set_xticks([0,1])
        ax.set_xticklabels(["switch", "block"])
        ax.locator_params(axis = 'y', nbins = 5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plot_range = all_means[np.product((trial_count>minimum_trial),axis=1)>0]
        if len(plot_range)>0:
            plot_max = np.max(plot_range)
            plot_min = np.min(plot_range)
            ax.set_ylim([plot_min-0.1,plot_max+0.1])
            def label_diff(text,y):
                x = 0.5
                dx = 1
                props = {'connectionstyle':'bar','arrowstyle':'-',\
                             'shrinkA':20,'shrinkB':20,'linewidth':1}
                if text is not None:
                    ax.annotate(text, xy=(0.45,y*0.92), zorder=10)
                    ax.annotate('', xy=(0,y*0.8), xytext=(1,y*0.8), arrowprops=props)
                else:
                    pass
            if p_value<=0.001:
                significance_level = "***"
            elif p_value<=0.01:
                significance_level = "**"
            elif p_value<=0.05:
                significance_level = "*"
            else:
                significance_level = None
            label_diff(significance_level,plot_max)
    if trial_type == "right" or trial_type == "left":
        fig.text(0.55, 0.5, "Target encoding", va='center', rotation='vertical')
    else:
        fig.text(0.07, 0.5, "Rule encoding", va='center', rotation='vertical')

    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()


def encoding_score_parallel_plot_RNN(fitted_table, trial_type, minimum_trial = 3, filename = None, message=False):
    """
    Parallel plot of the mean of each RNN/session's rule/target score for block and switch trials.
    """
    step_names = ["ITI", "Rule", "Delay", "Target", "Choice"]
    fig, axes = plt.subplots(nrows=1, ncols=len(step_names), figsize=(15,2.5))
    for t in range(len(step_names)):
        ax = axes[t]
        sessid = fitted_table[:,0]
        u = np.unique(sessid)
        all_means = np.zeros((len(u),2))
        trial_count = np.zeros((len(u),2))
        for i in range(len(u)):
            this_session = sessid==u[i]
            this_table = fitted_table[this_session,:]
            switch = this_table[:,3]==1
            block = np.logical_not(switch)
            if trial_type == "pro":
                TYPE = this_table[:,1]==1
                color = "green" 
            elif trial_type == "anti":
                TYPE = this_table[:,1]==0
                color = (1,0.35,0) 
            elif trial_type == "left":
                TYPE = this_table[:,2]==0
                color = "b"
            elif trial_type == "right":
                TYPE = this_table[:,2]==1
                color = "r"
            else:
                raise ValueError("trial_type must be either pro, anti, left or right")
            TYPE_switch = np.logical_and(TYPE,switch)
            TYPE_block = np.logical_and(TYPE,block)
            same_score = this_table[:,10+t]
            all_means[i,0] = np.mean(same_score[TYPE_switch])
            all_means[i,1] = np.mean(same_score[TYPE_block])

            trial_count[i,0] = np.sum(TYPE_switch>0)
            trial_count[i,1] = np.sum(TYPE_block>0)

        # Paired t-test
        valid_types = trial_count>minimum_trial
        valid_sessions = np.logical_and(valid_types[:,0],valid_types[:,1])
        _,p_value = ttest_rel(all_means[valid_sessions,0],all_means[valid_sessions,1])
        mean_switch = np.mean(all_means[valid_sessions,0])
        mean_block = np.mean(all_means[valid_sessions,1])
        if message:
            print "Time step %d" % (t,)
            print "%d sessions considered." % (np.sum(valid_sessions>0),)
            print "The mean normalized encoding score of switch trials is %f, block trials is %f" % (mean_switch,mean_block)
            print "The p value between switch and block trials is %f" % (p_value,)
        # Plot
        for i in range(len(u)):
            if np.product(trial_count[i,:]>minimum_trial):
                ax.plot(np.arange(2),all_means[i,:2],marker='o', color = color, alpha=0.5)
        ax.set_xlim([-0.4,1.4])
        ax.set_xticks([0,1])
        ax.set_xticklabels(["switch", "block"])
        ax.set_title(step_names[t])
        ax.locator_params(axis = 'y', nbins = 5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plot_range = all_means[np.product((trial_count>minimum_trial),axis=1)>0]
        if len(plot_range)>0:
            plot_max = np.max(plot_range)
            plot_min = np.min(plot_range)
            if t==0:
                ax.set_ylim([plot_min-0.15,plot_max+0.3])
            else:
                ax.set_ylim([plot_min-0.1,plot_max+0.1])
            def label_diff(text,y):
                x = 0.5
                dx = 1
                props = {'connectionstyle':'bar','arrowstyle':'-',\
                             'shrinkA':20,'shrinkB':20,'linewidth':1}
                if text is not None:
                    if t==0:
                        ax.annotate(text, xy=(0.4,y-0.01), zorder=10)
                    elif trial_type=="right" or trial_type=="left":
                        ax.annotate(text, xy=(0.4,y*0.9), zorder=10)
                    else:
                        ax.annotate(text, xy=(0.4,y*0.925), zorder=10)
                    if trial_type=="right" and t==2:
                        ax.annotate('', xy=(0,y*0.7), xytext=(1,y*0.7), arrowprops=props)
                    else:
                        ax.annotate('', xy=(0,y*0.85), xytext=(1,y*0.85), arrowprops=props)
                else:
                    pass
            if p_value<=0.001:
                significance_level = "***"
            elif p_value<=0.01:
                significance_level = "**"
            elif p_value<=0.05:
                significance_level = "*"
            else:
                significance_level = None
            label_diff(significance_level,plot_max)
    
    if trial_type == "right" or trial_type == "left":
        fig.text(0.07, 0.5, "Target encoding", va='center', rotation='vertical')
    else:
        fig.text(0.07, 0.5, "Rule encoding", va='center', rotation='vertical')

    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()


def fraction_significant_cell(single_AUC_p, pro_selective, anti_selective,time_steps = 5):
    """
    Plot the fraction of single rule/target selective cell for each time step.
    Only plot rule or target here.
    """
    
    threshold = 0.01
    ticks = ["ITI", "Rule", "Delay", "Target", "Choice"]
    significant_fraction = []
    for t in range(time_steps):
        significant_fraction.append(np.mean(single_AUC_p[:,2*t+1]<=threshold))

    # RNN
    pro_mean = np.mean(pro_selective,axis=0)
    anti_mean = np.mean(anti_selective,axis=0)
    rule_mean = pro_mean+anti_mean

    fig, ax = plt.subplots(figsize=(6,4.5))
    plt.plot([-10,10],[threshold,threshold],"k--")
    plot1 = plt.plot(np.arange(time_steps),rule_mean,color="orange", linewidth=3, marker="o")
    plot2 = plt.plot(np.arange(time_steps),significant_fraction,color="orange", linewidth=3, marker="o", linestyle="--")
    plt.xticks(np.arange(time_steps),ticks)
    plt.xlim([-0.5,4.5])
    plt.ylim([0,1])
    plt.ylabel("Fraction of Selective Cells")
    plt.xlabel("Time steps in one trial --->")
    plt.ylabel('Fraction of rule selective cells')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.show()

def fraction_significant_cell_combined(single_AUC_p, single_AUC_p_target, pro_selective, 
    anti_selective, right_selective, left_selective, filename = None):
    """
    Plot the fraction of single rule/target selective cell for each time step.
    Combine rule or target for rats and RNNs here.
    """

    time_steps = 5
    threshold = 0.01
    ticks = ["ITI", "Rule", "Delay", "Target", "Choice"]
    significant_fraction = []
    significant_fraction_target = []
    for t in range(time_steps):
        significant_fraction.append(np.mean(single_AUC_p[:,2*t+1]<=threshold))
        significant_fraction_target.append(np.mean(single_AUC_p_target[:,2*t+1]<=threshold))
    # RNN
    pro_mean = np.mean(pro_selective,axis=0)
    anti_mean = np.mean(anti_selective,axis=0)
    rule_mean = pro_mean+anti_mean

    right_mean = np.mean(right_selective,axis=0)
    left_mean = np.mean(left_selective,axis=0)
    target_mean = right_mean+left_mean

    fig, ax = plt.subplots(figsize=(6,4.5))
    plt.plot([-10,10],[threshold,threshold],"k--")
    plot1 = plt.plot(np.arange(time_steps),rule_mean,color="orange", linewidth=3, marker="o")
    plot2 = plt.plot(np.arange(time_steps),significant_fraction,color="orange", linewidth=3, marker="o", linestyle="--")
    plot3 = plt.plot(np.arange(time_steps),target_mean,color="mediumpurple", linewidth=3, marker="o")
    plot4 = plt.plot(np.arange(time_steps),significant_fraction_target,color="mediumpurple", linewidth=3, marker="o", linestyle="--")
    plt.xticks(np.arange(time_steps),ticks)
    plt.xlim([-0.5,4.5])
    plt.ylim([0,1.05])
    plt.ylabel("Fraction of Selective Cells")
    plt.xlabel("Time steps in one trial --->")
    plt.ylabel('Fraction of rule selective cells')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()



def save_weights(filename,weights):
    """
    Save numpy array into pkl file.
    """
    with open(filename,"wb") as f:
        pkl.dump(weights,f)

def load_weights(filename):
    """
    Load numpy array from pkl file.
    """
    with open(filename,"rb") as f:
        weights = pkl.load(f)
    return weights
