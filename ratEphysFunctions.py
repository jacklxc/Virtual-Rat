import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

from scipy.stats import zscore, binned_statistic
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
    Normalize the spike counts by z-scoring them.
    """
    num_session = SpikeCountsPerSession.size
    all_normalized_spike_count = []
    for session in range(num_session):
        normalized_spike_count = np.zeros(SpikeCountsPerSession[0,session].shape)
        for t in range(time_steps):
            trial_num = SpikeCountsPerSession[0,session][t,:,:].shape[-1]
            means = np.mean(SpikeCountsPerSession[0,session][t,:,:],axis=1)
            stds = np.std(SpikeCountsPerSession[0,session][t,:,:],axis=1)
            tiled_mean = np.tile(means,(trial_num,1)).T
            tiled_std = np.tile(stds,(trial_num,1)).T
            normalized_spike_count[t,:,:] = (SpikeCountsPerSession[0,session][t,:,:] - tiled_mean) / tiled_std
        normalized_spike_count[np.isnan(normalized_spike_count)] = 0 
        all_normalized_spike_count.append(normalized_spike_count)
    return all_normalized_spike_count

def train_SGD(all_normalized_spike_count, SessionInfo, time_steps=5, verbose=True, target = False):
    """
    Train (overfit) SGDClassifier (logistic regression classifier) using correctly responded block trials.
    """
    num_session = len(all_normalized_spike_count)
    accuracies = []
    clfs = []
    for session in range(num_session):
        for t in range(time_steps):
            if verbose:
                print "Computing session %d, time step %d" % (session, t)
            trial_num = all_normalized_spike_count[session].shape[-1]
            X = all_normalized_spike_count[session][t,:,:].T
            if target:
                pro = SessionInfo[0,session][:,1] # This pro actually means right!
            else:
                pro = SessionInfo[0,session][:,0]
            hit = SessionInfo[0,session][:,-1]==1
            accs = []
            best_acc = 0
            for r in range(100):
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
    return accuracies, clfs

def select_good_sessions(SessionInfo, time_steps=5, threshold = 0.7):
    """
    Select good sessions by picking sessions with both pro and anti block perfornmance > threshold.
    """
    num_session = SessionInfo.size
    good_sessions = []
    for session in range(num_session):
        pro_acc = np.sum(np.logical_and(SessionInfo[0,session][:,-1] ==1,  SessionInfo[0,session][:,0]==1))/\
                    float(np.sum(SessionInfo[0,session][:,0]==1))
        anti_acc = np.sum(np.logical_and(SessionInfo[0,session][:,-1] ==1,  SessionInfo[0,session][:,0]==0))/\
                float(np.sum(SessionInfo[0,session][:,0]==0))
        good_sessions.append(np.logical_and(pro_acc>threshold,anti_acc>threshold))
    good_sessions = np.repeat(good_sessions,time_steps)
    good_SGD_indices = np.where(good_sessions)[0]
    return good_SGD_indices

def test_clf(clf, X_test, y_test):
    """
    Test the performance of the Classifiers.
    """
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    score = clf.decision_function(X_test)
    pro_score = score[y_test>0]
    anti_score = score[y_test==0]
    return acc, score, pro_score, anti_score

def make_tables(all_normalized_spike_count, SessionInfo, ratindex, clfs, good_SGD_indices, 
    normalize = False, verbose=True, target=False, time_steps=5):
    """
    Make an array (later converted into MATLAB table) to store SGDClassifier's predictions and rule/target AUC.
    normalized_table has each trial as each row.
    session_table has each session as each row.
    """
    categories = 10 # session_index, time_step, score, pro, right, switch, hit, accuracy, rule_encoding, rat_index
    normalized_table = np.zeros((0,categories))
    session_categories = 6
    session_table = np.zeros((0,session_categories)) # session_index, time_step, accuracy, auc, p, rat_index
    for index in good_SGD_indices:
        session = index/time_steps
        t = index%time_steps
        if verbose:
            print "Computing session %d, time step %d" % (session, t)
        SGD_clf = clfs[index]
        trial_num = all_normalized_spike_count[session].shape[-1]
        sub_table = np.zeros((trial_num,categories))
        sub_session_table = np.zeros((1,session_categories))
        sub_table[:,0] = session
        sub_table[:,1] = t
        sub_session_table[:,0] = session
        sub_session_table[:,1] = t
        X = all_normalized_spike_count[session][t,:,:].T
        if target:
            pro = SessionInfo[0,session][:,1]==1 # This pro actually means right!
        else:
            pro = SessionInfo[0,session][:,0]==1
        anti = np.logical_not(pro)

        SGD_acc, SGD_score, SGD_pro_score, SGD_anti_score = test_clf(SGD_clf,X,pro)
        auc, p, _ = bootroc(SGD_pro_score,SGD_anti_score)

        if normalize:
            #SGD_score[pro] = (SGD_score[pro] - np.mean(SGD_pro_score)) / np.std(SGD_pro_score)
            #SGD_score[anti] = (SGD_score[anti] - np.mean(SGD_anti_score)) / np.std(SGD_anti_score)
            SGD_score = (SGD_score - np.mean(SGD_score)) / np.std(SGD_score)
        
        sub_table[:,2] = SGD_score
        sub_table[:,3:7] = SessionInfo[0,session]
        sub_table[:,7] = SGD_acc
        sub_session_table[:,2] = SGD_acc
        sub_session_table[:,-3] = auc
        sub_session_table[:,-2] = p
        sub_table[:,-2] = pro * SGD_score + (pro-1) * SGD_score # Rule encoding, aka. same_score
        sub_table[:,-1] = ratindex[session]
        sub_session_table[:,-1] = ratindex[session]
        normalized_table = np.append(normalized_table,sub_table,axis=0)
        session_table = np.append(session_table,sub_session_table,axis=0)
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

def single_neuron_AUC(single_AUC_p, target=False, save = False):
    """
    Plot the histogram of AUC of rat's single neuron rule/target encoding.
    White: all. Gray: p-value of AUC results, p<=0.05. Black: p<=0.01
    """
    bins=20
    step_names = ["ITI", "Rule", "Delay", "Target", "Choice"]
    for t in range(len(step_names)):
        fig, ax = plt.subplots(figsize=(6,4.5))
        p_values = single_AUC_p[:,2*t+1]
        auc = single_AUC_p[:,2*t]
        sig1 = p_values <= 0.01
        sig5 = p_values <= 0.05
        width = 0.04
        gray = [0.7,0.7,0.7]
        all_hist, _ = np.histogram(auc, weights=np.zeros_like(auc) + 1. / auc.size, bins=np.arange(bins+1)/float(bins))
        plt.bar(np.arange(bins)/float(bins),all_hist,width,color = 'w')
        sig5_hist, _ = np.histogram(auc[sig5], weights=np.zeros_like(auc[sig5]) + 1. / auc.size, bins=np.arange(bins+1)/float(bins))
        plt.bar(np.arange(bins)/float(bins),sig5_hist,width,color = gray)
        sig1_hist, _ = np.histogram(auc[sig1], weights=np.zeros_like(auc[sig1]) + 1. / auc.size, bins=np.arange(bins+1)/float(bins))
        plt.bar(np.arange(bins)/float(bins),sig1_hist,width,color = 'k')
        plt.xticks(np.arange(1,bins//2+1)/float(bins//2))
        plt.xlabel('Anti selective <-- AUC --> Pro selective')
        plt.ylabel('Fraction')
        plt.title(step_names[t])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if save:
            if target:
                TYPE = "target"
            else:
                TYPE = "rule"
            plt.savefig("figures/single_neuron_AUC_"+TYPE+"_"+step_names[t]+".pdf",dpi=600,bbox_inches='tight')
        else:
            plt.show()


def flip_neurons(single_AUC_p, CellIndexPerSession, threshold = 0.01):
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
    for session in range(364):
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

def plot_bins(fitted_table, bins=15, bins_fit=50, target = False, filename = None):
    """
    Plot the binned actual (normalized) hit rate and fitted curve by  (Generalized) Linear Mixed Effect model for rats.
    """
    same_score = fitted_table[:,-3]
    fitted = fitted_table[:,-1]
    hit = fitted_table[:,6]
    these_steps = fitted_table[:,1]!=0
    if target:
        pro = fitted_table[:,4]>0 # Here pro is actually right, anti is actually left.
        anti = fitted_table[:,4]==0
    else:
        pro = fitted_table[:,3]>0
        anti = fitted_table[:,3]==0

    plot_pro = np.logical_and(pro, these_steps)
    plot_anti = np.logical_and(anti, these_steps)
    binned_pro_fit = binned(same_score[plot_pro],fitted[plot_pro],bins_fit)
    binned_pro = binned(same_score[plot_pro],hit[plot_pro],bins)
    binned_anti_fit = binned(same_score[plot_anti],fitted[plot_anti],bins_fit)
    binned_anti = binned(same_score[plot_anti],hit[plot_anti],bins)

    mu_pro, SE_pro, binc_pro, _ = binned_pro
    mu_pro_fit, SE_pro_fit, binc_pro_fit, _ = binned_pro_fit
    mu_anti, SE_anti, binc_anti, _ = binned_anti
    mu_anti_fit, SE_anti_fit, binc_anti_fit, _ = binned_anti_fit

    notnan = np.logical_not(np.isnan(mu_pro_fit))
    popt, _ = curve_fit(sig4, binc_pro_fit[:-1][notnan], mu_pro_fit[notnan])
    x = np.linspace(-3, 3)
    y_pro = sig4(x, *popt)

    notnan = np.logical_not(np.isnan(mu_anti_fit))
    popt, _ = curve_fit(sig4, binc_anti_fit[:-1][notnan], mu_anti_fit[notnan])
    y_anti = sig4(x, *popt)

    if target:
        color1 = "r"
        color2 = "b"
    else:
        color1 = "green"
        color2 = (1,0.35,0)
    fig, ax = plt.subplots(figsize=(6,4.5))
    plt.errorbar(binc_pro[1:-1],mu_pro[1:],yerr=SE_pro[1:],linestyle='None',marker='o', color=color1,label='binned_pro_hit')
    #plt.scatter(binc_pro_fit[1:-1],mu_pro_fit[1:], facecolors='none', edgecolors=color1, label='fittted pro')
    plt.plot(x,y_pro, color=color1, linewidth = 3, label="fitted pro")
    plt.errorbar(binc_anti[1:-1],mu_anti[1:],yerr=SE_anti[1:],linestyle='None',marker='o',color=color2,label='binned anti hit')
    #plt.scatter(binc_anti_fit[1:-1],mu_anti_fit[1:], facecolors='none', edgecolors=color2, label='fitted anti')
    plt.plot(x,y_anti, color=color2, linewidth = 3, label="fitted anti")
    #plt.legend(loc='best')
    if target:
        plt.xlabel("Normalized target encoding score")
    else:
        plt.xlabel("Normalized rule encoding score")
    plt.ylabel("% Correct")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def plot_bins_RNN(fitted_table, bins=15, bins_fit=35, target = False,filename = None):
    """
    Plot the binned actual (normalized) hit rate and fitted curve by  (Generalized) Linear Mixed Effect model for RNNs.
    Dashed lines are for switch trials.
    """
    same_score = fitted_table[:,-2]
    fitted = fitted_table[:,-1]
    hit = fitted_table[:,-3]
    these_steps = fitted_table[:,1]!=0
    if target:
        pro = fitted_table[:,4]>0 # Here pro is actually right, anti is left
        anti = fitted_table[:,4]==0
    else:
        pro = fitted_table[:,3]>0
        anti = fitted_table[:,3]==0
    switch = fitted_table[:,5]>0
    block = fitted_table[:,5]==0
    pro_block = np.logical_and(pro, block)
    anti_block = np.logical_and(anti, block)
    pro_switch = np.logical_and(pro, switch)
    anti_switch = np.logical_and(anti, switch)

    plot_pro_block = np.logical_and(pro_block, these_steps)
    plot_anti_block = np.logical_and(anti_block, these_steps)
    plot_pro_switch = np.logical_and(pro_switch, these_steps)
    plot_anti_switch = np.logical_and(anti_switch, these_steps)

    binned_pro_block_fit = binned(same_score[plot_pro_block],fitted[plot_pro_block],bins_fit)
    binned_pro_block = binned(same_score[plot_pro_block],hit[plot_pro_block],bins)
    binned_anti_block_fit = binned(same_score[plot_anti_block],fitted[plot_anti_block],bins_fit)
    binned_anti_block = binned(same_score[plot_anti_block],hit[plot_anti_block],bins)

    binned_pro_switch_fit = binned(same_score[plot_pro_switch],fitted[plot_pro_switch],bins_fit)
    binned_pro_switch = binned(same_score[plot_pro_switch],hit[plot_pro_switch],bins)
    binned_anti_switch_fit = binned(same_score[plot_anti_switch],fitted[plot_anti_switch],bins_fit)
    binned_anti_switch = binned(same_score[plot_anti_switch],hit[plot_anti_switch],bins)

    x = np.linspace(0, 1.5)

    mu_pro_switch, SE_pro_switch, binc_pro_switch, _ = binned_pro_switch
    mu_pro_switch_fit, SE_pro_switch_fit, binc_pro_switch_fit, _ = binned_pro_switch_fit
    mu_anti_switch, SE_anti_switch, binc_anti_switch, _ = binned_anti_switch
    mu_anti_switch_fit, SE_anti_switch_fit, binc_anti_switch_fit, _ = binned_anti_switch_fit
    notnan = np.logical_not(np.isnan(mu_pro_switch_fit))
    popt, _ = curve_fit(sig4, binc_pro_switch_fit[:-1][notnan], mu_pro_switch_fit[notnan])
    y_pro_switch = sig4(x, *popt)

    notnan = np.logical_not(np.isnan(mu_anti_switch_fit))
    popt, _ = curve_fit(sig4, binc_anti_switch_fit[:-1][notnan], mu_anti_switch_fit[notnan],maxfev=10000)
    y_anti_switch = sig4(x, *popt)

    mu_pro_block, SE_pro_block, binc_pro_block, _ = binned_pro_block
    mu_pro_block_fit, SE_pro_block_fit, binc_pro_block_fit, _ = binned_pro_block_fit
    mu_anti_block, SE_anti_block, binc_anti_block, _ = binned_anti_block
    mu_anti_block_fit, SE_anti_block_fit, binc_anti_block_fit, _ = binned_anti_block_fit
    notnan = np.logical_not(np.isnan(mu_pro_block_fit))
    popt, _ = curve_fit(sig4, binc_pro_block_fit[:-1][notnan], mu_pro_block_fit[notnan])
    y_pro_block = sig4(x, *popt)

    notnan = np.logical_not(np.isnan(mu_anti_block_fit))
    popt, _ = curve_fit(sig4, binc_anti_block_fit[:-1][notnan], mu_anti_block_fit[notnan],maxfev=10000)
    y_anti_block = sig4(x, *popt)

    if target:
        color1 = "r"
        color2 = "b"
    else:
        color1 = "green"
        color2 = (1,0.35,0)
    fig, ax = plt.subplots(figsize=(6,4.5))
    plt.plot(x,y_pro_block, color=color1, linewidth=3)
    plt.plot(x,y_anti_block, color=color2, linewidth=3)
    plt.plot(x,y_pro_switch, color=color1, linestyle='--', linewidth=3)
    plt.plot(x,y_anti_switch, color=color2,linestyle='--', linewidth=3)

    plt.errorbar(binc_pro_block[1:-1],mu_pro_block[1:],yerr=SE_pro_block[1:],linestyle='None',
        marker='o', color=color1)
    plt.errorbar(binc_anti_block[1:-1],mu_anti_block[1:],yerr=SE_anti_block[1:],linestyle='None',
        marker='o',color=color2)
    plt.errorbar(binc_pro_switch[1:-1],mu_pro_switch[1:],yerr=SE_pro_switch[1:],linestyle='None',
        marker='o', color=color1, fillstyle = 'none')
    plt.errorbar(binc_anti_switch[1:-1],mu_anti_switch[1:],yerr=SE_anti_switch[1:],linestyle='None',
        marker='o',color=color2,fillstyle = 'none')
    plt.xlim([0,1.5])
    if target:
        plt.xlabel("Normalized target encoding score")
    else:
        plt.xlabel("Normalized rule encoding score")
    plt.ylabel("Normalized % Correct")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def encoding_score_parallel_plot(fitted_table, trial_type, time_step=None, minimum_trial = 3, filename = None):
    """
    Parallel plot of the mean of each RNN/session's rule/target score for block and switch trials.
    """
    sessid = fitted_table[:,0]
    u = np.unique(sessid)
    all_means = np.zeros((len(u),2))
    trial_count = np.zeros((len(u),2))
    for i in range(len(u)):
        this_session = sessid==u[i]
        if time_step:
            this_time_step = fitted_table[:,1]==time_step
        else:
            this_time_step = fitted_table[:,1]>-1 # all true
        this_table = fitted_table[np.logical_and(this_session,this_time_step),:]
        switch = this_table[:,5]==1
        block = np.logical_not(switch)
        if trial_type == "pro":
            TYPE = this_table[:,3]==1
            color = "green" 
        elif trial_type == "anti":
            TYPE = this_table[:,3]==0
            color = (1,0.35,0) 
        elif trial_type == "left":
            TYPE = this_table[:,4]==0
            color = "b"
        elif trial_type == "right":
            TYPE = this_table[:,4]==1
            color = "r"
        else:
            raise ValueError("trial_type must be either pro, anti, left or right")
        TYPE_switch = np.logical_and(TYPE,switch)
        TYPE_block = np.logical_and(TYPE,block)
        same_score = this_table[:,-3]
        all_means[i,0] = np.mean(same_score[TYPE_switch])
        all_means[i,1] = np.mean(same_score[TYPE_block])

        trial_count[i,0] = np.sum(TYPE_switch>0)
        trial_count[i,1] = np.sum(TYPE_block>0)

    # Plot
    fig, ax = plt.subplots(figsize=(3,1.8))
    for i in range(len(u)):
        if np.product(trial_count[i,:]>minimum_trial):
            plt.plot(np.arange(2),all_means[i,:2],marker='o', color = color, alpha=0.5)
    plt.xlim([-0.4,1.4])
    plt.xticks(np.arange(2),[trial_type+" switch", trial_type+" block"])
    ax.locator_params(axis = 'y', nbins = 5)
    #plt.title("Mean rule encoding score of each session")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()


def fraction_significant_cell(single_AUC_p, pro_selective, anti_selective):
    """
    Plot the fraction of single rule/target selective cell for each time step.
    Only plot rule or target here.
    """
    time_steps = 5
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
