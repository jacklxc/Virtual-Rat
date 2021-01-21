from Rat import Rat
import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
import matplotlib

FONT_SIZE = 20

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
plt.rc('legend',fontsize=FONT_SIZE) # using a size in points

def getData(num_rats=15):
    """
    Get data from MySQL database

    Inputs:
    - num_rats: the number of rats whose data need to be initiated. 

    Returns:
    -allRatsData: a dictionary with rat names as keys and numpy boolean array of 
    shape N * T * D as elements. In dimention 2, the bits represent pro_rule,
    target_on_right, trial_n=1, left, right, cpv (central poke violation) respectively.
    """
    CONN = db.Connection()
    CONN.use('pa')
    out = zip(*CONN.query('explain alldata'))
    all_rats = CONN.query('select distinct(ratname) from pa.alldata')

    MIN_PERF = 70;

    allRatsData = {}

    if num_rats > len(all_rats):
        num_rats = 15

    for rat in all_rats[:num_rats]:
        sqlstr=('select pro_rule, target_on_right, trial_n=1, '
                '(cpv=0 AND WR=0) as `left`, (cpv=0 AND WR = 1) as `right`, cpv '
                'from pa.alldata a, pa.pasess b where a.sessid=b.sessid '
                'and pro_perf>{} and anti_perf>{} and b.ratname=%s order by a.sessid, trial_n').format(MIN_PERF, MIN_PERF)

        out = CONN.query(sqlstr, (str(rat[0]),))
        data = np.array(out)
        allRatsData[str(rat[0])] = data
        print rat[0], data.shape
    return allRatsData


def preProcess(allRatsData, ratnames=[], Rats = {}):
    """
    Divide each rat's data into train, validation and test data for RNN

    Inputs:
    - allRatsData: a dictionary with rat names as keys and numpy boolean array of 
    shape N * T * D as elements. In dimention 2, the bits represent pro_rule,
    target_on_right, trial_n=1, left, right, cpv (central poke violation) respectively.
    - ratnames: a list of strings, containing rats' names. If not empty, only initialize
        the rats whose name is in this list.

    Returns:
    - Rats: a dictionary with rat names as keys and objects of class Rat as elements.
    """
    if len(ratnames) > 0:
        allratnames = ratnames
    else:
        allratnames = allRatsData.keys()

    for ratname in allratnames:
        rat = allRatsData[ratname]
        RAT = Rat(ratname,rat)
        Rats[ratname] = RAT
    return Rats

def preProcessNew(SessionInfo, ratIndices, Rats = {}):
    ratnames = ['A092',
    'A105',
    'A109',
    'A110',
    'A113',
    'A117',
    'A136',
    'C130',
    'J133',
    'J137',
    'J145',
    'J147',
    'J204',
    'J205',
    'Z014']
    session_num = SessionInfo.size
    ratindex = -1
    behavior_data = []
    for i in range(len(ratnames)):
        behavior_data.append(np.zeros((0,5)))
    for sess in range(session_num):        
        augmented = np.zeros((SessionInfo[sess].shape[0],SessionInfo[sess].shape[1]+1))
        augmented[:,:-1] = SessionInfo[sess]
        augmented[0,-1] = 1
        behavior_data[ratIndices[sess]] = np.append(behavior_data[ratIndices[sess]],augmented,axis=0)
    for i in range(len(ratnames)):
        Rats[ratnames[i]] = Rat(ratnames[i],behavior_data[i])
    return Rats

def figure_3d_matrix(rats = None, trial_window = 3):
    size = trial_window*2+1
    p2a_matrix = np.zeros((0,size))
    a2p_matrix = np.zeros((0,size))
    for ratname, rat in rats.iteritems():
        p2a_matrix = np.append(p2a_matrix, np.expand_dims(rat.real_p2a,axis=0), axis=0)
        a2p_matrix = np.append(a2p_matrix, np.expand_dims(rat.real_a2p,axis=0), axis=0)
    return p2a_matrix*100, a2p_matrix*100

def meanPerformance(rats, trial_window = 3):
    """
    Compute the switch cost of RNN output based on two ways of calculation.

    Inputs:
    - rats: a dictionary with ratname as keys and object Rat as elements.
    - trial_window: int, number of trials computed before and after swtiches.

    Returns:
    - p2a_mean: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around pro to anti swtich.
    - a2p_mean: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around anti to pro swtich.
    - p2a_mean2: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of the RNN choices around pro to anti swtich.
    - a2p_mean2: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of the RNN choices around anti to pro swtich.
    """
    size = trial_window*2+1
    p2a_mean = np.zeros(size)
    a2p_mean = np.zeros(size)

    for ratname, rat in rats.iteritems():
        p2a_mean += rat.p2a_prob
        a2p_mean += rat.a2p_prob

    p2a_mean /= len(rats)
    a2p_mean /= len(rats)

    return p2a_mean, a2p_mean


def sample_probabilities(probs, ratname, sample = 50):
    """
    Plots the first a few trial of the RNN softmax probability output.

    Inputs:
    - probs: numpy float array of shape (1, T, 3) which contains probabilities.
    - ratname: stirng, rat's name
    - sample: int, number of samples to display.
    """
    plt.plot(probs[0,:sample,0],'bo')
    plt.plot(probs[0,:sample,1],'ro')
    plt.plot(probs[0,:sample,2],'go')
    plt.xlabel('Trials')
    plt.ylabel('probs')
    plt.title('Probabilities of '+ ratname)
    plt.show()

def sample_correct_rate(rat, sample = 500):
    """
    Plots the first several trials of the prodicted correct rate.

    Inputs:
    - rat: object Rat.
    - sample: number of trials to display.
    """
    plt.plot(range(sample), rat.pro_prob[:sample],color='b')
    plt.plot(range(sample), rat.anti_prob[:sample],color='r')
    plt.xlabel('Trials')
    plt.ylabel('%Correct')
    plt.title('Correction rate')
    plt.show()
    
def draw_3d(p2a = None, a2p = None, p2a_matrix = None, a2p_matrix = None, trial_window = 3, fixed_size = True, shift=0.05, filename=None):
    """
    Plots figure 3-d in Duan et al.'s paper. 

    Inputs:
    - p2a: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around pro to anti swtich.
    - a2p: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around anti to pro swtich.
    - trial_window: int, number of trials computed before and after swtiches.
    """
    fig, ax = plt.subplots(figsize=(6,4.5))
    if (p2a_matrix is not None) and (a2p_matrix is not None):
        #p2a_matrix = p2a_matrix * 100
        #a2p_matrix = a2p_matrix * 100
        p2a = np.mean(p2a_matrix,axis=0)
        p2a_SE = np.std(p2a_matrix,axis=0) / np.sqrt(p2a_matrix.shape[0])
        a2p = np.mean(a2p_matrix,axis=0)
        a2p_SE = np.std(a2p_matrix,axis=0) / np.sqrt(a2p_matrix.shape[0])
    else:
        p2a = p2a * 100
        a2p = a2p * 100

    if fixed_size:
        plt.ylim([0,100])
    else:
        plt.ylim([np.min([np.min(p2a),np.min(a2p)])-20, 100])
    np.set_printoptions(precision=2)
    green = "green"
    orange = (1,0.35,0)
    
    plt.xlim([-trial_window-0.5,trial_window+0.5])
    p2aplot, = plt.plot(np.arange(-trial_window, 0)-shift, p2a[:trial_window], color=green, linewidth = 3, marker = "o")
    a2pplot, = plt.plot(np.arange(-trial_window, 0)+shift, a2p[:trial_window], color=orange, linewidth = 3,marker = "o")
    plt.plot(np.arange(trial_window+1)+shift, p2a[trial_window:], color=orange,linewidth = 2, marker = "o")
    plt.plot(np.arange(trial_window+1)-shift, a2p[trial_window:], color=green,linewidth = 2, marker = "o")
    plt.plot([-1-shift,0+shift],p2a[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1+shift,0-shift],a2p[trial_window - 1:trial_window + 1],'k--')

    if (p2a_matrix is not None) and (a2p_matrix is not None):
        plt.errorbar(np.arange(-trial_window, 0)-shift, p2a[:trial_window],yerr=p2a_SE[:trial_window],
            fmt = "None", ecolor = green, elinewidth = 2)
        plt.errorbar(np.arange(trial_window+1)+shift, p2a[trial_window:],yerr=p2a_SE[trial_window:],
            fmt = "None", ecolor = orange, elinewidth = 2)
        plt.errorbar(np.arange(-trial_window, 0)+shift, a2p[:trial_window],yerr=a2p_SE[:trial_window],
            fmt = "None", ecolor = orange, elinewidth = 2)
        plt.errorbar(np.arange(trial_window+1)-shift, a2p[trial_window:],yerr=a2p_SE[trial_window:],
            fmt = "None", ecolor = green, elinewidth = 2)

        alpha = 0.25
        for i in range(p2a_matrix.shape[0]):
            plt.plot(np.arange(-trial_window, 0)-shift, p2a_matrix[i,:trial_window], color=green, alpha = alpha)
            plt.plot(np.arange(-trial_window, 0)+shift, a2p_matrix[i,:trial_window], color=orange, alpha = alpha)
            plt.plot(np.arange(trial_window+1)+shift, p2a_matrix[i,trial_window:], color=orange,alpha = alpha)
            plt.plot(np.arange(trial_window+1)-shift, a2p_matrix[i,trial_window:], color=green,alpha = alpha)
            #plt.plot([-1-shift,0+shift],p2a_matrix[i,trial_window - 1:trial_window + 1],'k--', alpha = alpha)
            #plt.plot([-1+shift,0-shift],a2p_matrix[i,trial_window - 1:trial_window + 1],'k--', alpha = alpha)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.legend([p2aplot, a2pplot],["pro","anti"],loc = "lower right")
    plt.xlabel('Trial from switch')
    plt.ylabel('% Correct')
    #plt.title('Performance around switches')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def list2np(LIST):
    """
    Transform list of numpy arrays to numpy matrix. The numpy arrays in the list are not necessary to have the same length.
    """
    max_length = 0
    for i in range(len(LIST)):
        if LIST[i].size > max_length:
            max_length = LIST[i].size
            
    matrix = np.zeros((len(LIST), max_length))
    matrix[:] = np.nan
    for i in range(len(LIST)):
        length = LIST[i].size
        matrix[i,:length] = LIST[i]
    return matrix

def switch_cost_vs_time(rats, filename=None):
    pro_switch_cost = []
    anti_switch_cost = []
    switch_cost = []
    for ratname, rat in rats.iteritems():
        pro_switch_cost.append(rat.pro_switch_cost- rat.pro_switch_cost[-1])
        anti_switch_cost.append(rat.anti_switch_cost- rat.anti_switch_cost[-1])
        switch_cost.append(rat.pro_switch_cost- rat.pro_switch_cost[-1])
        switch_cost.append(rat.anti_switch_cost- rat.anti_switch_cost[-1])
    pro_switch_cost = list2np(pro_switch_cost)
    anti_switch_cost = list2np(anti_switch_cost)
    switch_cost = list2np(switch_cost) * 100
    
    pro_mean = np.nanmean(pro_switch_cost,axis=0)
    anti_mean = np.nanmean(anti_switch_cost,axis=0)
    SC_mean = np.nanmean(switch_cost,axis=0)
    pro_SE = np.nanstd(pro_switch_cost,axis=0) / np.sqrt(np.sum(np.isfinite(pro_switch_cost),axis=0))
    anti_SE = np.nanstd(anti_switch_cost,axis=0) / np.sqrt(np.sum(np.isfinite(anti_switch_cost),axis=0))
    SC_SE = np.nanstd(switch_cost,axis=0) / np.sqrt(np.sum(np.isfinite(switch_cost),axis=0))
    T = pro_mean.size
    
    fig, ax = plt.subplots(figsize=(6,4.5))
    mean_plot, = plt.plot(np.arange(15,170), SC_mean[15:170], color = "blue", linewidth = 2)
    plt.fill_between(np.arange(15,170), SC_mean[15:170]+SC_SE[15:170], SC_mean[15:170]-SC_SE[15:170],edgecolor="lightblue",\
                     facecolor = "lightblue")
    #plt.title("Switch cost of real rats vs. training time")
    plt.xlabel("Sessions")
    plt.ylabel("Normalized switch cost in %")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.locator_params(axis = 'x', nbins = 8)
    plt.ylim([-6,1])
    plt.xlim([0,180])
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()