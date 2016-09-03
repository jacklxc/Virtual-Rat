from helpers import DBUtilsClass as db
from Rat import Rat
import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

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

def downloadRNN(rnid):
    CONN = db.Connection()
    CONN.use('vrat')
    out = zip(*CONN.query('explain rnn'))
    RNNs = CONN.query('select distinct(rnid) from vrat.rnn')
    sqlstr = ('select ratname, hidden_dim, batch_size, learning_rate, model, rat')
    RNN = CONN.query(sqlstr, (str(rnid),))
    print type(RNN)
    return RNN

def preProcess(allRatsData, ratnames=[]):
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
    Rats = {}

    if len(ratnames) > 0:
        allratnames = ratnames
    else:
        allratnames = allRatsData.keys()

    for ratname in allratnames:
        Rats[ratname] = {}
        rat_data = Rats[ratname]
        rat = allRatsData[ratname]
        RAT = Rat(ratname,rat)
        Rats[ratname] = RAT
    return Rats

def uploadRNN(solver, ratname, rat, comments, lr, hidden_dim, batch_size, logical_acc,
 sequencial_acc, rp2a, ra2p, bias_p2a, bias_a2p, test = False):
    """
    A function to upload RNN itself and related training data to the database.
    
    Inputs:
    - solver: the solver object of training RNN.
    - ratname: string, name of the rat
    - rat: the object of Rat class
    - comments: string, comments added.
    - lr: float, learning rate.
    - hidden_dim: int, the hidden dimention of the RNN
    - batch_size: int, number of sessions of rat's data as training data in each iteration.
    - logical_acc: float, logical accuracy
    - sequencial_acc: float, sequencial accuracy
    - rp2a: float, correlation coefficient between rat's and RNN's output on pro to anti switch.
    - ra2p: float, correlation coefficient between rat's and RNN's output on anti to pro switch.
    - test: bool, True if training + validation data are trained and False if only training
            data is used to train.
    """
    dbc = db.Connection()
    D = {}

    model = solver.model

    D['ratname'] = ratname
    D['batch_size'] = batch_size
    D['average_loss'] = float(solver.average_loss_history[-1])
    D['hidden_dim'] = hidden_dim
    D['logical_accuracy'] = float(logical_acc)
    D['sequencial_accuracy'] = float(sequencial_acc)
    D['test'] = test
    D['learning_rate'] = float(lr)
    D['comments'] = comments
    D['model'] = pkl.dumps(model)
    D['rat'] = pkl.dumps(rat)
    D['rp2a'] = float(rp2a)
    D['ra2p'] = float(ra2p)
    D['bias_p2a'] = float(bias_p2a)
    D['bias_p2a'] = float(bias_p2a)

    print dbc.saveToDB('vrat.rnn',D)

def realRatSwitchCost(rats,trial_window = 3):
    """
    Computes the real rats' swtich cost.

    Inputs:
    - rats: a dictionary with ratname as keys and object Rat as elements.
    - trial_window: int, number of trials computed before and after swtiches.

    Returns:
    - p2a_mean: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of real rats around pro to anti swtich.
    - a2p_mean: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of real rats around anti to pro swtich.
    """
    p2a_mean = np.zeros(trial_window*2+1)
    a2p_mean = np.zeros(trial_window*2+1)
    for ratname, rat in rats.iteritems():
        p2a_mean += rat.real_p2a
        a2p_mean += rat.real_a2p
    p2a_mean /= len(rats)
    a2p_mean /= len(rats)

    return p2a_mean, a2p_mean

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
    p2a_mean2 = np.zeros(size)
    a2p_mean2 = np.zeros(size)
    for ratname, rat in rats.iteritems():
        p2a_mean += rat.p2a_prob
        a2p_mean += rat.a2p_prob
        p2a_mean2 += rat.p2a_prob2
        a2p_mean2 += rat.a2p_prob2
    p2a_mean /= len(rats)
    a2p_mean /= len(rats)
    p2a_mean2 /= len(rats)
    a2p_mean2 /= len(rats)
    return p2a_mean, a2p_mean, p2a_mean2, a2p_mean2

def corr(real, simulation):
    """
    Returns the correlation coefficient of input numpy array real and simulation.
    """
    return np.corrcoef(real, simulation)[0,1]

def bias(real, simulation):
    """
    Calculates the sum of square of bias between real and simulation.
    """

    return np.sum(np.square(real - simulation))

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

def loss_history(solver, ratname):
    """
    Plots the training loss history.

    Inputs:
    - solver: object Solver
    - ratname: string, name of the rat
    """
    plt.plot(solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history of '+ ratname)
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

def draw_3d(real_p2a, real_a2p, p2a, a2p, p2a2 = None, a2p2 = None, 
    exclude_cpv = False, trial_window = 3):
    """
    Plots figure 3-d in Duan et al.'s paper. 

    Inputs:
    - real_p2a: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of real rats around pro to anti swtich.
    - real_a2p: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of real rats around anti to pro swtich.
    - p2a: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around pro to anti swtich.
    - a2p: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around anti to pro swtich.
    - p2a2: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of the RNN choices around pro to anti swtich.
    - a2p2: a numpy float array of shape (trial_window*2+1,) that contains the 
        accuracy of the RNN choices around anti to pro swtich.
    - exclude_cpv: bool, True if plot p2a2 and a2p2, False if skip p2a2 and a2p2.
    - trial_window: int, number of trials computed before and after swtiches.
    """
    p2aplot, = plt.plot(range(-trial_window, 0), p2a[:trial_window], color='b')
    a2pplot, = plt.plot(range(-trial_window, 0), a2p[:trial_window], color='r')
    plt.plot(range(trial_window+1), p2a[trial_window:], color='r')
    plt.plot(range(trial_window+1), a2p[trial_window:], color='b')
    plt.plot([-1,0],p2a[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1,0],a2p[trial_window - 1:trial_window + 1],'k--')
    plt.scatter(range(-trial_window, 0), p2a[:trial_window], color='b')
    plt.scatter(range(-trial_window, 0), a2p[:trial_window], color='r')
    plt.scatter(range(trial_window+1), p2a[trial_window:], color='r')
    plt.scatter(range(trial_window+1), a2p[trial_window:], color='b')

    # Plot for excluding cpv
    if exclude_cpv:
        plt.plot(range(-trial_window, 0), p2a2[:trial_window], color='c')
        plt.plot(range(-trial_window, 0), a2p2[:trial_window], color='m')
        plt.plot(range(trial_window+1), p2a2[trial_window:], color='c')
        plt.plot(range(trial_window+1), a2p2[trial_window:], color='m')
        plt.plot([-1,0],p2a2[trial_window - 1:trial_window + 1],'y--')
        plt.plot([-1,0],a2p2[trial_window - 1:trial_window + 1],'y--')
        plt.scatter(range(-trial_window, 0), p2a2[:trial_window], color='c')
        plt.scatter(range(-trial_window, 0), a2p2[:trial_window], color='m')
        plt.scatter(range(trial_window+1), p2a2[trial_window:], color='m')
        plt.scatter(range(trial_window+1), a2p2[trial_window:], color='c')

    realp2aplot = plt.plot(range(-trial_window, 0), real_p2a[:trial_window], 'b--')
    reala2pplot = plt.plot(range(-trial_window, 0), real_a2p[:trial_window], 'r--')
    plt.plot(range(trial_window+1), real_p2a[trial_window:], 'r--')
    plt.plot(range(trial_window+1), real_a2p[trial_window:], 'b--')
    plt.plot([-1,0],real_p2a[trial_window - 1:trial_window + 1],'g--')
    plt.plot([-1,0],real_a2p[trial_window - 1:trial_window + 1],'g--')
    plt.scatter(range(-trial_window, 0), real_p2a[:trial_window], color='b')
    plt.scatter(range(-trial_window, 0), real_a2p[:trial_window], color='r')
    plt.scatter(range(trial_window+1), real_p2a[trial_window:], color='r')
    plt.scatter(range(trial_window+1), real_a2p[trial_window:], color='b')

    plt.legend([p2aplot, a2pplot, realp2aplot, reala2pplot],["RNN pro","RNN anti"
        ,"Real pro", "Real anti"])
    plt.xlabel('Trial from switch')
    plt.ylabel('Probability of correct')
    plt.title('Performance around switches')
    plt.show()
