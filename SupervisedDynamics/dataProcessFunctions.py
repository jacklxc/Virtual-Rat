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

def downloadRNN(rnid): # Problematic!!!
    CONN = db.Connection()
    CONN.use('vrat')
#    out = zip(*CONN.query('explain rnn'))
# That line did nothing.
    RNNs = CONN.query('select distinct(rnid) from vrat.rnn')
 # You would only do this if you wanted all the RNN? Also the distinct is not necessary since rnid is unique in vrat.rnn.    
    sqlstr = ('select ratname, hidden_dim, batch_size, learning_rate, model, rat from vrat.rnn where rnid = %s')
    # what table did you store this information in? D
    RNN = CONN.query(sqlstr, (rnid,))
    # This will give you a tuple of len 7
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

    for ratname, rat in rats.iteritems():
        p2a_mean += rat.p2a_prob
        a2p_mean += rat.a2p_prob

    p2a_mean /= len(rats)
    a2p_mean /= len(rats)

    return p2a_mean, a2p_mean

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

def sample_probabilities(probs, ratname, start = 0, end = 50):
    """
    Plots the first a few trial of the RNN softmax probability output.

    Inputs:
    - probs: numpy float array of shape (1, T, 3) which contains probabilities.
    - ratname: stirng, rat's name
    - sample: int, number of samples to display.
    """
    plt.plot(probs[0,start:end,0],'bo')
    plt.plot(probs[0,start:end,1],'ro')
    plt.plot(probs[0,start:end,2],'go')
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

def sample_correct_rate(rat, start=0, end=500):
    """
    Plots the first several trials of the prodicted correct rate.

    Inputs:
    - rat: object Rat.
    - sample: number of trials to display.
    """
    plt.plot(range(start, end), rat.pro_prob[start:end],color='blue')
    plt.plot(range(start, end), rat.anti_prob[start:end],color='red')

    plt.xlabel('Trials')
    plt.ylabel('%Correct')
    plt.title('Correct rate')
    plt.show()

def sample_correct_rate_new(rat, start=0, end=500):
    """
    Plots the first several trials of the prodicted correct rate.

    Inputs:
    - rat: object Rat.
    - sample: number of trials to display.
    """
    plt.plot(range(start, end), rat.pro_left_prob[start:end],color='blue')
    plt.plot(range(start, end), rat.anti_left_prob[start:end],color='red')
    plt.plot(range(start, end), rat.pro_right_prob[start:end],color='green')
    plt.plot(range(start, end), rat.anti_right_prob[start:end],color='orange')

    plt.xlabel('Trials')
    plt.ylabel('%Correct')
    plt.title('Correct rate')
    plt.show()

def draw_3d(p2a, a2p,real_p2a=None, real_a2p=None, trial_window = 3):
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

    if real_p2a and real_a2p:
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

    plt.legend([p2aplot, a2pplot],["pro","anti"])
    plt.xlabel('Trial from switch')
    plt.ylabel('Probability of correct')
    plt.title('Performance around switches')
    plt.show()

def draw_3d_new(p2a_left, p2a_right, a2p_left, a2p_right, trial_window = 3):
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
    - trial_window: int, number of trials computed before and after swtiches.
    """

    p2a_left_plot, = plt.plot(range(-trial_window, 0), p2a_left[:trial_window], color='b')
    a2p_left_plot, = plt.plot(range(-trial_window, 0), a2p_left[:trial_window], color='r')
    plt.plot(range(trial_window+1), p2a_left[trial_window:], color='r')
    plt.plot(range(trial_window+1), a2p_left[trial_window:], color='b')
    plt.plot([-1,0],p2a_left[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1,0],a2p_left[trial_window - 1:trial_window + 1],'k--')
    plt.scatter(range(-trial_window, 0), p2a_left[:trial_window], color='b')
    plt.scatter(range(-trial_window, 0), a2p_left[:trial_window], color='r')
    plt.scatter(range(trial_window+1), p2a_left[trial_window:], color='r')
    plt.scatter(range(trial_window+1), a2p_left[trial_window:], color='b')

    p2a_right_plot, = plt.plot(range(-trial_window, 0), p2a_right[:trial_window], color='green')
    a2p_right_plot, = plt.plot(range(-trial_window, 0), a2p_right[:trial_window], color='orange')
    plt.plot(range(trial_window+1), p2a_right[trial_window:], color='green')
    plt.plot(range(trial_window+1), a2p_right[trial_window:], color='orange')
    plt.plot([-1,0],p2a_right[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1,0],a2p_right[trial_window - 1:trial_window + 1],'k--')
    plt.scatter(range(-trial_window, 0), p2a_right[:trial_window], color='green')
    plt.scatter(range(-trial_window, 0), a2p_right[:trial_window], color='orange')
    plt.scatter(range(trial_window+1), p2a_right[trial_window:], color='green')
    plt.scatter(range(trial_window+1), a2p_right[trial_window:], color='orange')

    plt.legend([p2a_left_plot, a2p_left_plot, p2a_right_plot, a2p_right_plot],
        ["pro left","anti left", "pro right", "anti right"])
    plt.xlabel('Trial from switch')
    plt.ylabel('Probability of correct')
    plt.title('Performance around switches')
    plt.show()

def heatmap(h, start = 0, end = 1000):
    """
    Input:
    - h: numpy array in shape (T,N,H)
    """
    _, N, H = h.shape
    history = np.zeros((H,0))
    for n in range(N):
        history = np.append(history,h[:,n,:].T,axis=1)
    length = end-start
    magnify = int(length/H/3)

    history_slice = history[:,start:end]
    T = history_slice.shape[1]
    toPlot = np.zeros(((H-1)*magnify,T,3))


    for i in range(magnify):
        for j in range(H-2):
            for k in range(3):
                toPlot[magnify*j+i,:,k] = history_slice[j,:]
        pro = history_slice[j+1,:]==1
        anti = np.logical_not(pro)
        left =  history_slice[j+2,:]==0
        right = np.logical_not(left)

        toPlot[magnify*(j+1)+i,pro,2] = 1
        toPlot[magnify*(j+1)+i,anti,0] = 1
        toPlot[magnify*(j+1)+i,right,1] = 0.6
    plt.imshow(toPlot)
    plt.xlabel("Time step t")
    plt.title("Heatmap of hidden unit activations")
    plt.show()

def phasePlane(h, dim1=0, dim2=1, start = 0, end = 1000, xlim = (-1.2,1.2), 
        ylim = (-1.2,1.2), trajectory = True, scatter = True, mean = True):
    """
    Input:
    - h: numpy array in shape (T,N,H)
    """
    _, N, H = h.shape
    full_history = np.zeros((H,0))
    for n in range(N):
        full_history = np.append(full_history,h[:,n,:].T,axis=1)
    history = full_history[:,start:end]
    length = end-start
    dim1 = 0
    dim2 = 1

    pro = history[-2,:]==1
    anti = history[-2,:]==0
    left = history[-1,:]==0
    right = history[-1,:]==1
    switch = history[-2,1:]!=history[-2,:-1]
    switch = np.append([False],switch)
    not_switch = np.logical_not(switch)

    pro_left = np.logical_and(pro,left)
    pro_right = np.logical_and(pro,right)
    anti_left = np.logical_and(anti,left)
    anti_right = np.logical_and(anti,right)

    pro_left_switch = np.logical_and(pro_left,switch)
    pro_right_switch = np.logical_and(pro_right,switch)
    anti_left_switch = np.logical_and(anti_left,switch)
    anti_right_switch = np.logical_and(anti_right,switch)

    pro_left_not_switch = np.logical_and(pro_left,not_switch)
    pro_right_not_switch = np.logical_and(pro_right,not_switch)
    anti_left_not_switch = np.logical_and(anti_left,not_switch)
    anti_right_not_switch = np.logical_and(anti_right,not_switch)

    alpha1 = 0.5
    s1 = 5

    alpha2 = 0.2
    s2 = 60

    alpha3 = 0.8
    s3 = 250

    alpha4 = 0.8
    s4 = 400

    plt.xlim(xlim)
    plt.ylim(ylim)
    if trajectory:
        plt.plot(history[dim1,:],history[dim2,:],"k", alpha = 0.5)

    if scatter:
        PRO_LEFT = plt.scatter(history[dim1,pro_left_not_switch],
            history[dim2,pro_left_not_switch], color="blue", alpha = alpha1, s=s1)
        PRO_RIGHT = plt.scatter(history[dim1,pro_right_not_switch],
            history[dim2,pro_right_not_switch], color="green", alpha = alpha1, s=s1)
        ANTI_LEFT = plt.scatter(history[dim1,anti_left_not_switch],
            history[dim2,anti_left_not_switch], color = "red", alpha = alpha1, s=s1)
        ANTI_RIGHT = plt.scatter(history[dim1,anti_right_not_switch],
            history[dim2,anti_right_not_switch], color="orange", alpha = alpha1, s=s1)

        plt.scatter(history[dim1,pro_left_switch],
            history[dim2,pro_left_switch], color="blue", alpha = alpha2, s=s2)
        plt.scatter(history[dim1,pro_right_switch],
            history[dim2,pro_right_switch], color="green", alpha = alpha2, s=s2)
        plt.scatter(history[dim1,anti_left_switch],
            history[dim2,anti_left_switch], color="red", alpha = alpha2, s=s2)
        plt.scatter(history[dim1,anti_right_switch],
            history[dim2,anti_right_switch], color="orange", alpha = alpha2, s=s2)

        plt.legend([PRO_LEFT, PRO_RIGHT,ANTI_LEFT, ANTI_RIGHT],
        ["pro left","pro right", "anti left", "anti right"])

    if mean:
        plt.scatter(np.mean(history[dim1,pro_left_not_switch]),
            np.mean(history[dim2,pro_left_not_switch]), color="blue", alpha = alpha3, s=s3, marker = "^")
        plt.scatter(np.mean(history[dim1,pro_right_not_switch]),
            np.mean(history[dim2,pro_right_not_switch]), color="green", alpha = alpha3, s=s3, marker = "^")
        plt.scatter(np.mean(history[dim1,anti_left_not_switch]),
            np.mean(history[dim2,anti_left_not_switch]), color="red", alpha = alpha3, s=s3, marker = "^")
        plt.scatter(np.mean(history[dim1,anti_right_not_switch]),
            np.mean(history[dim2,anti_right_not_switch]), color="orange", alpha = alpha3, s=s3, marker = "^")

        plt.scatter(np.mean(history[dim1,pro_left_switch]),
            np.mean(history[dim2,pro_left_switch]), color="blue", alpha = alpha4, s=s4, marker = "*")
        plt.scatter(np.mean(history[dim1,pro_right_switch]),
            np.mean(history[dim2,pro_right_switch]), color="green", alpha = alpha4, s=s4, marker = "*")
        plt.scatter(np.mean(history[dim1,anti_left_switch]),
            np.mean(history[dim2,anti_left_switch]), color="red", alpha = alpha4, s=s4, marker = "*")
        plt.scatter(np.mean(history[dim1,anti_right_switch]),
            np.mean(history[dim2,anti_right_switch]), color="orange", alpha = alpha4, s=s4, marker = "*")
    
    
    plt.title("Phase plane of two dimensions of activations")
    plt.xlabel("Activation of dimension 1")
    plt.ylabel("Activation of dimension 2")
    plt.show()

def parallel_coordinate(h, start = 0, end = 1000, ylim = (-1.2,1.2)):
    """
    Input:
    - h: numpy array in shape (T,N,H)
    """
    _, N, H = h.shape
    full_history = np.zeros((H,0))
    for n in range(N):
        full_history = np.append(full_history,h[:,n,:].T,axis=1)
    history = full_history[:,start:end]
    length = end-start

    pro = history[-2,:]==1
    anti = history[-2,:]==0
    left = history[-1,:]==0
    right = history[-1,:]==1
    switch = history[-2,1:]!=history[-2,:-1]
    switch = np.append([False],switch)
    not_switch = np.logical_not(switch)

    pro_left = np.logical_and(pro,left)
    pro_right = np.logical_and(pro,right)
    anti_left = np.logical_and(anti,left)
    anti_right = np.logical_and(anti,right)

    pro_left_switch = np.logical_and(pro_left,switch)
    pro_right_switch = np.logical_and(pro_right,switch)
    anti_left_switch = np.logical_and(anti_left,switch)
    anti_right_switch = np.logical_and(anti_right,switch)

    pro_left_not_switch = np.logical_and(pro_left,not_switch)
    pro_right_not_switch = np.logical_and(pro_right,not_switch)
    anti_left_not_switch = np.logical_and(anti_left,not_switch)
    anti_right_not_switch = np.logical_and(anti_right,not_switch)

    #---------------------------------------#

    D = history.shape[0] -2 # Number of dimensions
    T = history.shape[1]

    plt.xlim((0.5,D+0.5))
    plt.ylim(ylim)
    alpha = 0.2
    for t in range(T):
        if pro_left[t]:
            color = "blue"
        elif pro_right[t]:
            color = "red"
        elif anti_left[t]:
            color = "green"
        else:
            color = "orange"

        if switch[t]:
            marker = "^"
        else:
            marker = "o"
        plt.plot(np.arange(1,D+1),history[:D,t], color = color, alpha = alpha)
        plt.scatter(np.arange(1,D+1),history[:D,t], color = color, marker = marker)

    plt.title("The Activations of Each Dimension")
    plt.xlabel("Dimension Number")
    plt.ylabel("Activation")
    plt.show()


def generateTrials(block_length,block_num):
    T = block_length * block_num
    X = np.zeros((1, T, 3), dtype = np.int8)
    y = np.zeros((1,T), dtype = np.int8)

    for i in xrange(T):
        quotient = i / block_length
        if quotient % 2 == 0:
            X[0,i,0] = 1
        else:
            X[0,i,0] = 0
        X[0,i,1] = np.random.randint(2)
    y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))

    return X, y

def randomLengthBlocks(minimum_length, block_num, Range = 7):
    block_length = np.random.randint(minimum_length, high = minimum_length+Range+1, 
        size = (block_num,))
    T = np.sum(block_length)
    X = np.zeros((1, T, 3), dtype = np.int8)
    y = np.zeros((1,T), dtype = np.int8)
    pro_rule = 1
    t = 0
    for length in block_length:
        count = length
        while count != 0:
            count -= 1
            X[0,t,0] = pro_rule
            X[0,t,1] = np.random.randint(2)
            t += 1
        pro_rule = np.bitwise_xor(pro_rule, 1)
    y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))

    return X, y

def save_weights(filename,weights):
    with open(filename,"wb") as f:
        pkl.dump(weights,f)

def load_weights(filename):
    with open(filename,"rb") as f:
        weights = pkl.load(f)
    return weights