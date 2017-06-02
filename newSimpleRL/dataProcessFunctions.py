import numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt

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
    p2a_left_mean = np.zeros(size)
    a2p_left_mean = np.zeros(size)
    p2a_right_mean = np.zeros(size)
    a2p_right_mean = np.zeros(size)

    num_exclude = 0

    for ratname, rat in rats.iteritems():
        if not rat.exclude:
            p2a_mean += rat.p2a_prob
            a2p_mean += rat.a2p_prob
            p2a_left_mean += rat.p2a_left_prob
            a2p_left_mean += rat.a2p_left_prob
            p2a_right_mean += rat.p2a_right_prob
            a2p_right_mean += rat.a2p_right_prob
        else:
            num_exclude += 1

    num_include = len(rats) - num_exclude
    p2a_mean /= num_include
    a2p_mean /= num_include
    p2a_left_mean /= num_include
    a2p_left_mean /= num_include
    p2a_right_mean /= num_include
    a2p_right_mean /= num_include

    return p2a_mean, a2p_mean, p2a_left_mean, a2p_left_mean, p2a_right_mean, \
    a2p_right_mean, num_include, num_exclude

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

def draw_3d(p2a, a2p,real_p2a=None, real_a2p=None, trial_window = 3, fixed_size = True):
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
    if fixed_size:
        plt.ylim([0,1])
    else:
        plt.ylim([np.min([np.min(p2a),np.min(a2p)])-0.05, 1])
    np.set_printoptions(precision=2)
    green = "green"
    orange = (1,0.35,0)
    plt.xlim([-trial_window-0.5,trial_window+0.5])
    p2aplot, = plt.plot(range(-trial_window, 0), p2a[:trial_window], color=green, linewidth = 2, marker = "o")
    a2pplot, = plt.plot(range(-trial_window, 0), a2p[:trial_window], color=orange, linewidth = 2,marker = "o")
    plt.plot(range(trial_window+1), p2a[trial_window:], color=orange,linewidth = 2, marker = "o")
    plt.plot(range(trial_window+1), a2p[trial_window:], color=green,linewidth = 2, marker = "o")
    plt.plot([-1,0],p2a[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1,0],a2p[trial_window - 1:trial_window + 1],'k--')

    if real_p2a and real_a2p:
        realp2aplot = plt.plot(range(-trial_window, 0), real_p2a[:trial_window], color = green,linewidth = 2, linestyle = "--")
        reala2pplot = plt.plot(range(-trial_window, 0), real_a2p[:trial_window], color = orange, linewidth = 2,linestyle = "--")
        plt.plot(range(trial_window+1), real_p2a[trial_window:], color = orange,linewidth = 2, linestyle = "--")
        plt.plot(range(trial_window+1), real_a2p[trial_window:], color = green, linewidth = 2,linestyle = "--")
        plt.plot([-1,0],real_p2a[trial_window - 1:trial_window + 1],'b--',linewidth = 2)
        plt.plot([-1,0],real_a2p[trial_window - 1:trial_window + 1],'b--', linewidth = 2)
        plt.scatter(range(-trial_window, 0), real_p2a[:trial_window], color = green,linewidth = 2)
        plt.scatter(range(-trial_window, 0), real_a2p[:trial_window], color=orange,linewidth = 2)
        plt.scatter(range(trial_window+1), real_p2a[trial_window:], color=orange,linewidth = 2)
        plt.scatter(range(trial_window+1), real_a2p[trial_window:], color = green,linewidth = 2)

    plt.legend([p2aplot, a2pplot],["pro","anti"],loc = "lower right")
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
    plt.ylim([0,1])
    np.set_printoptions(precision=2)

    plt.plot(range(-trial_window, 0), p2a_left[:trial_window], color='b')
    plt.plot(range(-trial_window, 0), a2p_left[:trial_window], color='r')
    anti_left, = plt.plot(range(trial_window+1), p2a_left[trial_window:], color='r')
    pro_left, = plt.plot(range(trial_window+1), a2p_left[trial_window:], color='b')
    plt.plot([-1,0],p2a_left[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1,0],a2p_left[trial_window - 1:trial_window + 1],'k--')
    plt.scatter(range(-trial_window, 0), p2a_left[:trial_window], color='b')
    plt.scatter(range(-trial_window, 0), a2p_left[:trial_window], color='r')
    plt.scatter(range(trial_window+1), p2a_left[trial_window:], color='r')
    plt.scatter(range(trial_window+1), a2p_left[trial_window:], color='b')

    plt.plot(range(-trial_window, 0), p2a_right[:trial_window], color='green')
    plt.plot(range(-trial_window, 0), a2p_right[:trial_window], color='orange')
    anti_right, = plt.plot(range(trial_window+1), p2a_right[trial_window:], color='orange')
    pro_right, = plt.plot(range(trial_window+1), a2p_right[trial_window:], color='green')
    plt.plot([-1,0],p2a_right[trial_window - 1:trial_window + 1],'k--')
    plt.plot([-1,0],a2p_right[trial_window - 1:trial_window + 1],'k--')
    plt.scatter(range(-trial_window, 0), p2a_right[:trial_window], color='green')
    plt.scatter(range(-trial_window, 0), a2p_right[:trial_window], color='orange')
    plt.scatter(range(trial_window+1), p2a_right[trial_window:], color='orange')
    plt.scatter(range(trial_window+1), a2p_right[trial_window:], color='green')

    plt.legend([pro_left, anti_left, pro_right, anti_right],
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
    full_history = h[:,0,:].T
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

def parallel_coordinate(h, rat, start = 0, end = 1000, mode = None, ylim = (-1.2,1.2)):
    """
    Input:
    - h: numpy array in shape (TT,N,H). 
        H[0:-2] is the activation. H[-2] is Pro/Anti: 1/0. H[-1] is right/left: 1/0.
    - rat: SimRat object that contains pre-calculated trial information.
    """
    _, N, H = h.shape
    full_history = h[:,0,:].T

    history = full_history[:,start:end] # history.shape = (D,T)
    D = history.shape[0] -2 # Number of dimensions
    T = history.shape[1]

    # Figure out indices corresponding trials.
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

    # Plot parallel coordinate.

    plt.xlim((0.5,D+0.5))
    plt.ylim(ylim)
    
    if mode == "average":
        # Average activations of corresponding trials.

        pro_left_switch_activation = np.mean(history[:D,pro_left_switch], axis=1)
        pro_right_switch_activation = np.mean(history[:D,pro_right_switch], axis=1)
        anti_left_switch_activation = np.mean(history[:D,anti_left_switch], axis=1)
        anti_right_switch_activation = np.mean(history[:D,anti_right_switch], axis=1)

        pro_left_not_switch_activation = np.mean(history[:D,pro_left_not_switch], axis=1)
        pro_right_not_switch_activation = np.mean(history[:D,pro_right_not_switch], axis=1)
        anti_left_not_switch_activation = np.mean(history[:D,anti_left_not_switch], axis=1)
        anti_right_not_switch_activation = np.mean(history[:D,anti_right_not_switch], axis=1)

        alpha = 1
        size = 50
        plt.plot(np.arange(1,D+1),pro_left_switch_activation, color = "blue", alpha = alpha, marker = "^")
        plt.plot(np.arange(1,D+1),pro_right_switch_activation, color = "red", alpha = alpha, marker = "^")
        plt.plot(np.arange(1,D+1),anti_left_switch_activation, color = "green", alpha = alpha, marker = "^")
        plt.plot(np.arange(1,D+1),anti_right_switch_activation, color = "orange", alpha = alpha, marker = "^")

        plt.plot(np.arange(1,D+1),pro_left_not_switch_activation, color = "blue", alpha = alpha,  marker = "o")
        plt.plot(np.arange(1,D+1),pro_right_not_switch_activation, color = "red", alpha = alpha,  marker = "o")
        plt.plot(np.arange(1,D+1),anti_left_not_switch_activation, color = "green", alpha = alpha,  marker = "o")
        plt.plot(np.arange(1,D+1),anti_right_not_switch_activation, color = "orange", alpha = alpha, marker = "o")

        plt.title("The Mean Activations of Each Dimension")

    elif mode == "accuracy":
        alpha = 0.2
        hit_rate = rat.hit_rate[start:end]
        min_hit_rate = np.min(hit_rate)
        max_hit_rate = np.max(hit_rate)
        interval = max_hit_rate - min_hit_rate

        for t in range(T):
            red = (max_hit_rate - hit_rate[t]) / interval
            blue = (hit_rate[t] - min_hit_rate) / interval

            if switch[t]:
                marker = "^"
            else:
                marker = "o"
            plt.plot(np.arange(1,D+1),history[:D,t], color = (red,0,blue), alpha = alpha)
            plt.scatter(np.arange(1,D+1),history[:D,t], color = (red,0,blue), marker = marker)
        plt.title("The Activations of Each Dimension")

    else:
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
    plt.xticks(np.arange(1,D+1))
    plt.show()

def meanLearningCurve(rats):
    for ratname, rat in rats.iteritems():
        size = rat.pro_rate.shape[0]
        break
    pro_matrix = np.zeros((0,size))
    anti_matrix = np.zeros((0,size))

    for ratname, rat in rats.iteritems():
        if not rat.exclude:
            pro_matrix = np.append(pro_matrix, np.expand_dims(rat.pro_rate, axis = 0), axis = 0)
            anti_matrix = np.append(anti_matrix, np.expand_dims(rat.anti_rate, axis = 0), axis = 0)
    pro_mean = np.nanmean(pro_matrix, axis = 0)
    anti_mean = np.nanmean(anti_matrix, axis = 0)
    return pro_mean, anti_mean

def learningCurve(pro_mean, anti_mean, rats = None, exclude = True):
    T = pro_mean.shape[0]
    plt.xlim([0,T+1])
    green = "green"
    orange = (1,0.35,0)
    pro_plot, = plt.plot(np.arange(T), pro_mean, color = green, linewidth = 3, marker = "o")
    anti_plot, = plt.plot(np.arange(T), anti_mean, color = orange, linewidth = 3, marker = "o")
    if rats:
        for ratname, rat in rats.iteritems():
            if not (exclude and rat.exclude):
                plt.plot(np.arange(T), rat.pro_rate, color = green, alpha = 0.2)
                plt.plot(np.arange(T), rat.anti_rate, color = orange, alpha = 0.2)
    plt.legend([pro_plot, anti_plot],["pro","anti"],loc = "lower right")
    plt.title("ProAnti learning curves")
    plt.xlabel("Sessions of Training")
    plt.ylabel("% Correct")
    plt.show()

def asymmetry_vs_ratio(all_rats, ratio = None, exclude = True):
    """
    Input:
    - all_rats: A list of dictionaries. Each dictionary contain all Rat object whose
     RNN model is trained with same Pro to Anti ratio.
    """
    pro_switch_costs = []
    anti_switch_costs = []
    if not ratio:
        ratio = np.arange(len(all_rats)) / float((len(all_rats) - 1))
    for i in range(len(all_rats)):
        rats = all_rats[i]
        pro_switch_costs.append([])
        anti_switch_costs.append([])
        for ratname, rat in rats.iteritems():
            if not (exclude and rat.exclude):
                pro_switch_cost = rat.pro_switch_cost
                anti_switch_cost = rat.anti_switch_cost
            else:
                pro_switch_cost = np.nan
                anti_switch_cost = np.nan
            pro_switch_costs[i].append(pro_switch_cost)
            anti_switch_costs[i].append(anti_switch_cost)
    pro_switch_costs = np.array(pro_switch_costs)
    anti_switch_costs = np.array(anti_switch_costs)

    pro_mean = np.nanmean(pro_switch_costs, axis=1)
    anti_mean = np.nanmean(anti_switch_costs, axis=1)

    plt.xlim([-0.2,1.2])
    plt.ylim([np.nanmin([np.nanmin(pro_switch_costs),np.nanmin(anti_switch_costs)])-0.05, \
        np.nanmax([np.nanmax(pro_switch_costs),np.nanmax(anti_switch_costs)])+0.05])
    plt.xticks(ratio)

    green = "green"
    orange = (1,0.35,0)

    pro_mean_plot, = plt.plot(ratio, pro_mean, color = green, linewidth=3, marker = "o")
    anti_mean_plot, = plt.plot(ratio, anti_mean, color = orange, linewidth=3, marker = "o")

    alpha = 0.2

    for i in range(pro_switch_costs.shape[1]):
        plt.scatter(ratio, pro_switch_costs[:,i], color = green, marker = "o", alpha = alpha)
        plt.scatter(ratio, anti_switch_costs[:,i], color = orange, marker = "o", alpha = alpha)

    plt.legend([pro_mean_plot, anti_mean_plot],["Pro","Anti"], loc = 4)

    plt.title("Switch cost vs Pro to Anti switch trial proportion during training")
    plt.xlabel("More Anti to Pro switches  <---   Pro to Anti switch trial proportion   --->  More Pro to Anti switches")
    plt.ylabel("Switch Cost")
    plt.show()

def asymmetry_difference_vs_ratio(all_rats, ratio = None, exclude = True):
    """
    Input:
    - all_rats: A list of dictionaries. Each dictionary contain all Rat object whose
     RNN model is trained with same Pro to Anti ratio.
    """
    pro_switch_costs = []
    anti_switch_costs = []
    if not ratio:
        ratio = np.arange(len(all_rats)) / float((len(all_rats) - 1))
    for i in range(len(all_rats)):
        rats = all_rats[i]
        pro_switch_costs.append([])
        anti_switch_costs.append([])
        for ratname, rat in rats.iteritems():
            if not (exclude and rat.exclude):
                pro_switch_cost = rat.pro_switch_cost
                anti_switch_cost = rat.anti_switch_cost
            else:
                pro_switch_cost = np.nan
                anti_switch_cost = np.nan
            pro_switch_costs[i].append(pro_switch_cost)
            anti_switch_costs[i].append(anti_switch_cost)
    pro_switch_costs = np.array(pro_switch_costs)
    anti_switch_costs = np.array(anti_switch_costs)

    switch_cost_difference = pro_switch_costs - anti_switch_costs

    difference_mean = np.nanmean(switch_cost_difference, axis=1)

    plt.xlim([-0.2,1.2])
    plt.ylim([np.nanmin(switch_cost_difference)-0.05, np.nanmax(switch_cost_difference)+0.05])
    plt.plot(np.arange(-1,3),np.zeros((4,)), color="black")
    plt.xticks(ratio)

    mean_plot, = plt.plot(ratio, difference_mean, color = "blue",linewidth=3, marker = "o")

    alpha = 0.2

    for i in range(switch_cost_difference.shape[1]):
        plt.scatter(ratio, switch_cost_difference[:,i], color = "blue", marker = "o", alpha = alpha)

    plt.title("Switch cost difference between Pro and Anti block vs Pro to Anti switch trial proportion during training")
    plt.xlabel("More Anti to Pro switches  <---   Pro to Anti switch trial proportion   --->  More Pro to Anti switches")
    plt.ylabel("Pro switch cost - Anti switch cost")
    plt.show()


def switch_cost_vs_block_length(all_rats, block_lengths, exclude = True):
    pro_switch_costs = []
    anti_switch_costs = []
    count=0
    for i in range(len(all_rats)):
        rats = all_rats[i]
        pro_switch_costs.append([])
        anti_switch_costs.append([])
        for ratname, rat in rats.iteritems():
            if rat.exclude:
                count+=1
                print count
            if not (exclude and rat.exclude):
                pro_switch_cost = rat.pro_switch_cost
                anti_switch_cost = rat.anti_switch_cost
            else:
                pro_switch_cost = np.nan
                anti_switch_cost = np.nan
            pro_switch_costs[i].append(pro_switch_cost)
            anti_switch_costs[i].append(anti_switch_cost)
    pro_switch_costs = np.array(pro_switch_costs)
    anti_switch_costs = np.array(anti_switch_costs)
    
    pro_mean = np.nanmean(pro_switch_costs, axis=1)
    anti_mean = np.nanmean(anti_switch_costs, axis=1)

    plt.ylim([np.nanmin([np.nanmin(pro_switch_costs),np.nanmin(anti_switch_costs)])-0.05, \
        np.max([np.nanmax(pro_switch_costs),np.nanmax(anti_switch_costs)])+0.05])
    plt.xticks(block_lengths)
 
    green = "green"
    orange = (1,0.35,0)

    pro_mean_plot, = plt.plot(block_lengths, pro_mean, color = green, marker = "o", linewidth = 3)
    anti_mean_plot, = plt.plot(block_lengths, anti_mean, color = orange, marker = "o", linewidth = 3)

    alpha = 0.2

    for i in range(pro_switch_costs.shape[1]):
        plt.scatter(block_lengths, pro_switch_costs[:,i], color = green, marker = "o", alpha = alpha)
        plt.scatter(block_lengths, anti_switch_costs[:,i], color = orange, marker = "o", alpha = alpha)

    plt.legend([pro_mean_plot, anti_mean_plot],["Pro","Anti"], loc = "lower left")

    plt.title("Switch cost vs block length during training")
    plt.xlabel("Block length during training")
    plt.ylabel("Switch Cost")
    plt.show()

def accuracy_vs_time(rats, epoch_per_loop, num_loop, exclude = True):
    pro_block_matrix = np.zeros((0,num_loop))
    pro_switch_matrix = np.zeros((0,num_loop))
    anti_block_matrix = np.zeros((0,num_loop))
    anti_switch_matrix = np.zeros((0,num_loop))

    for ratname, rat in rats.iteritems():
        if not (exclude and rat.exclude):
            pro_block_matrix = np.append(pro_block_matrix, 
                np.expand_dims(rat.pro_block_accuracy_history,axis=0),axis=0)
            pro_switch_matrix = np.append(pro_switch_matrix, 
                np.expand_dims(rat.pro_switch_accuracy_history,axis=0), axis=0)
            anti_block_matrix = np.append(anti_block_matrix, 
                np.expand_dims(rat.anti_block_accuracy_history,axis=0), axis=0)
            anti_switch_matrix = np.append(anti_switch_matrix, 
                np.expand_dims(rat.anti_switch_accuracy_history,axis=0), axis=0)
    
    pro_block_accuracy = np.mean(pro_block_matrix, axis=0)
    pro_switch_accuracy = np.mean(pro_switch_matrix, axis=0)
    anti_block_accuracy = np.mean(anti_block_matrix, axis=0)
    anti_switch_accuracy = np.mean(anti_switch_matrix, axis=0)

    episodes = np.arange(1,num_loop+1) * epoch_per_loop
    plt.xlim([0,episodes[-1]+epoch_per_loop])

    green = "green"
    orange = (1,0.35,0)

    pro_block, = plt.plot(episodes, pro_block_accuracy, color = green, linewidth = 3, marker = "o")
    pro_switch, = plt.plot(episodes, pro_switch_accuracy, color = green, linewidth = 3, marker = "^")
    anti_block, = plt.plot(episodes, anti_block_accuracy, color = orange, linewidth = 3, marker = "o")
    anti_switch, = plt.plot(episodes, anti_switch_accuracy, color = orange, linewidth = 3, marker = "^")

    alpha = 0.1
    for i in range(pro_block_matrix.shape[0]):
        plt.plot(episodes,pro_block_matrix[i,:], color = green, marker = "o", alpha = alpha)
        plt.plot(episodes,pro_switch_matrix[i,:], color = green, marker = "^", alpha = alpha)
        plt.plot(episodes,anti_block_matrix[i,:], color = orange, marker = "o", alpha = alpha)
        plt.plot(episodes,anti_switch_matrix[i,:], color = orange, marker = "^", alpha = alpha)

    plt.legend([pro_block, pro_switch,anti_block,anti_switch],\
        ["Pro block accuracy", "Pro switch accuracy","Anti block accuracy", "Anti switch accuracy"], loc=4)
    plt.xlabel("Number of training epochs")
    plt.ylabel("Test accuracy")
    plt.xticks(episodes)
    plt.title("Test Accuracy vs Number of Training Epochs")
    plt.show()

def report_accuracy(rat):
    print "Rat " + rat.name + " has pro block accuracy " + str(rat.pro_block_accuracy) + ", anti block accuracy " + \
    str(rat.anti_block_accuracy) + "."
    if rat.exclude:
        print "It is excluded from overall calculation."

def single_model_parameter_trajectory(parameter_history):
    params2color = {"b":"blue","temperature":"black","Wa":"pink","Wh":"red","ba":"cyan","Wx":"orange"}
    for k, vv in parameter_history.iteritems():
        v = vv.asnumpy()
        if len(v.shape)==1:
            plt.plot(np.arange(v.shape[0]),v[:],color=params2color[k])
        elif len(v.shape)==2:
            for i in range(v.shape[1]):
                plt.plot(np.arange(v.shape[0]),v[:,i],color=params2color[k])
        elif len(v.shape)==3:
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    plt.plot(np.arange(v.shape[0]),v[:,i,j],color=params2color[k])   
    
    plt.title("Trajectory of parameters dynamics in a single model")
    plt.xlabel("Iterations")
    plt.ylabel("Parameter values")
    plt.show()

def params2vector(parameter_history,index):
    """
    Convert the parameters at a time step index to vectors.
    """
    params = []
    for k, vv in parameter_history.iteritems():
        v = vv#.asnumpy()
        if len(v.shape)==1:
            params.append(v[index])
        elif len(v.shape)==2:
            for i in range(v.shape[1]):
                params.append(v[index,i])
        elif len(v.shape)==3:
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    params.append(v[index,i,j])
    return np.array(params)



def save_weights(filename,weights):
    with open(filename,"wb") as f:
        pkl.dump(weights,f)

def load_weights(filename):
    with open(filename,"rb") as f:
        weights = pkl.load(f)
    return weights