import numpy as np
import scipy
import scipy.stats as st
import cPickle as pkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scikits.bootstrap as bootstrap

FONT_SIZE = 20

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
plt.rc('legend',fontsize=FONT_SIZE) # using a size in points

def figure_3d_matrix(rats = None, trial_window = 3):
    size = trial_window*2+1
    p2a_matrix = np.zeros((0,size))
    a2p_matrix = np.zeros((0,size))
    for rat in rats:
        if not rat.exclude:
            p2a_matrix = np.append(p2a_matrix, np.expand_dims(rat.p2a_prob * 100, axis=0), axis=0)
            a2p_matrix = np.append(a2p_matrix, np.expand_dims(rat.a2p_prob * 100, axis=0), axis=0)
    return p2a_matrix, a2p_matrix

def draw_3d(p2a = None, a2p = None, p2a_matrix = None, a2p_matrix = None, trial_window = 3, fixed_size = True, shift=0.05, filename = None):
    """
    Plots figure 3-d in Duan et al.'s paper. 

    Inputs:
    - p2a: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around pro to anti swtich.
    - a2p: a numpy float array of shape (trial_window*2+1,) that contains the 
        RNN softmax probability around anti to pro swtich.
    - trial_window: int, number of trials computed before and after swtiches.
    """
    if (p2a_matrix is not None) and (a2p_matrix is not None):
        p2a = np.mean(p2a_matrix,axis=0)
        p2a_SE = np.std(p2a_matrix,axis=0) / np.sqrt(p2a_matrix.shape[0])
        a2p = np.mean(a2p_matrix,axis=0)
        a2p_SE = np.std(a2p_matrix,axis=0) / np.sqrt(a2p_matrix.shape[0])
    
    fig, ax = plt.subplots(figsize=(6,4.5))
    if fixed_size:
        plt.ylim([0,100])
    else:
        plt.ylim([np.min([np.min(p2a),np.min(a2p)])-31, 100])
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

        alpha = 0.1
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
    ax.locator_params(axis = 'y', nbins = 6)
    #plt.legend([p2aplot, a2pplot],["pro","anti"],loc = "lower right")
    plt.xlabel('Trial from switch')
    plt.ylabel('% Correct')
    #plt.title('Performance around switches')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def switch_cost_difference_histogram(switch_cost_difference, bins = 11, filename = None):
    fig, ax = plt.subplots(figsize=(6,4.5))
    switch_cost_difference = switch_cost_difference * 100
    plt.hist(switch_cost_difference,bins = bins, weights=np.zeros_like(switch_cost_difference) + 1. / switch_cost_difference.size, color="black")
    plt.ylabel("Fraction")
    plt.xlabel("Pro switch cost - Anti switch cost in %")
    #plt.title("Histogram of switch cost difference, n="+str(switch_cost_difference.shape[0]))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()
    
def hot2index(hot, start = None, end = None):
    T = hot.shape[0]
    series = np.arange(T)
    indices = series[hot]
    if start is None:
        start = 0
    if end is None:
        end = T
    want = np.logical_and(indices >= start,indices < T)
    return indices[want]


def _make_str_ticks(time_steps):
    """
    Inputs:
    - time_steps: numpy array, number of time steps at each stage.
    - label: list of strings, name of each stage.
    """
    label = ["ITI", "Rule", "Delay", "Target", "Go"]
    ticks = []
    for i in range(time_steps.shape[0]):
        for j in range(time_steps[i]):
            ticks.append(label[i])
    return ticks


def PETH(rat, errorbar = False, switch = True, filename=None, legend = True):
    """
    Plot perievent time histogram of each hidden units.
    """
    total_time_steps = np.sum(rat.time_steps)
    config_names = rat.config_names

    colors = ["forestgreen","limegreen",(1,0.35,0),"orange"]
    shift = 0.05 if errorbar else 0
    num_configs = len(config_names)

    plots = []
    for dim in range(rat.num_dim):
        fig, ax = plt.subplots(figsize=(6,4.5))
        if switch:
            for config in range(num_configs/2):
                plot, = plt.plot(np.arange(total_time_steps) + shift, rat.activation_matrix_mean[dim,config,:],
                    linewidth=3, marker = "o", color = colors[config % (num_configs/2)], linestyle = "dashed")
                plots.append(plot)
                if errorbar:
                    plt.errorbar(np.arange(total_time_steps) + shift,rat.activation_matrix_mean[dim,config,:],
                        yerr = rat.activation_matrix_SE[dim,config,:],fmt = "None",elinewidth=3, ecolor=colors[config % (num_configs/2)])
        for config in range(num_configs/2,num_configs):
            plot, = plt.plot(np.arange(total_time_steps), rat.activation_matrix_mean[dim,config,:],
                linewidth=3, marker = "o", color = colors[config % (num_configs/2)])
            plots.append(plot)
            if errorbar:
                plt.errorbar(np.arange(total_time_steps),rat.activation_matrix_mean[dim,config,:],
                    yerr = rat.activation_matrix_SE[dim,config,:],fmt = "None",elinewidth=3, ecolor=colors[config % (num_configs/2)])
        
        if switch:
            if legend:
                plt.legend(plots,config_names,bbox_to_anchor=(1, 0.5),loc= "center left")
        else:
            if legend:
                plt.legend(plots,["pro left", "pro right", "anti left", "anti right"],bbox_to_anchor=(1, 0.5),loc= "center left")
        ticks = _make_str_ticks(rat.time_steps)
        plt.xlim([-0.5,total_time_steps - 0.5])
        plt.ylim([-1.1,1.1])
        plt.xticks(np.arange(total_time_steps),ticks)
        plt.ylabel("Activation")
        plt.xlabel("Time steps in one trial --->")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #plt.title("PETH of a sample hidden unit")
        #plt.title("PETH of Hidden Unit No."+str(dim+1))
        
        if filename:
            plt.savefig(filename+"-"+str(dim)+".pdf",dpi=600,bbox_inches='tight')
        else:
            plt.show()

def ROC_matrix(rats):
    configs1 = np.zeros((0,5))
    configs2 = np.zeros((0,5))
    for rat in rats:
        config1, config2 = rat.pro_encoding, rat.anti_encoding
        configs1 = np.append(configs1, np.expand_dims(config1, axis=0),axis=0)
        configs2 = np.append(configs2, np.expand_dims(config2, axis=0),axis=0)
    return configs1, configs2
        
def ROC(config1, config2, config1_name = "pro", config2_name = "anti", filename=None):
    """
    Plot the fraction of hidden unit that are significantly encoding rule signals at each time step.
    """
    total_time_steps = 5
    mean1 = np.mean(config1,axis=0)
    mean2 = np.mean(config2,axis=0)
    mean = mean1+mean2
    fig, ax = plt.subplots(figsize=(6,4.5))
    plot1 = plt.bar(np.arange(5),mean,1, color = "deepskyblue")
    plt.plot([-10,10],[0.01,0.01],"k--")
    ticks = _make_str_ticks(np.ones((5,), dtype=np.int))
    plt.xlim([-0.2,total_time_steps + 0.2])
    plt.xticks(np.arange(total_time_steps)+0.5,ticks)
    plt.ylim([0,1])
    plt.ylabel("Fraction of Selective Cells")
    plt.xlabel("Time steps in one trial --->")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title("Time course of rule encoding (n="+str(config1.shape[0]*20)+")")
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def ROC_target_matrix(rats):
    configs1 = np.zeros((0,5))
    configs2 = np.zeros((0,5))
    for rat in rats:
        config1, config2 = rat.left_encoding, rat.right_encoding
        configs1 = np.append(configs1, np.expand_dims(config1, axis=0),axis=0)
        configs2 = np.append(configs2, np.expand_dims(config2, axis=0),axis=0)
    return configs1, configs2
        
def ROC_target(config1, config2, config1_name = "left", config2_name = "right", filename=None):
    """
    Plot the fraction of hidden unit that are significantly encoding rule signals at each time step.
    """
    total_time_steps = 5
    mean1 = np.mean(config1,axis=0)
    mean2 = np.mean(config2,axis=0)
    mean = mean1+mean2
    fig, ax = plt.subplots(figsize=(6,4.5))
    plot1 = plt.bar(np.arange(5),mean,1, color = "deepskyblue")
    plt.plot([-10,10],[0.01,0.01],"k--")
    ticks = _make_str_ticks(np.ones((5,), dtype=np.int))
    plt.xlim([-0.2,total_time_steps + 0.2])
    plt.xticks(np.arange(total_time_steps)+0.5,ticks)
    plt.ylim([0,1])
    plt.ylabel("Fraction of Selective Cells")
    plt.xlabel("Time steps in one trial --->")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title("Time course of target encoding (n="+str(config1.shape[0]*20)+")")
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()
        
def ROC_combine(pro, anti, left, right, filename=None):
    total_time_steps = 5
    pro_mean = np.mean(pro,axis=0)
    anti_mean = np.mean(anti,axis=0)
    rule_mean = pro_mean+anti_mean
    left_mean = np.mean(left,axis=0)
    right_mean = np.mean(right,axis=0)
    target_mean = left_mean+right_mean
    fig, ax = plt.subplots(figsize=(6,4.5))
    plot1 = plt.plot(np.arange(5)+0.5,rule_mean,color="orange", linewidth=3, marker="o")
    plot2 = plt.plot(np.arange(5)+0.5,target_mean,color="mediumpurple", linewidth=3, marker="o")
    plt.plot([-10,10],[0.01,0.01],"k--")
    ticks = _make_str_ticks(np.ones((5,), dtype=np.int))
    plt.xlim([-0.2,total_time_steps + 0.2])
    plt.xticks(np.arange(total_time_steps)+0.5,ticks)
    plt.ylim([0,1])
    plt.ylabel("Fraction of Selective Cells")
    plt.xlabel("Time steps in one trial --->")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title("Time course of rule encoding (n="+str(config1.shape[0]*20)+")")
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()
    
def AUC_matrix(rats, AUC_name = "AUC", significant_name = "AUC_significant"):
    auc = np.zeros((0,20,5))
    significant = np.zeros((0,20,5))
    for rat in rats:
        auc = np.append(auc, np.expand_dims(getattr(rat,AUC_name), axis=0), axis = 0)
        significant = np.append(significant, np.expand_dims(getattr(rat,significant_name), axis=0), axis = 0)
    return auc, significant


def AUC_flip_histogram(rats = None, AUC = None, significant = None, filename = None, hidden_dim = 20.0):
    if rats:
        AUC,significant = AUC_matrix(rats)
    # else AUC and significant should not be None
    fig, ax = plt.subplots(figsize=(6,4.5))
    num_rats = AUC.shape[0]
    # Exclude the effect of flipping for those are not significant
    AUC[np.logical_not(significant)] = 0.5
    flips = (AUC-0.5)[:,:,np.arange(4)] * (AUC-0.5)[:,:,np.arange(1,5)] < 0
    total_flips = np.sum(np.sum(flips,axis=2) > 0,axis=1)
    flips_fraction = total_flips / float(AUC.shape[1])
    plt.hist(flips_fraction,bins = np.arange(hidden_dim+1)/hidden_dim, weights=np.zeros_like(flips_fraction) + 1. / flips_fraction.size, color="black")
    #plt.xlim([0.6,1.05])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.title("Histogram of preference flips of each hidden unit")
    plt.xlabel("Fraction of flipping hidden units per RNN")
    plt.ylabel("Frequency")
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()
    
def AUC_target_matrix(rats):
    auc = np.zeros((0,20,5))
    for rat in rats:
        auc = np.append(auc, np.expand_dims(rat.AUC_target, axis=0), axis = 0)
    return auc
    
def LearningCurveMatrix(rats):
    """
    Extract learning curve information from a dictionary of Rat objects.
    """
    pro_matrix = []
    anti_matrix = []

    for rat in rats:
        if not rat.exclude:
            pro_matrix.append(rat.pro_rate*100)
            anti_matrix.append(rat.anti_rate*100)
    pro_matrix = list2np(pro_matrix)
    anti_matrix = list2np(anti_matrix)
    
    return pro_matrix, anti_matrix

def learningCurve(pro_matrix, anti_matrix, exclude = True, filename = None):
    """
    Plot learning curve.
    """
    pro_mean = np.nanmean(pro_matrix, axis = 0)
    anti_mean = np.nanmean(anti_matrix, axis = 0)
    pro_SE = np.nanstd(pro_matrix, axis = 0) / np.sqrt(np.sum(np.isfinite(pro_matrix),axis=0))
    anti_SE = np.nanstd(anti_matrix, axis = 0) / np.sqrt(np.sum(np.isfinite(anti_matrix),axis=0))
    
    plt.figure()
    T = pro_mean.shape[0]
    plt.xlim([0,T+1])
    green = "green"
    orange = (1,0.35,0)
    pro_plot, = plt.plot(np.arange(T), pro_mean, color = green, linewidth = 3, marker = "o")
    anti_plot, = plt.plot(np.arange(T), anti_mean, color = orange, linewidth = 3, marker = "o")
    plt.errorbar(np.arange(T), pro_mean, yerr=pro_SE,
            fmt = "None", ecolor = green, elinewidth = 3)
    plt.errorbar(np.arange(T), anti_mean, yerr=anti_SE,
            fmt = "None", ecolor = orange, elinewidth = 3)
    for i in range(pro_matrix.shape[0]):
        plt.plot(np.arange(T), pro_matrix[i,:], color = green, alpha = 0.2)
        plt.plot(np.arange(T), anti_matrix[i,:], color = orange, alpha = 0.2)
        
    plt.legend([pro_plot, anti_plot],["pro","anti"],loc = "lower right")
    plt.title("ProAnti learning curves")
    plt.xlabel("Sessions of Training")
    plt.ylabel("% Correct")
    if filename:
        plt.savefig(filename,dpi=600)
    else:
        plt.show()
        
def asymmetry_vs_ratio(all_rats = None, pro_switch_costs = None, anti_switch_costs = None, ratio = None, exclude = True, filename = None, individual = True):
    """
    Input:
    - all_rats: A list of dictionaries. Each dictionary contain all Rat object whose
     RNN model is trained with same Pro to Anti ratio.
    - ratio: numpy array of ratios.
    - exclude: whether to exclude bad-performance agents.
    """
    shift = 0.012
    fig, ax = plt.subplots(figsize=(6,4.5))
    if all_rats:
        pro_switch_costs, anti_switch_costs = rat2matrix(all_rats, exclude)
        
    if not ratio: # ratio is not specified, calculate it by equally dividing.
        ratio = np.linspace(0,1,pro_switch_costs.shape[0])
    
    pro_switch_costs = pro_switch_costs * 100
    anti_switch_costs = anti_switch_costs * 100
    
    pro_mean = np.nanmean(pro_switch_costs, axis=1)
    anti_mean = np.nanmean(anti_switch_costs, axis=1)
    pro_SE = np.nanstd(pro_switch_costs,axis=1) / np.sqrt(np.sum(np.isfinite(pro_switch_costs),axis=1))
    anti_SE = np.nanstd(anti_switch_costs,axis=1) / np.sqrt(np.sum(np.isfinite(anti_switch_costs),axis=1))

    plt.xlim([-0.1,1.1])
    #plt.ylim([np.nanmin([np.nanmin(pro_switch_costs),np.nanmin(anti_switch_costs)])-0.05, \
    #    np.nanmax([np.nanmax(pro_switch_costs),np.nanmax(anti_switch_costs)])+0.05])
    plt.xticks(ratio)

    green = "green"
    orange = (1,0.35,0)
    pro_mean_plot, = plt.plot(ratio, pro_mean, color = green, linewidth=3, marker = "o")
    anti_mean_plot, = plt.plot(ratio+shift, anti_mean, color = orange, linewidth=3, marker = "o")
    plt.errorbar(ratio, pro_mean, ecolor = green, yerr = pro_SE, elinewidth=3, fmt = "None")
    plt.errorbar(ratio+shift, anti_mean, ecolor = orange, yerr = anti_SE, elinewidth=3, fmt = "None")

    alpha = 0.1
    
    if individual:
        for i in range(pro_switch_costs.shape[0]):
            plt.scatter(np.repeat(ratio[i],pro_switch_costs.shape[1]), pro_switch_costs[i,:], color = green, marker = "o", alpha = alpha)
            plt.scatter(np.repeat(ratio[i]+shift,anti_switch_costs.shape[1]), anti_switch_costs[i,:], color = orange, marker = "o", alpha = alpha)

    #plt.legend([pro_mean_plot, anti_mean_plot],["Pro","Anti"], loc = "lower center")
    plt.xticks(np.arange(0,11,2)/10.0)
    #plt.title("Switch cost vs Pro to Anti switch trial proportion during training")
    plt.xlabel("More Anti to Pro  <-->  More Pro to Anti")
    plt.ylabel("Switch Cost in %")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def asymmetry_difference_vs_ratio(all_rats = None, pro_switch_costs = None, anti_switch_costs = None, ratio = None, exclude = True, filename = None, individual = True):
    """
    Inputs:
    - all_rats: A list of dictionaries. Each dictionary contain all Rat object whose
     RNN model is trained with same Pro to Anti ratio.
    - ratio: numpy array of ratios.
    - exclude: whether to exclude bad-performance agents.
    """
    if all_rats:
        pro_switch_costs, anti_switch_costs = rat2matrix(all_rats, exclude)
        
    if not ratio: # ratio is not specified, calculate it by equally dividing.
        ratio = np.linspace(0,1,pro_switch_costs.shape[0])
        
    difference = pro_switch_costs - anti_switch_costs
    difference_mean = np.nanmean(difference, axis=1)
    difference_SE = np.nanstd(difference,axis=1) / np.sqrt(np.sum(np.isfinite(difference),axis=1))
    
    fig, ax = plt.subplots(figsize=(6,4.5))
    plt.xlim([-0.2,1.2])
    #plt.ylim([np.nanmin(difference)-0.05, np.nanmax(difference)+0.05])
    plt.plot(np.arange(-1,3),np.zeros((4,)), color="black")
    plt.xticks(ratio)

    mean_plot, = plt.plot(ratio, difference_mean, color = "blue",linewidth=3, marker = "o")
    plt.errorbar(ratio, difference_mean,yerr = difference_SE, fmt = "None", ecolor = "blue",elinewidth=3)

    alpha = 0.1
    
    if individual:
        for i in range(difference.shape[0]):
            plt.scatter(np.repeat(ratio[i],difference.shape[1]), difference[i,:], color = "blue", marker = "o", alpha = alpha)

    plt.title("Switch cost difference between Pro and Anti block vs Pro to Anti switch trial proportion during training")
    plt.xlabel("More Anti to Pro switches  <---   Pro to Anti switch trial proportion   --->  More Pro to Anti switches")
    plt.ylabel("Pro switch cost - Anti switch cost")
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def rat2matrix(all_rats, exclude):
    """
    Pack switch cost values from all_rats to matrix, so that Rat objects can be got rid of.
    pro_switch_costs: numpy array in shape (num_of_block_length, num_of_rat_data)
    anti_switch_costs: numpy array in shape (num_of_block_length, num_of_rat_data)
    """
    pro_switch_costs = []
    anti_switch_costs = []
    num_include=0

    for i in range(len(all_rats)):
        rats = all_rats[i]
        pro_switch_cost_this_block = []
        anti_switch_cost_this_block = []
        num_include = 0
        for rat in rats:
            if not (exclude and rat.exclude):
                pro_switch_cost = rat.pro_switch_cost
                anti_switch_cost = rat.anti_switch_cost
                num_include += 1
            else:
                pro_switch_cost = np.nan
                anti_switch_cost = np.nan
            pro_switch_cost_this_block.append(pro_switch_cost)
            anti_switch_cost_this_block.append(anti_switch_cost)
        pro_switch_cost_this_block = np.array(pro_switch_cost_this_block)
        anti_switch_cost_this_block = np.array(anti_switch_cost_this_block)
        pro_switch_costs.append(pro_switch_cost_this_block)
        anti_switch_costs.append(anti_switch_cost_this_block)
    pro_switch_costs_np = list2np(pro_switch_costs)
    anti_switch_costs_np = list2np(anti_switch_costs)
    return pro_switch_costs_np, anti_switch_costs_np

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

def bootstrapCI(data):
    """
    Input:
    - data: numpy array from list2np.
    """
    CIs = np.zeros((data.shape[0],2))
    for i in range(data.shape[0]):
        datum = data[i,:]
        datum_new = datum[np.logical_not(np.isnan(datum))]
        #CI = bootstrap.ci(data=datum_new, statfunction=scipy.median)
        CI = bootstrap(datum_new)
        CIs[i,0] = CI[0]
        CIs[i,1] = CI[1]
    return CIs

def bootstrap(data):
    """
    Input:
    - data: numpy vector.
    """
    data = data[np.logical_not(np.isnan(data))]
    repeat = 1000
    medians = []
    for i in range(repeat):
        indices = np.random.randint(data.size,size=data.size)
        sampled = data[indices]
        medians.append(np.median(sampled))
    return (np.percentile(medians,40.0), np.percentile(medians,60.0))

def switch_cost_vs_block_length(block_lengths, all_rats = None, switch_costs = None, exclude = True, filename = None, individual = True):
    """
    Plot switch cost vs block length.
    Inputs:
    - all_rats: A list of dictionaries. Each dictionary contain all Rat object whose
     RNN model is trained with same Pro to Anti ratio.
    - block_lengths: block lengths during training. numpy array.
    - exclude: boolean, whether to exclude bad-performance agents or not. 
    """
    fig, ax = plt.subplots(figsize=(6,4.5))
    if all_rats:
        pro_switch_costs, anti_switch_costs = rat2matrix(all_rats, exclude)
        switch_costs = (pro_switch_costs + anti_switch_costs) / 2 
    switch_costs = switch_costs * 100
    SC_mean = np.nanmean(switch_costs,axis=1)
    SC_SE = np.nanstd(switch_costs,axis=1) / np.sqrt(np.sum(np.isfinite(switch_costs),axis=1))
    
    plt.xticks(block_lengths)

    alpha = 0.05
    shift = 1
    if individual:
        for i in range(switch_costs.shape[0]):
            block = block_lengths[i]
            switch_cost = switch_costs[i,:]
                  
            plt.scatter(np.repeat(block+shift,switch_cost.size), switch_cost, color = "blue", marker = "o", alpha = alpha)

    mean_plot, = plt.plot(block_lengths, SC_mean, color = "blue", marker = "o", linewidth = 3)
    plt.errorbar(block_lengths, SC_mean, yerr=SC_SE, fmt = "None", ecolor = "blue", elinewidth = 3)
        
    #plt.title("Switch cost vs block length during training")
    plt.xlabel("Block length during training")
    plt.ylabel("Switch Cost in %")
    plt.xlim([1,53])
    ax.locator_params(axis = 'y', nbins = 6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def accuracy_vs_time_make_matrix(rats,num_loop,exclude):
    """
    Inputs:
    - rats: list of objects VirtualRat.
    """
    pro_block_matrix = np.zeros((0,num_loop))
    pro_switch_matrix = np.zeros((0,num_loop))
    anti_block_matrix = np.zeros((0,num_loop))
    anti_switch_matrix = np.zeros((0,num_loop))
    threshold = 0.8
    # The order of the rat numbers must be from 0 to n, cannot be messed by dictionary's hashing.
    for i in range(len(rats)):
        rat = rats[i]
        if not (exclude and rat.exclude):
            pro_block_matrix = np.append(pro_block_matrix, 
                                         np.expand_dims(rat.pro_block_accuracy_history,axis=0),axis=0)
            pro_switch_matrix = np.append(pro_switch_matrix, 
                                          np.expand_dims(rat.pro_switch_accuracy_history,axis=0), axis=0)
            anti_block_matrix = np.append(anti_block_matrix, 
                                          np.expand_dims(rat.anti_block_accuracy_history,axis=0), axis=0)
            anti_switch_matrix = np.append(anti_switch_matrix, 
                                           np.expand_dims(rat.anti_switch_accuracy_history,axis=0), axis=0)
    return pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix
    
def accuracy_vs_time(epoch_per_loop, num_loop, rats = None, matrices = None, individual = False, ylim = None, filename = None, xlim1=(0,2000), xlim2=(7500,8000)):
    """
    Plot the test performance growth of the network as training continues.
    """
    if rats:
        pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = \
            accuracy_vs_time_make_matrix(rats,num_loop,exclude)
    else:
        pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = matrices
    pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = \
        pro_block_matrix * 100, pro_switch_matrix * 100, anti_block_matrix * 100, anti_switch_matrix * 100
    pro_block_accuracy = np.mean(pro_block_matrix, axis=0)
    anti_block_accuracy = np.mean(anti_block_matrix, axis=0)

    pro_block_SE = np.std(pro_block_matrix, axis=0) / np.sqrt(pro_block_matrix.shape[0])
    anti_block_SE = np.std(anti_block_matrix, axis=0) / np.sqrt(anti_block_matrix.shape[0])

    figure = plt.figure()
    
    anti_start_index = 30
    episodes_pro = np.arange(num_loop) * epoch_per_loop
    episodes_anti = np.arange(anti_start_index,num_loop) * epoch_per_loop

    plt.xlim([0,episodes_pro[-1]+epoch_per_loop])
    if ylim:
        plt.ylim(ylim)

    green = "green"
    orange = (1,0.35,0)

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    gs.update(wspace=0.3)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])


    ax.get_shared_y_axes().join(ax, ax2)

    for AX in (ax,ax2):

        pro_block, = AX.plot(episodes_pro, pro_block_accuracy, color = green, linewidth = 3, marker = "o")
        anti_block, = AX.plot(episodes_anti, anti_block_accuracy[anti_start_index:], color = orange, linewidth = 3, marker = "o")

        AX.errorbar(episodes_pro, pro_block_accuracy,yerr=pro_block_SE, fmt = "None", ecolor = green, elinewidth = 3)
        AX.errorbar(episodes_anti, anti_block_accuracy[anti_start_index:],yerr=anti_block_SE[anti_start_index:], fmt = "None", ecolor = orange, elinewidth = 3)

        alpha = 0.1
        if individual:
            for i in range(pro_block_matrix.shape[0]):
                AX.plot(episodes_pro,pro_block_matrix[i,:], color = green, alpha = alpha)
                AX.plot(episodes_anti,anti_block_matrix[i,anti_start_index:], color = orange, alpha = alpha)
    
    ax.locator_params(axis = 'x', nbins = 1)
    ax2.locator_params(axis = 'x', nbins = 4)
    #ax.set_xlim(1300,2500)
    ax.set_xlim(xlim1)
    ax2.set_xlim(xlim2)
    
    plt.ylim([0,105])
    
    # hide the spines between ax and ax2
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('right')
    # hide the spines between ax and ax2
    #ax.spines['right'].set_visible(False)
    #ax2.spines['left'].set_visible(False)
    #ax.yaxis.tick_left()
    #ax.tick_params(labelright='off')
    #ax2.yaxis.tick_right()

    #plt.legend([pro_block,anti_block],["Pro", "Anti"], loc=4)
    figure.axes[0].set_xlabel("Number of training epochs",x = 2.3)

    figure.axes[0].set_ylabel("% Correct")
    #plt.xticks(episodes)
    #plt.title("Test Accuracy vs Number of Training Epochs (n="+str(pro_block_matrix.shape[0])+")", x = 0.28)
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()

def switch_cost_vs_time(epoch_per_loop, num_loop, rats = None, matrices = None, individual = False, ylim = [-5.5,0], filename = None, xlim1=(0,2000), xlim2=(7500,8000), combine = False):
    """
    Plot the test performance growth of the network as training continues.
    """
    if rats:
        pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = \
            accuracy_vs_time_make_matrix(rats,num_loop,exclude)
    else:
        pro_block_matrix, pro_switch_matrix, anti_block_matrix, anti_switch_matrix = matrices
    
    pro_switch_cost_matrix = (pro_switch_matrix - pro_block_matrix) * 100
    anti_switch_cost_matrix = (anti_switch_matrix - anti_block_matrix) *100
    switch_cost_matrix = (pro_switch_cost_matrix + anti_switch_cost_matrix) / 2 
    
    pro_switch_cost = np.mean(pro_switch_cost_matrix, axis=0)
    anti_switch_cost = np.mean(anti_switch_cost_matrix, axis=0)
    switch_cost = np.mean(switch_cost_matrix, axis=0)
    
    pro_SE = np.std(pro_switch_cost_matrix, axis=0) / np.sqrt(pro_switch_cost_matrix.shape[0])
    anti_SE = np.std(anti_switch_cost_matrix, axis=0) / np.sqrt(anti_switch_cost_matrix.shape[0])
    SC_SE = np.std(switch_cost_matrix, axis=0) / np.sqrt(switch_cost_matrix.shape[0])

    figure = plt.figure()
    
    start_index = 31
    episodes = np.arange(start_index,num_loop) * epoch_per_loop

    plt.xlim([0,episodes[-1]+epoch_per_loop])
    if ylim:
        plt.ylim(ylim)

    green = "green"
    orange = (1,0.35,0)

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    gs.update(wspace=0.4)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])


    ax.get_shared_y_axes().join(ax, ax2)

    for AX in (ax,ax2):
        if combine:
            SC, = AX.plot(episodes, switch_cost[start_index:], color = "blue", linewidth = 3)
        else:
            pro_SC, = AX.plot(episodes, pro_switch_cost[start_index:], color = green, linewidth = 3, marker = "o")
            anti_SC, = AX.plot(episodes, anti_switch_cost[start_index:], color = orange, linewidth = 3, marker = "o")
        if combine:
            AX.fill_between(episodes, switch_cost[start_index:] + SC_SE[start_index:], \
                             switch_cost[start_index:] - SC_SE[start_index:], edgecolor="lightblue",facecolor = "lightblue")
            
        else:
            AX.errorbar(episodes, pro_switch_cost[start_index:],yerr=pro_SE[start_index:], fmt = "None", ecolor = green, elinewidth = 3)
            AX.errorbar(episodes, anti_switch_cost[start_index:],yerr=anti_SE[start_index:], fmt = "None", ecolor = orange, elinewidth = 3)

        alpha = 0.1
        if individual:
            for i in range(pro_block_matrix.shape[0]):
                if combine:
                    AX.plot(episodes,switch_cost_matrix[i,start_index:], color = "blue", marker = "o", alpha = alpha)
                else:
                    AX.plot(episodes,pro_switch_cost_matrix[i,start_index:], color = green, marker = "o", alpha = alpha)
                    AX.plot(episodes,anti_switch_cost_matrix[i,start_index:], color = orange, marker = "o", alpha = alpha)
    
    ax.locator_params(axis = 'x', nbins = 5)
    ax2.locator_params(axis = 'x', nbins = 2)
    ax.locator_params(axis = 'y', nbins = 5)
    ax2.locator_params(axis = 'y', nbins = 5)
    #ax.set_xlim(1300,2500)
    ax.set_xlim(xlim1)
    ax2.set_xlim(xlim2)
    
    # hide the spines between ax and ax2
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('right')
    #ax.yaxis.tick_left()
    #ax.tick_params(labelright='off')
    #ax2.yaxis.tick_right()
    #if not combine:
    #    plt.legend([pro_SC, anti_SC], ["Pro", "Anti"], loc=4)
    figure.axes[0].set_xlabel("Number of training epochs",x = 0.8)

    figure.axes[0].set_ylabel("Switch cost in %")
    #plt.xticks(episodes)
    #plt.title("Model switch cost vs Number of training epochs", x = -1.2)
    plt.ylim(ylim)
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()


def report_accuracy(rat):
    """
    Report the network's test accuracy and whether it should be excluded.
    """
    print "Rat " + rat.name + " has pro block accuracy " + str(rat.pro_block_accuracy) + ", anti block accuracy " + \
    str(rat.anti_block_accuracy) + "."
    if rat.exclude:
        print "It is excluded from overall calculation."


def mixActivationMedian(normalized, filename = None, original_config_name = "pro block", opposite_config_name = "anti block"):
    """
    Inputs:
    - normalized: numpy array of shape (num_of_rats, num_of_repetition, num_of_proportion)
    """
    mean_over_repetition = np.nanmean(normalized,axis=1)
    num_proportion = mean_over_repetition.shape[-1]
    fig, ax = plt.subplots(figsize=(6,4.5))

    green = "green"
    orange = (1,0.35,0)
    alpha = 0.2
    
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.25,1.5])
    
    plt.boxplot(mean_over_repetition, whis = [15,85], showfliers = False, labels = np.linspace(0,1,num=num_proportion))
               #,positions = np.arange(11)/10.0)
    
    #plt.title("Switch cost ratio vs. proportion of "+original_config_name+" and "+opposite_config_name+" activation")
    plt.xlabel("Original rule <--> Opposite rule")
    plt.ylabel("Relative switch cost")
    plt.xticks(np.arange(0,11,2)+1,np.arange(0,11,2)/10.0)
    ax.locator_params(axis = 'x', nbins = 6)
    ax.locator_params(axis = 'y', nbins = 6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if filename:
        plt.savefig(filename,dpi=600,bbox_inches='tight')
    else:
        plt.show()
    

def diluteActivationMedian(switch_costs,filename = None):
    """
    Inputs:
    - switch_costs: numpy array of shape (num_of_rats, num_of_proportion)
    """
    
    num_proportion = switch_costs.shape[-1]
    fig, ax = plt.subplots(figsize=(6,4.5))

    green = "green"
    orange = (1,0.35,0)
    alpha = 0.2
    
    plt.xlim([-0.1,1.1])
    plt.ylim([-1.5,2.5])
    
    plt.boxplot(switch_costs, whis = [20,80], showfliers = False, labels = np.linspace(0,1,num=num_proportion))

    #plt.title("Switch cost ratio vs. proportion of carry over of activation")
    plt.xlabel("None <- Proportion of carry over -> All")
    plt.ylabel("Relative switch cost")
    plt.xticks(np.arange(0,11,2)+1,np.arange(0,11,2)/10.0)
    ax.locator_params(axis = 'x', nbins = 6)
    ax.locator_params(axis = 'y', nbins = 6)
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