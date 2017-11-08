import numpy as np 
from scipy.stats import rankdata
from scipy.stats import ttest_ind as t_test
from sklearn.metrics import roc_auc_score
from VirtualRatBox import VirtualRatBox

class VirtualRat(object):
    def __init__(self, RNN, ratname = "VirtualRat", trial_window=3, 
        accuracy_threshold = 0.8, time_steps = (1,1,1,1,1)):

        """
        Create attributes and call methods to get their values.

        Inputs: 
        - RNN: Object of RNN model
        - ratname: string, the name of rat
        - trial_window: int, number of trials computed before and after swtiches.
        - time_steps: Number of time steps take for each period. (ITI, 
            rule, delay, orientation, response)
        """

        self.name = ratname
        self.trial_window = trial_window
        self.RNN = RNN # The temporal model
        self.training_history = np.zeros((0,5)) # (pro_rule, left/right, trial=1, action, reward)
        self.accuracy_threshold = accuracy_threshold # The threshold of block accuracy to judge if the model should be excluded.
        self.exclude = True
        self.time_steps = np.array(time_steps) #(ITI, rule, delay, orientation, response)
        self.total_time_steps = np.sum(self.time_steps)
         # For the experiment of tracking accracy vs training time
        self.pro_switch_accuracy_history = np.zeros((0,))
        self.pro_block_accuracy_history = np.zeros((0,))
        self.anti_switch_accuracy_history = np.zeros((0,))
        self.anti_block_accuracy_history = np.zeros((0,))

        self.num_dim = self.RNN.hidden_dim # 20

        # Different set of configuration names.
        self.basic_config_names = ["pro","anti","left","right","switch","block"]
        self.simple_config_names = ["pro switch", 
            "anti switch",
            "pro block",
            "anti block"]
        self.prev_config_names = ["pro prev left block",
            "pro prev right block",
            "anti prev left block",
            "anti prev right block"]
        self.config_names = ["pro left switch", 
            "pro right switch",
            "anti left switch",
            "anti right switch",
            "pro left block",
            "pro right block",
            "anti left block",
            "anti right block"]

        self.all_config_names = self.basic_config_names + self.simple_config_names + self.config_names + self.prev_config_names

    def predict(self, temporalX, temporalY):
        """
        Make prediction based on the trianed RNN.

        Inputs:
        - temporalX, temporalY: Inputs and labels of the validation/test data. 
            (N=1, T*total_time_steps, 5), (N, T*total_time_steps)

        Returns: 
        - self.probility: numpy array of shape (N, T, DD) giving the probability of each choice,
          where each element is a float. 
        """

        self.RNN.reset_history() # Reset RNN's activation history recorded before each forward pass.
        probability = self.RNN.forward(temporalX).asnumpy() # Run forward pass to make prediction.
        # Post RNN data process
        # Process probability itself.
        newX, y, self.probability = self._temporal2static(temporalX, temporalY, probability)
        self.trials = self._trial_type(newX)
        X = self._new2oldX(newX) # Convert input X from new format (1,T,5) to old format (1,T,3).
        # Exclude cpv probability and normalize.
        self.normalized_probs = self._normalizeProbs(self.probability) 
        # Combines the probabilities to get the probabilities of choosing correctly.
        self.hit_rate = self._hitRate(y, self.normalized_probs)
        self.T = self.hit_rate.shape[0]

        # Separate probability based on rule and direction.
        self.pro_prob, self.anti_prob = self._proAntiProbability(X, self.hit_rate)
        # Calculate switch cost
        self.pro_switch_cost, self.pro_switch_accuracy, self.pro_block_accuracy = self._switch_cost(self.pro_prob)
        self.anti_switch_cost, self.anti_switch_accuracy, self.anti_block_accuracy = self._switch_cost(self.anti_prob)

        if self.pro_block_accuracy > self.accuracy_threshold and self.anti_block_accuracy > self.accuracy_threshold:
            self.exclude = False # Exclude this rat's data when the performance is lower than threshold 

        # Prepare data for plotting.
        self.p2a_prob = self._calculate3d("anti switch", self.hit_rate)
        self.a2p_prob = self._calculate3d("pro switch", self.hit_rate)

        return self.probability

    def compute_PETH_ROC(self):
        # Compute perievent time histogram using activation history on pro left block, pro right block ....
        self.config_array, self.normalized_activation_matrices, self.activation_matrix_mean,\
            self.activation_matrix_SE = self._PETH(self.RNN.get_activation_history(),\
            self.hit_rate, self.config_names, self.trials)

        # Compute PETH on two kinds of configurations specified below. This self.two_config_normalized_activation_matrices
        # is used for self._ROC().
        self.categorized_activation_trajectory = \
            self._activationTrajectoryByTrials(self.RNN.get_activation_history(),self.all_config_names)

        self.pro_encoding, self.anti_encoding, self.AUC, self.AUC_significant = self._ROC(self.num_dim, "pro block", "anti block", 
             self.categorized_activation_trajectory)

        self.left_encoding, self.right_encoding, self.AUC_target, self.AUC_target_significant = self._ROC(self.num_dim, "left", "right", 
             self.categorized_activation_trajectory)


    def add_prediction_history(self):
        """
        After each predict() save prediction history in order to plot the change of model performance
        as training time increases.
        """
        self.pro_switch_accuracy_history = np.append(self.pro_switch_accuracy_history, self.pro_switch_accuracy)
        self.pro_block_accuracy_history = np.append(self.pro_block_accuracy_history, self.pro_block_accuracy)
        self.anti_switch_accuracy_history = np.append(self.anti_switch_accuracy_history, self.anti_switch_accuracy)
        self.anti_block_accuracy_history = np.append(self.anti_block_accuracy_history, self.anti_block_accuracy)

    def _temporal2static(self, temporalX, temporalY, temporal_probability):
        """
        Convert temporal sequences to static sequences. For probability, only extract the ones at "Go" time step.

        Inputs:
        temporalX, temporalY: Inputs and labels of the validation/test data. 
            (N=1, T*total_time_steps, 5), (N, T*total_time_steps)
        temporal_probability: (N=1, T*total_time_steps, 3)

        Returns:
        X: (N=1, T, 5)
        y: (N=1, T)
        probability: (N=1, T, 3)
        """
        TT = temporalX.shape[1]
        total_time_steps = np.sum(self.time_steps)
        T = TT / total_time_steps
        X = np.zeros((1,T,5),dtype=np.bool)
        y = np.zeros((1,T),dtype=np.int)
        probability = np.zeros((1,T,3))
        cumulative_time = np.cumsum(self.time_steps)
        for t in range(T):
            X[:,t,:] = np.sum(temporalX[:,t*total_time_steps : (t+1)*total_time_steps,:],axis=1) > 0
            # Convert responses in temporalY to the y we used before.
            y[0,t] = temporalY[0,t*total_time_steps+cumulative_time[3]]
            probability[0,t,:] = temporal_probability[0,t*total_time_steps+cumulative_time[3],:]
        return X, y, probability


    def _switch_cost(self, prob):
        """
        Calculate switch cost given probability record.

        Inputs:
        - prob: pro or anti accuracy. (With some None in opposite rules.) numpy array of shape (T,).

        Returns:
        - switch_cost_mean: switch cost, float.
        - switch_accuracy_mean: accuracy of switch trials, float.
        - block_accuracy_cost_mean: accuracy of block trials, float.
        """
        switch = 0
        block = np.zeros((0,))
        new_block = True
        switch_costs = np.zeros((0,))
        switch_accucary = np.zeros((0,)) # Save all switch_accucary
        block_accruacy = np.zeros((0,)) # Save all block_accucary
        for t in range(prob.shape[0]):
            if np.isnan(prob[t]):
                if not new_block:
                    switch_cost = switch - np.mean(block)
                    switch_costs = np.append(switch_costs,switch_cost)
                    block_accruacy = np.append(block_accruacy, block)
                    block = np.zeros((0,))
                    new_block = True

            else:
                if new_block:
                    switch = prob[t]
                    switch_accucary = np.append(switch_accucary, switch)
                    new_block = False
                else:
                    block = np.append(block,prob[t])
        if block.shape[0]>0:
            switch_cost = switch - np.mean(block)
            switch_costs = np.append(switch_costs,switch_cost)

        switch_cost_mean = np.mean(switch_costs)
        switch_accucary_mean = np.mean(switch_accucary)
        block_accruacy_mean = np.mean(block_accruacy)

        return switch_cost_mean, switch_accucary_mean, block_accruacy_mean

    def _new2oldX(self,X):
        """
        Convert new input X format to old format in order to compute stats.
        (pro, anti, left, right, trial=1) to (pro_rule,left,trial=1)  
        """
        N,T,_ = X.shape
        oldX = np.zeros((N,T,3),dtype = np.bool)
        pro = X[:,:,1]==0
        right = X[:,:,3]==1
        oldX[pro,0] = 1
        oldX[right,1] = 1
        oldX[:,:,2] = X[:,:,4]
        return oldX

    def addHistory(self, record):
        """
        Add training history from Box.
        """
        self.training_history = np.append(self.training_history,record,axis=0)

    def computeLearningCurve(self, session_length = 100):
        """
        Use raw training history to compute accuracy of each session to plot on learning curve based on session_length.
        """
        T, D = self.training_history.shape
        total_session = T/session_length
        self.pro_rate = np.zeros((total_session,))
        self.anti_rate = np.zeros((total_session))

        for i in range(total_session):
            history_piece = self.training_history[i*session_length:(i+1)*session_length,:]
            self.pro_rate[i], self.anti_rate[i] = self._sessionAccuracy(history_piece)

    def _sessionAccuracy(self,history):
        """
        history[0,:] =  (pro_rule, left/right, trial=1, action, reward), i.e. (X, action, reward)
        """
        switch = history[1:,0]!=history[:-1,0]
        switch = np.append([False],switch)

        block = np.logical_not(switch)
        pro = history[:,0]==1
        anti = history[:,0]==0
        pro_block = np.logical_and(pro, block)
        anti_block = np.logical_and(anti, block)

        correct = history[:,-1]>0
        pro_rate = np.mean(correct[pro_block])
        anti_rate = np.mean(correct[anti_block])

        return pro_rate, anti_rate

    def _normalizeProbs(self, probability):
        """
        Exclude the probability of cpv and normalize the probabilities of left and right.

        Inputs:
        - probability: numpy matrix shape (1,T,3)

        Returns:
        - normalized_probs: numpy array shape (1,T,2)
        """
        normalized_probs = np.zeros((probability.shape[0],probability.shape[1],
            probability.shape[2]-1))
        normalized_probs[:,:,0] = probability[:,:,0]/(probability[:,:,0]
         + probability[:,:,1])
        normalized_probs[:,:,1] = probability[:,:,1]/(probability[:,:,0] 
            + probability[:,:,1])
        return normalized_probs

    def _hitRate(self,trueY, normalized_probs):
        """
        Combines the probabilities to get the probabilities of choosing correctly.

        Inputs:
        - normalized_probs: numpy array shape (1,T,2)

        Returns:
        - hit_rate: numpy array shape (T,)
        """
        hit_rate = np.zeros(normalized_probs.shape[1])
        right = trueY[0,:] == 1
        left = trueY[0,:] == 0
        hit_rate[left] = normalized_probs[0,left,0]
        hit_rate[right] = normalized_probs[0,right,1]
        return hit_rate

    def _proAntiProbability(self,X, hit_rate):
        """
        Seperate self.hit_rate to probabilites of correct choice during pro and anti 
        for the purpose of plotting figure 3d (performances around switches). 
        Trials for different rules are filled with None.

        Inputs:
        - hit_rate: numpy array shape (T,)

        Returns:
        - pro_prob: numpy array shape (T,)
        - anti_prob: numpy array shape (T,)
        """
        pro_rules = X[0,:,0]
        pro_prob = np.copy(hit_rate)
        anti_prob = np.copy(hit_rate)
        pro_rules = pro_rules.astype(bool)
        pro_prob[np.logical_not(pro_rules)] = None
        anti_prob[pro_rules] = None
        return pro_prob, anti_prob


    def _calculate3d(self, switch_type, hit_rate):
        """
        Calculate the switch cost for this particular rat.

        Inputs:
        - switch_index: indices of trial numbers that switch occurs.
        - hit_rate: correct probability of each trial.

        Returns:
        - switch_prob: numpy float array that contains mean correct rate around switches
        """     
        # index i --> trial from switch = -trial_window + i
        switch_matrix = np.zeros((np.sum(self.trials[switch_type]), self.trial_window * 2 + 1))
        switches = np.where(self.trials[switch_type]==1)[0]
        for i in range(switches.size-1):
            switch = switches[i]
            switch_matrix[i,:] = hit_rate[(switch - self.trial_window):
             (switch + self.trial_window + 1)]
        swtich_cost = np.nanmean(switch_matrix,axis=0)
        return swtich_cost

    def _trial_type(self, X):
        """
        Figures out the input configurations of each trial based on static input X.
        All in one-hot style, i.e. [True, False, ...]

        Input:
        - X: static inputs in numpy array shape (1, T, 5)

        """
        trials = {}
        trials["anti"] = X[0,:,1] == 1
        trials["pro"]= np.logical_not(trials["anti"])
        trials["left"] = X[0,:,2] == 1
        trials["right"] = X[0,:,3] == 1

        trials["prev_pro"], trials["prev_anti"], trials["prev_left"], trials["prev_right"] \
            = self._prev_trial_type(X)

        switch = X[0,1:,0]!=X[0,:-1,0]
        trials["switch"] = np.append([False],switch)
        trials["block"] = np.logical_not(trials["switch"])
        trials["before_and_after_switch"] = np.logical_or(np.append(\
            trials["switch"][1:],[False]), trials["switch"])

        trials["pro block"] = np.logical_and(trials["pro"],trials["block"])
        trials["pro switch"] = np.logical_and(trials["pro"],trials["switch"])
        trials["anti block"] = np.logical_and(trials["anti"],trials["block"])
        trials["anti switch"] = np.logical_and(trials["anti"],trials["switch"])


        trials["pro left"] = np.logical_and(trials["pro"],trials["left"])
        trials["pro right"] = np.logical_and(trials["pro"],trials["right"])
        trials["anti left"] = np.logical_and(trials["anti"],trials["left"])
        trials["anti right"] = np.logical_and(trials["anti"],trials["right"])

        trials["pro left switch"] = np.logical_and(trials["pro left"],trials["switch"])
        trials["pro right switch"] = np.logical_and(trials["pro right"],trials["switch"])
        trials["anti left switch"] = np.logical_and(trials["anti left"],trials["switch"])
        trials["anti right switch"] = np.logical_and(trials["anti right"],trials["switch"])

        trials["pro left block"] = np.logical_and(trials["pro left"],trials["block"])
        trials["pro right block"] = np.logical_and(trials["pro right"],trials["block"])
        trials["anti left block"] = np.logical_and(trials["anti left"],trials["block"])
        trials["anti right block"] = np.logical_and(trials["anti right"],trials["block"])
        
        trials["pro prev left block"] = np.logical_and(trials["pro block"],trials["prev_left"])
        trials["pro prev right block"] = np.logical_and(trials["pro block"],trials["prev_right"])

        trials["anti prev left block"] = np.logical_and(trials["anti block"],trials["prev_left"])
        trials["anti prev right block"] = np.logical_and(trials["anti block"],trials["prev_right"])

        trials["pro left to left switch"] = np.logical_and(trials["pro left switch"],trials["prev_left"])
        trials["pro right to left switch"] = np.logical_and(trials["pro left switch"],trials["prev_right"])
        trials["pro left to right switch"] = np.logical_and(trials["pro right switch"],trials["prev_left"])
        trials["pro right to right switch"] = np.logical_and(trials["pro right switch"],trials["prev_right"])

        trials["anti left to left switch"] = np.logical_and(trials["anti left switch"],trials["prev_left"])
        trials["anti right to left switch"] = np.logical_and(trials["anti left switch"],trials["prev_right"])
        trials["anti left to right switch"] = np.logical_and(trials["anti right switch"],trials["prev_left"])
        trials["anti right to right switch"] = np.logical_and(trials["anti right switch"],trials["prev_right"])

        trials["pro left to left block"] = np.logical_and(trials["pro left block"],trials["prev_left"])
        trials["pro right to left block"] = np.logical_and(trials["pro left block"],trials["prev_right"])
        trials["pro left to right block"] = np.logical_and(trials["pro right block"],trials["prev_left"])
        trials["pro right to right block"] = np.logical_and(trials["pro right block"],trials["prev_right"])

        trials["anti left to left block"] = np.logical_and(trials["anti left block"],trials["prev_left"])
        trials["anti right to left block"] = np.logical_and(trials["anti left block"],trials["prev_right"])
        trials["anti left to right block"] = np.logical_and(trials["anti right block"],trials["prev_left"])
        trials["anti right to right block"] = np.logical_and(trials["anti right block"],trials["prev_right"])


        return trials

    def _prev_trial_type(self, X):
        """
        Special method for figuring out previous trial types for temporal inputs.
        """
        previous_anti = np.zeros(X.shape[1], dtype = np.bool)
        previous_anti[1:] = X[0,:-1,1]==1
        previous_pro = np.zeros(X.shape[1], dtype = np.bool)
        previous_pro[1:] = X[0,:-1,0]==1
        previous_left = np.zeros(X.shape[1], dtype = np.bool)
        previous_left[1:] = X[0,:-1,2]==1
        previous_right = np.zeros(X.shape[1], dtype = np.bool)
        previous_right[1:] = X[0,:-1,3]==1

        return previous_pro, previous_anti, previous_left, previous_right

    def _PETH(self, history, hit_rate, config_names, trials):
        """
        Computes perievent time histogram.
        Inputs:
        - history: activation history. Numpy array in shape (T*total_time_steps,_,hidden_dim)
        - hit_rate: Hit rate of each trial.
        config_names can be either ["pro left switch", ....] or can be ["pro","anti"].
        As long as the union of the categories are the universal set of trials.
        - config_names: list of set of configuration names.
        - trials: self.trials.

        Returns:
        - config_array: an integer version of trials. Each integer represents a configuration. Numpy array of Integers with shape (T,).
        - normalized_activation_matrices: A dictionary with config_name as keys and Numpy arrays of shape (t,total_time_steps, hidden dimension) 
            as values. Contains all of the PETH.
        - activation_matrix_mean: Mean of the PETH. In shape (num_dim,num_configs,total_time_steps).
        - activation_matrix_SE: Standard error of the PETH. In shape (num_dim,num_configs,total_time_steps).
        """
        total_time_steps = self.total_time_steps
        TT,_,num_dim = history.shape
        T = TT / total_time_steps
        num_configs = len(config_names)
        
        # Initialize arrays
        config_array = np.zeros((T,),dtype=np.int)
        normalized_activation_matrices = {}
        # A dictionary of numpy arrays. Each array for each configuration.
        # Each array has shape (trials, total_time_steps, num_hidden_activation)
        for i in range(num_configs):
            config_array[trials[config_names[i]]] = i
            normalized_activation_matrices[config_names[i]] = np.zeros((0, total_time_steps, num_dim))
        
        # Compute PETH
        for t in range(T):
            # PETH = Activation * hit_rate
            previous = "prev" in config_names[config_array[t]]
            if previous:
                if t==0:
                    hit = 0
                else:
                    hit = hit_rate[t-1]
            else:
                hit = hit_rate[t]
            normalized_activation = history[t * total_time_steps:(t+1) * \
                total_time_steps,0,:] * hit 
            # Append PETH of single trial to the big matrix
            normalized_activation_matrices[config_names[config_array[t]]] = \
                np.append(normalized_activation_matrices[config_names[config_array[t]]],\
                np.expand_dims(normalized_activation, axis=0), axis=0)

        activation_matrix_mean = np.zeros((num_dim,num_configs,total_time_steps))
        activation_matrix_SE = np.zeros((num_dim,num_configs,total_time_steps))
        for i in range(num_configs):
            activation_matrix_mean[:,i,:] = np.mean(normalized_activation_matrices[config_names[i]], axis=0).T
            SE = np.std(normalized_activation_matrices[config_names[i]], axis=0)\
                / np.sqrt(normalized_activation_matrices[config_names[i]].shape[0])
            activation_matrix_SE[:,i,:] = SE.T

        return config_array, normalized_activation_matrices, activation_matrix_mean, activation_matrix_SE

    def _ROC(self, num_dim, config1_name, config2_name, normalized_activation_matrices, threshold = 0.01):
        """
        Compute fraction of hidden units that are significant for coding config rules (e.g. pro rule).
        
        Inputs:
        - num_dim: number of hidden unit of the network.
        - config1_name: String, e.g. "pro".
        - config2_name: String, e.g. "anti".
        - normalized_activation_matrices: A dictionary with config_name as keys and Numpy arrays of shape (t,total_time_steps, hidden dimension) 
            as values. Contains all of the PETH.
        - threshold: significance level for statistical test.

        Returns:
        - config1: Numpy of shape (total_time_steps,). Fraction of hidden units that are significant 
            for coding config1 rules at each time step.
        - config2: Numpy of shape (total_time_steps,). Fraction of hidden units that are significant 
            for coding config2 rules at each time step.
        - AUC: Area under curve with respect to config1 and config2 of each hidden unit at each time step.
        """
        total_time_steps = self.total_time_steps
        AUC = np.zeros((num_dim, total_time_steps))
        ROC_p = np.zeros((num_dim, total_time_steps))

        for dim in range(num_dim):
            for t in range(total_time_steps):
                AUC[dim,t],ROC_p[dim,t],_  \
                    = self._bootroc(normalized_activation_matrices[config1_name][:,t,dim], 
                    normalized_activation_matrices[config2_name][:,t,dim], ttest = False)

        is_config1 = AUC >=0.5
        is_config2 = AUC < 0.5

        significant = ROC_p <= threshold

        self.raw_config1 = np.logical_and(is_config1,significant)
        self.raw_config2 = np.logical_and(is_config2,significant)
        config1 = np.mean(self.raw_config1, axis = 0)
        config2 = np.mean(self.raw_config2, axis = 0)
        return config1, config2, AUC, significant

    def _bootroc(self, A,B, BOOTS=100, CI=99, ttest = False):
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
        sd = self._sklearn_auc(A,B)
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
                boot_score[bx] = self._sklearn_auc(A,B)

            sd_p = self._get_p(sd,boot_score)

            half = (100 - CI)/2
            confidence_interval = np.percentile(boot_score,[(100-CI-half),CI+half])
        return sd, sd_p, confidence_interval

    
    def _sklearn_auc(self,stim,nostim):
        """
        Use module from scikit learn to compute area under curve.

        Inputs:
        - stim: numpy array in shape (a,n)
        - nostim: numpy array in shape (b,n)
        """
        labels = np.append(np.ones((stim.size)), np.zeros((nostim.size)))
        values = np.append(np.reshape(stim, stim.size), np.reshape(nostim, nostim.size))
        return roc_auc_score(labels, values)
        
    def _get_p(self, datum, dist, tails = 2, high=1, precision = 100):
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


    def _activationTrajectoryByTrials(self, activation_history, trial_types):
        """
        Categorize activation history by trial types.
        activation_history: (T*5,1,H)
        activations: dictionary of (T,5,H)
        """
        activations = {}
        for config_name in trial_types:
            trials = np.repeat(self.trials[config_name],self.RNN.total_time_steps)
            activations[config_name] = np.reshape(activation_history[trials,0,:],\
                (-1,self.RNN.total_time_steps,self.RNN.hidden_dim))
        return activations

    def mixActivation(self, original_config_name = "pro block", opposite_config_name = "anti block", repeat = 100, linespace_num = 10, verbose = False):
        pro_length = self.categorized_activation_trajectory[original_config_name].shape[0]
        pro_indices = np.random.randint(pro_length, size=repeat)

        anti_length = self.categorized_activation_trajectory[opposite_config_name].shape[0]
        anti_indices = np.random.randint(anti_length, size=repeat)
        if "pro" in original_config_name:
            mode = "pro_only"
        else:
            mode = "anti_only"
        box = VirtualRatBox(mode=mode,length=1,block_size=30,session = False)
        mix = []
        for i in range(repeat):
            if verbose:
                print "Calulating repetition %d" % (i,)
            mix.append([])
            original_h = self.categorized_activation_trajectory[original_config_name][pro_indices[i],-1,:]
            opposite_h = self.categorized_activation_trajectory[opposite_config_name][anti_indices[i],-1,:]

            for j in np.linspace(0,1, num=linespace_num+1):
                #print j
                if "pro" in original_config_name:
                    box.change_mode("pro_only")
                else:
                    box.change_mode("anti_only")
                h = (1-j) * original_h + j * opposite_h
                self.RNN.h0 = h
                ps = self.RNN.forward(box.X).asnumpy()
                prob = ps[0,-1,:]
                normalized_prob = [prob[0] / (prob[0] + prob[1]), prob[1] / (prob[0] + prob[1])]
                mix[i].append(normalized_prob[box.y[0,-1]])
        accuracy_matrix = np.array(mix)
        return accuracy_matrix