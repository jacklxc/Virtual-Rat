import numpy as np 
import minpy.numpy as mnp

class SimRat(object):
    def __init__(self, RNN, ratname = "SimRat", trial_window=3, accuracy_threshold = 0.8):

        """
        Create attributes and call methods to get their values.

        Inputs: 
        - RNN: Object of RNN model
        - ratname: string, the name of rat
        - trial_window: int, number of trials computed before and after swtiches.
        """

        self.name = ratname
        self.trial_window = trial_window
        self.RNN = RNN
        self.training_history = np.zeros((0,5)) # (pro_rule, left/right, trial=1, action, reward)
        self.accuracy_threshold = accuracy_threshold
        self.exclude = True
        self.pro_switch_accuracy_history = np.zeros((0,))
        self.pro_block_accuracy_history = np.zeros((0,))
        self.anti_switch_accuracy_history = np.zeros((0,))
        self.anti_block_accuracy_history = np.zeros((0,))

    def predict(self, X, y):
        """
        Make prediction based on the trianed RNN.

        Inputs:
        - X, y: Inputs and labels of the validation/test data.

        Returns: 
        - self.probility: numpy array of shape (N, T, DD) giving the probability of each choice,
          where each element is a float. 
        """
        self.X = X
        self.y = y
        self.session = np.where(self.X[0,:,2]==1)[0] # Stores trial number when new session starts
        self.pro_rules = self.X[0,:,0]
        self.RNN.reset_history() # Reset RNN's activation history recorded before each forward pass.
        self.probability = self.RNN.forward(X).asnumpy() # Run forward pass to make prediction.
        # Post RNN data process
        # Process probability itself.

        X = self._new2oldX(X) # Convert input X from new format (1,T,5) to old format (1,T,3).

        # Exclude cpv probability and normalize.
        self.normalized_probs = self._normalizeProbs(self.probability) 
        # Combines the probabilities to get the probabilities of choosing correctly.
        self.hit_rate = self._hitRate(y, self.normalized_probs)

        # Find out trial numbers when all kinds of switches occured.
        self.swtich, self.p2a, self.a2p, self.p2a_left, self.p2a_right, \
        self.a2p_left, self.a2p_right = self._switch(X)

        # Separate probability based on rule and direction.
        self.pro_prob, self.anti_prob = self._proAntiProbability(X, self.hit_rate)
        self.pro_left_prob, self.pro_right_prob, self.anti_left_prob, self.anti_right_prob \
            = self._proAntiLeftRightProbability(X, self.hit_rate)

        # Calculate switch cost
        self.pro_switch_cost, self.pro_switch_accuracy, self.pro_block_accuracy = self._switch_cost(self.pro_prob)
        self.anti_switch_cost, self.anti_switch_accuracy, self.anti_block_accuracy = self._switch_cost(self.anti_prob)
        if self.pro_block_accuracy > self.accuracy_threshold and self.anti_block_accuracy > self.accuracy_threshold:
            self.exclude = False

        # Prepare data for plotting.
        self.p2a_prob = self._calculatePerformance(self.p2a, self.hit_rate)
        self.a2p_prob = self._calculatePerformance(self.a2p, self.hit_rate)
        self.p2a_left_prob = self._calculatePerformance(self.p2a_left, self.hit_rate)
        self.a2p_left_prob = self._calculatePerformance(self.a2p_left, self.hit_rate)
        self.p2a_right_prob = self._calculatePerformance(self.p2a_right, self.hit_rate)
        self.a2p_right_prob = self._calculatePerformance(self.a2p_right, self.hit_rate)
        return self.probability

    def add_prediction_history(self):
        self.pro_switch_accuracy_history = np.append(self.pro_switch_accuracy_history, self.pro_switch_accuracy)
        self.pro_block_accuracy_history = np.append(self.pro_block_accuracy_history, self.pro_block_accuracy)
        self.anti_switch_accuracy_history = np.append(self.anti_switch_accuracy_history, self.anti_switch_accuracy)
        self.anti_block_accuracy_history = np.append(self.anti_block_accuracy_history, self.anti_block_accuracy)

    def _switch_cost(self, prob):
        """
        Calculate switch cost given probability record. 
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

    def computeLearningCurve(self, session_length = 600):
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

    def _switch(self, X):
        """
        Find out trial numbers when all kinds of switches occured.
        """
        SWITCH = []
        P2A = []
        A2P = []
        P2A_left = []
        P2A_right = []
        A2P_left = []
        A2P_right = []

        pro_rule = X[0,0,0]
        T = X.shape[1]
        for i in xrange(T):
            if X[0,i,0] != pro_rule:
                SWITCH.append(i)
                pro_rule = X[0,i,0]
                if i > self.trial_window and i < T - self.trial_window:
                    if pro_rule == 0:
                        P2A.append(i)
                        if X[0,i,1] == 0:
                            P2A_left.append(i)
                        elif X[0,i,1] == 1:
                            P2A_right.append(i)
                    else:
                        A2P.append(i)
                        if X[0,i,1] == 0:
                            A2P_left.append(i)
                        elif X[0,i,1] == 1:
                            A2P_right.append(i)
        switches = np.asarray(SWITCH)
        p2a_switch = np.asarray(P2A)
        a2p_switch = np.asarray(A2P)
        p2a_left_switch = np.asarray(P2A_left)
        a2p_left_switch = np.asarray(A2P_left)
        p2a_right_switch = np.asarray(P2A_right)
        a2p_right_switch = np.asarray(A2P_right)

        return switches, p2a_switch, a2p_switch, p2a_left_switch, p2a_right_switch, \
            a2p_left_switch, a2p_right_switch

    def _normalizeProbs(self, probability):
        """
        Exclude the probability of cpv and normalize the probabilities of left and right.
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
        """
        hit_rate = np.zeros(normalized_probs.shape[1])
        right = trueY[0,:] == 1
        left = trueY[0,:] == 0
        hit_rate[left] = normalized_probs[0,left,0]
        hit_rate[right] = normalized_probs[0,right,1]
        return hit_rate

    def _proAntiProbability(self,X, hit_rate):
        """
        Seperate self.hit_rate to probabilites of correct choice during pro and anti.
        """
        pro_rules = X[0,:,0]
        pro_prob = np.copy(hit_rate)
        anti_prob = np.copy(hit_rate)
        pro_rules = pro_rules.astype(bool)
        pro_prob[np.logical_not(pro_rules)] = None
        anti_prob[pro_rules] = None
        return pro_prob, anti_prob

    def _proAntiLeftRightProbability(self,X, hit_rate):
        """
        Seperate self.hit_rate to probabilites of correct choice during pro and anti.
        """
        pro_rules = X[0,:,0]
        left = X[0,:,1]==0
        right = X[0,:,1]==1

        pro_prob = np.copy(hit_rate)
        anti_prob = np.copy(hit_rate)
        pro_left_prob = np.copy(hit_rate)
        anti_left_prob = np.copy(hit_rate)
        pro_right_prob = np.copy(hit_rate)
        anti_right_prob = np.copy(hit_rate)

        pro = pro_rules.astype(bool)
        anti = np.logical_not(pro)
        # Assigning None actually is equivalent to NaN.
        pro_left_prob[np.logical_not(np.logical_and(pro,left))] = None
        pro_right_prob[np.logical_not(np.logical_and(pro,right))] = None
        anti_left_prob[np.logical_not(np.logical_and(anti,left))] = None
        anti_right_prob[np.logical_not(np.logical_and(anti,right))] = None
        return pro_left_prob, pro_right_prob, anti_left_prob, anti_right_prob


    def _calculatePerformance(self, switch_index, hit_rate):
        """
        Calculate the switch cost for this particular rat.

        Inputs:
        - switch_index: indices of trial numbers that switch occurs.
        - hit_rate: correct probability of each trial.

        Returns:
        - switch_prob: numpy float array that contains mean correct rate around switches
        """     
        # index i --> trial from switch = -trial_window + i
        switch_matrix = np.zeros((len(switch_index), self.trial_window * 2 + 1))
        for i in xrange(len(switch_index)):
            switch_matrix[i,:] = hit_rate[(switch_index[i] - self.trial_window): 
            (switch_index[i] + self.trial_window + 1)]
        switch_prob = np.nanmean(switch_matrix,axis=0)

        return switch_prob