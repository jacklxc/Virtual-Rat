import numpy as np
"""
This class saves each rat's behavioral data and compute switch cost.
"""
class Rat(object):
    def __init__(self,ratname,rat, trial_window=3):

        """
        Create attributes and call methods to get their values.

        Inputs: 
        - ratname: string, the name of rat
        - rat: numpy boolean array of shape N * T * D as elements. In dimention 2, the bits 
            represent pro_rule, target_on_right, trial_n=1, left, right, cpv (central poke 
            violation) respectively.
        - trial_window: int, number of trials computed before and after swtiches.
        """

        self.name = ratname
        if rat.shape[1] == 6:
            self.X, self.y, self.trueY = self._basicData(rat)
            self.new = False
        else:
            self.X, self.y, self.trueY = self._basicDataNew(rat)
            self.new = True
        self.trial_window = trial_window
        self.trials = self._trial_type(self.X)
        self.hit = self._realRatHit()
        self.real_p2a = self._realSwtichCost("anti switch") # For plotting figure 3d
        self.real_a2p = self._realSwtichCost("pro switch") # For plotting figure 3d

        self.p2a_accuracies, self.a2p_accuracies, self.pro_block_accuracies, self.anti_block_accuracies \
            = self.accuracyBySession(self.hit)

        self.pro_switch_cost = self.a2p_accuracies - self.pro_block_accuracies
        self.anti_switch_cost = self.p2a_accuracies - self.anti_block_accuracies

    def _trial_type(self, X):
        """
        Figures out the input configurations of each trial based on static input X.
        All in one-hot style, i.e. [True, False, ...]

        Input:
        - X: static inputs in numpy array shape (1, T, 3)

        """
        trials = {}
        trials["pro"] = X[0,:,0]==1
        trials["anti"] = X[0,:,0]==0
        trials["left"] = X[0,:,1]==0
        trials["right"] = X[0,:,1]==1

        switch = X[0,1:,0]!=X[0,:-1,0]
        trials["switch"] = np.append([False],switch)
        trials["block"] = np.logical_not(trials["switch"])

        trials["pro block"] = np.logical_and(trials["pro"],trials["block"])
        trials["pro switch"] = np.logical_and(trials["pro"],trials["switch"])
        trials["anti block"] = np.logical_and(trials["anti"],trials["block"])
        trials["anti switch"] = np.logical_and(trials["anti"],trials["switch"])

        trials["new session"] = X[0,:,2]==1
        return trials

    def compress_label(self, y):
        """
        Compress one hot encoded label to normal label. (0,1,2) style

        Returns:
        - yy: labels. Numpy array in shape (T,)
        """
        translator = np.array([0,1,2],dtype=np.int)
        yy = np.sum(y*translator,axis=2)
        return yy

    def _basicData(self,rat):
        """
        Format raw data to X, y and trueY

        Inputs:
        - rat: Numpy array of shape (T,6). The first 3 bits are directly copied to X. 
                The last 3 bits are actual reactions of rats.

        Outputs:
        - X: Inputs to rats. Numpy array in shape (T,3). (pro, right, first_trial_in_session)
        - y: Rat's response. Numpy array (one-hot) in shape (T,3). (left, right, cpv)
        - trueY: Ground truth (rational agent's) response. Numpy array (one-hot) in shape (T,3). (left, right, cpv)
        """
        X = np.zeros((1, rat.shape[0], 3), dtype=np.int)
        y = np.zeros((1, rat.shape[0], 3), dtype=np.int)
        trueY = np.zeros((1, rat.shape[0], 3), dtype=np.int)

        X[0,:,:] = rat[:,:3]

        # Reaction of rats
        y[0,rat[:,3]>0,0] = 1
        y[0,rat[:,4]>0,1] = 1
        y[0,rat[:,5]>0,2] = 1

        # Rational reaction (logically correct)
        true = np.logical_not(np.bitwise_xor(rat[:,0],rat[:,1]))
        trueY[0,true==0,0] = 1
        trueY[0,true==1,1] = 1
        return X, y, trueY

    def _basicDataNew(self,rat):
        """
        Convert new rat's data from .mat file into the same format as _basicData() does.
        """
        X = np.zeros((1, rat.shape[0], 3), dtype=np.int)
        y = np.zeros((1, rat.shape[0], 3), dtype=np.int)
        trueY = np.zeros((1, rat.shape[0], 3), dtype=np.int)

        X[0,:,:-1] = rat[:,:2]
        X[0,:,-1] = rat[:,-1]        

        # Rational reaction (logically correct)
        true = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        trueY[0,true==0,0] = 1
        trueY[0,true==1,1] = 1

        # Reconstruct reaction
        went_left_hit = np.logical_and(rat[:,-2]>0, true==0)
        went_left_wrong = np.logical_and(rat[:,-2]==0, true==1)
        went_left = np.logical_or(went_left_hit, went_left_wrong)
        went_right_hit = np.logical_and(rat[:,-2]>0, true==1)
        went_right_wrong = np.logical_and(rat[:,-2]==0, true==0)
        went_right = np.logical_or(went_right_hit, went_right_wrong)

        # Reaction of rats
        y[0,went_left>0,0] = 1
        y[0,went_right>0,1] = 1
        y[0,np.isnan(rat[:,-2]),2] = 1

        return X, y, trueY


    def _realRatHit(self):
        """
        Returns:
        hit: a numpy boolean array of rat's real performance (for each trial, hit or not).
            for cpv, put NaN.
\        """
        y = self.compress_label(self.y)
        trueY = self.compress_label(self.trueY)
        cpv = (y[0,:] == 2)
        hit_trials = (y[0,:] == trueY[0,:])
        hit = np.zeros(y.shape[1])
        hit[hit_trials] = 1
        hit[cpv] = np.nan
        return hit

    def _realSwtichCost(self, switch_type):
        """
        Calculates the switch cost of the real rat's performance.
        Returns:
        - switch_cost: a numpy float array of shape (2 * self.trial_window + 1,). Mean percentage
         of correct orientation before and after each pro_rule switch form pro to anti.
            (index i --> trial from switch = -trial_window + i)
        """
        switch_matrix = np.zeros((np.sum(self.trials[switch_type]), self.trial_window * 2 + 1))
        switches = np.where(self.trials[switch_type]==1)[0]
        for i in range(switches.size-1):
            switch = switches[i]
            switch_matrix[i,:] = self.hit[(switch - self.trial_window):
             (switch + self.trial_window + 1)]
        swtich_cost = np.nanmean(switch_matrix,axis=0)
        return swtich_cost

    def accuracyBySession(self, hit):
        """
        Computes each session's accuracy
        """
        session_index = np.where(self.X[0,:,2]==1)[0]
        p2a_accuracies = []
        a2p_accuracies = []
        pro_block_accuracies = []
        anti_block_accuracies = []
        T = hit.size
        prev_i = session_index[0]
        for i in session_index[1:]:
            this_session = np.zeros((T,))
            this_session[prev_i:i] = 1
            a2p_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["pro switch"])]))
            p2a_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["anti switch"])]))
            pro_block_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["pro block"])]))
            anti_block_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["anti block"])]))
        # Discard Last session
        #this_session = np.zeros((T,))
        #this_session[i:] = 1
        #a2p_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["pro switch"])]))
        #p2a_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["anti switch"])]))
        #pro_block_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["pro block"])]))
        #anti_block_accuracies.append(np.nanmean(hit[np.logical_and(this_session, self.trials["anti block"])]))

        p2a_accuracies = np.array(p2a_accuracies)
        a2p_accuracies = np.array(a2p_accuracies)
        pro_block_accuracies = np.array(pro_block_accuracies)
        anti_block_accuracies = np.array(anti_block_accuracies)

        return p2a_accuracies, a2p_accuracies, pro_block_accuracies, anti_block_accuracies


