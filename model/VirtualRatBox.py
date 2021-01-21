import numpy as np

class VirtualRatBox(object):
    """
    This class is an environment for virtual rat agent, like a bpod.
    It does not matter if supervised learning or reinforcement learning is used.
    It provides inputs to virtual rat and record the rat's response and adjust inputs accordingly.
    """
    def __init__(self, mode = "alternative", length = 20000,
                block_size = 30, std = 1, trial_per_episode = 30,
                repeat = False, p2a=0.5, block_correction = False, reward = 1,
                left_right_correction = False, time_steps = (1,1,1,1,1), 
                extra_anti_length = 0, session = True):
        """
        Input:
        - X: (N, T, D)
        - y: (N, T)
        - length: length of the sequence of trial generated each time.
        - block_size: block length
        - repeat: if true, then always use the same X sequence without changing.
        - time_steps: Number of time steps take for each period. (ITI, 
            rule, delay, orientation, response)
        """
        self.length = length
        self.block_size = block_size
        self.std = std
        self.repeat = repeat
        self.block_correction = block_correction
        self.left_right_correction = left_right_correction
        self.session = session
        self.t = 0 # Normal step count, reset when reaches self.length
        self.tt = 0 # Special step count, reset when done, i.e. when parameter update occurs
        self.previous_rule = -1
        self.previous_block = -1
        self.previous_trials = 10
        self.episode = trial_per_episode # How many trials per parameter update occurs
        self.p2a = p2a # p2a ratio
        self.history = np.zeros((0,5)) # (pro_rule, left/right, trial=1, action, reward)
        self.time_steps = np.array(time_steps) #(ITI, rule, delay, orientation, response)
        self.reward = reward
        self.extra_anti_length = extra_anti_length

        self.change_mode(mode, self.p2a)

    def change_time_steps(self,time_steps):
        """
        Change the allocation of time steps. (ITI, rule, delay, orientation, response).
        Input:
        - time_steps: a tuple with lenth 5. Default (1,1,1,1,1).
        """
        time_steps = np.array(time_steps)
        if time_steps.shape[0] != 5:
            raise ValueError("time_steps should be a numpy array with 5 integers!")
        self.time_steps = time_steps

    def change_mode(self,mode,p2a=None):
        """
        Obtain a series of inputs under the specified mode.
        - self.X, self.y: numpy array of (1,T*total_time_steps,5) and (1, T*total_time_steps)
        """
        self.mode = mode
        if p2a:
            self.p2a = p2a
        else:
            p2a = self.p2a
        self.t = 0
        if mode == "no_rule":
            newX, newY = self._no_rule()
        elif mode == "no_target":
            newX, newY = self._no_target()
        else:
            if mode == "pro_only":
                X, y = self._pro_only()
            elif mode == "anti_only":
                X, y = self._anti_only()
            elif mode == "switch_ratio":
                X, y = self._switch_ratio(p2a)
            elif mode == "interleave":
                X, y = self._interleave()
            else:
                X, y = self._alternate()
            newX, newY = self._old2newX(X), y
        self.X, self.y = self._static2temporal(newX, newY, self.time_steps)

    def _old2newX(self,X):
        """
        Convert (pro_rule,left,trial=1) to (pro, anti, left, right, trial=1)
        """
        N,T,_ = X.shape
        newX = np.zeros((N,T,5), dtype = np.bool)
        pro = X[:,:,0] == 1
        anti = X[:,:,0] == 0
        left = X[:,:,1] == 0
        right = X[:,:,1] == 1
        newX[pro,0] = 1
        newX[anti,1] = 1
        newX[left,2] = 1
        newX[right,3] = 1
        newX[:,:,4] = X[:,:,2]
        return newX

    def _static2temporal(self,X,y,time_steps):
        """
        Convert static inputs to temporal.
        """
        T = X.shape[1]
        total_time_steps = np.sum(time_steps)
        temporalX = np.zeros((1,T*total_time_steps,5), dtype = np.int)
        temporalY = np.zeros((1,T*total_time_steps), dtype = np.int)
        cumulative_time = np.cumsum(time_steps)
        for t in range(T):
            # ITI step 
            # all of bits are silent except trial=1 bit
            temporalX[0,t*total_time_steps,-1] = X[0,t,-1]
            # rule step
            # convert pro/anti
            temporalX[0,(t*total_time_steps+cumulative_time[0]):(t*total_time_steps+cumulative_time[1]),:2] = X[0,t,:2]
            # delay step, silent input
            # orientation step, convert left/right
            temporalX[0,(t*total_time_steps+cumulative_time[2]):(t*total_time_steps+cumulative_time[3]),2:4] = X[0,t,2:4]
            # response step, no input

            # Except response step, always choose do_nothing
            temporalY[0,(t*total_time_steps):(t*total_time_steps+cumulative_time[3])] = 2
            temporalY[0,(t*total_time_steps+cumulative_time[3]):((t+1)*total_time_steps)] = y[0,t]
        return temporalX, temporalY

    def stop_repeat(self):
        """
        Stop the VirtualRatBox to output repeated trials.
        """
        self.repeat = False

    def reset(self):
        return self.X[:,self.t,:]

    def output_history(self):
        """
        Output training history and empty it.
        """
        history = np.copy(self.history)
        self.history = np.zeros((0,5))
        return history

    def _no_rule(self):
        """
        Generate inputs of no rule bit.
        """
        X = np.zeros((1,self.length,5),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,-1] = 1
        y[0,:] = np.random.randint(2, size=self.length)
        X[0,:,3] = y[0,:]
        X[0,:,2] = np.logical_not(y[0,:])
        return X,y
    
    
    def _interleave(self):
        """
        Output sequences interleaved pro/anti.
        """
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        if self.session:
            X[0,0,2] = 1
        X[0,:,0] = np.random.randint(2, size=self.length)
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y
    
    def _pro_only(self):
        """
        Output pro only trials.
        """
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        if self.session:
            X[0,0,2] = 1
        X[0,:,0] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _anti_only(self):
        """
        Output anti only trials.
        """
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        if self.session:
            X[0,0,2] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _alternate(self):
        """
        Output block based pro/anti trials.
        """
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        if self.session:
            X[0,0,2] = 1
        pro_rule = np.random.randint(2, size=1)[0]
        prev_t = 0
        t = self.block_size + np.around(np.random.normal(0, self.std, 1)[0]).astype(int)
        #t = self.block_size + np.random.normal(0, self.std, 1)[0]
        while t<self.length:
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += self.block_size + np.around(np.random.normal(0, self.std, 1)[0]).astype(int)
            #t += self.block_size + np.random.normal(0, self.std, 1)[0]
            if not pro_rule:
                t+= self.extra_anti_length
        X[0,prev_t:,0] = pro_rule

        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _switch_ratio(self, p2a):
        """
        Concatenate Pro+Anti and Anti+Pro block based on binomial trial with probability "p2a"
        """
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        pro_rule = 0 if np.random.binomial(1,p2a,1)[0] else 1
        prev_t = 0
        t = self.block_size + np.around(np.random.normal(0, self.std, 1)[0]).astype(int)
        while t<self.length:
            mode = np.random.binomial(1,p2a,1)[0] #a2p or p2a
            X[0,prev_t,2] = 1
            pro_rule = mode
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += self.block_size + np.around(np.random.normal(0, self.std, 1)[0]).astype(int)
            if t<self.length:
                X[0,prev_t:t,0] = pro_rule
                prev_t = t
                t += self.block_size + np.around(np.random.normal(0, self.std, 1)[0]).astype(int)
                if not pro_rule:
                    t+= self.extra_anti_length
            else:
                break # Actually this is not necessary but for bug free, I add break here.
        X[0,prev_t:,0] = pro_rule
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X, y

    def step(self, action):
        """
        Called by Solver each time step. Keeps track of the agent's response and changes inputs accordingly
        to reduce pro/anti bias and left/right bias in agent's performance.
        """
        total_time_steps = np.sum(self.time_steps)
        #record = np.zeros((1,5)) # (pro_rule, left/right, trial=1, action, reward)
        X = self.X[0,self.t,:]
        y = self.y[0,self.t]
        # Before updating t!
        if self.t % total_time_steps==total_time_steps-1 and np.sum(self.X[0,self.t,:1])>0 \
            and self.X[0,self.t,:1] != self.previous_rule:
            # Switch trial. (response trial && exclude no_rule case && new rule bit)
            # Give switch trials a stronger reward or punishment.

            # If the past block had low accuracy, repeat that block.
            if self.block_correction and self.t>self.previous_trials and self._low_accuracy():
                self.t = self.previous_block
            else:
                self.previous_block = self.t
                self.previous_rule = self.X[0,self.t,0] # 1 if pro, 0 if anti
            
        else:
            if self.t % total_time_steps >= np.cumsum(self.time_steps)[-2]:
                reward = self.reward if action == y else -1
            else:
                reward = 0 if action == y else -1

        
        # Move to a new time step
        self.t+=1
        self.tt+=1

        done = True if self.tt>=self.episode*total_time_steps else False
        if done:
            self.tt=0

        if self.t % total_time_steps == 0: # Only for response time steps
            # Dynamically modify the future trials based on previous performance.
            if self.left_right_correction and self.t>self.previous_trials * total_time_steps:
                right = self._go_right(self.X[:,self.t+1,1])
                left = np.logical_not(right).astype(int)
                ## Not adapted to total_time_steps yet.
                self.X[:,self.t+3,2] = left
                self.X[:,self.t+3,3] = right
                self.y[0,self.t+4] = np.logical_not(np.bitwise_xor(self.X[0,self.t+1,1],self.X[0,self.t+3,2]))

            # Make record.
            # (pro_rule, left/right, trial=1, action, reward)
            record = np.zeros((1,5)) 
            record[0,0] = np.logical_not(self.X[:,self.t-4,1])
            record[0,1] = self.X[:,self.t-2,3]
            record[0,2] = self.X[:,self.t - total_time_steps,-1]
            record[0,-2] = action
            record[0,-1] = reward
            self.history = np.append(self.history,record,axis=0)

        if self.t>=self.length * total_time_steps-1:
            if not self.repeat:
                self.change_mode(self.mode, self.p2a)
            self.t=0
        x = self.X[:,self.t,:]
        return x, reward, done, y 

    def _low_accuracy(self):
        """
        Check if the past a few trials had accuracy lower than 70%.
        """
        return np.mean(self.history[-self.previous_trials:,-1]) < 0.4 # Equivalent to accuracy = 70%

    def _left_right_accuracy(self):
        """
        Check the accuracy to left and right in the past a few trials.
        """
        history = self.history[-self.previous_trials:,:]
        left = history[:, 1] == 0
        right = history[:, 1] == 1
        left_accuracy = np.sum(history[left,-1]>0).astype(float) / (np.sum(left) + 1e-8)
        right_accuracy = np.sum(history[right,-1]>0).astype(float) / (np.sum(right) + 1e-8)
        return left_accuracy, right_accuracy

    def _go_right(self,anti):
        """
        Compute the probability of target on the right in the next trial.
        """
        # If left_accuracy is high, it's more likely to choose to go right
        left_accuracy, right_accuracy = self._left_right_accuracy()
        p = left_accuracy / (left_accuracy + right_accuracy + 1e-8)
        if np.isnan(p):
            p = 0.5
        return np.random.binomial(1,1-p) if anti else np.random.binomial(1,p)


