import numpy as np
class simpleBox(object):
    """
    This class is an environment for reinforcement learning virtual rat, like a bpod.
    It provides inputs to virtual rat and record the rat's response.
    """
    def __init__(self, mode = "alternative", X=None, y=None, length = 10000,
     block_size = 20, random_range = 5, trial_per_episode = 3,
    repeat = False, reward_ratio = 1, p2a=0.5, block_correction = True, left_right_correction = True):
        """
        Input:
        - X: (N, T, D)
        - y: (N, T)
        - length: length of the sequence of trial generated each time.
        - block_size: block length
        - repeat: if true, then always use the same X sequence without changing.
        """
        self.realX = X
        self.realy = y
        self.length = length
        self.block_size = block_size
        self.random_range = random_range
        self.repeat = repeat
        self.reward_ratio = reward_ratio
        self.block_correction = block_correction
        self.left_right_correction = left_right_correction
        self.t = 0 # Normal step count, reset when reaches self.length
        self.tt = 0 # Special step count, reset when done, i.e. when paprameter update occurs
        self.previous_block = 0
        self.previous_trials = 10
        self.episode = trial_per_episode # How many trials per parameter update occurs
        self.p2a = p2a
        self.history = np.zeros((0,5)) # (pro_rule, left/right, trial=1, action, reward)

        self.change_mode(mode, self.p2a)

    def change_mode(self,mode,p2a=None):
        self.mode = mode
        if not p2a:
            p2a = self.p2a
        self.t = 0
        if mode == "real":
            X = self.realX[:,:self.length,:]
            y = self.realy[:,:self.length,:]
            self.X, self.y = self._old2newX(X), y
        elif mode == "no_rule":
            self.X, self.y = self._no_rule()
        elif mode == "pro_only":
            X, y = self._pro_only()
            self.X, self.y = self._old2newX(X), y
        elif mode == "anti_only":
            X, y = self._anti_only()
            self.X, self.y = self._old2newX(X), y
        elif mode == "switch_ratio":
            X, y = self._switch_ratio(p2a) 
            self.X, self.y = self._old2newX(X), y           
        else:
            X, y = self._alternate()
            self.X, self.y = self._old2newX(X), y

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

    def stop_repeat(self):
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
        X = np.zeros((1,self.length,5),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,-1] = 1
        y[0,:] = np.random.randint(2, size=self.length)
        X[0,:,3] = y[0,:]
        X[0,:,2] = np.logical_not(y[0,:])
        return X,y

    def _pro_only(self):
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,2] = 1
        X[0,:,0] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _anti_only(self):
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,2] = 1
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X,y

    def _alternate(self):
        X = np.zeros((1,self.length,3),dtype = np.int)
        y = np.zeros((1,self.length),dtype = np.int)
        X[0,0,2] = 1
        pro_rule = 1
        prev_t = 0
        t = self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
        while t<self.length:
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
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
        t = self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
        while t<self.length:
            mode = np.random.binomial(1,p2a,1)[0] #a2p or p2a
            X[0,prev_t,2] = 1
            pro_rule = mode
            X[0,prev_t:t,0] = pro_rule
            pro_rule = np.logical_not(pro_rule)
            prev_t = t
            t += self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
            if t<self.length:
                X[0,prev_t:t,0] = pro_rule
                prev_t = t
                t += self.block_size + np.random.randint(-self.random_range,high=self.random_range+1)
            else:
                break # Actually this is not necessary but for bug free, I add break here.
        X[0,prev_t:,0] = pro_rule
        X[0,:,1] = np.random.randint(2, size=self.length)
        y[0,:] = np.logical_not(np.bitwise_xor(X[0,:,0],X[0,:,1]))
        return X, y


    def render(self):
        """
        Empty method that might be called by Solver.
        """
        return

    def step(self, action):
        record = np.zeros((1,5)) # (pro_rule, left/right, trial=1, action, reward)
        X = self.X[0,self.t,:]
        y = self.y[0,self.t]
        if self.t>0 and self.X[0,self.t,0]!=self.X[0,self.t-1,0]:
            # Switch trial.
            # Give switch trials a stronger reward or punishment.
            reward = self.reward_ratio if action == y else -1
            if self.block_correction and self.t>self.previous_trials and self._low_accuracy():
                self.t = self.previous_block
            else:
                self.previous_block = self.t
            
        else:
            reward = 1 if action == y else -1

        record[0,0] = np.logical_not(X[1])
        record[0,1] = X[3]
        record[0,2] = X[-1]
        record[0,-2] = action
        record[0,-1] = reward

        self.t+=1
        self.tt+=1
        done = True if self.tt>=self.episode else False
        if done:
            self.tt=0

        if self.left_right_correction and self.t>self.previous_trials:
            right = self._go_right(self.X[:,self.t,1])
            left = np.logical_not(right).astype(int)
            self.X[:,self.t,2] = left
            self.X[:,self.t,3] = right
            self.y[0,self.t] = np.logical_not(np.bitwise_xor(self.X[0,self.t,1],self.X[0,self.t,2]))

        x = self.X[:,self.t,:]
        if self.t>=self.length-1:
            if not self.repeat:
                self.change_mode(self.mode, self.p2a)
            self.t=0
        self.history = np.append(self.history,record,axis=0)
        return x, reward, done, y 

    def _low_accuracy(self):
        return np.mean(self.history[-self.previous_trials:,-1]) < 0.4 # Equivalent to accuracy = 70%

    def _left_right_accuracy(self):
        history = self.history[-self.previous_trials:,:]
        left = history[:, 1] == 0
        right = history[:, 1] == 1
        left_accuracy = np.sum(history[left,-1]>0).astype(float) / np.sum(left)
        right_accuracy = np.sum(history[right,-1]>0).astype(float) / np.sum(right)
        return left_accuracy, right_accuracy

    def _go_right(self,anti):
        # If left_accuracy is high, it's more likely to choose to go right
        left_accuracy, right_accuracy = self._left_right_accuracy()
        p = left_accuracy / (left_accuracy + right_accuracy)
        if np.isnan(p):
            p = 0.5
        return np.random.binomial(1,1-p) if anti else np.random.binomial(1,p)


