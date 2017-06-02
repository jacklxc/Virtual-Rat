from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import time
import pickle
import os

import minpy.numpy as np
from minpy import core
from minpy.nn.solver import Solver
from minpy.nn import optim
from nn import init

class SimpleRLPolicyGradientSolver(object):
    """A custom `Solver` for models trained using policy gradient.
    Specifically, the model should provide:
        .forward(X)
        .choose_action(p)
        .loss(xs, ys, rs)
        .discount_rewards(rs)
        .preprocessor
    """

    def __init__(self, model, env, **kwargs):
        """ Construct a new `RLPolicyGradientSolver` instance.
        Parameters
        ----------
        model : ModelBase
            A model that supports policy gradient training (see above).
        env : gym.Environment
            A `gym` Environment, e.g. Pong-v0.
        Other Parameters
        ----------------
        num_episodes : int, optional
            Number of episodes to train for.
        update_every : int, optional
            Update model parameters every `update_every` episodes.
        save_every : int, optional
            Save model parameters in the `save_dir` directory every `save_every` episodes.
        save_dir : str, optional
            Directory to save model parameters in.
        resume_from : str, optional
            Loads a parameter file at this location, resuming model training with those parameters.
        """
        self.model = model
        self.env = env
        self.supervised = kwargs.pop('supervised', False)
        self.num_episodes = kwargs.pop('num_episodes', 10000)
        self.update_every = kwargs.pop('update_every', 1)
        self.save_every = kwargs.pop('save_every', 10)
        self.save_dir = kwargs.pop('save_dir', './')
        self.resume_from = kwargs.pop('resume_from', None)

        self.running_reward = None
        self.episode_reward = 0

        self.init_rule = kwargs.pop('init_rule', 'xavier')
        self.init_config = kwargs.pop('init_config', {})
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def change_settings(self,learning_rate = None, num_episodes = None):
        if learning_rate is not None:
            self.optim_config = {'learning_rate': learning_rate}
        if num_episodes is not None:
            self.num_episodes = num_episodes

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.param_configs:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        # Overwrite it if the model specify the rules

        # Make a deep copy of the init_config for each parameter
        # and set each param to their own init_rule and init_config
        self.init_rules = {}
        self.init_configs = {}
        for p in self.model.param_configs:
            if 'init_rule' in self.model.param_configs[p]:
                init_rule = self.model.param_configs[p]['init_rule']
                init_config = self.model.param_configs[p].get('init_config',
                                                              {})
            else:
                init_rule = self.init_rule
                init_config = {k: v for k, v in self.init_config.items()}
            # replace string name with actual function
            if not hasattr(init, init_rule):
                raise ValueError('Invalid init_rule "%s"' % init_rule)
            init_rule = getattr(init, init_rule)
            self.init_rules[p] = init_rule
            self.init_configs[p] = init_config


    def init(self):
        """
        Init model parameters based on the param_configs in model
        """
        for name, config in self.model.param_configs.items():
            self.model.params[name] = self.init_rules[name](
                config['shape'], self.init_configs[name])
        for name, value in self.model.aux_param_configs.items():
            self.model.aux_params[name] = value

    def load_params(self, params):
        """
        An alternative of init() when previously created parameters are loaded.
        """
        for k, v in params.iteritems():
            self.model.params[k] = v

    def save_params(self):
        params = {}
        for k, v in self.model.params.iteritems():
            params[k] = v.asnumpy()
        return params

    def run_episode(self):
        """Run an episode using the current model to generate training data.
        Specifically, this involves repeatedly getting an observation from the environment,
        performing a forward pass using the single observation to get a distribution over actions
        (in binary case a probability of a single action), and choosing an action.
        Finally, rewards are discounted when the episode completes.
        Returns
        -------
        (xs, ys, rs) : tuple
            The N x input_size observations, N x 1 action labels, and N x 1 discounted rewards
            obtained from running the episode's N steps.
        """
        observation = self.env.reset()
        self.episode_reward = 0

        xs, ys, rs = [], [], []
        done = False
        trial_number = 1
        game_start = time.time()
        self.model.reset_h()
        while not done:
            x = observation # preprocessor can work here
            p = self.model.step_forward(x)
            a, y = self.model.choose_action(p.asnumpy())
            observation, r, done, info = self.env.step(a)
            xs.append(x)
            ys.append(y)
            rs.append(r)
            self.episode_reward += r
            game_time = time.time() - game_start
            if self.verbose:
                print('game %d complete (%.2fs), reward: %f' % (trial_number, game_time, r))
            trial_number += 1
            game_start = time.time()

        # Episode finished.
        self.running_reward = self.episode_reward if self.running_reward is None else (
            0.99*self.running_reward + 0.01*self.episode_reward)
        xs = np.vstack(xs)
        ys = np.vstack(ys)
        rs = np.expand_dims(self.model.discount_rewards(rs), axis=1)
        rs = np.vstack(rs)
        return xs, ys, rs

    def _trial_complete(self, reward):
        return reward != 0

    def train(self):
        """Trains the model for `num_episodes` iterations.
        On each iteration, runs an episode (see `.run_episode()`) to generate three matrices of
        observations, labels and rewards (xs, ys, rs) containing data for the _entire_ episode.
        Then the parameter gradients are found using these episode matrices.
        Specifically, auto-grad is performed on `loss_func`, which does a single forward pass
        with the episode's observations `xs` then computes the loss using the output of the forward
        pass and the episode's labels `ys` and discounted rewards `rs`.
        This two-step approach of generating episode data then doing a single forward/backward pass
        is done to conserve memory during the auto-grad computation.
        """

        # Accumulate gradients since updates are only performed every `update_every` iterations.
        grad_buffer = self._init_grad_buffer()
        for episode_number in xrange(1, self.num_episodes):
            episode_start = time.time()
            # Generate an episode of training data.
            xs, ys, rs = self.run_episode()
            if False:
                #print("xs")
                #print(xs)
                print("ys")
                print(ys)
                #print("rs")
                #print(rs)
            # Performs a forward pass and computes loss using an entire episode's data.
            def loss_func(*params):
                xss = np.expand_dims(xs,axis=0)
                ps = self.model.forward(xss)
                if False:
                    print("ps")
                    print(ps)
                loss = self.model.loss(ps, ys, rs)
                if self.supervised:
                    loss = self.model.loss(ps, self.env.y.T, np.ones(rs.shape))
                return loss

            # Compute gradients with auto-grad on `loss_func` (duplicated from `Solver`).
            param_arrays = list(self.model.params.values())
            param_keys = list(self.model.params.keys())
            grad_and_loss_func = core.grad_and_loss(loss_func, argnum=range(len(param_arrays)))
            backward_start = time.time()
            grad_arrays, loss = grad_and_loss_func(*param_arrays)
            backward_time = time.time() - backward_start
            grads = dict(zip(param_keys, grad_arrays))

            # Accumulate gradients until an update is performed.
            for k, v in grads.iteritems():
                grad_buffer[k] += v

            # Misc. diagnostic info.
            self.loss_history.append(loss.asnumpy())
            episode_time = time.time() - episode_start
            if self.verbose:
                print('Backward pass complete (%.2fs)' % backward_time)
            if self.verbose or episode_number % self.print_every == 0:
                print('Episode %d complete (%.2fs), loss: %s, reward: %s, running reward: %s' %
                      (episode_number, episode_time, loss, self.episode_reward, self.running_reward))

            # Perform parameter update and reset the `grad_buffer` when appropriate.
            if episode_number % self.update_every == 0:
                for p, w in self.model.params.items():
                    dw = grad_buffer[p]
                    if False:
                        print(p)
                        print(w)
                        print(dw)
                    config = self.optim_configs[p]
                    next_w, next_config = self.update_rule(w, dw, config)
                    self.model.params[p] = next_w
                    self.optim_configs[p] = next_config
                    grad_buffer[p] = np.zeros_like(w)

            # Save model parameters to `save_dir` when appropriate..
            if False:
                if episode_number: #% self.save_every == 0:
                    if self.verbose:
                        print('Saving model parameters...')
                    file_name = os.path.join(self.save_dir, 'params_%d.p' % episode_number)
                    with open(file_name, 'w') as f:
                        pickle.dump({k: v.asnumpy() for k, v in self.model.params.iteritems()}, f)
                    if self.verbose:
                        print('Wrote parameter file %s' % file_name)

    def _init_grad_buffer(self):
        return {k: np.zeros_like(v) for k, v in self.model.params.iteritems()}