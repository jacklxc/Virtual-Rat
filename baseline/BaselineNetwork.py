from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as npp
import minpy.numpy as np
from minpy.nn.model import ModelBase
from minpy.nn import layers

from minpy.nn import optim
from minpy.nn import init
from minpy import core

class BaselineNetwork(ModelBase):
	def __init__(self, input_dim, hidden_dim=5, output_dim=1):
		super(BaselineNetwork, self).__init__()
		self.add_param(name = 'W1', shape = (input_dim, hidden_dim))
		self.add_param(name = 'b1', shape = (hidden_dim,))
		self.add_param(name = 'W2', shape = (hidden_dim, output_dim))
		self.add_param(name = 'b2', shape = (output_dim,))

	def forward(self, X):
		h = layers.affine(X, self.params['W1'], self.params["b1"])
		H = layers.relu(h)
		baseline = layers.affine(H, self.params['W2'], self.params["b2"])
		return baseline

	def loss(self, x, y):
		return layers.l2_loss(x,y)

class BaselineNetworkSolver(object):

    def __init__(self, model, **kwargs):
        self.model = model
        # Unpack keyword arguments
        self.init_rule = kwargs.pop('init_rule', 'xavier')
        self.init_config = kwargs.pop('init_config', {})
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.train_acc_num_samples = kwargs.pop('train_acc_num_samples', 1000)

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
        self.init()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.loss_history = []

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

    def getBaseline(self, X, y):
        # Compute loss and gradient
        def loss_func(*params):
            # It seems that params are not used in forward function. But since we will pass
            # model.params as arguments, we are ok here.
            predict = self.model.forward(X)
            return self.model.loss(predict, y)

        param_arrays = list(self.model.params.values())
        param_keys = list(self.model.params.keys())
        grad_and_loss_func = core.grad_and_loss(
            loss_func, argnum=range(len(param_arrays)))
        grad_arrays, loss = grad_and_loss_func(*param_arrays)
        grads = dict(zip(param_keys, grad_arrays))

        self.loss_history.append(loss.asnumpy())

        predict = self.model.forward(X)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

        return predict.asnumpy()

    def init(self):
        """
        Init model parameters based on the param_configs in model
        """
        for name, config in self.model.param_configs.items():
            self.model.params[name] = self.init_rules[name](
                config['shape'], self.init_configs[name])
        for name, value in self.model.aux_param_configs.items():
            self.model.aux_params[name] = value


