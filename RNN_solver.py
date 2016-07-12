import numpy as np
from VirtualRat import optim
class RNNsolver(object):
	def __init__(self, model, data, **kwargs):
		self.model = model
    	self.data = data

    	# Unpack keyword arguments
		self.update_rule = kwargs.pop('update_rule', 'sgd')
		self.optim_config = kwargs.pop('optim_config', {})
		self.lr_decay = kwargs.pop('lr_decay', 1.0)
		self.batch_size = kwargs.pop('batch_size', 1)
		self.num_epochs = kwargs.pop('num_epochs', 10)

		self.print_every = kwargs.pop('print_every', 100)
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
		for p in self.model.params:
			d = {k: v for k, v in self.optim_config.iteritems()}
			self.optim_configs[p] = d

	def _step(self):
		"""
		Make a single gradient update. This is called by train() and should not
		be called manually.
		"""

		# Compute loss and gradient
		loss, grads = self.model.loss(features, captions) ###########
		self.loss_history.append(loss)

		# Perform a parameter update
		for p, w in self.model.params.iteritems():
			dw = grads[p]
			config = self.optim_configs[p]
			next_w, next_config = self.update_rule(w, dw, config)
			self.model.params[p] = next_w
			self.optim_configs[p] = next_config

	def train(self):
		"""
		Run optimization to train the model.
		"""
		num_train = self.data['train_captions'].shape[0]
		iterations_per_epoch = max(num_train / self.batch_size, 1)
		num_iterations = self.num_epochs * iterations_per_epoch

		for t in xrange(num_iterations):
			self._step()

			# Maybe print training loss
			if self.verbose and t % self.print_every == 0:
				print '(Iteration %d / %d) loss: %f' % (
					t + 1, num_iterations, self.loss_history[-1])

			# At the end of every epoch, increment the epoch counter and decay the
			# learning rate.
		  	epoch_end = (t + 1) % iterations_per_epoch == 0
		    if epoch_end:
		    	self.epoch += 1
		    	for k in self.optim_configs:
		      		self.optim_configs[k]['learning_rate'] *= self.lr_decay