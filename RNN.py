import numpy as np
from helpers import DBUtilsClass as db

class FirstRNN():
	"""
	One RNN for each rat. input dimention N * D * T, where N = 2 (rational agent and rat),
	D = 3 (pro_rule, target_on_right, trial_n=1), T is the length of the time sequence.
	Output dimention = 3, (left, right, central poke violation)
	"""
	def __init__(self):
		self.data = None

	