"""
Written by Matteo Dunnhofer - 2020

Class that defines a tracker based on stored results
"""
import os
import numpy as np


class ResultsTracker(object):

	def __init__(self, name):
		self.name = name

	def init(self, image, bbox, **kwargs):

		result_dir = kwargs['result_dir']
		seq_name = kwargs['seq_name']
		seq_results_path = os.path.join(result_dir, self.name, seq_name + '.txt')

		self.result = self.load_predictions(seq_results_path)

		self.step = 0

	def update(self, image):
		self.step += 1

		return np.array(self.result[self.step])

	def load_predictions(self, result_file_path):
		"""
		Load the predictions of the tracker for the given sequence
		"""

		#if 'GOT-10k' in self.dataset:
		#	f_name = seq_name + '/' + seq_name + '_001'
		#else:
		#	f_name = seq_name

		#self.seq_pred_file_path = os.path.join(self.tracker_path, f_name + '.txt')

		with open(result_file_path) as f:
			content = f.readlines()

			f.close()

		predictions = []
		for c in content:
			ccs = c.split(',')
			x, y, w, h = float(ccs[0]), float(ccs[1]), float(ccs[2]), float(ccs[3])

			predictions.append([x, y, w, h])

		return predictions


