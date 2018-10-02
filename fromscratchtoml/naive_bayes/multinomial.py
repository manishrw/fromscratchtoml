#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author - Manish Wadhwani <manishwr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT

import numpy as np

from ..base import BaseModel
from ..toolbox.exceptions import InvalidArgumentError

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultinomialNaiveBayes(BaseModel):
	"""Implements the multinomial model for naive bayes classifier

	Examples
	--------
	>>> from fromscratchtoml import np
	>>> from fromscratchtoml.naive_bayes import MultinomialNaiveBayes as MultinomialNaiveBayes
	>>> from fromscratchtoml.toolbox.random import Distribution
	>>> X = np.random.randint(10, size=(1000, 10))
	>>> Y = np.random.randint(2, size =(1000))
	>>> testX = np.random.randint(10, size=(100, 10))
	>>> testY = np.random.randint(2, size =(100))
	>>> model = MultinomialNaiveBayes()
	>>> model.fit(X, Y)
	>>> model.test(testX, testY)

	Parameters
	----------
	alpha : float, optional (default=1.0)
			Additive Laplace smoothing parameter (0 for no smoothing).
	fit_prior : boolean, optional (default=True)
			Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
	class_prior : array-like, size (n_classes,), optional (default=None)
			Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
	"""

	def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
		self.n = self.m = 0
		self.class_prior = class_prior
		self.fit_prior = fit_prior
		self.alpha = alpha
	
	def fit(self, X, Y):
		self.n = n = len(Y)
		self.m = m = len(X[0])

		# summarize class data and calculate prior probaility of class
		if not self.class_prior and self.fit_prior: # calculate class prior from data
			class_bin_count = dict()
			for val in Y:
				class_bin_count[val] = class_bin_count[val] + 1 if val in class_bin_count else 1

			self.class_prior = dict()
			for key in class_bin_count:
				self.class_prior[key] = class_bin_count[key] / self.n
		
		elif not self.fit_prior: # uniform class prior
			class_bin_count = dict()
			for val in Y:
				class_bin_count[val] = 1
			
			class_len = len(class_bin_count)
			self.class_prior = dict()
			for key, val in class_bin_count.items:
				self.class_prior[key] = val / class_len

		# summarize feature data and calculate likelihood of the feature given the class is true
		feature_cnt_by_class = dict()
		feature_cnt_sum_by_class = dict()
		
		for i in range(n):
			c_val = Y[i]
			
			if c_val not in feature_cnt_by_class:
				feature_cnt_by_class[c_val] = [0] * m
				feature_cnt_sum_by_class[c_val] = 0
			
			for j in range(m):
				feature_cnt_by_class[c_val][j] += X[i][j]
				feature_cnt_sum_by_class[c_val] += X[i][j]

		self.feature_cnt_by_class = feature_cnt_by_class
		self.feature_cnt_sum_by_class = feature_cnt_sum_by_class
	
	def predict(self, X):
		class_predictions = dict()

		for c_key, c_val in self.class_prior.items():
			class_predictions[c_key] = c_val
			
			for i in range(self.m):
				tmp = (self.feature_cnt_by_class[c_key][i] + self.alpha) 
				tmp /= (self.feature_cnt_sum_by_class[c_key] + self.alpha * self.m)
				class_predictions[c_key] *= tmp

		return class_predictions
	
	def test(self, testX, testY):
		n = len(testX)
		cnt = 0
		for i in range(n):
			result = self.predict(testX[i])
			if max(result.items(), key=operator.itemgetter(1))[0] == testY[i]:
				cnt += 1
		return cnt * 100 / n