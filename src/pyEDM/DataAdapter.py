"""
Data adapter for handling separate X/Y and train/test arrays.

Provides bridge from modern SKLearn style to EDM single-array style
"""
from typing import Optional, Tuple

import numpy

class DataAdapter:

	def __init__(self, X_train: numpy.ndarray, Y_train: numpy.ndarray, X_test: Optional[numpy.ndarray] = None,
				 Y_test: Optional[numpy.ndarray] = None,
				 trainTime: Optional[numpy.ndarray] = None, testTime: Optional[numpy.ndarray] = None):
		"""
		Data adapter init
		:param X_train: 	training features
		:param X_test: 		testing features
		:param Y_train: 	training value to predict, should be just a single column
		:param Y_test: 		testing value to predict, should be juse a single column
		:param trainTime: 	time labels for train data
		:param testTime: 	time labels for test data
		"""

		self.X_train = X_train
		self.X_test = X_test
		self.Y_train = Y_train.squeeze()[:, None]
		if Y_test is not None:
			self.Y_test = Y_test.squeeze()[:, None]
		self.trainTime = trainTime
		self.testTime = testTime
		self.hasTime = False

		train = numpy.hstack([X_train, Y_train])

		if Y_test is not None:
			test = numpy.hstack([X_test, Y_test])
			data = numpy.vstack([train, test])
		else:
			data = train

		# add time if not none
		if trainTime is not None:
			self.trainTime = self.trainTime.squeeze()
			if testTime is not None:
				self.testTime = self.testTime.squeeze()
				time = numpy.concatenate([self.trainTime, self.testTime])
			else:
				time = self.trainTime
			data = numpy.hstack([time[:, None], data])
			self.hasTime = True

		self.fullData = data

	@property
	def HasTime(self) -> bool:
		return self.hasTime

	@property
	def TrainIndices(self) -> Tuple[int, int]:
		return (0, self.X_train.shape[0])

	@property
	def TestIndices(self) -> Tuple[int, int]:
		if self.Y_test is not None:
			return (self.X_train.shape[0], self.fullData.shape[0])
		else:
			raise ValueError('No test data')

	@property
	def XIndices(self) -> Tuple[int, int]:
		"""
		Indices for X variables. The end is Inclusive!
		:return:
		"""
		return (0 + int(self.hasTime), self.X_train.shape[1] + int(self.hasTime) - 1)

	@property
	def YIndex(self) -> int:
		"""
		Index for Y variable, assumes we only do one
		:return:
		"""
		return self.fullData.shape[1] - 1