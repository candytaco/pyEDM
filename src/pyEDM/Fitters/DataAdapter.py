"""
Data adapter for handling separate X/Y and train/test arrays.

Provides bridge from modern SKLearn style to EDM single-array style.

Note: the EDM-style API, when given indices along the time dimension, are stop-inclusive, which is
Counter to the normal stop-exclusive indexing style in python
"""
from typing import Optional, Tuple

import numpy

class DataAdapter:

	def __init__(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: Optional[numpy.ndarray] = None,
				 YTest: Optional[numpy.ndarray] = None, TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0,
				 trainTime: Optional[numpy.ndarray] = None, testTime: Optional[numpy.ndarray] = None):
		"""
		Data adapter init
		:param TrainStart:
		:param TestStart:
		:param XTrain: 		training features
		:param XTest: 		testing features
		:param YTrain: 		training value to predict, should be just a single column
		:param YTest: 		testing value to predict, should be just a single column
		:param TrainStart:	index at which to start the train data; used to provide history for the first train sample
		:param TrainEnd:	number of additional data samples at end of train data to ignore
		:param TestEnd:		number of additional data samples at end of test data to ignore
		:param TestStart:	index at which to start the test data; used to provide history for the first test sample
		:param trainTime: 	time labels for train data
		:param testTime: 	time labels for test data
		"""

		self.X_train = XTrain
		self.X_test = XTest
		self.Y_train = YTrain.squeeze()[:, None]
		self.trainTime = trainTime
		self.testTime = testTime
		self.hasTime = False
		if YTest is not None:
			self.Y_test = YTest.squeeze()[:, None]

		train = numpy.hstack([XTrain, YTrain])
		self.trainOffset = TrainStart
		self.trainEnd = TrainEnd
		self.testOffset = TestStart + TrainStart + TrainEnd
		self.testEnd = TestEnd

		if YTest is not None:
			test = numpy.hstack([XTest, YTest])
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
		# returning with 1 subtracted because EDM functions are stop-inclusive
		return (self.trainOffset, self.X_train.shape[0] - 1 + self.trainOffset - self.trainEnd)

	@property
	def TestIndices(self) -> Tuple[int, int]:
		if self.Y_test is not None:
			return (self.X_train.shape[0] + self.testOffset, self.fullData.shape[0] - 1 - self.testEnd)
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