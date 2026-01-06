"""
CCM wrapper for sklearn-like API.
"""
from typing import Optional, List

import numpy

from pyEDM.EDM.CCM import CCM
from .EDMFitter import EDMFitter


class CCMFitter(EDMFitter):
	"""
	Wrapper class for CCM that provides sklearn-like API.
	CCM is done on two timeseries only! Even though the EDM API allows you to pass a list
	of columns to it, in reality the math can only work on single timeseries pairs, so we enforce that here
	"""

	def __init__(self,
				 TrainSizes: Optional[List[int]] = None,
				 numRepeats: int = 0,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False):
		"""
		Init.

		:param XTrain: 				1d timeseries of one variable
		:param YTrain: 				1d timeseries of another variable
		:param TrainSizes: 			train sizes to explore
		:param numRepeats: 			Number of repeats at each training size
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Verbose: 			Print diagnostic messages
		"""

		super().__init__()

		self.TrainSizes = TrainSizes
		self.Sample = numRepeats
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose

		self.CCM = None

	def Fit(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: numpy.ndarray = None, YTest: numpy.ndarray = None,
			TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0, TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		Data = self.GetEDMData()
		NoTime = not self.HasTime()

		# Columns and target are hard-coded because
		# we al;ways only have a single pair of things

		self.CCM = CCM(
			data = Data,
			columns = [0],
			target = [1],
			trainSizes = self.TrainSizes,
			sample = self.Sample,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			noTime = NoTime,
			verbose = self.Verbose
		)

		self.CCM.FwdMap.EmbedData()
		self.CCM.FwdMap.RemoveNan()
		self.CCM.RevMap.EmbedData()
		self.CCM.RevMap.RemoveNan()

		self.Result = self.CCM.Run()
		return self.Result
