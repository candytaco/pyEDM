"""
Simplex wrapper for sklearn-like API.
"""
from typing import Optional

import numpy

from pyEDM.EDM.Simplex import Simplex
from .EDMFitter import EDMFitter


class SimplexFitter(EDMFitter):
	"""
	Wrapper class for Simplex that provides sklearn-like API.
	"""

	def __init__(self,
				 XTrain: numpy.ndarray,
				 YTrain: numpy.ndarray,
				 XTest: numpy.ndarray,
				 YTest: numpy.ndarray,
				 TrainStart: int = 0,
				 TrainEnd: int = 0,
				 TestStart: int = 0,
				 TestEnd: int = 0,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Embedded: bool = False,
				 TrainTime: Optional[numpy.ndarray] = None,
				 TestTime: Optional[numpy.ndarray] = None,
				 Verbose: bool = False):
		"""
		Initialize Simplex wrapper with sklearn-style separate arrays.

		:param XTrain: 				Training feature data
		:param YTrain: 				Training target data
		:param XTest: 				Test feature data
		:param YTest: 				Test target data
		:param TrainStart: 			Start index for train data
		:param TrainEnd: 			End index for train data
		:param TestStart: 			Start index for test data
		:param TestEnd: 			End index for test data
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Embedded: 			Whether data is already embedded
		:param TrainTime: 			Time labels for train data
		:param TestTime: 			Time labels for test data
		:param Verbose: 			Print diagnostic messages
		"""

		super().__init__(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd,
						 TrainTime = TrainTime, TestTime = TestTime)

		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Embedded = Embedded
		self.Verbose = Verbose

		self.Simplex = None

	def Run(self):
		"""
		Run Simplex prediction.

		:return: Prediction results
		"""
		Data = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices = self.GetTestIndices()
		YIndex = self.GetYIndex()
		NoTime = not self.HasTime()

		XStart, XEnd = self.GetXIndices()
		Columns = list(range(XStart, XEnd + 1))
		Target = YIndex

		self.Simplex = Simplex(
			data = Data,
			columns = Columns,
			target = Target,
			train = TrainIndices,
			test = TestIndices,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			noTime = NoTime,
			verbose = self.Verbose,
			embedded = self.Embedded
		)

		return self.Simplex.Run()
