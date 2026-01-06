"""
Multiview wrapper for sklearn-like API.
"""
from typing import Optional

import numpy

from ..EDM.Multiview import Multiview
from .EDMFitter import EDMFitter


class MultiviewFitter(EDMFitter):
	"""
	Wrapper class for Multiview that provides sklearn-like API.
	"""

	def __init__(self,
				 dimensions: int = 0,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 NumMultiview: int = 0,
				 ExclusionRadius: int = 0,
				 TrainLib: bool = True,
				 ExcludeTarget: bool = False,
				 Verbose: bool = False):
		"""
		Initialize Multiview wrapper with sklearn-style separate arrays.

		:param XTrain: 			Training feature data
		:param YTrain: 			Training target data
		:param XTest: 			Test feature data
		:param YTest: 			Test target data
		:param dimensions: 		State-space dimension
		:param TrainStart: 		Start index for train data
		:param TrainEnd: 		Number of samples at end of train data to ignore
		:param TestStart: 		Start index for test data
		:param TestEnd: 		Number of samples at end of test data to ignore
		:param EmbedDimensions: Embedding dimension (E)
		:param PredictionHorizon:	Prediction time horizon (Tp)
		:param KNN: 			Number of nearest neighbors
		:param Step: 			Time delay step size (tau)
		:param NumMultiview: 	Number of top-ranked predictions
		:param ExclusionRadius: Temporal exclusion radius for neighbors
		:param TrainLib: 		Evaluation strategy for ranking
		:param ExcludeTarget: 	Whether to exclude target column
		:param TrainTime: 		Time labels for train data
		:param TestTime: 		Time labels for test data
		:param Verbose: 		Print diagnostic messages
		"""

		super().__init__()

		self.dimensions = dimensions
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.NumMultiview = NumMultiview
		self.ExclusionRadius = ExclusionRadius
		self.TrainLib = TrainLib
		self.ExcludeTarget = ExcludeTarget
		self.Verbose = Verbose

		self.Multiview = None

	def Fit(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: numpy.ndarray, YTest: numpy.ndarray,
			TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0, TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		Data = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices = self.GetTestIndices()
		YIndex = self.GetYIndex()

		XStart, XEnd = self.GetXIndices()
		Columns = list(range(XStart, XEnd + 1))
		Target = YIndex

		self.Multiview = Multiview(
			data = Data,
			columns = Columns,
			target = Target,
			train = TrainIndices,
			test = TestIndices,
			D = self.dimensions,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			multiview = self.NumMultiview,
			exclusionRadius = self.ExclusionRadius,
			trainLib = self.TrainLib,
			excludeTarget = self.ExcludeTarget,
			verbose = self.Verbose
		)

		return self.Multiview.Run()
