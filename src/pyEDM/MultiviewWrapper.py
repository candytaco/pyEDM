"""
Multiview wrapper for sklearn-like API.
"""
from typing import Optional, List

import numpy

from .EDMWrapper import EDMWrapper
from .Multiview import Multiview

class MultiviewWrapper(EDMWrapper):
	"""
	Wrapper class for Multiview that provides sklearn-like API.
	"""

	def __init__(self,
				 XTrain: numpy.ndarray,
				 YTrain: numpy.ndarray,
				 XTest: numpy.ndarray,
				 YTest: numpy.ndarray,
				 D: int = 0,
				 Columns: Optional[List[int]] = None,
				 Target: Optional[int] = None,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 NumMultiview: int = 0,
				 ExclusionRadius: int = 0,
				 TrainLib: bool = True,
				 ExcludeTarget: bool = False,
				 TrainTime: Optional[numpy.ndarray] = None,
				 TestTime: Optional[numpy.ndarray] = None,
				 Verbose: bool = False,
				 XTestHistory = None,
				 YTestHistory = None,
				 TestHistoryTime = None):
		"""
		Initialize Multiview wrapper with sklearn-style separate arrays.

		Parameters
		----------
		XTrain : numpy.ndarray
			Training feature data
		YTrain : numpy.ndarray
			Training target data
		XTest : numpy.ndarray
			Test feature data
		YTest : numpy.ndarray
			Test target data
		D : int, default=0
			State-space dimension
		Columns : list of int, optional
			Column indices to use for embedding
		Target : int, optional
			Target column index
		EmbedDimensions : int, default=0
			Embedding dimension (E)
		PredictionHorizon : int, default=1
			Prediction time horizon (Tp)
		KNN : int, default=0
			Number of nearest neighbors
		Step : int, default=-1
			Time delay step size (tau)
		NumMultiview : int, default=0
			Number of top-ranked predictions
		ExclusionRadius : int, default=0
			Temporal exclusion radius for neighbors
		TrainLib : bool, default=True
			Evaluation strategy for ranking
		ExcludeTarget : bool, default=False
			Whether to exclude target column
		TrainTime : numpy.ndarray, optional
			Time labels for train data
		TestTime : numpy.ndarray, optional
			Time labels for test data
		Verbose : bool, default=False
			Print diagnostic messages
		"""

		super().__init__(XTrain, YTrain, XTest, YTest, XTestHistory, YTestHistory,
						 TrainTime, TestTime, TestHistoryTime)

		self.D = D
		self.Columns = Columns
		self.Target = Target
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

	def Run(self):
		"""
		Run Multiview prediction.

		Returns
		-------
		MultiviewResult
			Multiview results
		"""
		Data = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices = self.GetTestIndices()
		YIndex = self.GetYIndex()

		# Determine columns to use
		XStart, XEnd = self.GetXIndices()
		if self.Columns is not None:
			# Map wrapper columns to EDM data columns
			Columns = [XStart + col for col in self.Columns]
		else:
			# Use all X columns
			Columns = list(range(XStart, XEnd + 1))

		# Determine target
		if self.Target is not None:
			# Map wrapper target to EDM data columns
			Target = XStart + self.Target
		else:
			Target = YIndex


		self.Multiview = Multiview(
			data=Data,
			columns=Columns,
			target=Target,
			train=TrainIndices,
			test=TestIndices,
			D=self.D,
			embedDimensions=self.EmbedDimensions,
			predictionHorizon=self.PredictionHorizon,
			knn=self.KNN,
			step=self.Step,
			multiview=self.NumMultiview,
			exclusionRadius=self.ExclusionRadius,
			trainLib=self.TrainLib,
			excludeTarget=self.ExcludeTarget,
			verbose=self.Verbose
		)

		return self.Multiview.Run()
