"""
MDE wrapper for sklearn-like API.
"""
from typing import Optional, List

import numpy

from pyEDM.EDM.MDE import MDE
from .EDMFitter import EDMFitter

class MDEFitter(EDMFitter):
	"""
	Wrapper class for MDE that provides sklearn-like API.
	"""

	def __init__(self,
				 MaxD: int = 5,
				 IncludeTarget: bool = True,
				 Convergent: bool = True,
				 Metric: str = "correlation",
				 BatchSize: int = 1000,
				 Columns: Optional[List[int]] = None,
				 Target: Optional[int] = None,
				 Embed: bool = False,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0,
				 nThreads: int = -1):
		"""
		Initialize MDE wrapper with sklearn-style separate arrays.

		:param MaxD: 				Maximum number of features to select
		:param IncludeTarget: 		Whether to start with target in feature list
		:param Convergent: 			Whether to use convergence checking
		:param Metric: 				Metric to use: "correlation" or "MAE"
		:param BatchSize: 			Number of features to process in each batch
		:param Columns: 			Column indices to use for embedding
		:param Target: 				Target column index
		:param Embde:				whether to embed the data or not
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Verbose: 			Print diagnostic messages
		:param UseSMap: 			Whether to use SMap instead of Simplex
		:param Theta: 				S-Map localization parameter
		"""

		super().__init__()

		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.Columns = Columns
		self.Target = Target
		self.EmbedDimensions = EmbedDimensions
		self.PredictionHorizon = PredictionHorizon
		self.KNN = KNN
		self.Step = Step
		self.ExclusionRadius = ExclusionRadius
		self.Verbose = Verbose
		self.UseSMap = UseSMap
		self.Theta = Theta
		self.nThreads = nThreads
		self.Embed = Embed

		self.MDE = None

	def Fit(self, XTrain: numpy.ndarray, YTrain: numpy.ndarray, XTest: numpy.ndarray, YTest: numpy.ndarray,
			TrainStart = 0, TrainEnd = 0, TestStart = 0, TestEnd = 0, TrainTime: Optional[numpy.ndarray] = None,
			TestTime: Optional[numpy.ndarray] = None):
		super().Fit(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime, TestTime)

		Data = self.GetEDMData()
		TrainIndices = self.GetTrainIndices()
		TestIndices = self.GetTestIndices()
		YIndex = self.GetYIndex()
		NoTime = not self.HasTime()

		# Determine columns to use
		XStart, XEnd = self.GetXIndices()
		if self.Columns is not None:
			Columns = [XStart + col for col in self.Columns]
		else:
			Columns = list(range(XStart, XEnd + 1))

		# Determine target
		if self.Target is not None:
			Target = XStart + self.Target
		else:
			Target = YIndex

		self.MDE = MDE(
			data = Data,
			target = Target,
			maxD = self.MaxD,
			include_target = self.IncludeTarget,
			convergent = self.Convergent,
			metric = self.Metric,
			batch_size = self.BatchSize,
			columns = Columns,
			train = TrainIndices,
			test = TestIndices,
			embedded = not self.Embed,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			noTime = NoTime,
			verbose = self.Verbose,
			useSMap = self.UseSMap,
			theta = self.Theta,
			nThreads = self.nThreads
		)

		self.Result = self.MDE.Run()
		return self.Result
