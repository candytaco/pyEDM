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
				 XTrain: numpy.ndarray,
				 YTrain: numpy.ndarray,
				 XTest: numpy.ndarray,
				 YTest: numpy.ndarray,
				 TrainStart: int = 0,
				 TrainEnd: int = 0,
				 TestStart: int = 0,
				 TestEnd: int = 0,
				 MaxD: int = 5,
				 IncludeTarget: bool = True,
				 Convergent: bool = True,
				 Metric: str = "correlation",
				 BatchSize: int = 1000,
				 Columns: Optional[List[int]] = None,
				 Target: Optional[int] = None,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 TrainTime: Optional[numpy.ndarray] = None,
				 TestTime: Optional[numpy.ndarray] = None,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0):
		"""
		Initialize MDE wrapper with sklearn-style separate arrays.

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
		MaxD : int, default=5
			Maximum number of features to select
		IncludeTarget : bool, default=True
			Whether to start with target in feature list
		Convergent : bool, default=True
			Whether to use convergence checking
		Metric : str, default="correlation"
			Metric to use: "correlation" or "MAE"
		BatchSize : int, default=1000
			Number of features to process in each batch
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
		ExclusionRadius : int, default=0
			Temporal exclusion radius for neighbors
		TrainTime : numpy.ndarray, optional
			Time labels for train data
		TestTime : numpy.ndarray, optional
			Time labels for test data
		Verbose : bool, default=False
			Print diagnostic messages
		UseSMap : bool, default=False
			Whether to use SMap instead of Simplex
		Theta : float, default=0.0
			S-Map localization parameter
		"""

		super().__init__(XTrain, YTrain, XTest, YTest, TrainStart, TrainEnd, TestStart, TestEnd, TrainTime = TrainTime,
						 TestTime = TestTime)

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

		self.MDE = None

	def Run(self):
		"""
		Run MDE feature selection.

		Returns
		-------
		MDEResult
			MDE results
		"""
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
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			noTime = NoTime,
			verbose = self.Verbose,
			useSMap = self.UseSMap,
			theta = self.Theta
		)

		return self.MDE.Run()
