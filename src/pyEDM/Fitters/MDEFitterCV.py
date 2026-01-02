"""
MDECV wrapper for sklearn-like API.
"""
from typing import Optional, List

import numpy

from pyEDM.EDM.MDECV import MDECV


class MDEFitterCV:
	"""
	Wrapper class for MDECV that provides sklearn-like API.
	"""

	def __init__(self,
				 XTrain: numpy.ndarray,
				 YTrain: numpy.ndarray,
				 XTest: numpy.ndarray,
				 YTest: numpy.ndarray,
				 MaxD: int = 5,
				 IncludeTarget: bool = True,
				 Convergent: bool = True,
				 Metric: str = "correlation",
				 BatchSize: int = 1000,
				 Folds: int = 5,
				 TestSize: float = 0.2,
				 FinalFeatureMode: str = "best_fold",
				 Columns: Optional[List[int]] = None,
				 Target: Optional[int] = None,
				 EmbedDimensions: int = 0,
				 PredictionHorizon: int = 1,
				 KNN: int = 0,
				 Step: int = -1,
				 ExclusionRadius: int = 0,
				 Verbose: bool = False,
				 UseSMap: bool = False,
				 Theta: float = 0.0):
		"""
		Initialize MDECV wrapper with sklearn-style separate arrays.

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
		Folds : int, default=5
			Number of cross-validation folds
		TestSize : float, default=0.2
			Proportion of data to use for test set
		FinalFeatureMode : str, default="best_fold"
			Method for selecting final features
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
		Verbose : bool, default=False
			Print diagnostic messages
		UseSMap : bool, default=False
			Whether to use SMap instead of Simplex
		Theta : float, default=0.0
			S-Map localization parameter
		"""

		self.XTrain = XTrain
		self.YTrain = YTrain
		self.XTest = XTest
		self.YTest = YTest
		self.MaxD = MaxD
		self.IncludeTarget = IncludeTarget
		self.Convergent = Convergent
		self.Metric = Metric
		self.BatchSize = BatchSize
		self.Folds = Folds
		self.TestSize = TestSize
		self.FinalFeatureMode = FinalFeatureMode
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

		self.MDECV = None

	def Fit(self):
		"""
		Fit MDECV model using cross-validation.

		Returns
		-------
		MDECVResult
			Cross-validation results
		"""
		# Combine train data
		TrainData = numpy.hstack([self.XTrain, self.YTrain])

		# Determine columns to use
		if self.Columns is not None:
			Columns = self.Columns
		else:
			Columns = list(range(self.XTrain.shape[1]))

		# Determine target
		if self.Target is not None:
			Target = self.Target
		else:
			Target = self.XTrain.shape[1]

		self.MDECV = MDECV(
			trainData = TrainData,
			target = Target,
			maxD = self.MaxD,
			include_target = self.IncludeTarget,
			convergent = self.Convergent,
			metric = self.Metric,
			batch_size = self.BatchSize,
			folds = self.Folds,
			test_size = self.TestSize,
			final_feature_mode = self.FinalFeatureMode,
			columns = Columns,
			embedDimensions = self.EmbedDimensions,
			predictionHorizon = self.PredictionHorizon,
			knn = self.KNN,
			step = self.Step,
			exclusionRadius = self.ExclusionRadius,
			verbose = self.Verbose,
			useSMap = self.UseSMap,
			theta = self.Theta
		)

		self.MDECV.fit()
		return self.MDECV

	def Predict(self):
		"""
		Predict using the fitted MDECV model.

		Returns
		-------
		MDECVResult
			Prediction results
		"""
		if self.MDECV is None:
			raise RuntimeError("Model not fitted. Call Fit() first.")

		# Combine all data for prediction
		TestData = numpy.hstack([self.XTest, self.YTest])
		return self.MDECV.predict(TestData)
