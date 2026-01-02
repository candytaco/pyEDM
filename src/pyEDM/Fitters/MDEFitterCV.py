"""
MDECV wrapper for sklearn-like API.
"""

import numpy

from ..EDM.MDECV import MDECV


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

		:param XTrain: 				Training feature data
		:param YTrain: 				Training target data
		:param XTest: 				Test feature data
		:param YTest: 				Test target data
		:param MaxD: 				Maximum number of features to select
		:param IncludeTarget: 		Whether to start with target in feature list
		:param Convergent: 			Whether to use convergence checking
		:param Metric: 				Metric to use: "correlation" or "MAE"
		:param BatchSize: 			Number of features to process in each batch
		:param Folds: 				Number of cross-validation folds
		:param TestSize: 			Proportion of data to use for test set
		:param FinalFeatureMode: 	Method for selecting final features
		:param EmbedDimensions: 	Embedding dimension (E)
		:param PredictionHorizon: 	Prediction time horizon (Tp)
		:param KNN: 				Number of nearest neighbors
		:param Step: 				Time delay step size (tau)
		:param ExclusionRadius: 	Temporal exclusion radius for neighbors
		:param Verbose: 			Print diagnostic messages
		:param UseSMap: 			Whether to use SMap instead of Simplex
		:param Theta: 				S-Map localization parameter
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

		:return: Cross-validation results
		"""
		# Combine train data
		TrainData = numpy.hstack([self.XTrain, self.YTrain])

		Columns = list(range(0, self.XTrain.shape[0] - 1))
		Target = TrainData.shape[1] - 1

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

		:return: Prediction results
		"""
		if self.MDECV is None:
			raise RuntimeError("Model not fitted. Call Fit() first.")

		# Combine all data for prediction
		TestData = numpy.hstack([self.XTest, self.YTest])
		return self.MDECV.predict(TestData)
