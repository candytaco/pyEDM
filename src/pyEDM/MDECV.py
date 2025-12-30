"""Multivariate Delay Embedding Cross-Validation for pyEDM.

This module provides classes for cross-validated multivariate feature
selection using Empirical Dynamic Modeling methods. The MDECV class
performs feature selection within each cross-validation fold and
provides methods for selecting final features.
"""

from typing import List, Optional, Tuple
import numpy
from sklearn.model_selection import train_test_split, KFold
from .Parameters import EDMParameters, DataSplit, SMapParameters, MDEParameters, MDECVParameters
from .Results import MDECVResult, MDEResult
from .MDE import MDE


class MDECV:
	"""Multivariate Delay Embedding with Cross-Validation.

	This class extends MDE with cross-validation functionality, allowing
	feature selection to be performed within each fold and providing
	methods for selecting final features based on different criteria.
	"""

	def __init__(self,
				 trainData: numpy.ndarray,
				 MDEparameters: MDEParameters,
				 EDMparameters: EDMParameters,
				 CrossValidationParameters: MDECVParameters,
				 useSMap: bool = False,
				 SMapParameters: Optional[SMapParameters] = None):
		"""Initialize MDECV with data and parameters.

		Parameters
		----------
		trainData : numpy.ndarray
			train data for cross-validation 0 is time (unless noTime=True)
		MDEparameters : MDEParameters
			MDE-specific parameters
		EDMparameters : EDMParameters
			Core EDM parameters
		CrossValidationParameters : MDECVParameters
			Cross-validation specific parameters
		useSMap : bool, default=False
			Whether to use SMap instead of Simplex
		SMapParameters : SMapParameters, optional
			S-Map specific parameters (required if use_smap = True)
		"""
		self.data = trainData
		self.MDEParameters = MDEparameters
		self.EDMParameters = EDMparameters
		self.CrossValidationParameters = CrossValidationParameters
		self.useSMap = useSMap
		self.SMapParameters = SMapParameters

		# Cross-validation results
		self.fold_results = []
		self.test_accuracy = []
		self.bestFold = None
		self.best_fold_features = None
		self.best_fold_accuracy = None

	def fit(self) -> None:
		"""Perform cross-validation using MDE in each fold."""

		# Build fold indices
		if self.CrossValidationParameters.folds <= 1:
			trainIndices = numpy.arange(len(self.data))
			fold_indices = [(trainIndices, trainIndices)]
		else:
			kFolds = KFold(n_splits = self.CrossValidationParameters.folds, shuffle = False)
			fold_indices = [
				(train_idx, val_idx)
				for train_idx, val_idx in kFolds.split(self.data)
			]

		# Process each fold
		for fold, (trainIndices, validationIndices) in enumerate(fold_indices, start = 1):
			fold_result = self.fitSingleFold(self.data[trainIndices], self.data[validationIndices])
			self.fold_results.append(fold_result)

		# Identify best fold
		self.test_accuracy = [r.compute_error()["correlation"] for r in self.fold_results]
		self.bestFold = numpy.argmax(self.test_accuracy)
		self.best_fold_accuracy = self.test_accuracy[self.bestFold]
		self.best_fold_features = self.fold_results[self.bestFold].selectedVariables

	def fitSingleFold(self, train_data : numpy.ndarray, val_data: numpy.ndarray) -> MDEResult:
		"""Process a single cross-validation fold.

		Parameters
		----------
		train_data : numpy.ndarray
			Training data for this fold
		val_data : numpy.ndarray
			Validation data for this fold

		Returns
		-------
		MDEResult
			Results from this fold
		"""
		# Create fold-specific data
		fold_data = numpy.vstack([train_data, val_data])

		# Create split parameters for this fold
		train_split = DataSplit(
			train = (0, len(train_data) - 1),
			test = (len(train_data), len(fold_data) - 1)
		)

		# Run MDE on this fold
		mde = MDE(
			data = fold_data,
			MDEparameters = self.MDEParameters,
			EDMparameters = self.EDMParameters,
			split = train_split,
			useSMap = self.useSMap,
			SMapParameters = self.SMapParameters
		)

		return mde.Run()

	def predict(self, testData: numpy.ndarray) -> MDECVResult:
		"""Predict using the final chosen feature set on the test set.
		Note that this is different than the original implementation where the test data is
		internally generated from the data array

		:param testData:	test data with same variables as train data

		:return: Cross-validation results including final prediction
		"""
		# Decide final features based on mode
		if self.CrossValidationParameters.final_feature_mode == "best_fold":
			features = self.best_fold_features
		elif self.CrossValidationParameters.final_feature_mode == "frequency":
			features = self._get_frequency_features()
		else:
			features = self._get_best_n_features()

		stackedData = numpy.vstack((self.data, testData))

		# Run final prediction on test set
		test_split = DataSplit(
			train = (0, self.data.shape[0] - 1),
			test = (self.data.shape[0] - 1, stackedData.shape[0] - 1)
		)

		mde = MDE(
			data = stackedData,
			MDEparameters = self.MDEParameters,
			EDMparameters = self.EDMParameters,
			split = test_split,
			useSMap = self.useSMap,
			SMapParameters = self.SMapParameters
		)

		final_result = mde.Run()

		return MDECVResult(
			final_forecast = final_result.final_forecast,
			selected_features = features,
			fold_results = self.fold_results,
			accuracy = self.test_accuracy,
			best_fold = self.bestFold
		)

	def _get_frequency_features(self) -> List[int]:
		"""Get most frequent features across folds.

		Returns
		-------
		list of int
			List of most frequent feature indices
		"""
		all_features = []
		for result in self.fold_results:
			all_features.extend(result.selectedVariables)

		# Count frequency of each feature
		feature_counts = {}
		for feat in all_features:
			feature_counts[feat] = feature_counts.get(feat, 0) + 1

		# Sort by frequency and return top features
		sorted_features = sorted(feature_counts.items(), key = lambda x: x[1], reverse = True)
		return [feat for feat, count in sorted_features[:self.MDEParameters.maxD]]

	def _get_best_n_features(self) -> List[int]:
		"""Get top N features based on incremental prediction.

		Returns
		-------
		list of int
			List of top N feature indices
		"""
		# Get frequency features first
		freq_features = self._get_frequency_features()

		# Perform incremental prediction to find best N
		# This is a simplified version
		return freq_features[:len(freq_features) // 2]
