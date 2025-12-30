"""Multivariate Delay Embedding (MDE) for pyEDM.

This module provides classes for multivariate feature selection using
Empirical Dynamic Modeling methods. The MDE class performs iterative
feature selection by evaluating combinations of features using Simplex
or S-Map predictions with parallel processing.
"""

from typing import List, Optional, Tuple

import numpy
from joblib import Parallel, delayed

from .Parameters import EDMParameters, DataSplit, SMapParameters, MDEParameters
from .Results import MDEResult, SimplexResult
from .SMap import SMap
from .Simplex import Simplex


class MDE:
	"""Multivariate Delay Embedding for feature selection.

	This class implements the iterative feature selection algorithm that
	evaluates combinations of features using EDM methods and selects the
	best performing features based on convergence criteria.
	"""

	def __init__(self,
				 data: numpy.ndarray,
				 MDEparameters: MDEParameters,
				 EDMparameters: EDMParameters,
				 split: DataSplit,
				 useSMap: bool = False,
				 SMapParameters: Optional[SMapParameters] = None):
		"""Initialize MDE with data and parameters.

		Parameters
		----------
		data : numpy.ndarray
			2D numpy array where column 0 is time (unless noTime=True)
		MDEparameters : MDEParameters
			MDE-specific parameters
		EDMparameters : EDMParameters
			Core EDM parameters
		split : DataSplit
			Train/test split configuration
		useSMap : bool, default=False
			Whether to use SMap instead of Simplex
		SMapParameters : SMapParameters, optional
			S-Map specific parameters (required if use_smap=True)
		"""
		self.data = data
		self.MDEParameters = MDEparameters
		self.EDMParameters = EDMparameters
		self.split = split
		self.useSMap = useSMap
		self.SMapParameters = SMapParameters

		# Initialize feature selection state
		self.selectedVariables = []
		self.accuracy = []
		self.ccm_values = []

	def Run(self) -> MDEResult:
		"""Execute MDE feature selection and return results.

		Returns
		-------
		MDEResult
			Results containing final prediction, selected features, accuracy,
			and CCM values
		"""
		# TODO: If the embedding dimensionality is not specified, estimate it

		# variable selection
		self._select_features()

		# Final training and testing
		finalPrediction = self._final_prediction()

		return MDEResult(
			final_forecast = finalPrediction,
			selected_features = self.selectedVariables,
			accuracy = self.accuracy,
			ccm_values = self.ccm_values
		)

	def _select_features(self) -> None:
		"""Perform iterative feature selection with parallel processing."""

		# do we include Y in the predictors that we select?
		if self.MDEParameters.include_target:
			self.selectedVariables = [self.MDEParameters.target]
		else:
			self.selectedVariables = []

		# Initial prediction
		initial_result = self._run_edm(self.EDMParameters.columns)
		score = self._compute_performance(initial_result)
		self.accuracy.append(score)

		# Build the list of possible variables to check
		remaining_variables = self._get_remaining_variables()

		# Iteratively add variables up to maxD
		while (len(self.selectedVariables) < self.MDEParameters.maxD) and (len(remaining_variables) > 0):
			# Break up remaining columns into parallel-friendly batches
			batches = [
				remaining_variables[i:(i + self.MDEParameters.batch_size)]
				for i in range(0, len(remaining_variables), self.MDEParameters.batch_size)
			]

			# Evaluate correlation/MAE for each possible addition in parallel
			batch_results = Parallel(n_jobs = -1)(
				delayed(self._evaluate_batch)(batch) for batch in batches
			)

			# Flatten results and sort
			metric_results = [item for sublist in batch_results for item in sublist]
			metric_results.sort(key = lambda x: x[1] if x[1] is not None else -numpy.inf, reverse = True)

			best_var = None
			best_score = None

			# If conv=True, use first convergent variable
			if self.MDEParameters.convergent:
				for c, score in metric_results:
					if c is None or numpy.isnan(score):
						continue
					# Check convergence
					check = self._check_convergence(c)
					if check[0]:
						best_var = c
						best_score = score
						self.ccm_values.append(check[1])
						break
					else:
						remaining_variables.remove(c)
			else:
				# Pick top scoring candidate
				if metric_results and not numpy.isnan(metric_results[0][1]):
					best_var = metric_results[0][0]
					best_score = metric_results[0][1]

			# Add best variable if found
			if best_var is not None:
				self.selectedVariables.append(best_var)
				remaining_variables.remove(best_var)
				self.accuracy.append(best_score)
			else:
				# No more valid candidates
				break

	def _evaluate_batch(self, batch: List[int]) -> List[Tuple[int, float]]:
		"""Evaluate a batch of candidate variables in parallel.

		Parameters
		----------
		batch : list of int
			List of variable indices to evaluate

		Returns
		-------
		list of tuple
			List of (column_index, metric_value) tuples
		"""
		results = []

		# TODO: this is quite suboptimal because:
		#  1. the performance is calculated on each item rather than array broadcast
		#  2. the EDM performance function calculates multiple metricx even though we only use one
		for var in batch:
			thesePredictors = [var] + self.selectedVariables
			result = self._run_edm(thesePredictors)
			score = self._compute_performance(result)
			results.append((var, score))
		return results

	def _run_edm(self, variables: List[int]) -> SimplexResult:
		"""Run EDM prediction with given variable indices.

		Parameters
		----------
		variables : list of int
			Column indices to use for prediction

		Returns
		-------
		SimplexResult or SMapResult
			Prediction results
		"""
		# Create EDM parameters with specified columns
		theseVariables = EDMParameters(
			data = self.data,
			columns = variables,
			target = self.MDEParameters.target,
			embedDimensions = self.EDMParameters.embedDimensions,
			predictionHorizon = self.EDMParameters.predictionHorizon,
			knn = self.EDMParameters.knn,
			step = self.EDMParameters.step,
			exclusionRadius = self.EDMParameters.exclusionRadius,
			embedded = self.EDMParameters.embedded,
			validLib = self.EDMParameters.validLib,
			noTime = self.EDMParameters.noTime,
			ignoreNan = self.EDMParameters.ignoreNan,
			verbose = self.EDMParameters.verbose
		)

		# Create split parameters
		split = DataSplit(
			train = self.split.train,
			test = self.split.test
		)

		# Run prediction
		if self.useSMap:
			smap = SMap(
				params = theseVariables,
				split = split,
				smap = self.SMapParameters
			)
			result = smap.Run()
			return result.prediction_result
		else:
			simplex = Simplex(
				params = theseVariables,
				split = split
			)
			return simplex.Run()

	def _compute_performance(self, result: SimplexResult) -> float:
		"""Compute optimization metric from prediction result.

		Parameters
		----------
		result : SimplexResult
			Prediction result

		Returns
		-------
		float
			Metric value (correlation or MAE)
		"""
		if self.MDEParameters.metric == "correlation":
			return result.compute_error()["correlation"]
		else:
			return result.compute_error()["MAE"]

	def _get_remaining_variables(self) -> List[int]:
		"""Get list of remaining columns to consider.

		Returns
		-------
		list of int
			List of column indices not yet selected
		"""
		all_columns = list(range(self.data.shape[1]))
		excluded = []
		if not self.MDEParameters.include_target:
			excluded.append(self.MDEParameters.target)
		excluded += self.selectedVariables

		return [c for c in all_columns if c not in excluded and c != 0]

	def _check_convergence(self, column: int) -> Tuple[bool, float]:
		"""Check convergence for a candidate feature.

		Parameters
		----------
		column : int
			Column index to check

		Returns
		-------
		tuple of (bool, float)
			(convergent, ccm_value) tuple
		"""
		# Simplified convergence check
		# In full implementation, this would use CCM
		# TODO: implement CCM convergence check
		return (True, 0.5)

	def _final_prediction(self) -> numpy.ndarray:
		"""Run final prediction with selected features.

		Returns
		-------
		numpy.ndarray
			Final prediction array [Time, Observations, Predictions]
		"""
		result = self._run_edm(self.selectedVariables)
		return result.projection
