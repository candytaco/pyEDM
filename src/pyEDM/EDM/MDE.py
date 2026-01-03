"""Multivariate Delay Embedding (MDE) for pyEDM.

This module provides classes for multivariate feature selection using
Empirical Dynamic Modeling methods. The MDE class performs iterative
feature selection by evaluating combinations of features using Simplex
or S-Map predictions with parallel processing.
"""

from typing import List, Tuple

import numpy
from tqdm import tqdm as ProgressBar
from joblib import Parallel, delayed

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
				 target: int,
				 maxD: int = 5,
				 include_target: bool = True,
				 convergent: bool = True,
				 metric: str = "correlation",
				 batch_size: int = 1000,
				 columns=None,
				 train=None,
				 test=None,
				 embedDimensions=0,
				 predictionHorizon=1,
				 knn=0,
				 step=-1,
				 exclusionRadius=0,
				 embedded=False,
				 validLib=None,
				 noTime=False,
				 ignoreNan=True,
				 verbose=False,
				 useSMap: bool = False,
				 theta: float = 0.0,
				 solver=None):
		"""Initialize MDE with data and parameters.

		Parameters
		----------
		data : numpy.ndarray
			2D numpy array where column 0 is time (unless noTime=True)
		target : int
			Column index of the target column to forecast
		maxD : int, default=5
			Maximum number of features to select (including target if include_target=True)
		include_target : bool, default=True
			Whether to start with target in feature list
		convergent : bool, default=True
			Whether to use convergence checking for feature selection
		metric : str, default="correlation"
			Metric to use: "correlation" or "MAE"
		batch_size : int, default=1000
			Number of features to process in each parallel batch
		columns : list of int, optional
			Column indices to use for embedding (defaults to all except time)
		train : tuple of (int, int), optional
			Training set indices [start, end]
		test : tuple of (int, int), optional
			Test set indices [start, end]
		embedDimensions : int, default=0
			Embedding dimension (E). If 0, will be set by Validate()
		predictionHorizon : int, default=1
			Prediction time horizon (Tp)
		knn : int, default=0
			Number of nearest neighbors. If 0, will be set to E+1 by Validate()
		step : int, default=-1
			Time delay step size (tau). Negative values indicate lag
		exclusionRadius : int, default=0
			Temporal exclusion radius for neighbors
		embedded : bool, default=False
			Whether data is already embedded
		validLib : list, optional
			Boolean mask for valid library points
		noTime : bool, default=False
			Whether first column is time or data
		ignoreNan : bool, default=True
			Remove NaN values from embedding
		verbose : bool, default=False
			Print diagnostic messages
		useSMap : bool, default=False
			Whether to use SMap instead of Simplex
		theta : float, default=0.0
			S-Map localization parameter. theta=0 is global linear map,
			larger values increase localization
		solver : object, optional
			Solver to use for S-Map regression. If None, uses numpy.linalg.lstsq.
			Can be any sklearn-compatible regressor.
		"""
		self.data = data
		self.target = target
		self.maxD = maxD
		self.include_target = include_target
		self.convergent = convergent
		self.metric = metric
		self.batch_size = batch_size
		self.columns = columns
		self.train = train
		self.test = test
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.embedded = embedded
		self.validLib = validLib if validLib is not None else []
		self.noTime = noTime
		self.ignoreNan = ignoreNan
		self.verbose = verbose
		self.useSMap = useSMap
		self.theta = theta
		self.solver = solver

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
		if self.include_target:
			self.selectedVariables = [self.target]
		else:
			self.selectedVariables = []

		# Initial prediction
		initial_result = self._run_edm(self.columns)
		score = self._compute_performance(initial_result)
		self.accuracy.append(score)

		# Build the list of possible variables to check
		remaining_variables = self._get_remaining_variables()

		# Iteratively add variables up to maxD
		progressBar = ProgressBar(total = self.maxD, desc = 'Selecting variables', leave = False)
		while (len(self.selectedVariables) < self.maxD) and (len(remaining_variables) > 0):
			# Break up remaining columns into parallel-friendly batches
			batches = [
				remaining_variables[i:(i + self.batch_size)]
				for i in range(0, len(remaining_variables), self.batch_size)
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
			if self.convergent:
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
				progressBar.update(1)
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
		# Run prediction
		if self.useSMap:
			smap = SMap(
				data = self.data,
				columns = variables,
				target = self.target,
				train = self.train,
				test = self.test,
				embedDimensions = self.embedDimensions,
				predictionHorizon = self.predictionHorizon,
				knn = self.knn,
				step = self.step,
				exclusionRadius = self.exclusionRadius,
				theta = self.theta,
				solver = self.solver,
				embedded = self.embedded,
				validLib = self.validLib,
				noTime = self.noTime,
				ignoreNan = self.ignoreNan,
				verbose = self.verbose
			)
			result = smap.Run()
			return result
		else:
			simplex = Simplex(
				data = self.data,
				columns = variables,
				target = self.target,
				train = self.train,
				test = self.test,
				embedDimensions = self.embedDimensions,
				predictionHorizon = self.predictionHorizon,
				knn = self.knn,
				step = self.step,
				exclusionRadius = self.exclusionRadius,
				embedded = self.embedded,
				validLib = self.validLib,
				noTime = self.noTime,
				ignoreNan = self.ignoreNan,
				verbose = self.verbose
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
		if self.metric == "correlation":
			return result.compute_error()
		else:
			return result.compute_error("MAE")

	def _get_remaining_variables(self) -> List[int]:
		"""Get list of remaining columns to consider.

		Returns
		-------
		list of int
			List of column indices not yet selected
		"""
		all_columns = list(range(self.data.shape[1]))
		excluded = []
		if not self.include_target:
			excluded.append(self.target)
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
