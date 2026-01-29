"""Multivariate Delay Embedding (MDE) for pyEDM.

This module provides classes for multivariate feature selection using
Empirical Dynamic Modeling methods. The MDE class performs iterative
feature selection by evaluating combinations of features using Simplex
or S-Map predictions with parallel processing.
"""

from typing import List, Tuple

import numpy
from tqdm import tqdm as ProgressBar
import torch

from .NeighborFinder import PairwiseDistanceNeighborFinder
from .Results import MDEResult, SimplexResult
from .SMap import SMap
from .Simplex import Simplex

from ._MDE import ElementwisePairwiseDistance, RowwiseCorrelation, \
	IncrementPairwiseDistance, FloorArray, MinAxis1, ComputeWeights, SumAxis1, \
	ComputePredictions
from .. import FindOptimalEmbeddingDimensionality


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
				 use_half_precision: bool = False,
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
				 solver=None,
				 nThreads = -1,
				 stdThreshold: float = 1e-3,
				 CCMLibrarySizes = None,
				 CCMSampleSize: int = 100,
				 CCMConvergenceThreshold: float = 0.01,
				 MinPredictionThreshold: float = 0.0,
				 EmbedDimCorrelationMin: float = 0.0,
				 FirstEMax: bool = False,
				 TimeDelay: int = 0):
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
			Number of features to process in each batch
		use_half_precision : bool, default=False
			Use float16 instead of float32 for GPU tensors to save memory
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
		CCMLibrarySizes : list, optional
			Library sizes for CCM testing as [start, stop, increment].
			If None, defaults to [10, 100, 10]
		CCMSampleSize : int, default=100
			Number of random samples per library size for CCM
		CCMConvergenceThreshold : float, default=0.01
			Minimum slope threshold for CCM convergence
		MinPredictionThreshold : float, default=0.0
			Minimum correlation threshold for candidate filtering
		EmbedDimCorrelationMin : float, default=0.0
			Minimum correlation for E selection
		FirstEMax : bool, default=False
			Use first local maximum in E-rho curve instead of global max
		TimeDelay : int, default=0
			Time delay analysis depth. If 0, time delay analysis is disabled
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
		self.nThreads = nThreads
		self.stdThreshold = stdThreshold
		self.use_half_precision = use_half_precision
		self.CCMLibrarySizes = CCMLibrarySizes if CCMLibrarySizes is not None else [10, 100, 10]
		self.CCMSampleSize = CCMSampleSize
		self.CCMConvergenceThreshold = CCMConvergenceThreshold
		self.MinPredictionThreshold = MinPredictionThreshold
		self.EmbedDimCorrelationMin = EmbedDimCorrelationMin
		self.FirstEMax = FirstEMax
		self.TimeDelay = TimeDelay
		self.optimalEmbeddingDimensions = {}

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.dtype = torch.float16 if use_half_precision else torch.float32

		self.rankings_ = None # performances of adding each variable at each iteration
		self.all_distances = None
		self.current_best_distance_matrix = None

		# Initialize feature selection state
		self.selectedVariables = []
		self.accuracy = []
		self.ccm_values = []
		self.results_ = None
		self.trainData = None
		self.testData = None
		self.timeDelayResults = None

	def Run(self) -> MDEResult:
		"""Execute MDE feature selection and return results.

		Returns
		-------
		MDEResult
			Results containing final prediction, selected features, accuracy,
			and CCM values
		"""
		# TODO: If the embedding dimensionality is not specified, estimate it
		if self.embedDimensions == 0:
			self.embedDimensions = FindOptimalEmbeddingDimensionality(self.data, [self.target], self.target, self.maxD,
																	  train = self.train, test = self.test, predictionHorizon = self.predictionHorizon,
																	  noTime = self.noTime)
		if self.knn == 0:
			self.knn = self.embedDimensions + 1

		# variable selection
		self._select_features()

		# Final training and testing
		finalPrediction = self._final_prediction()

		self.results_ = MDEResult(
			final_forecast = finalPrediction,
			selected_features = self.selectedVariables,
			accuracy = self.accuracy,
			ccm_values = self.ccm_values,
			rankings = self.rankings_,
			timeDelayResults = self.timeDelayResults
		)
		return self.results_

	def _select_features(self) -> None:
		"""Perform iterative feature selection with parallel processing."""

		self.selectedVariables = []
		if self.include_target:
			self.selectedVariables.append(self.target)
		#
		# we use this to correctly get indices for calculating the distance tensor
		dummy = Simplex(
			data = self.data,
			columns = numpy.arange(self.data.shape[1]).tolist(),
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
		dummy.EmbedData()
		trainData = dummy.Embedding[dummy.trainIndices, :]
		testData = dummy.Embedding[dummy.testIndices, :]
		self.trainData = trainData
		self.testData = testData

		nTrain = trainData.shape[0]
		nTest = testData.shape[0]
		nVars = self.data.shape[1]
		
		all_columns = numpy.arange(nVars - 1) # ignore the Y var, which is the last column
		# ignore all variables with stdev less than threshold
		excluded = numpy.argwhere(numpy.std(self.data, axis = 0) < self.stdThreshold).squeeze().tolist()
		if not self.noTime: # time is first column if true and we exclude that
			excluded.append(0)
		if not self.include_target:
			excluded.append(self.target)
		excluded += self.selectedVariables

		remaining_variables = [c for c in all_columns if c not in excluded]

		# Filter convergent variables BEFORE selection if convergent=True
		if self.convergent:
			remaining_variables = self._filter_convergent_variables(remaining_variables)

		# Iteratively add variables up to maxD
		progressBar = ProgressBar(total = self.maxD, desc = 'Selecting variables', leave = False)

		# rankings is a numpy array because it's apparently more multithreading friendly
		# than just storing the lists that come out?
		self.rankings_ = numpy.zeros([self.maxD, self.data.shape[1]])

		trainData_tensor = torch.tensor(trainData, device = self.device, dtype = self.dtype)
		testData_tensor = torch.tensor(testData, device = self.device, dtype = self.dtype)
		current_best_distance_matrix = torch.tensor(dummy._BuildExclusionMask(), device = self.device, dtype = self.dtype)
		train_y = self.data[:, self.target]
		train_y_tensor = torch.tensor(train_y, device = self.device, dtype = self.dtype)
		test_y = testData[:, self.target]
		test_y_tensor = torch.tensor(test_y, device = self.device, dtype = self.dtype)

		for i in range(self.maxD):
			# Process remaining variables in batches to avoid OOM
			metric_results = []

			for batch_start in range(0, len(remaining_variables), self.batch_size):
				batch_end = min(batch_start + self.batch_size, len(remaining_variables))
				batch_vars = remaining_variables[batch_start:batch_end]
				batch_size = len(batch_vars)

				# Compute distances for this batch of variables
				batch_distances = torch.zeros([batch_size, nTrain, nTest], device = self.device, dtype = self.dtype)
				for j, var in enumerate(batch_vars):
					diff = trainData_tensor[:, var].unsqueeze(1) - testData_tensor[:, var].unsqueeze(0)
					batch_distances[j, :, :] = diff * diff

				# Add current best distances
				candidateDistances = batch_distances + current_best_distance_matrix.unsqueeze(0)

				# find k nearest neighbors
				nearestNeighbors = torch.topk(candidateDistances, self.knn, dim = 1, largest = False)[1]
				neighborDistances = torch.gather(candidateDistances, 1, nearestNeighbors)
				FloorArray(neighborDistances, 1e-6)
				nearestNeighbors = nearestNeighbors + self.predictionHorizon

				minDistances = MinAxis1(neighborDistances)
				weights = ComputeWeights(neighborDistances, minDistances)
				weightSum = SumAxis1(weights)
				select = train_y_tensor[nearestNeighbors]
				predictions = ComputePredictions(weights, select, weightSum)

				# calculate performances
				perfs = torch.zeros(batch_size, device = self.device, dtype = self.dtype)
				RowwiseCorrelation(test_y_tensor, predictions, perfs)

				# Convert to list of tuples
				perfs_numpy = perfs.cpu().numpy()
				batch_results = [(var, perfs_numpy[j]) for j, var in enumerate(batch_vars)]
				metric_results.extend(batch_results)

			metric_results.sort(key=lambda x: x[1] if not numpy.isnan(x[1]) else -numpy.inf, reverse=True)

			# Apply correlation threshold filtering
			if self.MinPredictionThreshold > 0:
				original_count = len(metric_results)
				metric_results = [(var, score) for var, score in metric_results if not numpy.isnan(score) and score >= self.MinPredictionThreshold]
				if self.verbose and len(metric_results) < original_count:
					print(f"Filtered {original_count - len(metric_results)} candidates below correlation threshold {self.MinPredictionThreshold}")

			# Flatten results and sort
			# # NOTE: there's nothing about aborting if performance doesn't increase
			# metric_results = [item for sublist in batch_results for item in sublist]
			# metric_results.sort(key = lambda x: x[1] if x[1] is not None else -numpy.inf, reverse = True)

			r = numpy.array(metric_results) if len(metric_results) > 0 else numpy.array([]).reshape(0, 2)
			if len(r) > 0:
				self.rankings_[i, r[:, 0].astype(int)] = r[:, 1]

			best_var = None
			best_score = None

			# Pick top scoring candidate (convergence already checked if convergent=True)
			if metric_results and not numpy.isnan(metric_results[0][1]):
				best_var = metric_results[0][0]
				best_score = metric_results[0][1]

			# Add best variable if found
			if best_var is not None:
				self.selectedVariables.append(best_var)
				remaining_variables.remove(best_var)
				self.accuracy.append(best_score)

				# calc distance matrix update
				train = trainData_tensor[:, best_var]
				test = testData_tensor[:, best_var]
				distances = (train.unsqueeze(1) - test.unsqueeze(0)) ** 2
				current_best_distance_matrix += distances
				progressBar.update(1)
			else:
				# No more valid candidates
				break

		# Time delay analysis
		if self.TimeDelay > 0:
			if self.verbose:
				print(f"Starting time delay analysis with max delay {self.TimeDelay}")

			self.timeDelayResults = []
			best_accuracy = max(self.accuracy) if len(self.accuracy) > 0 else 0

			for var in self.selectedVariables:
				for delay in range(1, self.TimeDelay + 1):
					# Create time-delayed version by shifting the column
					delayed_data = numpy.roll(self.data[:, var], delay)
					# Zero out the first delay values to avoid wrap-around
					delayed_data[:delay] = numpy.nan

					# Temporarily add delayed column to data
					augmented_data = numpy.column_stack([self.data, delayed_data])
					delayed_col_idx = augmented_data.shape[1] - 1

					# Evaluate with delayed variable added
					test_variables = self.selectedVariables + [delayed_col_idx]

					# Save original data and restore after
					original_data = self.data
					self.data = augmented_data

					try:
						result = self._run_edm(test_variables)
						score = self._compute_performance(result)

						improvement = score - best_accuracy
						self.timeDelayResults.append((var, delay, improvement, score))

						if self.verbose:
							print(f"Variable {var} with delay {delay}: score={score:.4f}, improvement={improvement:.4f}")

					except Exception as e:
						if self.verbose:
							print(f"Warning: Time delay evaluation failed for var {var}, delay {delay}: {e}")
					finally:
						self.data = original_data

		self.current_best_distance_matrix = current_best_distance_matrix.cpu().numpy()

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
		#  2. each Simplex/SMap object redoes the entire validation, embedding, and index building
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
		# distance matrix
		# the new one to be added is always the first one in the list
		var = variables[0]
		train = self.trainData[:, var]
		test = self.testData[:, var]
		distances = numpy.subtract.outer(train, test)
		distances **= 2
		if self.current_best_distance_matrix is not None:
			distances += self.current_best_distance_matrix

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
			smap.knnThreads = 1
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
			simplex.EmbedData()
			simplex.RemoveNan()
			neighborFinder = PairwiseDistanceNeighborFinder(None)
			neighborFinder.distanceMatrix = distances
			neighborFinder.numNeighbors = simplex.knn_
			knn_distances, knn_neighbors = neighborFinder.requery()
			simplex.knn_distances, simplex.knn_neighbors = simplex.MapKNNIndicesToData(knn_neighbors, knn_distances)
			simplex.Project()
			simplex.FormatProjection()
			res = SimplexResult(projection = simplex.Projection,
								embedDimensions = 0,
								predictionHorizon = 0)
			return res

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

	def _filter_convergent_variables(self, candidate_columns: List[int]) -> List[int]:
		"""
		Filter candidate variables to only include convergent ones using BatchedCCM.

		:param candidate_columns: Column indices to check for convergence
		:type candidate_columns: List[int]
		:return: Tuple of convergent column indices and their CCM slopes
		:rtype: Tuple[List[int], List[float]]
		"""
		from .CCM_batch import BatchedCCM
		from sklearn.linear_model import LinearRegression

		if len(candidate_columns) == 0:
			return []

		train_size = len(self.data) if self.train is None else self.train[1] - self.train[0]
		lib_start, lib_stop, lib_increment = self.CCMLibrarySizes
		lib_sizes = list(range(lib_start, min(lib_stop + 1, train_size), lib_increment))

		if len(lib_sizes) < 2:
			return candidate_columns

		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float)
		lib_sizes_normalized = (lib_sizes_normalized - lib_sizes_normalized.min()) / (lib_sizes_normalized.max() - lib_sizes_normalized.min())

		X = self.data[:, candidate_columns]
		Y = self.data[:, self.target]

		batchedCCM = BatchedCCM(
			X = X,
			Y = Y,
			trainSizes = lib_sizes,
			sample = self.CCMSampleSize,
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn if self.knn > 0 else self.embedDimensions + 1,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			seed = None,
			embedded = self.embedded,
			validLib = self.validLib,
			includeData = False,
			ignoreNan = self.ignoreNan,
			includeReverse = False,
			device = self.device
		)

		try:
			result = batchedCCM.Run()
			forward_correlations = result.forward_correlations

			convergent_vars = []
			convergent_slopes = []

			lr = LinearRegression()
			for i, col in enumerate(candidate_columns):
				corr_values = forward_correlations[:, i]
				lr.fit(lib_sizes_normalized.reshape(-1, 1), corr_values)
				slope = lr.coef_[0]

				if slope > self.CCMConvergenceThreshold:
					convergent_vars.append(col)

			return convergent_vars

		except Exception as e:
			return candidate_columns

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
		from .CCM import CCM
		from sklearn.linear_model import LinearRegression
		from scipy.signal import argrelextrema

		if self.embedDimensions > 0:
			best_e = self.embedDimensions
		elif column not in self.optimalEmbeddingDimensions:
		# Determine optimal E for this column if not cached
			e_results = FindOptimalEmbeddingDimensionality(
				self.data,
				[column],
				self.target,
				self.maxD,
				train = self.train,
				test = self.test,
				predictionHorizon = self.predictionHorizon,
				noTime = self.noTime
			)

			# Apply firstEMax logic
			correlations = e_results[:, 1]

			if self.FirstEMax:
				# Find local maxima
				local_max_indices = argrelextrema(correlations, numpy.greater)[0]
				if len(local_max_indices) > 0:
					best_e_idx = local_max_indices[0]
				else:
					best_e_idx = len(correlations) - 1
			else:
				# Use global maximum
				best_e_idx = numpy.argmax(correlations)

			best_e = int(e_results[best_e_idx, 0])
			best_e_correlation = correlations[best_e_idx]

			# Check if correlation meets minimum threshold
			if best_e_correlation < self.EmbedDimCorrelationMin:
				return (False, 0.0)

			self.optimalEmbeddingDimensions[column] = best_e
		else:
			best_e = self.optimalEmbeddingDimensions[column]

		# Compute library sizes for CCM
		train_size = len(self.data) if self.train is None else self.train[1] - self.train[0]
		lib_start, lib_stop, lib_increment = self.CCMLibrarySizes
		lib_sizes = list(range(lib_start, min(lib_stop + 1, train_size), lib_increment))

		if len(lib_sizes) < 2:
			if self.verbose:
				print(f"Warning: Not enough library sizes for CCM convergence check on column {column}")
			return (True, 0.5)

		# Normalize library sizes to [0, 1] for slope calculation
		lib_sizes_normalized = numpy.array(lib_sizes, dtype = float)
		lib_sizes_normalized = (lib_sizes_normalized - lib_sizes_normalized.min()) / (lib_sizes_normalized.max() - lib_sizes_normalized.min())

		# Run CCM
		ccm = CCM(
			data = self.data,
			columns = [column],
			target = [self.target],
			trainSizes = self.CCMLibrarySizes,
			sample = self.CCMSampleSize,
			embedDimensions = best_e,
			predictionHorizon = self.predictionHorizon,
			knn = self.knn if self.knn > 0 else best_e + 1,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			noTime = self.noTime,
			ignoreNan = self.ignoreNan,
			verbose = False
		)
		ccm.sequential = True

		try:
			ccm_result = ccm.Run()

			# Extract forward correlation values (column 1 of libMeans)
			forward_correlations = ccm_result.libMeans[:, 1]

			# Fit linear regression to check convergence slope
			lr = LinearRegression()
			lr.fit(lib_sizes_normalized.reshape(-1, 1), forward_correlations)
			slope = lr.coef_[0]

			if self.verbose:
				print(f"Column {column}: CCM slope = {slope:.4f}, threshold = {self.CCMConvergenceThreshold}")

			return (slope > self.CCMConvergenceThreshold, slope)

		except Exception as e:
			if self.verbose:
				print(f"Warning: CCM failed for column {column}: {e}")
			return (False, 0.0)

	def _final_prediction(self) -> numpy.ndarray:
		"""Run final prediction with selected features.

		Returns
		-------
		numpy.ndarray
			Final prediction array [Time, Observations, Predictions]
		"""
		result = self._run_edm(self.selectedVariables)
		return result.projection
