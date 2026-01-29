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

from ._MDE import ElementwisePairwiseDistance, ColumnwiseCorrelation, \
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
				 stdThreshold: float = 1e-3):
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

		if torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.dtype = torch.float16 if use_half_precision else self.dtype

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
			rankings = self.rankings_
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
				ColumnwiseCorrelation(test_y_tensor, predictions, perfs)

				# Convert to list of tuples
				perfs_numpy = perfs.cpu().numpy()
				batch_results = [(var, perfs_numpy[j]) for j, var in enumerate(batch_vars)]
				metric_results.extend(batch_results)

			metric_results.sort(key=lambda x: x[1] if not numpy.isnan(x[1]) else -numpy.inf, reverse=True)

			# Flatten results and sort
			# # NOTE: there's nothing about aborting if performance doesn't increase
			# metric_results = [item for sublist in batch_results for item in sublist]
			# metric_results.sort(key = lambda x: x[1] if x[1] is not None else -numpy.inf, reverse = True)

			r = numpy.array(metric_results)
			self.rankings_[i, r[:, 0].astype(int)] = r[:, 1]

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

				# calc distance matrix update
				train = trainData_tensor[:, best_var]
				test = testData_tensor[:, best_var]
				distances = (train.unsqueeze(1) - test.unsqueeze(0)) ** 2
				current_best_distance_matrix += distances
				progressBar.update(1)
			else:
				# No more valid candidates
				break

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
