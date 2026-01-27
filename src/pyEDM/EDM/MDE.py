"""Multivariate Delay Embedding (MDE) for pyEDM.

This module provides classes for multivariate feature selection using
Empirical Dynamic Modeling methods. The MDE class performs iterative
feature selection by evaluating combinations of features using Simplex
or S-Map predictions with parallel processing.
"""

from typing import List, Tuple

import numpy
from tqdm import tqdm as ProgressBar
from joblib import Parallel, delayed, cpu_count

from .NeighborFinder import PairwiseDistanceNeighborFinder
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
		self.nThreads = nThreads
		if self.nThreads < 1:
			self.nThreads = cpu_count()
		self.stdThreshold = stdThreshold

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
		self.trainData = dummy.Embedding[dummy.trainIndices, :]
		self.testData = dummy.Embedding[dummy.testIndices, :]

		initial_result = self._run_edm(self.columns if self.columns is not None else [self.target])
		score = self._compute_performance(initial_result)
		self.accuracy.append(score)
		
		all_columns = list(range(self.data.shape[1] - 1)) # ignore the Y var, which is the last column
		# ignore all variables with stdev less than threshold
		excluded = numpy.argwhere(numpy.std(self.data, axis = 0) < self.stdThreshold).squeeze().tolist()
		if not self.noTime: # time is first column if true and we exclude that
			excluded.append(0)
		if not self.include_target:
			excluded.append(self.target)
		excluded += self.selectedVariables

		remaining_variables = [c for c in all_columns if c not in excluded]

		# make batch size smaller if we have lots of threads
		jobsPerThread = int(len(remaining_variables) / self.nThreads)
		if jobsPerThread < self.batch_size:
			self.batch_size = jobsPerThread


		# TODO: This can be much more optimal
		# - the current thing uses kdtree, which gets suboptimal rapidly with increasing numbers of
		#   selected variables, the time is like O(n) to O(n log n) or something with n dims, not to mention
		#	the overhead of building the trees each time
		# - an optimal way is to
		#		- use squared Euclidean distances and argsort
		#		- precompute distance matrices based on each feature
		#		- this would be basically a bunch of outer products
		#			- result = A[:, np.newaxis, :] - B[np.newaxis, :, :]
		#			- then each var will be the last dimension in result
		#		- and for each new search, add just add the distances based on indexing
		#		- this is O(1) because we calculate the same same number of numbers per iteration
		#		- in fact this should get slightly faster with each iteration because we check fewer variables
		# - We can further speed things up by breaking the prediction logic out of the Simplex (and SMap) classes
		# 	because they are matrix maths. We would probably use numexpr or numba to speed up the matrix additions
		#	and neighbor findings, and then do the predictions all in one go

		# as an intermediate step we can compute all the neighbors, and then use the joblib threading to spread
		# them across threads, and manually fix all the simplex/smap objects with the neighbors and only have them
		# predict and score

		"""
		Here's a outline for matrix-izing basically all of this.
		Train X: M samples x F features
		Train Y: M samples x 1
		Test X: N samples x F features
		Test Y: N samples x 1
		
		[numba parallel]
		for each feature k we calculate the pairwise distance between the train and test samples to create
		distances: M x N x F
		
		best distance: M x N distance matrix : this is accumulated over iterations
		
		on each iteration we have:
		[numba parallel]
		candidate distances M x N x F = distances + best distance (broadcast add over K)
		find k nearest neighbors in M for each N in each F
		indices: k x N x F
		
		[numba parallel]
		weight matrix: M x N x F -- a sparse matrix in which only the k entries for each N x F slice are non-zero
			these weights are determined by the simplex or smap algorithms.
			these weights the best way to use train M samples to predict the test N samples
		
		[possibly numba parallel]
		vector-tensor multiply Train Y with weight matrix to produce predictions N x F
		for numpy.matmul the notes say:
			If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
		predictions: N x F : Y_hat if we include each F into the picture
		
		broadcast correlate Y with predictions to produce
		performance: 1 x F 
		find best feature f		
		store candidate distance [:, :, best feature] into best distances
		
		repeat until satisfied
		"""



		# Iteratively add variables up to maxD
		progressBar = ProgressBar(total = self.maxD, desc = 'Selecting variables', leave = False)

		# rankings is a numpy array because it's apparently more multithreading friendly
		# than just storing the lists that come out?
		self.rankings_ = numpy.zeros([self.maxD, self.data.shape[1]])

		self.current_best_distance_matrix = None

		for i in range(self.maxD):
			# Break up remaining columns into parallel-friendly batches
			batches = [
				remaining_variables[i:(i + self.batch_size)]
				for i in range(0, len(remaining_variables), self.batch_size)
			]

			# Evaluate correlation/MAE for each possible addition in parallel
			batch_results = Parallel(n_jobs = self.nThreads)(
				delayed(self._evaluate_batch)(batch) for batch in batches
			)

			# Flatten results and sort
			# NOTE: there's nothing about aborting if performance doesn't increase
			metric_results = [item for sublist in batch_results for item in sublist]
			metric_results.sort(key = lambda x: x[1] if x[1] is not None else -numpy.inf, reverse = True)

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
				train = self.trainData[:, best_var]
				test = self.testData[:, best_var]
				distances = numpy.subtract.outer(train, test)
				distances **= 2
				if self.current_best_distance_matrix is None:
					self.current_best_distance_matrix = distances
				else:
					self.current_best_distance_matrix += distances
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
