import numpy
import torch
from numpy import zeros, array, mean
from numpy.random import default_rng

from pyEDM.EDM._MDE import ElementwisePairwiseDistance, FloorArray, MinAxis1, ComputeWeights, SumAxis1, \
	ComputePredictions, RowwiseCorrelation


class BatchedCCM:
	"""
	BatchedCCM class: Vectorized CCM where M predictor variables predict the same target simultaneously.
	Uses torch batching for parallelization across variables instead of Python multiprocessing.
	Only supports pairwise distance mode (no KDTree).
	"""

	def __init__(self,
				 X,
				 Y,
				 trainSizes = None,
				 sample = 0,
				 embedDimensions = 0,
				 predictionHorizon = 1,
				 knn = 0,
				 step = -1,
				 exclusionRadius = 0,
				 seed = None,
				 embedded = False,
				 validLib = None,
				 includeData = False,
				 ignoreNan = True,
				 includeReverse = False):
		"""
		Initialize BatchedCCM.

		:param X: 					2D numpy array of predictor variables (N_timepoints, M_variables)
		:param Y: 					1D or 2D numpy array of target variable (N_timepoints,) or (N_timepoints, 1)
		:param trainSizes: 			Library sizes to evaluate [start, stop, increment]
		:param sample: 				Number of random samples at each library size
		:param embedDimensions: 	Embedding dimension (E)
		:param predictionHorizon: 	Prediction time horizon (Tp)
		:param knn: 				Number of nearest neighbors
		:param step: 				Time delay step size (tau)
		:param exclusionRadius: 	Temporal exclusion radius for neighbors
		:param seed: 				Random seed for reproducible sampling
		:param embedded: 			Whether data is already embedded
		:param validLib:			Boolean mask for valid library points
		:param includeData: 		Whether to include detailed prediction statistics
		:param ignoreNan: 			Remove NaN values from embedding
		:param includeReverse: 		Whether to compute reverse direction (target -> columns)
		"""

		self.name = 'BatchedCCM'
		self.X = X
		self.Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
		self.numVariables = X.shape[1]
		self.embedDimensions = embedDimensions
		self.predictionHorizon = predictionHorizon
		self.knn = knn
		self.step = step
		self.exclusionRadius = exclusionRadius
		self.embedded = embedded
		self.validLib = validLib if validLib is not None else []
		self.ignoreNan = ignoreNan
		self.includeReverse = includeReverse

		self.trainSizes = trainSizes if trainSizes is not None else []
		self.sample = sample
		self.seed = seed
		self.includeData = includeData

		self.train = self.test = [1, self.X.shape[0]]

		self.libMeansFwd = None
		self.libMeansRev = None
		self.PredictStatsFwd = None
		self.PredictStatsRev = None

	def Run(self):
		"""
		Execute BatchedCCM and return BatchedCCMResult.
		"""
		self.Project()

		from .Results import BatchedCCMResult
		return BatchedCCMResult(
			libMeansFwd = self.libMeansFwd,
			libMeansRev = self.libMeansRev,
			embedDimensions = self.embedDimensions,
			predictionHorizon = self.predictionHorizon,
			predictStatsFwd = self.PredictStatsFwd if self.includeData else None,
			predictStatsRev = self.PredictStatsRev if self.includeData else None
		)

	def Project(self):
		"""
		Execute batched cross-mapping for all predictor variables.
		"""

		FwdResult = self.BatchedCrossMap(reverse = False)

		self.libMeansFwd = zeros([len(self.trainSizes), 1 + self.numVariables])
		for i, size in enumerate(self.trainSizes):
			self.libMeansFwd[i, 0] = size
			for m in range(self.numVariables):
				self.libMeansFwd[i, 1 + m] = FwdResult['libcorrelation'][size][m]

		if self.includeData:
			self.PredictStatsFwd = FwdResult['predictStats']
		#
		# if self.includeReverse:
		# 	RevResult = self.BatchedCrossMap(reverse = True)
		#
		# 	self.libMeansRev = zeros([len(self.trainSizes), 1 + self.numVariables])
		# 	for i, size in enumerate(self.trainSizes):
		# 		self.libMeansRev[i, 0] = size
		# 		for m in range(self.numVariables):
		# 			self.libMeansRev[i, 1 + m] = RevResult['libcorrelation'][size][m]
		#
		# 	if self.includeData:
		# 		self.PredictStatsRev = RevResult['predictStats']

	def BatchedCrossMap(self, reverse: bool = False):
		"""
		Perform batched cross-mapping across M predictor variables.
		"""

		from .Embed import Embed

		RNG = default_rng(self.seed)

		if not reverse:
			predictorData = self.X
			targetData = self.Y
		else:
			predictorData = self.Y
			targetData = self.X

		trainIndices = array(range(self.train[0] - 1, self.train[1]))
		testIndices = array(range(self.test[0] - 1, self.test[1]))

		numPredictors = predictorData.shape[1]

		embeddings = []
		for varIndex in range(numPredictors):
			if self.embedded:
				embedding = predictorData[:, varIndex].reshape(-1, 1)
			else:
				embedding = Embed(data = predictorData,
								  columns = [varIndex],
								  embeddingDimensions = self.embedDimensions,
								  stepSize = self.step,
								  includeTime = False)
			embeddings.append(embedding)

		targetVector = targetData[:, 0]

		libraryIndices = trainIndices.copy()
		N_libraryIndices = len(libraryIndices)

		libcorrelationMap = {}
		libStatMap = {}

		numOutputs = numPredictors

		trainEmbeddings = torch.tensor([embedding[libraryIndices, :] for embedding in embeddings])
		testEmbeddings = torch.tensor([embedding[testIndices, :] for embedding in embeddings])

		fullDistances = torch.zeros([numPredictors, trainEmbeddings.shape[1], trainEmbeddings.shape[1]])
		for i in range(numPredictors):
			# these are the distance matrices summed across all embedding dimensions
			# for each predictor
			d = torch.zeros([trainEmbeddings.shape[2], trainEmbeddings.shape[1], trainEmbeddings.shape[1]])
			ElementwisePairwiseDistance(trainEmbeddings[i, :, :], testEmbeddings[i, :, :], d)
			fullDistances[i, :, :] = torch.sum(d, dim = 0)

		for libSize in self.trainSizes:
			correlations = zeros([self.sample, numOutputs])
			if self.includeData:
				predictStats = [[None] * self.sample for _ in range(numOutputs)]

			for s in range(self.sample):
				subsampleIndices = RNG.choice(numpy.arange(fullDistances.shape[1]),
											  size = min(libSize, N_libraryIndices),
											  replace = False)

				distances = torch.zeros_like(fullDistances)
				distances.copy_(fullDistances)
				mask = numpy.ones(distances.shape[1], dtype = bool)
				mask[subsampleIndices] = False
				distances[:, mask, :] = numpy.inf

				neighbors = torch.topk(distances, self.knn, dim = 1, largest = False)[1]
				distances = torch.gather(distances, 1, neighbors)
				FloorArray(distances, 1e-6)

				minDistances = MinAxis1(distances)
				weights = ComputeWeights(distances, minDistances)
				weightSum = SumAxis1(weights)
				select = targetVector[neighbors]
				predictions = ComputePredictions(weights, select, weightSum)

				# calculate performances
				perfs_ = torch.zeros(numPredictors)
				RowwiseCorrelation(torch.tensor(targetVector), predictions, perfs_)
				correlations[s, :] = perfs_.cpu().numpy()

			meanCorrelations = mean(correlations, axis = 0)
			libcorrelationMap[libSize] = meanCorrelations

			if self.includeData:
				libStatMap[libSize] = predictStats

		if self.includeData:
			return {'libcorrelation': libcorrelationMap, 'predictStats': libStatMap}
		else:
			return {'libcorrelation': libcorrelationMap}
