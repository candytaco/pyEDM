import numpy
import torch
from tqdm import tqdm as ProgressBar
from numpy import zeros, mean
from numpy.random import default_rng

from pyEDM.EDM.Simplex import Simplex
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
				 includeReverse = False,
				 device = 'cuda',
				 batchSize = 10000,
				 useHalfPrecision = False):
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
		:param device: 				Device for torch tensors ('cpu', 'cuda', or torch.device object)
		:param batchSize: 			Number of variables to process per batch to limit VRAM usage
		:param useHalfPrecision: 	Use float16 instead of float32 to save VRAM
		"""

		self.name = 'BatchedCCM'
		self.X = X
		self.Y = Y[:, None] if Y.ndim == 1 else Y
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
		self.batchSize = batchSize

		self.trainSizes = trainSizes if trainSizes is not None else []
		self.sample = sample
		self.seed = seed
		self.includeData = includeData

		self.device = torch.device(device) if isinstance(device, str) else device
		self.dtype = torch.float16 if useHalfPrecision else torch.float32

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

		FwdResult = self.CrossMap(self.X, self.Y)

		self.libMeansFwd = zeros([len(self.trainSizes), 1 + self.numVariables])
		for i, size in enumerate(self.trainSizes):
			self.libMeansFwd[i, 0] = size
			for m in range(self.numVariables):
				self.libMeansFwd[i, 1 + m] = FwdResult['libcorrelation'][size][m]

		if self.includeData:
			self.PredictStatsFwd = FwdResult['predictStats']

		if self.includeReverse:
			RevResult = self.CrossMap(self.Y, self.X)

			self.libMeansRev = zeros([len(self.trainSizes), 1 + self.numVariables])
			for i, size in enumerate(self.trainSizes):
				self.libMeansRev[i, 0] = size
				for m in range(self.numVariables):
					self.libMeansRev[i, 1 + m] = RevResult['libcorrelation'][size][m]

			if self.includeData:
				self.PredictStatsRev = RevResult['predictStats']

	def CrossMap(self, X, Y):
		from .Embed import Embed

		RNG = default_rng(self.seed)

		dummy = Simplex(
			data = X,
			columns = numpy.arange(X.shape[1]).tolist(),
			target = 0,
			train = self.train,
			test = self.test,
			embedDimensions = self.embedDimensions,
			predictionHorizon = 0,
			knn = self.knn,
			step = self.step,
			exclusionRadius = self.exclusionRadius,
			embedded = self.embedded,
			validLib = self.validLib,
			noTime = True,
			ignoreNan = self.ignoreNan,
			verbose = False
		)
		dummy.EmbedData()

		numPredictors = X.shape[1]

		embeddings = []
		for varIndex in range(numPredictors):
			if self.embedded:
				embedding = X[:, varIndex].reshape(-1, 1)
			else:
				embedding = Embed(data = X,
								  columns = [varIndex],
								  embeddingDimensions = self.embedDimensions,
								  stepSize = self.step,
								  includeTime = False)
			embeddings.append(embedding)

		libraryIndices = dummy.trainIndices.copy()
		N_libraryIndices = len(libraryIndices)
		targetVector = Y[libraryIndices, 0]

		libcorrelationMap = {libSize: zeros([self.sample, numPredictors]) for libSize in self.trainSizes}
		libStatMap = {}
		if self.includeData:
			for libSize in self.trainSizes:
				libStatMap[libSize] = [[None] * self.sample for _ in range(numPredictors)]

		target = torch.tensor(targetVector, dtype = self.dtype, device = self.device)

		d = torch.zeros([embeddings[0].shape[1], N_libraryIndices, N_libraryIndices],
						dtype = self.dtype, device = self.device)
		fullDistances = torch.zeros([self.batchSize, N_libraryIndices, N_libraryIndices],
									dtype = self.dtype, device = self.device)

		for batchStart in ProgressBar(range(0, numPredictors, self.batchSize), desc = 'Variable batch'):
			batchEnd = min(batchStart + self.batchSize, numPredictors)
			batchEmbeddings = embeddings[batchStart:batchEnd]
			batchNumPredictors = len(batchEmbeddings)

			trainEmbeddings = torch.tensor(numpy.array([embedding[libraryIndices, :] for embedding in batchEmbeddings]), dtype = self.dtype, device = self.device)
			for i in range(batchNumPredictors):
				ElementwisePairwiseDistance(trainEmbeddings[i, :, :], trainEmbeddings[i, :, :], d)
				fullDistances[i, :, :] = torch.sum(d, dim = 0)
			fullDistances[:batchNumPredictors, :, :] = torch.sqrt(fullDistances[:batchNumPredictors, :, :])
			perfs_ = torch.zeros(batchNumPredictors, dtype = self.dtype, device = self.device)
			maskedDistances = torch.zeros([batchNumPredictors, N_libraryIndices, N_libraryIndices], dtype = self.dtype,
									device = self.device)

			distances = torch.zeros([batchNumPredictors, self.knn, N_libraryIndices], dtype = self.dtype, device = self.device)
			neighbors = torch.zeros([batchNumPredictors, self.knn, N_libraryIndices], dtype = torch.long, device = self.device)
			minDistances = torch.zeros([batchNumPredictors, N_libraryIndices], dtype = self.dtype, device = self.device)
			weights = torch.zeros([batchNumPredictors, self.knn, N_libraryIndices], dtype = self.dtype, device = self.device)
			weightSum = torch.zeros([batchNumPredictors, N_libraryIndices], dtype = self.dtype, device = self.device)
			select = torch.zeros([batchNumPredictors, self.knn, N_libraryIndices], dtype = self.dtype, device = self.device)
			predictions = torch.zeros([batchNumPredictors, N_libraryIndices], dtype = self.dtype, device = self.device)
			mask = torch.ones(N_libraryIndices, dtype = torch.bool, device = self.device)

			if self.exclusionRadius == 0:
				diagIndices = torch.arange(fullDistances.shape[1], device = self.device)
				for i in range(batchNumPredictors):
					fullDistances[i, diagIndices, diagIndices] = float('inf')

			for libSize in ProgressBar(self.trainSizes, desc = 'CCM library sizes', leave = False):
				for s in ProgressBar(range(self.sample), desc = 'Repeats', leave = False):
					subsampleIndices = RNG.choice(N_libraryIndices,
												  size = min(libSize, N_libraryIndices),
												  replace = False)

					maskedDistances.copy_(fullDistances[:batchNumPredictors, :, :])
					mask.fill_(True)
					mask[subsampleIndices] = False
					maskedDistances[:, mask, :] = float('inf')

					torch.topk(maskedDistances, self.knn, dim = 1, largest = False, out = (distances, neighbors))
					FloorArray(distances, 1e-6)

					minDistances[:] = MinAxis1(distances)
					weights[:] = ComputeWeights(distances, minDistances)
					weightSum[:] = SumAxis1(weights)
					select.copy_(target[neighbors])
					predictions[:] = ComputePredictions(weights, select, weightSum)

					RowwiseCorrelation(target, predictions, perfs_)
					libcorrelationMap[libSize][s, batchStart:batchEnd] = perfs_.cpu().numpy()

			del trainEmbeddings
			del maskedDistances
			del perfs_
			del distances
			del neighbors
			del minDistances
			del weights
			del weightSum
			del select
			del predictions
			del mask
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		for libSize in self.trainSizes:
			libcorrelationMap[libSize] = mean(libcorrelationMap[libSize], axis = 0)

		if self.includeData:
			return {'libcorrelation': libcorrelationMap, 'predictStats': libStatMap}
		else:
			return {'libcorrelation': libcorrelationMap}