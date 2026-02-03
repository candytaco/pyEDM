"""
Functional programming interface to Empirical Dynamic Modeling (EDM) pyEDM.
While the underlying classes have been refactored, these functions should
return roughly the same data structures returned by the original pyEDM functions
"""

from itertools import repeat
# python modules
from multiprocessing import get_context
from typing import Any, Dict, List, Tuple, Union
import numpy
import torch

# local modules
from .EDM import PoolFunc
from .Utils import IsNonStringIterable, ComputeError
from .EDM.CCM import CCM
from .EDM.Multiview import Multiview
from .EDM.SMap import SMap
from .EDM.Simplex import Simplex


def FitSimplex(data: numpy.ndarray,
			   columns: List[int] = None,
			   target: int = None,
			   train: Tuple[int, int] = None,
			   test: Tuple[int, int] = None,
			   embedDimensions: int = 0,
			   predictionHorizon: int = 1,
			   knn: int = 0,
			   step: int = -1,
			   exclusionRadius: float = 0,
			   embedded: bool = False,
			   validLib: List = [],
			   noTime: bool = False,
			   generateSteps: int = 0,
			   generateConcat: bool = False,
			   verbose: bool = False,
			   ignoreNan: bool = True,
			   returnObject: bool = False) -> Union[numpy.ndarray, Simplex]:
	"""
	Simplex prediction.

	:param data: 			2D numpy array where column 0 is time
	:param columns: 		Column indices to use for embedding (defaults to all except time)
	:param target: 			Target column index (defaults to column 1)
	:param train: 			Train indices [start, end]
	:param test: 			Test indices [start, end]
	:param embedDimensions: Embedding dimension
	:param predictionHorizon: Prediction horizon
	:param knn: 			Number of nearest neighbors
	:param step: 			Step size for embedding
	:param exclusionRadius: Exclusion radius
	:param embedded: 		Whether data is already embedded
	:param validLib: 		Valid library indices
	:param noTime: 			Whether to exclude time column
	:param generateSteps: 	Number of generation steps
	:param generateConcat: 	Whether to concatenate generated predictions
	:param verbose: 		Print diagnostic messages
	:param ignoreNan: 		Whether to ignore NaN values
	:param returnObject: 	Whether to return Simplex object instead of projection
	:return: Prediction projection array or Simplex object
	"""

	# Instantiate SimplexClass object
	S = Simplex(data = data,
				columns = columns,
				target = target,
				train = train,
				test = test,
				embedDimensions = embedDimensions,
				predictionHorizon = predictionHorizon,
				knn = knn,
				step = step,
				exclusionRadius = exclusionRadius,
				embedded = embedded,
				validLib = validLib,
				noTime = noTime,
				generateSteps = generateSteps,
				generateConcat = generateConcat,
				ignoreNan = ignoreNan,
				verbose = verbose)

	if generateSteps:
		result = S.Generate()
	else:
		result = S.Run()

	if returnObject:
		return S
	else:
		return result.projection


def FitSMap(data: numpy.ndarray,
			columns: List[int] = None,
			target: int = None,
			train: Tuple[int, int] = None,
			test: Tuple[int, int] = None,
			embedDimensions: int = 0,
			predictionHorizon: int = 1,
			knn: int = 0,
			step: int = -1,
			theta: float = 0,
			exclusionRadius: float = 0,
			solver: Any = None,
			embedded: bool = False,
			validLib: List = [],
			noTime: bool = False,
			generateSteps: int = 0,
			generateConcat: bool = False,
			ignoreNan: bool = True,
			verbose: bool = False,
			returnObject: bool = False) -> Union[Dict[str, Any], SMap]:
	"""
	S-Map prediction.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use for embedding (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param embedDimensions: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param theta: 				Localization parameter
	:param exclusionRadius: 	Exclusion radius
	:param solver: 				solver
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param noTime: 				Whether to exclude time column
	:param generateSteps: 		Number of generation steps
	:param generateConcat: 		Whether to concatenate generated predictions
	:param verbose: 			Print diagnostic messages
	:param ignoreNan: 			Whether to ignore NaN values
	:param returnObject: 		Whether to return SMap object instead of prediction dict
	:return: Dictionary with predictions, coefficients, and singular values or SMap object
	"""

	# Instantiate SMapClass object
	S = SMap(data = data,
			 columns = columns,
			 target = target,
			 train = train,
			 test = test,
			 embedDimensions = embedDimensions,
			 predictionHorizon = predictionHorizon,
			 knn = knn,
			 step = step,
			 theta = theta,
			 exclusionRadius = exclusionRadius,
			 solver = solver,
			 embedded = embedded,
			 validLib = validLib,
			 noTime = noTime,
			 generateSteps = generateSteps,
			 generateConcat = generateConcat,
			 ignoreNan = ignoreNan,
			 verbose = verbose)

	if generateSteps:
		result = S.Generate()
	else:
		result = S.Run()

	if returnObject:
		return S
	else:
		SMapDict = {'predictions': result.projection,
					'coefficients': S.Coefficients,
					'singularValues': S.SingularValues}
		return SMapDict


# TODO: mpMethod and sequential are redundant given the execution enum includes a sequential method
def FitCCM(data: numpy.ndarray,
		   columns: List[int] = None,
		   target: int = None,
		   trainSizes: Any = None,
		   sample: int = 0,
		   embedDimensions: int = 0,
		   predictionHorizon: int = 0,
		   knn: int = 0,
		   step: int = -1,
		   exclusionRadius: float = 0,
		   seed: Any = None,
		   embedded: bool = False,
		   validLib: List = [],
		   includeData: bool = False,
		   noTime: bool = False,
		   ignoreNan: bool = True,
		   mpMethod: Any = None,
		   sequential: bool = False,
		   verbose: bool = False,
		   returnObject: bool = False,
		   kdTree: bool = False) -> Union[Dict[str, Any], CCM]:
	"""
	Convergent Cross Mapping.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param trainSizes: 			Library sizes to evaluate
	:param sample: 				Sample size for each library
	:param embedDimensions: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param seed: 				Random seed
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param includeData: 		Whether to include detailed prediction statistics
	:param noTime: 				Whether to exclude time column
	:param ignoreNan: 			Whether to ignore NaN values
	:param mpMethod: 			Multiprocessing method
	:param sequential: 			Whether to run sequentially
	:param verbose: 			Print diagnostic messages
	:param returnObject: 		Whether to return CCM object instead of libMeans
	:param kdTree:				Use kdtree for neighbors
	:return: Library means array or CCM object
	"""

	# Instantiate CCMClass object
	C = CCM(data = data,
			columns = columns,
			target = [target],
			trainSizes = trainSizes,
			sample = sample,
			embedDimensions = embedDimensions,
			predictionHorizon = predictionHorizon,
			knn = knn,
			step = step,
			exclusionRadius = exclusionRadius,
			seed = seed,
			embedded = embedded,
			validLib = validLib,
			includeData = includeData,
			noTime = noTime,
			ignoreNan = ignoreNan,
			mpMethod = mpMethod,
			sequential = sequential,
			verbose = verbose,
			kdTree = kdTree)

	# Embedding of Forward & Reverse mapping
	C.FwdMap.EmbedData()
	C.FwdMap.RemoveNan()
	C.RevMap.EmbedData()
	C.RevMap.RemoveNan()

	result = C.Run()

	if returnObject:
		return C
	else:
		if includeData:
			return {'LibMeans': result.libMeans,
					'PredictStats1': result.predictStats1,
					'PredictStats2': result.predictStats2}
		else:
			return result.libMeans


def FitMultiview(data: numpy.ndarray,
				 columns: List[int] = None,
				 target: int = None,
				 train: Tuple[int, int] = None,
				 test: Tuple[int, int] = None,
				 D: int = 0,
				 embedDimensions: int = 1,
				 predictionHorizon: int = 1,
				 knn: int = 0,
				 step: int = -1,
				 multiview: int = 0,
				 exclusionRadius: float = 0,
				 trainLib: bool = True,
				 excludeTarget: bool = False,
				 ignoreNan: bool = True,
				 verbose: bool = False,
				 numProcess: int = 4,
				 mpMethod: Any = None,
				 chunksize: int = 1,
				 returnObject: bool = False) -> Union[Dict[str, Any], Multiview]:
	"""
	Multiview prediction.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param D: 					State-space dimension
	:param embedDimensions: 	Embedding dimension for each variable
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param multiview: 			Multiview parameter
	:param exclusionRadius: 	Exclusion radius
	:param trainLib: 			Whether to use training library
	:param excludeTarget: 		Whether to exclude target from columns
	:param ignoreNan: 			Whether to ignore NaN values
	:param verbose: 			Print diagnostic messages
	:param numProcess: 			Number of processes for multiprocessing
	:param mpMethod: 			Multiprocessing method
	:param chunksize: 			Chunk size for pool.starmap
	:param returnObject: 		Whether to return Multiview object instead of prediction dict
	:return: Dictionary with predictions and view rankings or Multiview object
	"""

	# Instantiate MultiviewClass object
	M = Multiview(data = data,
				  columns = columns,
				  target = target,
				  train = train,
				  test = test,
				  D = D,
				  embedDimensions = embedDimensions,
				  predictionHorizon = predictionHorizon,
				  knn = knn,
				  step = step,
				  multiview = multiview,
				  exclusionRadius = exclusionRadius,
				  trainLib = trainLib,
				  excludeTarget = excludeTarget,
				  ignoreNan = ignoreNan,
				  verbose = verbose,
				  numProcess = numProcess,
				  mpMethod = mpMethod,
				  chunksize = chunksize)

	result = M.Run()

	if returnObject:
		return M
	else:
		return {'Predictions': result.projection, 'View': result.view}


def FindOptimalEmbeddingDimensionality(data: numpy.ndarray,
									   columns: List[int] = None,
									   target: int = None,
									   maxE: int = 10,
									   train: Tuple[int, int] = None,
									   test: Tuple[int, int] = None,
									   predictionHorizon: int = 1,
									   step: int = -1,
									   exclusionRadius: float = 0,
									   embedded: bool = False,
									   validLib: List = [],
									   noTime: bool = False,
									   ignoreNan: bool = True,
									   batched: bool = False) -> numpy.ndarray:
	"""
	Estimate optimal embedding dimension [1:maxE] using GPU-accelerated Simplex.

	When batched=False, each E gets its own proper train/test indices derived
	from that E's embedding. When batched=True, the maxE indices (most
	restrictive NaN filtering) are used for all E values, which enables
	shared distance precomputation but slightly penalizes lower E values
	by excluding a few extra rows.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param maxE: 				Maximum embedding dimension to test
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param predictionHorizon: 	Prediction horizon
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param noTime: 				Whether to exclude time column
	:param ignoreNan: 			Whether to ignore NaN values
	:param batched: 			Use shared maxE indices for all E (faster, slightly less accurate for low E)
	:return: Array with columns [E, correlation]
	"""

	Evals = list(range(1, maxE + 1))

	if batched:
		correlations = _FindOptimalEmbeddingDimensionalityBatched(
			data, columns, target, maxE, Evals, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan)
	else:
		correlations = _FindOptimalEmbeddingDimensionalityIterative(
			data, columns, target, Evals, train, test,
			predictionHorizon, step, exclusionRadius, embedded,
			validLib, noTime, ignoreNan)

	return numpy.column_stack([Evals, correlations])


def _FindOptimalEmbeddingDimensionalityIterative(data, columns, target, Evals,
												  train, test, predictionHorizon,
												  step, exclusionRadius, embedded,
												  validLib, noTime, ignoreNan):
	"""
	Evaluate each E with its own proper train/test indices.
	Each E creates a Simplex and runs GPU-accelerated FindNeighbors/Project.
	"""
	correlations = []

	for E in Evals:
		S = Simplex(data=data, columns=columns, target=target,
					train=train, test=test, embedDimensions=E,
					predictionHorizon=predictionHorizon, knn=0,
					step=step, exclusionRadius=exclusionRadius,
					embedded=embedded, validLib=validLib,
					noTime=noTime, ignoreNan=ignoreNan)

		result = S.Run()
		correlation = ComputeError(result.projection[:, 1], result.projection[:, 2], None)
		correlations.append(correlation)

	return correlations


def _FindOptimalEmbeddingDimensionalityBatched(data, columns, target, maxE, Evals,
												train, test, predictionHorizon,
												step, exclusionRadius, embedded,
												validLib, noTime, ignoreNan):
	"""
	Evaluate all E values using shared maxE indices and precomputed
	cumulative per-column distances on GPU. Uses the most restrictive
	NaN filtering (from maxE) for all E values.
	"""
	# Create a Simplex at maxE to get proper indices, embedding, and target
	S = Simplex(data=data, columns=columns, target=target,
				train=train, test=test, embedDimensions=maxE,
				predictionHorizon=predictionHorizon, knn=0,
				step=step, exclusionRadius=exclusionRadius,
				embedded=embedded, validLib=validLib,
				noTime=noTime, ignoreNan=ignoreNan)

	S.EmbedData()
	S.RemoveNan()

	device = S.device
	dtype = S.dtype

	trainEmbedding = S.Embedding[S.trainIndices, :]
	testEmbedding = S.Embedding[S.testIndices, :]
	nTrain = len(S.trainIndices)
	nTest = len(S.testIndices)
	numEmbeddingColumns = trainEmbedding.shape[1]

	trainTensor = torch.tensor(trainEmbedding, device=device, dtype=dtype)
	testTensor = torch.tensor(testEmbedding, device=device, dtype=dtype)
	targetVector = torch.tensor(S.targetVec.squeeze(), device=device, dtype=dtype)

	# Compute per-column squared pairwise distances: [numCols, nTrain, nTest]
	perColumnDistancesSq = torch.zeros(numEmbeddingColumns, nTrain, nTest, device=device, dtype=dtype)
	for c in range(numEmbeddingColumns):
		diff = trainTensor[:, c].unsqueeze(1) - testTensor[:, c].unsqueeze(0)
		perColumnDistancesSq[c] = diff * diff

	# Cumulative sum gives squared distances for each E
	cumulativeDistancesSq = torch.cumsum(perColumnDistancesSq, dim=0)

	del perColumnDistancesSq, trainTensor, testTensor

	# Build exclusion mask once (same for all E since indices are shared)
	exclusionMask = S._BuildExclusionMask()
	hasMask = exclusionMask.any()
	if hasMask:
		maskTensor = torch.tensor(exclusionMask, device=device, dtype=torch.bool)

	correlations = []

	for E in Evals:
		knn = E + 1

		# For multi-column embeddings, E dimensions use E * len(columns) actual columns
		embeddingColumnsForE = E * len(S.columns) if not embedded else E
		if embeddingColumnsForE > numEmbeddingColumns:
			embeddingColumnsForE = numEmbeddingColumns
		distancesSq = cumulativeDistancesSq[embeddingColumnsForE - 1]

		distances = torch.sqrt(distancesSq)

		if hasMask:
			distances[maskTensor] = float('inf')

		topkDistances, topkIndices = torch.topk(distances, knn, dim=0, largest=False)

		neighborDistances = topkDistances.t()
		neighborIndices = topkIndices.t()

		# Compute weighted predictions
		minDist = neighborDistances[:, 0].clone()
		torch.clamp_min(minDist, 1e-6, out=minDist)
		scaledDistances = neighborDistances / minDist.unsqueeze(1)
		weights = torch.exp(-scaledDistances)
		weightRowSum = torch.sum(weights, dim=1)

		neighborIndicesData = neighborIndices.cpu().numpy()
		neighborIndicesData = S._MapKNNIndicesToLibraryIndices(neighborIndicesData)
		neighborIndicesDataTp = torch.tensor(neighborIndicesData + predictionHorizon, device=device, dtype=torch.long)

		libTargetValues = targetVector[neighborIndicesDataTp]
		predictions = torch.sum(weights * libTargetValues, dim=1) / weightRowSum

		observationIndices = S.testIndices + predictionHorizon
		validObsIndices = observationIndices[observationIndices < len(S.targetVec)]
		observations = S.targetVec[validObsIndices, 0]

		predictionsNumpy = predictions.cpu().numpy()
		nValid = len(validObsIndices)
		correlation = ComputeError(observations[:nValid], predictionsNumpy[:nValid], None)
		correlations.append(correlation)

	del cumulativeDistancesSq
	if hasMask:
		del maskTensor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return correlations


def FindOptimalPredictionHorizon(data: numpy.ndarray,
								 columns: List[int] = None,
								 target: int = None,
								 train: Tuple[int, int] = None,
								 test: Tuple[int, int] = None,
								 maxTp: int = 10,
								 embedDimensions: int = 1,
								 step: int = -1,
								 exclusionRadius: float = 0,
								 embedded: bool = False,
								 validLib: List = [],
								 noTime: bool = False,
								 ignoreNan: bool = True,
								 numProcess: int = 4,
								 mpMethod: Any = None,
								 chunksize: int = 1) -> numpy.ndarray:
	"""
	Estimate optimal prediction interval [1:maxTp].

	:param data: 			2D numpy array where column 0 is time
	:param columns: 		Column indices to use (defaults to all except time)
	:param target: 			Target column index (defaults to column 1)
	:param maxTp: 			Maximum prediction horizon to test
	:param train: 			Train indices [start, end]
	:param test: 			Test indices [start, end]
	:param embedDimensions: Embedding dimension
	:param step: 			Step size for embedding
	:param exclusionRadius: Exclusion radius
	:param embedded: 		Whether data is already embedded
	:param validLib: 		Valid library indices
	:param noTime: 			Whether to exclude time column
	:param ignoreNan: 		Whether to ignore NaN values
	:param numProcess: 		Number of processes for multiprocessing
	:param mpMethod: 		Multiprocessing method
	:param chunksize: 		Chunk size for pool.starmap
	:return: Array with columns [predictionHorizon, correlation]
	"""

	# Setup Pool
	Evals = [predictionHorizon for predictionHorizon in range(1, maxTp + 1)]
	args = {'columns': columns,
			'target': target,
			'train': train,
			'test': test,
			'embedDims': embedDimensions,
			'step': step,
			'exclusionRadius': exclusionRadius,
			'embedded': embedded,
			'validLib': validLib,
			'noTime': noTime,
			'ignoreNan': ignoreNan}

	# Create iterable for Pool.starmap, use repeated copies of data, args
	poolArgs = zip(Evals, repeat(data), repeat(args))

	# Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
	mpContext = get_context(mpMethod)
	with mpContext.Pool(processes = numProcess) as pool:
		correlationList = pool.starmap(PoolFunc.PredictIntervalSimplexFunc, poolArgs,
									   chunksize = chunksize)

	import numpy as np
	result = np.column_stack([Evals, correlationList])

	return result


def FindSMapNeighborhood(data: numpy.ndarray,
						 columns: List[int] = None,
						 target: int = None,
						 theta: Any = None,
						 train: Tuple[int, int] = None,
						 test: Tuple[int, int] = None,
						 embedDimensions: int = 1,
						 predictionHorizon: int = 1,
						 knn: int = 0,
						 step: int = -1,
						 exclusionRadius: float = 0,
						 solver: Any = None,
						 embedded: bool = False,
						 validLib: List = [],
						 noTime: bool = False,
						 ignoreNan: bool = True,
						 numProcess: int = 4,
						 mpMethod: Any = None,
						 chunksize: int = 1) -> numpy.ndarray:
	"""
	Estimate the best neighborhood size for SMap, i.e. the
	exponential decay factor for weighing neighbors by distance.

	:param data: 				2D numpy array where column 0 is time
	:param columns: 			Column indices to use (defaults to all except time)
	:param target: 				Target column index (defaults to column 1)
	:param theta: 				Theta values to test
	:param train: 				Train indices [start, end]
	:param test: 				Test indices [start, end]
	:param embedDimensions: 	Embedding dimension
	:param predictionHorizon: 	Prediction horizon
	:param knn: 				Number of nearest neighbors
	:param step: 				Step size for embedding
	:param exclusionRadius: 	Exclusion radius
	:param solver: 				SMap solver
	:param embedded: 			Whether data is already embedded
	:param validLib: 			Valid library indices
	:param noTime: 				Whether to exclude time column
	:param ignoreNan: 			Whether to ignore NaN values
	:param numProcess: 			Number of processes for multiprocessing
	:param mpMethod: 			Multiprocessing method
	:param chunksize: 			Chunk size for pool.starmap
	:return: Array with columns [theta, correlation]
	"""

	if theta is None:
		theta = [0.01, 0.1, 0.3, 0.5, 0.75, 1,
				 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
	elif not IsNonStringIterable(theta):
		theta = [float(t) for t in theta.split()]

	# Setup Pool
	args = {'columns': columns,
			'target': target,
			'train': train,
			'test': test,
			'embedDims': embedDimensions,
			'predictionHorizon': predictionHorizon,
			'knn': knn,
			'step': step,
			'exclusionRadius': exclusionRadius,
			'solver': solver,
			'embedded': embedded,
			'validLib': validLib,
			'noTime': noTime,
			'ignoreNan': ignoreNan}

	# Create iterable for Pool.starmap, use repeated copies of data, args
	poolArgs = zip(theta, repeat(data), repeat(args))

	# Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
	mpContext = get_context(mpMethod)
	with mpContext.Pool(processes = numProcess) as pool:
		correlationList = pool.starmap(PoolFunc.PredictNLSMapFunc, poolArgs,
									   chunksize = chunksize)

	import numpy as np
	result = np.column_stack([theta, correlationList])

	return result
