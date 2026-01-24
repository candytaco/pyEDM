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

# local modules
from .EDM import PoolFunc
from .Utils import IsNonStringIterable
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
									   numProcess: int = 4,
									   mpMethod: Any = None,
									   chunksize: int = 1) -> numpy.ndarray:
	"""
	Estimate optimal embedding dimension [1:maxE].

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
	:param numProcess: 			Number of processes for multiprocessing
	:param mpMethod: 			Multiprocessing method
	:param chunksize: 			Chunk size for pool.starmap
	:return: Array with columns [E, correlation]
	"""

	# Setup Pool
	Evals = [E for E in range(1, maxE + 1)]
	args = {'columns': columns,
	        'target': target,
	        'train': train,
	        'test': test,
	        'predictionHorizon': predictionHorizon,
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
		correlationList = pool.starmap(PoolFunc.EmbedDimSimplexFunc, poolArgs,
									   chunksize = chunksize)

	result = numpy.column_stack([Evals, correlationList])

	return result


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
