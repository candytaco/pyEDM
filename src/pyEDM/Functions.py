"""
Functional programming interface to Empirical Dynamic Modeling (EDM) pyEDM.
While the underlying classes have been refactored, these functions should 
return roughly the same data structures returned by the original pyEDM functions
"""

from itertools import repeat
# python modules
from multiprocessing import get_context


# local modules
from .EDM import PoolFunc
from .Utils import IsNonStringIterable
from .EDM.CCM import CCM
from .EDM.Multiview import Multiview
from .EDM.SMap import SMap
from .EDM.Simplex import Simplex


def FitSimplex(data = None,
               columns = None,
               target = None,
               train = None,
               test = None,
               embedDimensions = 0,
               predictionHorizon = 1,
               knn = 0,
               step = -1,
               exclusionRadius = 0,
               embedded = False,
               validLib = [],
               noTime = False,
               generateSteps = 0,
               generateConcat = False,
               verbose = False,
               ignoreNan = True,
               returnObject = False):
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


def FitSMap(data = None,
            columns = None,
            target = None,
            train = None,
            test = None,
            embedDimensions = 0,
            predictionHorizon = 1,
            knn = 0,
            step = -1,
            theta = 0,
            exclusionRadius = 0,
            solver = None,
            embedded = False,
            validLib = [],
            noTime = False,
            generateSteps = 0,
            generateConcat = False,
            ignoreNan = True,
            verbose = False,
            returnObject = False):
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
def FitCCM(data = None,
           columns = None,
           target = None,
           trainSizes = None,
           sample = 0,
           embedDimensions = 0,
           predictionHorizon = 0,
           knn = 0,
           step = -1,
           exclusionRadius = 0,
           seed = None,
           embedded = False,
           validLib = [],
           includeData = False,
           noTime = False,
           ignoreNan = True,
           mpMethod = None,
           sequential = False,
           verbose = False,
           returnObject = False):
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
	:return: Library means array or CCM object
	"""

	# Instantiate CCMClass object
	C = CCM(data = data,
			columns = columns,
			target = target,
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
			verbose = verbose)

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


def FitMultiview(data = None,
                 columns = None,
                 target = None,
                 train = None,
                 test = None,
                 D = 0,
                 embedDimensions = 1,
                 predictionHorizon = 1,
                 knn = 0,
                 step = -1,
                 multiview = 0,
                 exclusionRadius = 0,
                 trainLib = True,
                 excludeTarget = False,
                 ignoreNan = True,
                 verbose = False,
                 numProcess = 4,
                 mpMethod = None,
                 chunksize = 1,
                 returnObject = False):
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


def FindOptimalEmbeddingDimensionality(data = None,
                                       columns = None,
                                       target = None,
                                       maxE = 10,
                                       train = None,
                                       test = None,
                                       predictionHorizon = 1,
                                       step = -1,
                                       exclusionRadius = 0,
                                       embedded = False,
                                       validLib = [],
                                       noTime = False,
                                       ignoreNan = True,
                                       verbose = False,
                                       numProcess = 4,
                                       mpMethod = None,
                                       chunksize = 1, ):
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
	:param verbose: 			Print diagnostic messages
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

	import numpy as np
	result = np.column_stack([Evals, correlationList])

	return result


def FindOptimalPredictionHorizon(data = None,
                                 columns = None,
                                 target = None,
                                 train = None,
                                 test = None,
                                 maxTp = 10,
                                 embedDimensions = 1,
                                 step = -1,
                                 exclusionRadius = 0,
                                 embedded = False,
                                 validLib = [],
                                 noTime = False,
                                 ignoreNan = True,
                                 verbose = False,
                                 numProcess = 4,
                                 mpMethod = None,
                                 chunksize = 1, ):
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
	:param verbose: 		Print diagnostic messages
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


def FindSMapNeighborhood(data = None,
                         columns = None,
                         target = None,
                         theta = None,
                         train = None,
                         test = None,
                         embedDimensions = 1,
                         predictionHorizon = 1,
                         knn = 0,
                         step = -1,
                         exclusionRadius = 0,
                         solver = None,
                         embedded = False,
                         validLib = [],
                         noTime = False,
                         ignoreNan = True,
                         verbose = False,
                         numProcess = 4,
                         mpMethod = None,
                         chunksize = 1, ):
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
	:param verbose: 			Print diagnostic messages
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
