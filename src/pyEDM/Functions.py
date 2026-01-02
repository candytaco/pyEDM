"""Functional programming interface to Empirical Dynamic Modeling (EDM) pyEDM.
While the underlying classes have been refactored, these functions should return roughly the same data structures
returned by the original pyEDM functions"""

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
	"""Simplex prediction.

	:param data: 2D numpy array where column 0 is time
	:param columns: Column indices to use for embedding (defaults to all except time)
	:param target: Target column index (defaults to column 1)
	columns : list of int or None
		Column indices to use for embedding (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)
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
	"""S-Map prediction.

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use for embedding (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)
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
	"""Convergent Cross Mapping.

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use (defaults to all except time)
	target : int or list of int or None
		Target column index (defaults to column 1)
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
	"""Multiview prediction

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)
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
	"""Estimate optimal embedding dimension [1:maxE].

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)

	Returns:
	numpy.ndarray, shape (maxE, 2)
		Column 0: E values, Column 1: correlation values
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
	"""Estimate optimal prediction interval [1:maxTp].

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)

	Returns:
	numpy.ndarray, shape (maxTp, 2)
		Column 0: predictionHorizon values, Column 1: correlation values
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
	"""Estimate the best neighboorhood size for SMap, i.e. the
	exponential decay factor for weighing neighbors by distance

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)

	Returns:
	numpy.ndarray, shape (len(theta), 2)
		Column 0: theta values, Column 1: correlation values
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
