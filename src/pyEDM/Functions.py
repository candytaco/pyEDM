"""Functional programming interface to Empirical Dynamic Modeling (EDM) pyEDM.
While the underlying classes have been refactored, these functions should return roughly the same data structures
returned by the original pyEDM functions"""

from itertools import repeat
# python modules
from multiprocessing import get_context


import pyEDM.PoolFunc as PoolFunc
# local modules
from .Utils import IsIterable
from .CCM import CCM as CCMClass
from .Multiview import Multiview as MultiviewClass
from .SMap import SMap as SMapClass
from .Simplex import Simplex as SimplexClass
from .Parameters import (EDMParameters, DataSplit, GenerationParameters,
                         SMapParameters, CCMParameters, MultiviewParameters,
                         ExecutionParameters)


def Simplex(data = None,
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
	"""Simplex prediction using numpy array data.

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use for embedding (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)
	"""

	# Create parameter objects
	params = EDMParameters(
		data = data,
		columns = columns,
		target = target,
		embedDimensions = embedDimensions,
		predictionHorizon = predictionHorizon,
		knn = knn,
		step = step,
		exclusionRadius = exclusionRadius,
		embedded = embedded,
		validLib = validLib,
		noTime = noTime,
		ignoreNan = ignoreNan,
		verbose = verbose
	)

	split = DataSplit(train = train, test = test)
	generation = GenerationParameters(generateSteps = generateSteps, generateConcat = generateConcat)

	# Instantiate SimplexClass object using parameter objects
	S = SimplexClass(params = params, split = split, generation = generation)

	if generateSteps:
		result = S.Generate()
	else:
		result = S.Run()

	if returnObject:
		return S
	else:
		return result.projection


def SMap(data = None,
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
	"""S-Map prediction using numpy array data.

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use for embedding (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)
	"""

	# Create parameter objects
	params = EDMParameters(
		data = data,
		columns = columns,
		target = target,
		embedDimensions = embedDimensions,
		predictionHorizon = predictionHorizon,
		knn = knn,
		step = step,
		exclusionRadius = exclusionRadius,
		embedded = embedded,
		validLib = validLib,
		noTime = noTime,
		ignoreNan = ignoreNan,
		verbose = verbose
	)

	split = DataSplit(train = train, test = test)
	generation = GenerationParameters(generateSteps = generateSteps, generateConcat = generateConcat)
	smap_params = SMapParameters(theta = theta, solver = solver)

	# Instantiate SMapClass object using parameter objects
	S = SMapClass(params = params, split = split, generation = generation, smap = smap_params)

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


def CCM(data = None,
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

	# Create parameter objects
	params = EDMParameters(
		data = data,
		columns = columns,
		target = target,
		embedDimensions = embedDimensions,
		predictionHorizon = predictionHorizon,
		knn = knn,
		step = step,
		exclusionRadius = exclusionRadius,
		embedded = embedded,
		validLib = validLib,
		noTime = noTime,
		ignoreNan = ignoreNan,
		verbose = verbose
	)

	ccm_params = CCMParameters(
		trainSizes = trainSizes if trainSizes is not None else [],
		sample = sample,
		seed = seed,
		includeData = includeData
	)

	execution = ExecutionParameters(mpMethod = mpMethod, sequential = sequential)

	# Instantiate CCMClass object using parameter objects
	C = CCMClass(params = params, ccm = ccm_params, execution = execution)

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


def Multiview(data = None,
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
	"""Multiview prediction using numpy array data.

	Parameters:
	data : numpy.ndarray, shape (n_samples, n_features)
		2D numpy array where column 0 is time
	columns : list of int or None
		Column indices to use (defaults to all except time)
	target : int or None
		Target column index (defaults to column 1)
	"""

	# Create parameter objects
	params = EDMParameters(
		data = data,
		columns = columns,
		target = target,
		embedDimensions = embedDimensions,
		predictionHorizon = predictionHorizon,
		knn = knn,
		step = step,
		exclusionRadius = exclusionRadius,
		ignoreNan = ignoreNan,
		verbose = verbose
	)

	split = DataSplit(train = train, test = test)

	multiview_params = MultiviewParameters(
		D = D,
		multiview = multiview,
		trainLib = trainLib,
		excludeTarget = excludeTarget
	)

	execution = ExecutionParameters(
		numProcess = numProcess,
		mpMethod = mpMethod,
		chunksize = chunksize
	)

	# Instantiate MultiviewClass object using parameter objects
	M = MultiviewClass(params = params, split = split, multiview = multiview_params, execution = execution)

	result = M.Run()

	if returnObject:
		return M
	else:
		return {'Predictions': result.projection, 'View': result.view}


def EmbedDimension(data = None,
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


def PredictInterval(data = None,
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


def PredictNonlinear(data = None,
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
	"""Estimate S-map localisation over theta.

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
	elif not IsIterable(theta):
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
