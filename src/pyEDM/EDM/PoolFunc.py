
# python modules

# package modules

# local modules
from .. import Functions
from ..Utils import ComputeError

#------------------------------------------------------
# Function to evaluate multiview predictions top combos
#------------------------------------------------------
def MultiviewSimplexPred( combo, data, args ) :
	"""
	Function to evaluate multiview predictions top combos

	:param combo: Column combination tuple
	:param data: Embedded data array
	:param args: Dictionary of Simplex arguments
	:return: Prediction projection array
	"""
	projection = Functions.FitSimplex(data       = data,
                                columns         = list( combo ),
                                target          = args['target'],
                                train             = args['train'],
                                test            = args['test'],
                                embedDimensions = args['embedDims'],
                                predictionHorizon              = args['predictionHorizon'],
                                step             = args['step'],
                                exclusionRadius = args['exclusionRadius'],
                                embedded        = args['embedded'],
                                noTime          = args['noTime'],
                                ignoreNan       = args['ignoreNan'])
	return projection

#----------------------------------------------------
# Function to evaluate combo rank (correlation)
#----------------------------------------------------
def MultiviewSimplexcorrelation( combo, data, args ) :
	"""
	Function to evaluate combo rank (correlation)

	:param combo: Column combination tuple
	:param data: Embedded data array
	:param args: Dictionary of Simplex arguments
	:return: Correlation value
	"""
	projection = Functions.FitSimplex(data       = data,
                                columns         = list( combo ),
                                target          = args['target'],
                                train             = args['train'],
                                test            = args['test'],
                                embedDimensions = args['embedDims'],
                                predictionHorizon              = args['predictionHorizon'],
                                step             = args['step'],
                                exclusionRadius = args['exclusionRadius'],
                                embedded        = args['embedded'],
                                noTime          = args['noTime'],
                                ignoreNan       = args['ignoreNan'])

	# projection is numpy array: Column 1 is Observations, Column 2 is Predictions
	return ComputeError(projection[:, 1], projection[:, 2], None)

#----------------------------------------------------
# Function to evaluate Simplex in EmbedDimension Pool
#----------------------------------------------------
def EmbedDimSimplexFunc( embedDimensions, data, args ) :
	"""
	Function to evaluate Simplex in EmbedDimension Pool

	:param embedDimensions: Embedding dimension to test
	:param data: Data array
	:param args: Dictionary of Simplex arguments
	:return: Correlation value
	"""
	projection = Functions.FitSimplex(data       = data,
                                columns         = args['columns'],
                                target          = args['target'],
                                train             = args['train'],
                                test            = args['test'],
                                embedDimensions = embedDimensions,
                                predictionHorizon              = args['predictionHorizon'],
                                step             = args['step'],
                                exclusionRadius = args['exclusionRadius'],
                                embedded        = args['embedded'],
                                validLib        = args['validLib'],
                                noTime          = args['noTime'],
                                ignoreNan       = args['ignoreNan'])

	# projection is numpy array: Column 1 is Observations, Column 2 is Predictions
	return ComputeError(projection[:, 1], projection[:, 2], None)

#-----------------------------------------------------
# Function to evaluate Simplex in PredictInterval Pool
#----------------------------------------------------
def PredictIntervalSimplexFunc( predictionHorizon, data, args ) :
	"""
	Function to evaluate Simplex in PredictInterval Pool

	:param predictionHorizon: Prediction horizon to test
	:param data: Data array
	:param args: Dictionary of Simplex arguments
	:return: Correlation value
	"""
	projection = Functions.FitSimplex(data       = data,
                                columns         = args['columns'],
                                target          = args['target'],
                                train             = args['train'],
                                test            = args['test'],
                                embedDimensions = args['embedDims'],
                                predictionHorizon              = predictionHorizon,
                                step             = args['step'],
                                exclusionRadius = args['exclusionRadius'],
                                embedded        = args['embedded'],
                                validLib        = args['validLib'],
                                noTime          = args['noTime'],
                                ignoreNan       = args['ignoreNan'])

	# projection is numpy array: Column 1 is Observations, Column 2 is Predictions
	return ComputeError(projection[:, 1], projection[:, 2], None)

#-----------------------------------------------------
# Function to evaluate SMap in PredictNonlinear Pool
#----------------------------------------------------
def PredictNLSMapFunc( theta, data, args ) :
	"""
	Function to evaluate SMap in PredictNonlinear Pool

	:param theta: Localization parameter to test
	:param data: Data array
	:param args: Dictionary of SMap arguments
	:return: Correlation value
	"""
	S = Functions.FitSMap(data       = data,
                    columns         = args['columns'],
                    target          = args['target'],
                    train             = args['train'],
                    test            = args['test'],
                    embedDimensions = args['embedDims'],
                    predictionHorizon              = args['predictionHorizon'],
                    knn             = args['knn'],
                    step             = args['step'],
                    theta           = theta,
                    exclusionRadius = args['exclusionRadius'],
                    solver          = args['solver'],
                    embedded        = args['embedded'],
                    validLib        = args['validLib'],
                    noTime          = args['noTime'],
                    ignoreNan       = args['ignoreNan'])

	projection = S['predictions']
	# projection is numpy array: Column 1 is Observations, Column 2 is Predictions
	return ComputeError(projection[:, 1], projection[:, 2], None)
