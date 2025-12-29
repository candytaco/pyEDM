
# python modules

# package modules

# local modules
import pyEDM.Functions as API
from .Utils import ComputeError

#------------------------------------------------------
# Function to evaluate multiview predictions top combos
#------------------------------------------------------
def MultiviewSimplexPred( combo, data, args ) :

    projection = API.Simplex( data       = data,
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
                              ignoreNan       = args['ignoreNan'] )
    return projection

#----------------------------------------------------
# Function to evaluate combo rank (correlation)
#----------------------------------------------------
def MultiviewSimplexcorrelation( combo, data, args ) :

    projection = API.Simplex( data       = data,
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
                              ignoreNan       = args['ignoreNan'] )

    # projection is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( projection[:, 1], projection[:, 2] )
    return err['correlation']

#----------------------------------------------------
# Function to evaluate Simplex in EmbedDimension Pool
#----------------------------------------------------
def EmbedDimSimplexFunc( embedDimensions, data, args ) :

    projection = API.Simplex( data       = data,
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
                              ignoreNan       = args['ignoreNan'] )

    # projection is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( projection[:, 1], projection[:, 2] )
    return err['correlation']

#-----------------------------------------------------
# Function to evaluate Simplex in PredictInterval Pool
#----------------------------------------------------
def PredictIntervalSimplexFunc( predictionHorizon, data, args ) :

    projection = API.Simplex( data       = data,
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
                              ignoreNan       = args['ignoreNan'] )

    # projection is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( projection[:, 1], projection[:, 2] )
    return err['correlation']

#-----------------------------------------------------
# Function to evaluate SMap in PredictNonlinear Pool
#----------------------------------------------------
def PredictNLSMapFunc( theta, data, args ) :

    S = API.SMap( data       = data,
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
                  ignoreNan       = args['ignoreNan'] )

    projection = S['predictions']
    # projection is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( projection[:, 1], projection[:, 2] )
    return err['correlation']
