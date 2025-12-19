
# python modules

# package modules

# local modules
import pyEDM.API as API
from .AuxFunc import ComputeError

#------------------------------------------------------
# Function to evaluate multiview predictions top combos
#------------------------------------------------------
def MultiviewSimplexPred( combo, data, args ) :

    df = API.Simplex( data       = data,
                      columns         = list( combo ),
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )
    return df

#----------------------------------------------------
# Function to evaluate combo rank (correlation)
#----------------------------------------------------
def MultiviewSimplexcorrelation( combo, data, args ) :

    df = API.Simplex( data       = data,
                      columns         = list( combo ),
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    # df is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( df[:, 1], df[:, 2] )
    return err['correlation']

#----------------------------------------------------
# Function to evaluate Simplex in EmbedDimension Pool
#----------------------------------------------------
def EmbedDimSimplexFunc( E, data, args ) :

    df = API.Simplex( data       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = E, 
                      Tp              = args['Tp'],
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    # df is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( df[:, 1], df[:, 2] )
    return err['correlation']

#-----------------------------------------------------
# Function to evaluate Simplex in PredictInterval Pool
#----------------------------------------------------
def PredictIntervalSimplexFunc( Tp, data, args ) :

    df = API.Simplex( data       = data,
                      columns         = args['columns'],
                      target          = args['target'], 
                      lib             = args['lib'],
                      pred            = args['pred'],
                      E               = args['E'], 
                      Tp              = Tp,
                      tau             = args['tau'],
                      exclusionRadius = args['exclusionRadius'],
                      embedded        = args['embedded'],
                      validLib        = args['validLib'],
                      noTime          = args['noTime'],
                      ignoreNan       = args['ignoreNan'] )

    # df is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( df[:, 1], df[:, 2] )
    return err['correlation']

#-----------------------------------------------------
# Function to evaluate SMap in PredictNonlinear Pool
#----------------------------------------------------
def PredictNLSMapFunc( theta, data, args ) :

    S = API.SMap( data       = data,
                  columns         = args['columns'],
                  target          = args['target'], 
                  lib             = args['lib'],
                  pred            = args['pred'],
                  E               = args['E'], 
                  Tp              = args['Tp'],
                  knn             = args['knn'],
                  tau             = args['tau'],
                  theta           = theta,
                  exclusionRadius = args['exclusionRadius'],
                  solver          = args['solver'],
                  embedded        = args['embedded'],
                  validLib        = args['validLib'],
                  noTime          = args['noTime'],
                  ignoreNan       = args['ignoreNan'] )

    df = S['predictions']
    # df is numpy array: Column 1 is Observations, Column 2 is Predictions
    err = ComputeError( df[:, 1], df[:, 2] )
    return err['correlation']
