'''Interface to Empirical Dynamic Modeling (EDM) pyEDM'''

# python modules
from multiprocessing import get_context
from itertools       import repeat

# package modules
from matplotlib.pyplot import show, axhline

# local modules
from .AuxFunc   import IsIterable, PlotObsPred, PlotCoeff, ComputeError
from .Simplex   import Simplex   as SimplexClass
from .SMap      import SMap      as SMapClass
from .CCM       import CCM       as CCMClass
from .Multiview import Multiview as MultiviewClass

import pyEDM.PoolFunc as PoolFunc

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Simplex( data               = None,
             columns            = None,
             target             = None,
             train                = "",
             test               = "",
             embedDimensions    = 0,
             predictionHorizon                 = 1,
             knn                = 0,
             step                = -1,
             exclusionRadius    = 0,
             embedded           = False,
             validLib           = [],
             noTime             = False,
             generateSteps      = 0,
             generateConcat     = False,
             verbose            = False,
             showPlot           = False,
             ignoreNan          = True,
             returnObject       = False ):
    '''Simplex prediction using numpy array data.

    Parameters:
    data : numpy.ndarray, shape (n_samples, n_features)
        2D numpy array where column 0 is time
    columns : list of int or None
        Column indices to use for embedding (defaults to all except time)
    target : int or None
        Target column index (defaults to column 1)
    '''

    # Instantiate SimplexClass object
    #    Constructor assigns data to self, calls Validate(),
    #    CreateIndices(), and assigns targetVec, time
    S = SimplexClass(data = data,
                     columns         = columns,
                     target          = target,
                     train             = train,
                     test            = test,
                     embedDimensions = embedDimensions,
                     predictionHorizon              = predictionHorizon,
                     knn             = knn,
                     step             = step,
                     exclusionRadius = exclusionRadius,
                     embedded        = embedded,
                     validLib        = validLib,
                     noTime          = noTime,
                     generateSteps   = generateSteps,
                     generateConcat  = generateConcat,
                     ignoreNan       = ignoreNan,
                     verbose         = verbose)

    if generateSteps :
        S.Generate()
    else :
        S.Run()

    if showPlot :
        PlotObsPred( S.Projection, "", S.embedDimensions, S.predictionHorizon )

    if returnObject :
        return S
    else :
        return S.Projection

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def SMap( data               = None,
          columns            = None,
          target             = None,
          train                = "",
          test               = "",
          embedDimensions    = 0,
          predictionHorizon                 = 1,
          knn                = 0,
          step                = -1,
          theta              = 0,
          exclusionRadius    = 0,
          solver             = None,
          embedded           = False,
          validLib           = [],
          noTime             = False,
          generateSteps      = 0,
          generateConcat     = False,
          ignoreNan          = True,
          showPlot           = False,
          verbose            = False,
          returnObject       = False ):
    '''S-Map prediction using numpy array data.

    Parameters:
    data : numpy.ndarray, shape (n_samples, n_features)
        2D numpy array where column 0 is time
    columns : list of int or None
        Column indices to use for embedding (defaults to all except time)
    target : int or None
        Target column index (defaults to column 1)
    '''

    # Validate solver if one was provided
    if solver is not None :
        supportedSolvers = [ 'function',  'lstsq',
                             'LinearRegression', 'SGDRegressor',
                             'Ridge',      'RidgeCV',
                             'Lasso',      'LassoCV',
                             'Lars',       'LarsCV',
                             'LassoLars',  'LassoLarsCV', 'LassoLarsIC',
                             'ElasticNet', 'ElasticNetCV',
                             'OrthogonalMatchingPursuit',
                             'OrthogonalMatchingPursuitCV' ]
        if not solver.__class__.__name__ in supportedSolvers :
            msg = f'SMap(): Invalid solver {solver.__name__}.\n' +\
                  f'Supported solvers: {supportedSolvers}'
            raise Exception( msg )

    # Instantiate SMapClass object
    #    Constructor assigns data to self, calls Validate(),
    #    CreateIndices(), and assigns targetVec, time
    S = SMapClass(data = data,
                  columns         = columns,
                  target          = target,
                  train             = train,
                  test            = test,
                  embedDimensions = embedDimensions,
                  predictionHorizon              = predictionHorizon,
                  knn             = knn,
                  step             = step,
                  theta           = theta,
                  exclusionRadius = exclusionRadius,
                  solver          = solver,
                  embedded        = embedded,
                  validLib        = validLib,
                  noTime          = noTime,
                  generateSteps   = generateSteps,
                  generateConcat  = generateConcat,
                  ignoreNan       = ignoreNan,
                  verbose         = verbose)

    if generateSteps :
        S.Generate()
    else :
        S.Run()

    if showPlot :
        PlotObsPred(S.Projection,   "", S.embedDimensions, S.predictionHorizon)
        PlotCoeff  (S.Coefficients, "", S.embedDimensions, S.predictionHorizon)

    if returnObject :
        return S
    else :
        SMapDict = { 'predictions'    : S.Projection,
                     'coefficients'   : S.Coefficients,
                     'singularValues' : S.SingularValues }
        return SMapDict

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def CCM( data                = None,
         columns             = None,
         target              = None,
         libSizes            = "",
         sample              = 0,
         embedDimensions     = 0,
         predictionHorizon                  = 0,
         knn                 = 0,
         step                 = -1,
         exclusionRadius     = 0,
         seed                = None,
         embedded            = False,
         validLib            = [],
         includeData         = False,
         noTime              = False,
         ignoreNan           = True,
         mpMethod            = None,
         sequential          = False,
         verbose             = False,
         showPlot            = False,
         returnObject        = False ) :
    '''Convergent Cross Mapping.

    Parameters:
    data : numpy.ndarray, shape (n_samples, n_features)
        2D numpy array where column 0 is time
    columns : list of int or None
        Column indices to use (defaults to all except time)
    target : int or list of int or None
        Target column index (defaults to column 1)
    '''

    # Instantiate CCMClass object
    # __init__ creates .FwdMap & .RevMap
    C = CCMClass(data            = data,
                 columns         = columns,
                 target          = target,
                 embedDimensions = embedDimensions,
                 predictionHorizon              = predictionHorizon,
                 knn             = knn,
                 step             = step,
                 exclusionRadius = exclusionRadius,
                 libSizes        = libSizes,
                 sample          = sample,
                 seed            = seed,
                 includeData     = includeData,
                 embedded        = embedded,
                 validLib        = validLib,
                 noTime          = noTime,
                 ignoreNan       = ignoreNan,
                 mpMethod        = mpMethod,
                 sequential      = sequential,
                 verbose         = verbose)

    # Embedding of Forward & Reverse mapping
    C.FwdMap.EmbedData()
    C.FwdMap.RemoveNan()
    C.RevMap.EmbedData()
    C.RevMap.RemoveNan()

    C.Project()

    if showPlot :
        import matplotlib.pyplot as plt
        title = f'E = {C.embedDimensions}'
        fig, ax = plt.subplots()

        # C.libMeans is numpy array: Column 0 is LibSize, rest are correlation values
        if C.libMeans.shape[1] == 3 :
            # CCM of two different variables
            ax.plot(C.libMeans[:, 0], C.libMeans[:, 1], linewidth=3, label='Col 1')
            ax.plot(C.libMeans[:, 0], C.libMeans[:, 2], linewidth=3, label='Col 2')
            ax.legend()
        elif C.libMeans.shape[1] == 2 :
            # CCM of degenerate columns : target
            ax.plot(C.libMeans[:, 0], C.libMeans[:, 1], linewidth=3)

        ax.set( xlabel = "Library Size", ylabel = "CCM correlation", title=title )
        axhline( y = 0, linewidth = 1 )
        show()

    if returnObject :
        return C
    else :
        if includeData :
            return { 'LibMeans'      : C.libMeans,
                     'PredictStats1' : C.PredictStats1,
                     'PredictStats2' : C.PredictStats2 }
        else :
            return C.libMeans

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def Multiview( data               = None,
               columns            = None,
               target             = None,
               train                = "",
               test               = "",
               D                  = 0,
               embedDimensions    = 1,
               predictionHorizon                 = 1,
               knn                = 0,
               step                = -1,
               multiview          = 0,
               exclusionRadius    = 0,
               trainLib           = True,
               excludeTarget      = False,
               ignoreNan          = True,
               verbose            = False,
               numProcess         = 4,
               mpMethod           = None,
               chunksize          = 1,
               showPlot           = False,
               returnObject       = False ):
    '''Multiview prediction using numpy array data.

    Parameters:
    data : numpy.ndarray, shape (n_samples, n_features)
        2D numpy array where column 0 is time
    columns : list of int or None
        Column indices to use (defaults to all except time)
    target : int or None
        Target column index (defaults to column 1)
    '''

    # Instantiate MultiviewClass object
    # __init__ creates .Simplex_, calls Validate(), Setup()
    M = MultiviewClass( data            = data,
                        columns         = columns,
                        target          = target,
                        train             = train,
                        test            = test,
                        D               = D,
                        embedDimensions = embedDimensions,
                        predictionHorizon              = predictionHorizon,
                        knn             = knn,
                        step             = step,
                        multiview       = multiview,
                        exclusionRadius = exclusionRadius,
                        trainLib        = trainLib,
                        excludeTarget   = excludeTarget,
                        ignoreNan       = ignoreNan,
                        verbose         = verbose,
                        numProcess      = numProcess,
                        mpMethod        = mpMethod,
                        chunksize       = chunksize,
                        returnObject    = returnObject )

    M.Rank()
    M.Project()

    # multiview averaged prediction
    # M.topRankProjections is dict of combo : numpy array
    # Each array has columns: [Time, Observations, Predictions, Pred_Variance]
    import numpy as np

    # Get first projection for Time and Observations
    first_proj = next(iter(M.topRankProjections.values()))

    # Collect all predictions (column 2) and average them
    all_predictions = [proj[:, 2] for proj in M.topRankProjections.values()]
    multiviewPredict = np.mean(all_predictions, axis=0)

    # Create result array: [Time, Observations, Predictions]
    M.Projection = np.column_stack([first_proj[:, 0], first_proj[:, 1], multiviewPredict])

    # Create View: rankings of column combinations
    colCombos = list(M.topRankProjections.keys())

    topRankStats = {}
    for combo in colCombos :
        proj = M.topRankProjections[combo]
        # proj columns: 0=Time, 1=Observations, 2=Predictions, 3=Variance
        topRankStats[combo] = ComputeError(proj[:, 1], proj[:, 2])

    M.topRankStats = topRankStats

    # Build View array: each row is [combo_as_str, correlation, MAE, CAE, RMSE]
    view_rows = []
    for combo in colCombos:
        stats = topRankStats[combo]
        view_rows.append([str(combo), stats['correlation'], stats['MAE'], stats['CAE'], stats['RMSE']])

    M.View = view_rows  # List of lists for now

    if showPlot :
        PlotObsPred( M.Projection, "", M.D, M.predictionHorizon )

    if returnObject :
        return M
    else :
        return { 'Predictions' : M.Projection, 'View' : M.View }

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def EmbedDimension( data            = None,
                    columns         = None,
                    target          = None,
                    maxE            = 10,
                    train             = "",
                    test            = "",
                    predictionHorizon              = 1,
                    step             = -1,
                    exclusionRadius = 0,
                    embedded        = False,
                    validLib        = [],
                    noTime          = False,
                    ignoreNan       = True,
                    verbose         = False,
                    numProcess      = 4,
                    mpMethod        = None,
                    chunksize       = 1,
                    showPlot        = True ):
    '''Estimate optimal embedding dimension [1:maxE].

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
    '''

    # Setup Pool
    Evals = [ E for E in range( 1, maxE + 1 ) ]
    args = { 'columns'         : columns,
             'target'          : target,
             'train'             : train,
             'test'            : test,
             'predictionHorizon'              : predictionHorizon,
             'step'             : step,
             'exclusionRadius' : exclusionRadius,
             'embedded'        : embedded,
             'validLib'        : validLib,
             'noTime'          : noTime,
             'ignoreNan'       : ignoreNan }

    # Create iterable for Pool.starmap, use repeated copies of data, args
    poolArgs = zip( Evals, repeat( data ), repeat( args ) )

    # Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
    mpContext = get_context( mpMethod )
    with mpContext.Pool( processes = numProcess ) as pool :
        correlationList = pool.starmap( PoolFunc.EmbedDimSimplexFunc, poolArgs,
                                chunksize = chunksize )

    import numpy as np
    result = np.column_stack([Evals, correlationList])

    if showPlot :
        import matplotlib.pyplot as plt
        title = "predictionHorizon=" + str(predictionHorizon)
        fig, ax = plt.subplots()
        ax.plot(result[:, 0], result[:, 1], linewidth=3)
        ax.set( xlabel = "Embedding Dimension",
                ylabel = "Prediction Skill correlation",
                title = title )
        show()

    return result

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictInterval( data               = None,
                     columns            = None,
                     target             = None,
                     train                = "",
                     test               = "",
                     maxTp              = 10,
                     embedDimensions    = 1,
                     step                = -1,
                     exclusionRadius    = 0,
                     embedded           = False,
                     validLib           = [],
                     noTime             = False,
                     ignoreNan          = True,
                     verbose            = False,
                     numProcess         = 4,
                     mpMethod           = None,
                     chunksize          = 1,
                     showPlot           = True ):
    '''Estimate optimal prediction interval [1:maxTp].

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
    '''

    # Setup Pool
    Evals = [ predictionHorizon for predictionHorizon in range( 1, maxTp + 1 ) ]
    args = { 'columns'         : columns,
             'target'          : target,
             'train'             : train,
             'test'            : test,
             'embedDims'               : embedDimensions,
             'step'             : step,
             'exclusionRadius' : exclusionRadius,
             'embedded'        : embedded,
             'validLib'        : validLib,
             'noTime'          : noTime,
             'ignoreNan'       : ignoreNan }

    # Create iterable for Pool.starmap, use repeated copies of data, args
    poolArgs = zip( Evals, repeat( data ), repeat( args ) )

    # Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
    mpContext = get_context( mpMethod )
    with mpContext.Pool( processes = numProcess ) as pool :
        correlationList = pool.starmap( PoolFunc.PredictIntervalSimplexFunc, poolArgs,
                                chunksize = chunksize )

    import numpy as np
    result = np.column_stack([Evals, correlationList])

    if showPlot :
        if embedded :
            if IsIterable( columns ) :
                embedDimensions = len( columns )
            else :
                embedDimensions = 1
        title = "Embedding Dims = " + str( embedDimensions )
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(result[:, 0], result[:, 1], linewidth=3)
        ax.set( xlabel = "Forecast Interval",
                ylabel = "Prediction Skill correlation",
                title = title )
        show()

    return result

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def PredictNonlinear( data               = None,
                      columns            = None,
                      target             = None,
                      theta              = None,
                      train                = "",
                      test               = "",
                      embedDimensions    = 1,
                      predictionHorizon                 = 1,
                      knn                = 0,
                      step                = -1,
                      exclusionRadius    = 0,
                      solver             = None,
                      embedded           = False,
                      validLib           = [],
                      noTime             = False,
                      ignoreNan          = True,
                      verbose            = False,
                      numProcess         = 4,
                      mpMethod           = None,
                      chunksize          = 1,
                      showPlot           = True ):
    '''Estimate S-map localisation over theta.

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
    '''

    if theta is None :
        theta = [ 0.01, 0.1, 0.3, 0.5, 0.75, 1,
                  1.5, 2, 3, 4, 5, 6, 7, 8, 9 ]
    elif not IsIterable( theta ) :
        theta = [float(t) for t in theta.split()]

    # Setup Pool
    args = { 'columns'         : columns,
             'target'          : target,
             'train'             : train,
             'test'            : test,
             'embedDims'               : embedDimensions,
             'predictionHorizon'              : predictionHorizon,
             'knn'             : knn,
             'step'             : step,
             'exclusionRadius' : exclusionRadius,
             'solver'          : solver,
             'embedded'        : embedded,
             'validLib'        : validLib,
             'noTime'          : noTime,
             'ignoreNan'       : ignoreNan }

    # Create iterable for Pool.starmap, use repeated copies of data, args
    poolArgs = zip( theta, repeat( data ), repeat( args ) )

    # Multiargument starmap : EmbedDimSimplexFunc in PoolFunc
    mpContext = get_context( mpMethod )
    with mpContext.Pool( processes = numProcess ) as pool :
        correlationList = pool.starmap( PoolFunc.PredictNLSMapFunc, poolArgs,
                                chunksize = chunksize )

    import numpy as np
    result = np.column_stack([theta, correlationList])

    if showPlot :
        if embedded :
            if IsIterable( columns ) :
                embedDimensions = len( columns )
            else :
                embedDimensions = 1
        title = "Embedding Dims = " + str( embedDimensions )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(result[:, 0], result[:, 1], linewidth=3)
        ax.set( xlabel = "S-map Localisation (theta)",
                ylabel = "Prediction Skill correlation",
                title = title )
        show()

    return result
