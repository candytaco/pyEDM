
# python modules
from multiprocessing import get_context

# package modules
from numpy import array, exp, fmax, divide, mean, nan, roll, sum, zeros, column_stack
from numpy.random import default_rng

from .Utils import ComputeError, IsIterable
# local modules
from .Simplex import Simplex as SimplexClass
from .Results import CCMResult


#------------------------------------------------------------
class CCM:
    '''CCM class : Base class. Contains two Simplex instances'''

    def __init__(self,
                 data            = None,
                 columns = None,
                 target = None,
                 embedDimensions = 0,
                 predictionHorizon              = 0,
                 knn             = 0,
                 step             = -1,
                 exclusionRadius = 0,
                 trainSizes        = [],
                 sample          = 0,
                 seed            = None,
                 includeData     = False,
                 embedded        = False,
                 validLib        = [],
                 noTime          = False,
                 ignoreNan       = True,
                 mpMethod        = None,
                 sequential      = False,
                 verbose         = False):
        '''Initialize CCM.'''

        # Assign parameters from API arguments
        self.name            = 'CCM'
        self.Data            = data
        self.columns         = columns
        self.target          = target
        self.embedDimensions = embedDimensions
        self.predictionHorizon              = predictionHorizon
        self.knn             = knn
        self.step             = step
        self.exclusionRadius = exclusionRadius
        self.trainSizes        = trainSizes
        self.sample          = sample
        self.seed            = seed
        self.includeData     = includeData
        self.embedded        = embedded
        self.validLib        = validLib
        self.noTime          = noTime
        self.ignoreNan       = ignoreNan
        self.mpMethod        = mpMethod
        self.sequential      = sequential
        self.verbose         = verbose

        # Set full train & test
        self.train = self.test = [ 1, self.Data.shape[0] ]

        self.CrossMapList  = None # List of CrossMap results
        self.libMeans      = None # DataFrame of CrossMap results
        self.PredictStats1 = None # DataFrame of CrossMap stats
        self.PredictStats2 = None # DataFrame of CrossMap stats

        # Setup
        self.Validate() # CCM Method

        # Instantiate Forward and Reverse Mapping objects
        # Each __init__ calls EDM.Validate() & EDM.CreateIndices()
        # and sets up targetVec, allTime
        # EDM.Validate sets default knn, overrides E if embedded
        self.FwdMap = SimplexClass(data = data,
                                   columns         = columns,
                                   target          = target,
                                   train             = self.train,
                                   test            = self.test,
                                   embedDimensions = embedDimensions,
                                   predictionHorizon              = predictionHorizon,
                                   knn             = knn,
                                   step             = step,
                                   exclusionRadius = exclusionRadius,
                                   embedded        = embedded,
                                   validLib        = validLib,
                                   noTime          = noTime,
                                   ignoreNan       = ignoreNan,
                                   verbose         = verbose)

        self.RevMap = SimplexClass(data = data,
                                   columns         = target,
                                   target          = columns,
                                   train             = self.train,
                                   test            = self.test,
                                   embedDimensions = embedDimensions,
                                   predictionHorizon              = predictionHorizon,
                                   knn             = knn,
                                   step             = step,
                                   exclusionRadius = exclusionRadius,
                                   embedded        = embedded,
                                   validLib        = validLib,
                                   noTime          = noTime,
                                   ignoreNan       = ignoreNan,
                                   verbose         = verbose)

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run( self ) :
        """Execute CCM and return CCMResult.

        Returns
        -------
        CCMResult
            CCM results with library means and optional detailed statistics
        """
        self.Project()

        return CCMResult(
            libMeans=self.libMeans,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon,
            predictStats1=self.PredictStats1 if self.includeData else None,
            predictStats2=self.PredictStats2 if self.includeData else None
        )

    #-------------------------------------------------------------------
    def Project( self, sequential = False ) :
        '''CCM both directions with CrossMap()'''

        if self.verbose:
            print( f'{self.name}: Project()' )

        if self.sequential : # Sequential alternative to multiprocessing
            FwdCM = self.CrossMap( 'FWD' )
            RevCM = self.CrossMap( 'REV' )
            self.CrossMapList = [ FwdCM, RevCM ]
        else :
            # multiprocessing Pool CrossMap both directions simultaneously
            poolArgs = [ 'FWD', 'REV' ]
            mpContext = get_context( self.mpMethod )
            with mpContext.Pool( processes = 2 ) as pool :
                CrossMapList = pool.map( self.CrossMap, poolArgs )

            self.CrossMapList = CrossMapList

        FwdCM, RevCM = self.CrossMapList

        # Create libMeans array: shape (n_lib_sizes, 3)
        # Column 0: LibSize, Column 1: Fwd correlation, Column 2: Rev correlation
        lib_sizes = array(list(FwdCM['libcorrelation'].keys()))
        fwd_correlations = array(list(FwdCM['libcorrelation'].values()))
        rev_correlations = array(list(RevCM['libcorrelation'].values()))

        self.libMeans = column_stack([lib_sizes, fwd_correlations, rev_correlations])

        if self.includeData :
            FwdCMStats = FwdCM['predictStats'] # key libSize : list of CE dicts
            RevCMStats = RevCM['predictStats']

            # Build PredictStats1 array
            # Each row is a sample with: LibSize, correlation, mae, rmse, mse, nrmse
            stats1_rows = []
            for libSize in FwdCMStats.keys() :
                LibSize  = [libSize] * self.sample # this libSize sample times
                libStats = FwdCMStats[libSize]     # sample ComputeError dicts

                for s in range(self.sample):
                    stats = libStats[s]
                    row = [libSize[s], stats['correlation'], stats['mae'], stats['rmse'],
                           stats['mse'], stats['nrmse']]
                    stats1_rows.append(row)

            self.PredictStats1 = array(stats1_rows)

            # Build PredictStats2 array
            stats2_rows = []
            for libSize in RevCMStats.keys() :
                LibSize  = [libSize] * self.sample # this libSize sample times
                libStats = RevCMStats[libSize]     # sample ComputeError dicts

                for s in range(self.sample):
                    stats = libStats[s]
                    row = [libSize[s], stats['correlation'], stats['mae'], stats['rmse'],
                           stats['mse'], stats['nrmse']]
                    stats2_rows.append(row)

            self.PredictStats2 = array(stats2_rows)

    #-------------------------------------------------------------------
    # 
    #-------------------------------------------------------------------
    def CrossMap( self, direction ) :
        if self.verbose:
            print( f'{self.name}: CrossMap()' )

        if direction == 'FWD' :
            S = self.FwdMap
        elif direction == 'REV' :
            S = self.RevMap
        else :
            raise RuntimeError( f'{self.name}: CrossMap() Invalid Map' )

        # Create random number generator : None sets random state from OS
        RNG = default_rng( self.seed )

        # Copy S.lib_i since it's replaced every iteration
        lib_i   = S.trainIndices.copy()
        N_lib_i = len( lib_i )

        libcorrelationMap  = {} # Output dict libSize key : mean correlation value
        libStatMap = {} # Output dict libSize key : list of ComputeError dicts

        # Loop for library sizes
        for libSize in self.trainSizes :
            correlations = zeros( self.sample )
            if self.includeData :
                predictStats = [None] * self.sample

            # Loop for subsamples
            for s in range( self.sample ) :
                # Generate library row indices for this subsample
                rng_i = RNG.choice( lib_i, size = min( libSize, N_lib_i ),
                                    replace = False )

                S.trainIndices = rng_i

                S.FindNeighbors() # Depends on S.lib_i

                # Code from Simplex:Project ---------------------------------
                # First column is minimum distance of all N test rows
                minDistances = S.knn_distances[:,0]
                # In case there is 0 in minDistances: minWeight = 1E-6
                minDistances = fmax( minDistances, 1E-6 )

                # Divide each column of N x k knn_distances by minDistances
                scaledDistances = divide(S.knn_distances, minDistances[:,None])
                weights         = exp( -scaledDistances )  # Npred x k
                weightRowSum    = sum( weights, axis = 1 ) # Npred x 1

                # Matrix of knn_neighbors + predictionHorizon defines library target values
                knn_neighbors_Tp = S.knn_neighbors + self.predictionHorizon      # Npred x k

                libTargetValues = zeros( knn_neighbors_Tp.shape ) # Npred x k
                for j in range( knn_neighbors_Tp.shape[1] ) :
                    libTargetValues[ :, j ][ :, None ] = \
                        S.targetVec[ knn_neighbors_Tp[ :, j ] ]
                # Code from Simplex:Project ----------------------------------

                # Projection is average of weighted knn library target values
                projection_ = sum( weights * libTargetValues,
                                   axis = 1) / weightRowSum

                # Align observations & predictions as in FormatProjection()
                # Shift projection_ by predictionHorizon
                projection_ = roll( projection_, S.predictionHorizon )
                if S.predictionHorizon > 0 :
                    projection_[ :S.predictionHorizon ] = nan
                elif S.predictionHorizon < 0 :
                    projection_[ S.predictionHorizon: ] = nan

                err = ComputeError(S.targetVec[ S.testIndices, 0],
                                   projection_, digits = 5)

                correlations[ s ] = err['correlation']

                if self.includeData :
                    predictStats[s] = err

            libcorrelationMap[ libSize ] = mean( correlations )

            if self.includeData :
                libStatMap[ libSize ] = predictStats

        # Reset S.lib_i to original
        S.trainIndices = lib_i

        if self.includeData :
            return { 'columns' : S.columns, 'target' : S.target,
                     'libcorrelation' : libcorrelationMap, 'predictStats' : libStatMap }
        else :
            return {'columns':S.columns, 'target':S.target, 'libcorrelation':libcorrelationMap}

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if not len(self.trainSizes) :
            raise RuntimeError(f'{self.name} Validate(): LibSizes required.')
        if not IsIterable(self.trainSizes) :
            self.trainSizes = [int(L) for L in self.trainSizes.split()]

        if self.sample == 0:
            raise RuntimeError(f'{self.name} Validate(): ' +\
                               'sample must be non-zero.')

        # libSizes
        #   if 3 arguments presume [start, stop, increment]
        #      if increment < stop generate the library sequence.
        #      if increment > stop presume list of 3 library sizes.
        #   else: Already list of library sizes.
        if len(self.trainSizes) == 3 :
            # Presume ( start, stop, increment ) sequence arguments
            start, stop, increment = [int( s ) for s in self.trainSizes]

            # If increment < stop, presume start : stop : increment
            # and generate the sequence of library sizes
            if increment < stop :
                if increment < 1 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes increment {increment} is invalid.'
                    raise RuntimeError( msg )

                if start > stop :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} stop {stop} are invalid.'
                    raise RuntimeError( msg )

                if start < self.embedDimensions :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than E {self.embedDimensions}'
                    raise RuntimeError( msg )
                elif start < 3 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than 3.'
                    raise RuntimeError( msg )

                # Fill in libSizes sequence
                self.trainSizes = [i for i in range(start, stop + 1, increment)]

        if self.trainSizes[-1] > self.Data.shape[0] :
            msg = f'{self.name} Validate(): ' +\
                  f'Maximum libSize {self.trainSizes[-1]}' +\
                  f' exceeds data size {self.Data.shape[0]}.'
            raise RuntimeError( msg )

        if self.trainSizes[0] < self.embedDimensions + 2 :
            msg = f'{self.name} Validate(): ' +\
                  f'Minimum libSize {self.trainSizes[0]}' +\
                  f' invalid for E={self.embedDimensions}. Minimum {self.embedDimensions + 2}.'
            raise RuntimeError( msg )

        if self.predictionHorizon < 0 :
            embedShift = abs( self.step ) * (self.embedDimensions - 1)
            maxLibSize = self.trainSizes[-1]
            maxAllowed = self.Data.shape[0] - embedShift + (self.predictionHorizon + 1)
            if maxLibSize > maxAllowed :
                msg = f'{self.name} Validate(): Maximum libSize {maxLibSize}' +\
                    f' too large for predictionHorizon {self.predictionHorizon}, E {self.embedDimensions}, step {self.step}' +\
                    f' Maximum is {maxAllowed}'
                raise RuntimeError( msg )
