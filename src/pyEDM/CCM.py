
# python modules
from multiprocessing import get_context

# package modules
from numpy  import array, exp, fmax, divide, mean, nan, roll, sum, zeros, column_stack
from numpy.random import default_rng

# local modules
from .Simplex import Simplex as SimplexClass
from .AuxFunc import ComputeError, IsIterable

#------------------------------------------------------------
class CCM:
    '''CCM class : Base class. Contains two Simplex instances'''

    def __init__( self,
                  data            = None,
                  columns         = "",
                  target          = "",
                  E               = 0,
                  Tp              = 0,
                  knn             = 0,
                  tau             = -1,
                  exclusionRadius = 0,
                  libSizes        = [],
                  sample          = 0,
                  seed            = None,
                  includeData     = False,
                  embedded        = False,
                  validLib        = [],
                  noTime          = False,
                  ignoreNan       = True,
                  mpMethod        = None,
                  sequential      = False,
                  verbose         = False ):
        '''Initialize CCM.'''

        # Assign parameters from API arguments
        self.name            = 'CCM'
        self.Data            = data
        self.columns         = columns
        self.target          = target
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn
        self.tau             = tau
        self.exclusionRadius = exclusionRadius
        self.libSizes        = libSizes
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

        # Set full lib & pred
        self.lib = self.pred = [ 1, self.Data.shape[0] ]

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
                                   lib             = self.lib,
                                   pred            = self.pred,
                                   E               = E,
                                   Tp              = Tp,
                                   knn             = knn,
                                   tau             = tau,
                                   exclusionRadius = exclusionRadius,
                                   embedded        = embedded,
                                   validLib        = validLib,
                                   noTime          = noTime,
                                   ignoreNan       = ignoreNan,
                                   verbose         = verbose)

        self.RevMap = SimplexClass(data = data,
                                   columns         = target,
                                   target          = columns,
                                   lib             = self.lib,
                                   pred            = self.pred,
                                   E               = E,
                                   Tp              = Tp,
                                   knn             = knn,
                                   tau             = tau,
                                   exclusionRadius = exclusionRadius,
                                   embedded        = embedded,
                                   validLib        = validLib,
                                   noTime          = noTime,
                                   ignoreNan       = ignoreNan,
                                   verbose         = verbose)

    #-------------------------------------------------------------------
    # Methods
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
        # Column 0: LibSize, Column 1: Fwd rho, Column 2: Rev rho
        lib_sizes = array(list(FwdCM['libRho'].keys()))
        fwd_rhos = array(list(FwdCM['libRho'].values()))
        rev_rhos = array(list(RevCM['libRho'].values()))

        self.libMeans = column_stack([lib_sizes, fwd_rhos, rev_rhos])

        if self.includeData :
            FwdCMStats = FwdCM['predictStats'] # key libSize : list of CE dicts
            RevCMStats = RevCM['predictStats']

            # Build PredictStats1 array
            # Each row is a sample with: LibSize, rho, mae, rmse, mse, nrmse
            stats1_rows = []
            for libSize in FwdCMStats.keys() :
                LibSize  = [libSize] * self.sample # this libSize sample times
                libStats = FwdCMStats[libSize]     # sample ComputeError dicts

                for s in range(self.sample):
                    stats = libStats[s]
                    row = [libSize[s], stats['rho'], stats['mae'], stats['rmse'],
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
                    row = [libSize[s], stats['rho'], stats['mae'], stats['rmse'],
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

        libRhoMap  = {} # Output dict libSize key : mean rho value
        libStatMap = {} # Output dict libSize key : list of ComputeError dicts

        # Loop for library sizes
        for libSize in self.libSizes :
            rhos = zeros( self.sample )
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
                # First column is minimum distance of all N pred rows
                minDistances = S.knn_distances[:,0]
                # In case there is 0 in minDistances: minWeight = 1E-6
                minDistances = fmax( minDistances, 1E-6 )

                # Divide each column of N x k knn_distances by minDistances
                scaledDistances = divide(S.knn_distances, minDistances[:,None])
                weights         = exp( -scaledDistances )  # Npred x k
                weightRowSum    = sum( weights, axis = 1 ) # Npred x 1

                # Matrix of knn_neighbors + Tp defines library target values
                knn_neighbors_Tp = S.knn_neighbors + self.Tp      # Npred x k

                libTargetValues = zeros( knn_neighbors_Tp.shape ) # Npred x k
                for j in range( knn_neighbors_Tp.shape[1] ) :
                    libTargetValues[ :, j ][ :, None ] = \
                        S.targetVec[ knn_neighbors_Tp[ :, j ] ]
                # Code from Simplex:Project ----------------------------------

                # Projection is average of weighted knn library target values
                projection_ = sum( weights * libTargetValues,
                                   axis = 1) / weightRowSum

                # Align observations & predictions as in FormatProjection()
                # Shift projection_ by Tp
                projection_ = roll( projection_, S.Tp )
                if S.Tp > 0 :
                    projection_[ :S.Tp ] = nan
                elif S.Tp < 0 :
                    projection_[ S.Tp: ] = nan

                err = ComputeError(S.targetVec[ S.testIndices, 0],
                                   projection_, digits = 5)

                rhos[ s ] = err['rho']

                if self.includeData :
                    predictStats[s] = err

            libRhoMap[ libSize ] = mean( rhos )

            if self.includeData :
                libStatMap[ libSize ] = predictStats

        # Reset S.lib_i to original
        S.trainIndices = lib_i

        if self.includeData :
            return { 'columns' : S.columns, 'target' : S.target,
                     'libRho' : libRhoMap, 'predictStats' : libStatMap }
        else :
            return {'columns':S.columns, 'target':S.target, 'libRho':libRhoMap}

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if not len( self.libSizes ) :
            raise RuntimeError(f'{self.name} Validate(): LibSizes required.')
        if not IsIterable( self.libSizes ) :
            self.libSizes = [ int(L) for L in self.libSizes.split() ]

        if self.sample == 0:
            raise RuntimeError(f'{self.name} Validate(): ' +\
                               'sample must be non-zero.')

        # libSizes
        #   if 3 arguments presume [start, stop, increment]
        #      if increment < stop generate the library sequence.
        #      if increment > stop presume list of 3 library sizes.
        #   else: Already list of library sizes.
        if len( self.libSizes ) == 3 :
            # Presume ( start, stop, increment ) sequence arguments
            start, stop, increment = [ int( s ) for s in self.libSizes ]

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

                if start < self.E :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than E {self.E}'
                    raise RuntimeError( msg )
                elif start < 3 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than 3.'
                    raise RuntimeError( msg )

                # Fill in libSizes sequence
                self.libSizes = [i for i in range(start, stop+1, increment)]

        if self.libSizes[-1] > self.Data.shape[0] :
            msg = f'{self.name} Validate(): ' +\
                  f'Maximum libSize {self.libSizes[-1]}'    +\
                  f' exceeds data size {self.Data.shape[0]}.'
            raise RuntimeError( msg )

        if self.libSizes[0] < self.E + 2 :
            msg = f'{self.name} Validate(): ' +\
                  f'Minimum libSize {self.libSizes[0]}'    +\
                  f' invalid for E={self.E}. Minimum {self.E + 2}.'
            raise RuntimeError( msg )

        if self.Tp < 0 :
            embedShift = abs( self.tau ) * ( self.E - 1 )
            maxLibSize = self.libSizes[-1]
            maxAllowed = self.Data.shape[0] - embedShift + (self.Tp + 1)
            if maxLibSize > maxAllowed :
                msg = f'{self.name} Validate(): Maximum libSize {maxLibSize}'  +\
                    f' too large for Tp {self.Tp}, E {self.E}, tau {self.tau}' +\
                    f' Maximum is {maxAllowed}'
                raise RuntimeError( msg )
