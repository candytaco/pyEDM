# python modules

# package modules
from numpy  import apply_along_axis, insert, isnan, isfinite, exp
from numpy  import full, integer, linspace, mean, nan, power, sum, array
from numpy  import column_stack

from numpy.linalg import lstsq # from scipy.linalg import lstsq

# local modules
from .EDM import EDM as EDMClass

#-----------------------------------------------------------
class SMap( EDMClass ):
    '''SMap class : child of EDM'''

    def __init__(self,
                 data       = None,
                 columns         = "",
                 target          = "",
                 train             = "",
                 test            = "",
                 embedDimensions = 0,
                 predictionHorizon              = 1,
                 knn             = 0,
                 step             = -1,
                 theta           = 0.,
                 exclusionRadius = 0,
                 solver          = None,
                 embedded        = False,
                 validLib        = [],
                 noTime          = False,
                 generateSteps   = 0,
                 generateConcat  = False,
                 ignoreNan       = True,
                 verbose         = False):
        '''Initialize SMap as child of EDM.
           Set data object to dataFrame.
           Setup : Validate(), CreateIndices(), get targetVec, time'''

        # Instantiate EDM class: inheret all members to self
        super(SMap, self).__init__(data, isEmbedded=False, name='SMap')

        # Assign parameters from API arguments
        self.columns         = columns
        self.target          = target
        self.train             = train
        self.test            = test
        self.embedDimensions = embedDimensions
        self.predictionHorizon              = predictionHorizon
        self.knn             = knn
        self.step             = step
        self.theta           = theta
        self.exclusionRadius = exclusionRadius
        self.solver          = solver
        self.embedded        = embedded
        self.validLib        = validLib
        self.noTime          = noTime
        self.generateSteps   = generateSteps
        self.generateConcat  = generateConcat
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # Map API parameter names to EDM base class names
        self.predictionHorizon = predictionHorizon
        self.embedStep         = step
        self.isEmbedded        = embedded

        # SMap storage
        self.Coefficients   = None # DataFrame SMap API output
        self.SingularValues = None # DataFrame SMap API output
        self.coefficients   = None # ndarray SMap output (N_pred, E+1)
        self.singularValues = None # ndarray SMap output (N_pred, E+1)

        # Setup
        self.Validate()      # EDM Method: set knn default, E if embedded
        self.CreateIndices() # Generate lib_i & pred_i, validLib

        self.targetVec = self.Data[:, [self.target[0]]]

        if self.noTime :
            # Generate a time/index vector, store as ndarray
            timeIndex = [ i for i in range( 1, self.Data.shape[0] + 1 ) ]
            self.time = array( timeIndex, dtype = int )
        else :
            # 1st data column is time
            self.time = self.Data[:, 0]

        if self.solver is None :
            self.solver = lstsq

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run( self ) :
    #-------------------------------------------------------------------
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighbors()
        self.Project()
        self.FormatProjection()

    #-------------------------------------------------------------------
    def Project( self ) :
    #-------------------------------------------------------------------
        '''For each prediction row compute projection as the linear
           combination of regression coefficients (C) of weighted
           embedding vectors (A) against target vector (B) : AC = B.

           Weights reflect the SMap theta localization of the knn
           for each prediction. Default knn = len( lib_i ). 

           Matrix A has (weighted) constant (1) first column
           to enable a linear intercept/bias term.

           Sugihara (1994) doi.org/10.1098/rsta.1994.0106
        '''

        if self.verbose:
            print( f'{self.name}: Project()' )

        N_pred = len(self.testIndices)
        N_dim  = self.embedDimensions + 1

        self.projection     = full( N_pred, nan, dtype = float )
        self.variance       = full( N_pred, nan, dtype = float )
        self.coefficients   = full( (N_pred, N_dim), nan, dtype = float )
        self.singularValues = full( (N_pred, N_dim), nan, dtype = float )

        embedding = self.Embedding # reference to ndarray

        # Compute average distance for knn test rows into a vector
        distRowMean = mean( self.knn_distances, axis = 1 )

        # Weight matrix of row vectors
        if self.theta == 0 :
            W = full( self.knn_distances.shape, 1., dtype = float )
        else :
            distRowScale = self.theta / distRowMean
            W = exp( -distRowScale[:,None] * self.knn_distances )

        # knn_neighbors + predictionHorizon
        knn_neighbors_Tp = self.knn_neighbors + self.predictionHorizon # N_pred x knn

        # Function to select targetVec for rows of Boundary condition matrix
        def GetTargetRow( knn_neighbor_row ) :
            return self.targetVec[ knn_neighbor_row ][:,0]

        # Boundary condition matrix of knn + predictionHorizon targets : N_pred x knn
        B = apply_along_axis( GetTargetRow, 1, knn_neighbors_Tp )

        if self.targetVecNan :
            # If there are nan in the targetVec need to remove them
            # from B since Solver returns nan. B_valid is matrix of
            # B row booleans of valid data for test rows
            # Function to apply isfinite to rows
            def FiniteRow( B_row ) :
                return isfinite( B_row )

            B_valid = apply_along_axis( FiniteRow, 1, B )

        # Weighted boundary condition matrix of targets : N_pred x knn
        wB = W * B

        # Process each prediction row
        for row in range( N_pred ) :
            # Allocate array
            A = full( ( self.knn, N_dim ), nan, dtype = float )

            A[:,0] = W[row,:] # Intercept bias terms in column 0 (weighted)

            libRows = self.knn_neighbors[ row, : ] # 1 x knn

            for j in range( 1, N_dim ) :
                A[ :, j ] = W[ row, : ] * embedding[ libRows, j-1 ]

            wB_ = wB[row,:]

            if self.targetVecNan :
                # Redefine A, wB_ to remove targetVec nan
                valid_i = B_valid[ row, : ]
                A       = A [ valid_i, : ]
                wB_     = wB[ row, valid_i ]

            # Linear mapping of theta weighted embedding A onto weighted target B
            C, SV = self.Solver( A, wB_ )

            self.coefficients  [ row, : ] = C
            self.singularValues[ row, : ] = SV

            # Prediction is local linear projection.
            if isnan( C[0] ) :
                projection_ = 0
            else :
                projection_ = C[0]

            for e in range( 1, N_dim ) :
                projection_ = projection_ + \
                    C[e] * embedding[ self.testIndices[ row], e - 1]

            self.projection[ row ] = projection_

            # "Variance" estimate assuming weights are probabilities
            if self.targetVecNan :
                deltaSqr = power( B[ row, valid_i ] - projection_, 2 )
                self.variance[ row ] = \
                    sum( W[ row, valid_i ] * deltaSqr ) \
                    / sum( W[ row, valid_i ] )
            else :
                deltaSqr = power( B[row,:] - projection_, 2 )
                self.variance[ row ] = sum(W[row]*deltaSqr) / sum(W[row])

    #-------------------------------------------------------------------
    def Solver( self, A, wB ) :
    #-------------------------------------------------------------------
        '''Call SMap solver. Default is numpy.lstsq'''

        if self.solver.__class__.__name__ in \
           [ 'function', '_ArrayFunctionDispatcher' ] and \
           self.solver.__name__ == 'lstsq' :
            # numpy default lstsq or scipy lstsq
            C, residuals, rank, SV = self.solver( A, wB, rcond = None )
            return C, SV

        # Otherwise, sklearn.linear_model passed as solver
        # Coefficient matrix A has weighted unity vector in the first
        # column to create a bias (intercept) term. sklearn.linear_model's
        # include an intercept term by default. Ignore first column of A.
        LM = self.solver.fit( A[:,1:], wB )
        C  = LM.coef_
        if hasattr( LM, 'intercept_' ) :
            C = insert( C, 0, LM.intercept_ ) 
        else :
            C = insert( C, 0, nan ) # Insert nan for intercept term

        if self.solver.__class__.__name__ == 'LinearRegression' :
            SV = LM.singular_ # Only LinearRegression has singular_
            SV = insert( SV, 0, nan )
        else :
            SV = None # full( A.shape[0], nan )

        return C, SV

    #-------------------------------------------------------------------
    def Generate( self ) :
    #-------------------------------------------------------------------
        '''SMap Generation
           Given train: override test to be single prediction at end of train
           Replace self.Projection with G.Projection

           Note: Generation with datetime time values fails: incompatible
                 numpy.datetime64, timedelta64 and python datetime, timedelta
        '''
        if self.verbose:
            print( f'{self.name}: Generate()' )

        # Local references for convenience
        N      = self.Data.shape[0]
        column = self.columns[0]
        target = self.target[0]
        train    = self.train

        # Override test for single prediction at end of train
        test = [ train[-1] - 1, train[-1] ]
        if self.verbose:
            print(f'{self.name}: Generate(): test overriden to {test}')

        # Output numpy arrays to replace self.Projection, self.Coefficients...
        nOutRows  = self.generateSteps

        # Projection array: shape (n_samples, 4)
        # Column 0: Time, Column 1: Observations, Column 2: Predictions, Column 3: Pred_Variance
        generated = full( (nOutRows, 4), nan )

        # Coefficients array: shape (n_samples, E+2)
        # Column 0: Time, Column 1: C0, Columns 2-E+1: coefficients
        genCoeff = full((nOutRows, self.embedDimensions + 2), nan)

        # SingularValues array: shape (n_samples, E+2)
        # Column 0: Time, Columns 1-E+1: singular values
        genSV = full((nOutRows, self.embedDimensions + 2), nan)

        # Allocate vector for univariate column data
        # At each iteration the prediction is stored in columnData
        # timeData and columnData are copied to newData for next iteration
        columnData     = full( N + nOutRows, nan )
        columnData[:N] = self.Data[:, column] # First col only

        # Allocate output time vector & newData DataFrame
        timeData = full( N + nOutRows, nan )
        if self.noTime :
            # If noTime create a time vector and join into self.Data
            timeData[:N] = linspace( 1, N, N )
            # Create new data array with time column
            newData = column_stack([timeData[:N], columnData[:N]])
        else :
            timeData[:N] = self.time # Presume column 0 is time
            # Create new data array with time column
            newData = column_stack([timeData[:N], columnData[:N]])

        #-------------------------------------------------------------------
        # Loop for each feedback generation step
        #-------------------------------------------------------------------
        for step in range( self.generateSteps ) :
            if self.verbose :
                print( f'{self.name}: Generate(): step {step} {"="*50}')

            # Local SMapClass for generation
            G = SMap(data = newData,
                     columns         = column,
                     target          = target,
                     train             = train,
                     test            = test,
                     embedDimensions = self.embedDimensions,
                     predictionHorizon              = self.predictionHorizon,
                     knn             = self.knn,
                     step             = self.step,
                     theta           = self.theta,
                     exclusionRadius = self.exclusionRadius,
                     solver          = self.solver,
                     embedded        = self.embedded,
                     validLib        = self.validLib,
                     noTime          = self.noTime,
                     generateSteps   = self.generateSteps,
                     generateConcat  = self.generateConcat,
                     ignoreNan       = self.ignoreNan,
                     verbose         = self.verbose)

            # 1) Generate prediction ----------------------------------
            G.Run()

            if self.verbose :
                print( 'G.Projection' )
                print( G.Projection ); print()

            newPrediction = G.Projection[:, 2]  # Column 2 is Predictions
            newTime       = G.Projection[-1, 0]  # Column 0 is time

            # 2) Save prediction in generated --------------------------
            generated[step, 0] = newTime
            generated[step, 1] = nan  # Observations (not applicable for generation)
            generated[step, 2] = newPrediction
            generated[step, 3] = nan  # Pred_Variance (not applicable for generation)

            # Save coefficients and singular values
            genCoeff[step, 0] = newTime
            genCoeff[step, 1:] = G.Coefficients[-1, 1:]  # Skip time column
            genSV[step, 0] = newTime
            genSV[step, 1:] = G.SingularValues[-1, 1:]  # Skip time column

            if self.verbose :
                print( f'2) generated step {step}' )

            # 3) Increment library by adding another row index ---------
            # Dynamic library not implemented

            # 4) Increment prediction indices --------------------------
            test = [ p + 1 for p in test ]

            if self.verbose:
                print(f'4) test {test}')

            # 5) Add 1-step ahead projection to newData for next Project()
            columnData[ N + step ] = newPrediction
            timeData  [ N + step ] = newTime

            # JP : for big data this is likely not efficient
            newData = column_stack([timeData[:(N + step + 1)],
                                   columnData[:(N + step + 1)]])

            if self.verbose:
                print(f'5) newData: {newData.shape}')
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Loop for each feedback generation step
        #----------------------------------------------------------

        # Replace self.Projection with generated
        if self.generateConcat :
            # Concatenate original data observations with generated predictions
            # Original data: columns 0 (time), 1 (observations)
            # Generated: columns 0 (time), 1 (obs), 2 (test), 3 (var)
            # Result: columns 0 (time), 1 (obs), 2 (test), 3 (var)
            timeName = 0  # Column 0 is time
            data_obs = column_stack([self.Data[:, timeName], self.Data[:, column]])
            self.Projection = column_stack([data_obs, generated[:, 2:4]])

        else :
            self.Projection = generated

        self.Coefficients   = genCoeff
        self.SingularValues = genSV
