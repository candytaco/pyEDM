
# python modules

# package modules
from numpy import array, divide, exp, fmax, full, nan
from numpy import linspace, power, subtract, sum, zeros, column_stack

# local modules
from .EDM import EDM
from .Results import SimplexResult

#-----------------------------------------------------------
class Simplex(EDM):
    """
    Simplex class : child of EDM
    CCM & Multiview are composed of Simplex instances
    TODO: Neighbor ties
    """

    def __init__(self,
                 data,
                 columns=None,
                 target=None,
                 train=None,
                 test=None,
                 embedDimensions=0,
                 predictionHorizon=1,
                 knn=0,
                 step=-1,
                 exclusionRadius=0,
                 embedded=False,
                 validLib=None,
                 noTime=False,
                 ignoreNan=True,
                 verbose=False,
                 generateSteps=0,
                 generateConcat=False):
        """
        Initialize Simplex as child of EDM.

        :param data: 2D numpy array where column 0 is time (unless noTime=True)
        :param columns: Column indices to use for embedding (defaults to all except time)
        :param target: Target column index (defaults to column 1)
        :param train: Training set indices [start, end]
        :param test: Test set indices [start, end]
        :param embedDimensions: Embedding dimension (E). If 0, will be set by Validate()
        :param predictionHorizon: Prediction time horizon (Tp)
        :param knn: Number of nearest neighbors. If 0, will be set to E+1 by Validate()
        :param step: Time delay step size (tau). Negative values indicate lag
        :param exclusionRadius: Temporal exclusion radius for neighbors
        :param embedded: Whether data is already embedded
        :param validLib: Boolean mask for valid library points
        :param noTime: Whether first column is time or data
        :param ignoreNan: Remove NaN values from embedding
        :param verbose: Print diagnostic messages
        :param generateSteps: Number of iterative generation steps. If 0, uses standard prediction.
        :param generateConcat: Whether to concatenate generated predictions
        """

        # Instantiate EDM class: inheret EDM members to self
        super(Simplex, self).__init__(data, isEmbedded=False, name='Simplex')

        self.columns         = columns
        self.target          = target
        self.embedDimensions = embedDimensions
        self.predictionHorizon = predictionHorizon
        self.knn             = knn
        self.step            = step
        self.exclusionRadius = exclusionRadius
        self.embedded        = embedded
        self.validLib        = validLib if validLib is not None else []
        self.noTime          = noTime
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # Assign split parameters
        self.train = train if train is not None else []
        self.test = test if test is not None else []

        # Assign generation parameters
        self.generateSteps = generateSteps
        self.generateConcat = generateConcat

        # Map API parameter names to EDM base class names
        self.embedStep         = self.step
        self.isEmbedded        = self.embedded

        # Setup
        self.Validate()      # EDM Method: set knn default, E if embedded
        self.CreateIndices() # Generate lib_i & pred_i, validLib : EDM Method

        self.targetVec = self.Data[:, [self.target[0]]]

        if self.noTime :
            # Generate a time/index vector, store as ndarray
            timeIndex = [ i for i in range( 1, self.Data.shape[0] + 1 ) ]
            self.time = array( timeIndex, dtype = int )
        else :
            # 1st data column is time
            self.time = self.Data[:, 0]

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run( self ):
    #-------------------------------------------------------------------
        """
        Execute standard prediction and return SimplexResult.

        :return: Prediction results with projection array and metadata
        """
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighbors()
        self.Project()
        self.FormatProjection()

        return SimplexResult(
            projection=self.Projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    #-------------------------------------------------------------------
    def Project( self ) :
    #-------------------------------------------------------------------
        """
        Simplex Projection
        Sugihara & May (1990) doi.org/10.1038/344734a0
        """
        if self.verbose:
            print( f'{self.name}: Project()' )

        # First column of knn_distances is minimum distance of all N test rows
        minDistances = self.knn_distances[:,0]
        # In case there is 0 in minDistances: minWeight = 1E-6
        minDistances = fmax( minDistances, 1E-6 )

        # Divide each column of the N x k knn_distances matrix by N row
        # column vector minDistances
        scaledDistances = divide( self.knn_distances, minDistances[:,None] )

        weights      = exp( -scaledDistances )  # N x k
        weightRowSum = sum( weights, axis = 1 ) # N x 1

        # Matrix of knn_neighbors + predictionHorizon defines library target values
        knn_neighbors_Tp = self.knn_neighbors + self.predictionHorizon     # N x k
        libTargetValues = self.targetVec[knn_neighbors_Tp].squeeze()


        # Projection is average of weighted knn library target values
        self.projection = sum(weights * libTargetValues, axis=1) / weightRowSum

        # "Variance" estimate assuming weights are probabilities
        libTargetPredDiff = subtract( libTargetValues, self.projection[:,None] )
        deltaSqr          = power( libTargetPredDiff, 2 )
        self.variance     = sum( weights * deltaSqr, axis = 1 ) / weightRowSum

    #-------------------------------------------------------------------
    def Generate( self ) :
    #-------------------------------------------------------------------
        """
        Simplex Generation
        Given train: override test for single prediction at end of train
        Replace self.Projection with G.Projection

        Note: Generation with datetime time values fails: incompatible
        numpy.datetime64, timedelta64 and python datetime, timedelta
        """
        if self.verbose:
            print( f'{self.name}: Generate()' )

        # Local references for convenience
        N      = self.Data.shape[0]
        column = self.columns[0]
        target = self.target[0]
        train    = self.train

        if self.verbose:
            print(f'\tData shape: {self.Data.shape}')
            print(f'\ttrain: {train}')

        # Override test for single prediction at end of train
        test = [ train[-1] - 1, train[-1] ]
        if self.verbose:
            print(f'\tGenerate(): test overriden to {test}')

        # Output numpy array to replace self.Projection
        # Shape: (n_samples, 4)
        # Column 0: Time
        # Column 1: Observations
        # Column 2: Predictions
        # Column 3: Pred_Variance
        nOutRows  = self.generateSteps
        generated = full( (nOutRows, 4), nan )

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
            timeData[:N] = self.time
            # Create new data array with time column
            newData = column_stack([timeData[:N], columnData[:N]])

        #-------------------------------------------------------------------
        # Loop for each feedback generation step
        #-------------------------------------------------------------------
        for step in range( self.generateSteps ) :
            if self.verbose :
                print( f'{self.name}: Generate(): step {step} {"="*50}')

            # Local SimplexClass for generation
            G = Simplex(data = newData,
                        columns         = [column],
                        target          = target,
                        train             = train,
                        test            = test,
                        embedDimensions = self.embedDimensions,
                        predictionHorizon              = self.predictionHorizon,
                        knn             = self.knn,
                        step             = self.step,
                        exclusionRadius = self.exclusionRadius,
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
                print( '1) G.Projection' )
                print( G.Projection ); print()

            newPrediction = G.Projection[-1, 2]  # Column 2 is Predictions
            newTime       = G.Projection[-1, 0]  # Column 0 is time

            # 2) Save prediction in generated --------------------------
            generated[step, 0] = newTime
            generated[step, 1] = nan  # Observations (not applicable for generation)
            generated[step, 2] = newPrediction
            generated[step, 3] = nan  # Pred_Variance (not applicable for generation)

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

        # Replace self.Projection with Generated
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

        return SimplexResult(
            projection=self.Projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )
