import unittest
from datetime import datetime
from warnings import filterwarnings, catch_warnings

from numpy import nan, array, array_equal, ndarray
import numpy
from pandas import DataFrame, read_csv

from pyEDM import Functions as EDM
import pyEDM.EDM.Embed
from pyEDM.ExampleData import dataFileNames
import importlib.resources

# because the example data are now numpy arrays but these tests read in dataframes
sampleDataFrames = {}
for fileName, dataName in dataFileNames:

    filePath = "data/" + fileName

    ref = importlib.resources.files('pyEDM') / filePath

    with importlib.resources.as_file( ref ) as filePath_ :
        # Read CSV using pandas, convert to numpy array
        df = read_csv( filePath_ )
        sampleDataFrames[ dataName ] = df


#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_EDM( unittest.TestCase ):
    """The examples.py and smapSolverTest.py must also run.

    NOTE: Bizarre default of unittest class presumes
          methods names to be run begin with "test_" 
    """
    # JP How to pass command line arg to class? verbose = True
    def __init__( self, *args, **kwargs):
        super( test_EDM, self ).__init__( *args, **kwargs )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    @classmethod
    def setUpClass( self ):
        self.verbose = False
        self.GetValidFiles( self )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    def GetValidFiles( self ):
        """Create dictionary of DataFrame values from file name keys"""
        self.ValidationFiles = {}

        validFiles = [ 'CCM_anch_sst_valid.csv',
                       'CCM_Lorenz5D_MV_Space_valid.csv',
                       'CCM_Lorenz5D_MV_valid.csv',
                       'CCM_nan_valid.csv',
                       'EmbedDim_valid.csv',
                       'Multiview_combos_valid.csv',
                       'Multiview_pred_valid.csv',
                       'PredictInterval_valid.csv',
                       'PredictNonlinear_valid.csv',
                       'SMap_circle_E2_embd_valid.csv',
                       'SMap_circle_E4_valid.csv',
                       'SMap_nan_valid.csv',
                       'SMap_noTime_valid.csv',
                       'Smplx_DateTime_valid.csv',
                       'Smplx_disjointLib_valid.csv',
                       'Smplx_disjointPred_nan_valid.csv',
                       'Smplx_E3_block_3sp_valid.csv',
                       'Smplx_E3_embd_block_3sp_valid.csv',
                       'Smplx_exclRadius_valid.csv',
                       'Smplx_nan2_valid.csv',
                       'Smplx_nan_valid.csv',
                       'Smplx_negTp_block_3sp_valid.csv',
                       'Smplx_validLib_valid.csv' ]

        # Create map of module validFiles pathnames in validFiles
        for file in validFiles:
            filename = "validation/" + file
            self.ValidationFiles[ file] = read_csv(filename)

    #------------------------------------------------------------
    # API
    #------------------------------------------------------------
    def test_API_1( self ):
        """API 1"""
        if self.verbose : print ( " --- API 1 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_2( self ):
        """API 2"""
        if self.verbose : print ( "--- API 2 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_3( self ):
        """API 3"""
        if self.verbose : print ( "--- API 3 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index1 = df_.columns.get_loc('V1')
        col_index2 = df_.columns.get_loc('V3')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index1, col_index2], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_4( self ):
        """API 4"""
        if self.verbose : print ( "--- API 4 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index1 = df_.columns.get_loc('V1')
        col_index2 = df_.columns.get_loc('V3')
        target_index1 = df_.columns.get_loc('V5')
        target_index2 = df_.columns.get_loc('V2')
        df  = EDM.FitSimplex(data = data,
                             columns = [col_index1, col_index2], target = [target_index1, target_index2],
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_5( self ):
        """API 5"""
        if self.verbose : print ( "--- API 5 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5, knn = 0)

    def test_API_6( self ):
        """API 6"""
        if self.verbose : print ( "--- API 6 ---" )
        df_ = sampleDataFrames['Lorenz5D']
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V5')
        df  = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                             train = [1, 300], test = [301, 310], embedDimensions = 5, step = -2)

    def test_API_7( self ):
        """API 7"""
        if self.verbose : print ( "--- API 7 Column names with space ---" )
        df_ = sampleDataFrames["columnNameSpace"]
        data = df_.values
        col_index1 = df_.columns.get_loc('Var 1')
        col_index2 = df_.columns.get_loc('Var 2')
        target_index = df_.columns.get_loc('Var 5 1')
        df = EDM.FitSimplex(data = data, columns = [col_index1, col_index2], target = target_index,
                            train = [1, 80], test = [81, 85], embedDimensions = 5, predictionHorizon = 1,
                            knn = 0, step = -1, exclusionRadius = 0,
                            embedded = False, validLib = [], noTime = False, generateSteps = 0,
                            generateConcat = False, verbose = False, ignoreNan = True, returnObject = False)

    #------------------------------------------------------------
    # Embed
    #------------------------------------------------------------
    def test_embed( self ):
        """Embed"""
        if self.verbose : print ( "--- Embed ---" )
        df_ = sampleDataFrames['circle']
        x = df_.columns.get_loc('x')
        df  = pyEDM.EDM.Embed.Embed(df_.values, [x], 3, -1,  False)

    def test_embed2( self ):
        """Embed multivariate"""
        if self.verbose : print ( "--- Embed multivariate ---" )
        df_ = sampleDataFrames['circle']
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df  = pyEDM.EDM.Embed.Embed(df_.values, [x, y], 3, -1,  False)

    def test_embed3( self ):
        """Embed multivariate"""
        if self.verbose : print ( "--- Embed includeTime ---" )
        df_ = sampleDataFrames['circle']
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df  = pyEDM.EDM.Embed.Embed(df_.values, [x, y], 3, -1,  True)

    #------------------------------------------------------------
    # Simplex
    #------------------------------------------------------------
    def test_simplex( self ):
        """embedded = False"""
        if self.verbose : print ( "--- Simplex embedded = False ---" )
        df_ = sampleDataFrames["block_3sp"]
        col_index = df_.columns.get_loc('x_t')
        target_index = df_.columns.get_loc('x_t')
        df = EDM.FitSimplex(df_.values, [col_index], [target_index],
                            [1, 100], [101, 195], 3, 1, 0, -1, 0,
                            False, [], False, 0, False, False, False)

        dfv = self.ValidationFiles["Smplx_E3_block_3sp_valid.csv"]

        S1 = dfv.get('Predictions')[1:95].to_numpy() # Skip row 0 Nan
        S2 = df[1:95, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex2( self ):
        """embedded = True"""
        if self.verbose : print ( "--- Simplex embedded = True ---" )
        df_ = sampleDataFrames["block_3sp"]

        x = df_.columns.get_loc('x_t')
        y = df_.columns.get_loc('y_t')
        z = df_.columns.get_loc('z_t')
        df = EDM.FitSimplex(df_.values, [x, y, z], [x],
                            [1, 99], [100, 198], 3, 1, 0, -1, 0,
                            True, [], False, 0, False, False, False)

        dfv = self.ValidationFiles["Smplx_E3_embd_block_3sp_valid.csv"]

        S1 = dfv.get('Predictions')[1:98].to_numpy() # Skip row 0 Nan
        S2 = df[1:98, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex3( self ):
        """negative predictionHorizon"""
        if self.verbose : print ( "--- negative predictionHorizon ---" )
        df_ = sampleDataFrames["block_3sp"]
        x = df_.columns.get_loc('x_t')
        y = df_.columns.get_loc('y_t')
        z = df_.columns.get_loc('z_t')
        df = EDM.FitSimplex(df_.values, [x], [y],
                            [1, 100], [50, 80], 3, -2, 0, -1, 0,
                            False, [], False, 0, False, False, False)

        dfv = self.ValidationFiles["Smplx_negTp_block_3sp_valid.csv"]

        S1 = dfv.get('Predictions')[1:98].to_numpy() # Skip row 0 Nan
        S2 = df[1:98, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex4( self ):
        """validLib"""
        if self.verbose : print ( "--- validLib ---" )
        df_ = sampleDataFrames["circle"]
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df = EDM.FitSimplex(data = df_.values, columns = [x], target = [y],
                            train = [1,200], test = [1,200], embedDimensions = 2, predictionHorizon = 1,
                            validLib = df_.eval('x > 0.5 | x < -0.5'))

        dfv = self.ValidationFiles["Smplx_validLib_valid.csv"]

        S1 = dfv.get('Predictions')[1:195].to_numpy() # Skip row 0 Nan
        S2 = df[1:195, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex5( self ):
        """disjoint train"""
        if self.verbose : print ( "--- disjoint train ---" )
        df_ = sampleDataFrames["circle"]

        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df = EDM.FitSimplex(data = df_.values, columns = [x], target = [x],
                            train = [1,40, 50,130], test = [80,170],
                            embedDimensions = 2, predictionHorizon = 1, step = -3)

        dfv = self.ValidationFiles["Smplx_disjointLib_valid.csv"]

        S1 = dfv.get('Predictions')[1:195].to_numpy() # Skip row 0 Nan
        S2 = df[1:195, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex6( self ):
        """disjoint test w/ nan"""
        if self.verbose : print ( "--- disjoint test w/ nan ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        df_.iloc[ [8,50,501], [1,2] ] = nan
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V1')
        df = EDM.FitSimplex(data = df_.values, columns= [col_index], target = [target_index],
                            embedDimensions = 5, predictionHorizon = 2, train = [1,50,101,200,251,500],
                            test = [1,10,151,155,551,555,881,885,991,1000])

        dfv = self.ValidationFiles["Smplx_disjointPred_nan_valid.csv"]

        S1 = dfv.get('Predictions')[1:195].to_numpy() # Skip row 0 Nan
        S2 = df[1:195, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-5))

    #------------------------------------------------------------
    def test_simplex7( self ):
        """exclusion radius"""
        if self.verbose : print ( "--- exclusion radius ---" )
        df_ = sampleDataFrames["circle"]

        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')
        df = EDM.FitSimplex(data = df_.values, columns = [x], target = y,
                            train = [1,100], test = [21,81], embedDimensions = 2, predictionHorizon = 1,
                            exclusionRadius = 5)

        dfv = self.ValidationFiles["Smplx_exclRadius_valid.csv"]

        S1 = dfv.get('Predictions')[1:60].to_numpy() # Skip row 0 Nan
        S2 = df[1:60, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex8( self ):
        """nan"""
        if self.verbose : print ( "--- nan ---" )
        df_ = sampleDataFrames["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')

        df = EDM.FitSimplex(dfn.values, columns = [x], target = [y],
                            train = [1,100], test = [1,95], embedDimensions = 2, predictionHorizon = 1)

        dfv = self.ValidationFiles["Smplx_nan_valid.csv"]

        S1 = dfv.get('Predictions')[1:90].to_numpy() # Skip row 0 Nan
        S2 = df[1:90, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex9( self ):
        """nan"""
        if self.verbose : print ( "--- nan ---" )
        df_ = sampleDataFrames["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan
        x = df_.columns.get_loc('x')
        y = df_.columns.get_loc('y')

        df = EDM.FitSimplex(dfn.values, columns = [y], target = [x],
                            train = [1,200], test = [1,195], embedDimensions = 2, predictionHorizon = 1)

        dfv = self.ValidationFiles["Smplx_nan2_valid.csv"]

        S1 = dfv.get('Predictions')[1:190].to_numpy() # Skip row 0 Nan
        S2 = df[1:190, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex10( self ):
        """DateTime"""
        if self.verbose : print ( "--- DateTime ---" )
        df_ = sampleDataFrames["SumFlow_1980-2005"]
        col_index = df_.columns.get_loc('S12.C.D.S333')
        target_index = df_.columns.get_loc('S12.C.D.S333')
        df = EDM.FitSimplex(data = df_.values,
                            columns = [col_index], target = [target_index],
                            train = [1,800], test = [801,1001], embedDimensions = 3, predictionHorizon = 1)

        self.assertTrue( isinstance( df['Time'][0],  datetime ) )

        dfv = self.ValidationFiles["Smplx_DateTime_valid.csv"]

        S1 = dfv.get('Predictions')[1:200].to_numpy() # Skip row 0 Nan
        S2 = df[1:200, 2] # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_simplex11( self ):
        """knn = 1"""
        if self.verbose : print ( "--- knn = 1 ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        col_index = df_.columns.get_loc('V5')
        target_index = df_.columns.get_loc('V5')

        df = EDM.FitSimplex(data = df_.values, columns= [col_index], target = target_index,
                            train = [301,400], test = [350,355],
                            knn = 1, embedded = True, returnObject = True)

        knn = df.knn_neighbors
        knnValid = array( [322,334,362,387,356,355] )[:,None]
        self.assertTrue( array_equal( knn, knnValid ) )

    #------------------------------------------------------------
    def test_simplex12( self ):
        """exclusion Radius """
        if self.verbose : print ( "--- exclusion Radius ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        x   = [i+1 for i in range(1000)]
        df_ = DataFrame({'Time':df_['Time'],'X':x,'V1':df_['V1']})
        x = df_.columns.get_loc('X')
        V1 = df_.columns.get_loc('V1')

        df = EDM.FitSimplex(data = df_.values, columns= [x], target = [V1],
                            train = [1,100], test = [101,110],
                            embedDimensions = 5, exclusionRadius = 10, returnObject = True)

        knn = df.knn_neighbors[:,0]
        knnValid = array( [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] )
        self.assertTrue( array_equal( knn, knnValid ) )

    #------------------------------------------------------------
    # S-map
    #------------------------------------------------------------
    def test_smap( self ):
        """SMap"""
        if self.verbose : print ( "--- SMap ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('x')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index,
                        train = [1,100], test = [110,160], embedDimensions = 4, predictionHorizon = 1,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        solver = None, embedded = False, validLib = [], noTime = False, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv = self.ValidationFiles["SMap_circle_E4_valid.csv"]
        df  = S['predictions']

        S1 = dfv.get('Predictions')[1:50].to_numpy()  # Skip row 0 Nan
        S2 = df[1:50, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_smap2( self ):
        """SMap embedded = True"""
        if self.verbose : print ( "--- SMap embedded = True ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index1 = df_.columns.get_loc('x')
        col_index2 = df_.columns.get_loc('y')
        target_index = df_.columns.get_loc('x')
        S = EDM.FitSMap(data = data, columns = [col_index1, col_index2], target = target_index,
                        train = [1,200], test = [1,200], embedDimensions = 2, predictionHorizon = 1,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        solver = None, embedded = True, validLib = [], noTime = False, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv  = self.ValidationFiles["SMap_circle_E2_embd_valid.csv"]

        df  = S['predictions']
        dfc = S['coefficients']

        S1 = dfv.get('Predictions')[1:195].to_numpy()  # Skip row 0 Nan
        S2 = df[1:195, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

        # self.assertTrue( dfc['∂x/∂x'].mean().round(5) == 0.99801 )
        # self.assertTrue( dfc['∂x/∂y'].mean().round(5) == 0.06311 )
        # TODO: check if these actually are correct for getting those value
        self.assertTrue( dfc[:, 0].mean().round(5) == 0.99801 )
        self.assertTrue( dfc[:, 1].mean().round(5) == 0.06311 )

    #------------------------------------------------------------
    def test_smap3( self ):
        """SMap nan"""
        if self.verbose : print ( "--- SMap nan ---" )
        df_ = sampleDataFrames["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan
        data = dfn.values
        col_index = dfn.columns.get_loc('x')
        target_index = dfn.columns.get_loc('y')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index,
                        train = [1,50], test = [1,50], embedDimensions = 2, predictionHorizon = 1,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        solver = None, embedded = False, validLib = [], noTime = False, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv = self.ValidationFiles["SMap_nan_valid.csv"]
        df  = S['predictions']

        S1 = dfv.get('Predictions')[1:50].to_numpy()  # Skip row 0 Nan
        S2 = df[1:150, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    def test_smap4( self ):
        """DateTime"""
        if self.verbose : print ( "--- noTime ---" )
        df_ = sampleDataFrames["circle_noTime"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('y')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index,
                        train = [1,100], test = [101,150], embedDimensions = 2,
                        knn = 0, step = -1, theta = 3., exclusionRadius = 0,
                        solver = None, embedded = False, validLib = [], noTime = True, generateSteps = 0,
                        generateConcat = False, ignoreNan = True, verbose = False, returnObject = False)

        dfv = self.ValidationFiles["SMap_noTime_valid.csv"]
        df  = S['predictions']

        S1 = dfv.get('Predictions')[1:50].to_numpy()  # Skip row 0 Nan
        S2 = df[1:50, 2]  # Skip row 0 Nan
        self.assertTrue(numpy.allclose(S1, S2, atol = 1e-6))

    #------------------------------------------------------------
    # CCM
    #------------------------------------------------------------
    def test_ccm( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM ---" )
            df_ = sampleDataFrames['sardine_anchovy_sst']
            data = df_.values
            col_index = df_.columns.get_loc('anchovy')
            target_index = df_.columns.get_loc('np_sst')
            df = EDM.FitCCM(data = data, columns = [col_index], target = target_index,
                            trainSizes = [10, 20, 30, 40, 50, 60, 70, 75], sample = 100,
                            embedDimensions = 3, predictionHorizon = 0, knn = 0, step = -1, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = self.ValidationFiles["CCM_anch_sst_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-2))

    #------------------------------------------------------------
    def test_ccm2( self ):
        """CCM Multivariate"""
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM multivariate ---" )
            df_ = sampleDataFrames['Lorenz5D']
            data = df_.values
            col_index1 = df_.columns.get_loc('V3')
            col_index2 = df_.columns.get_loc('V5')
            target_index = df_.columns.get_loc('V1')
            df = EDM.FitCCM(data = data, columns = [col_index1, col_index2], target = target_index,
                            trainSizes = [20, 200, 500, 950], sample = 30, embedDimensions = 5,
                            predictionHorizon = 10, knn = 0, step = -5, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = self.ValidationFiles["CCM_Lorenz5D_MV_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, rtol = 1e-4))

    #------------------------------------------------------------
    def test_ccm3( self ):
        """CCM nan"""
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM nan ---" )
            df_ = sampleDataFrames['circle']
            dfn = df_.copy()
            dfn.iloc[ [5,6,12], 1 ] = nan
            dfn.iloc[ [10,11,17], 2 ] = nan
            data = dfn.values
            col_index = dfn.columns.get_loc('x')
            target_index = dfn.columns.get_loc('y')
            df = EDM.FitCCM(data = data, columns = [col_index], target = target_index,
                            trainSizes = [10, 190, 10], sample = 20, embedDimensions = 2,
                            predictionHorizon = 5, knn = 0, step = -1, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = self.ValidationFiles["CCM_nan_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, rtol = 1e-4))

    #------------------------------------------------------------
    def test_ccm4( self ):
        """CCM Multivariate names with spaces"""

        # TODO: I don't understand the point of multivariate CCM and need to look in the main repo
        self.assertTrue(False)

        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM multivariate name spaces ---" )
            df_ = sampleDataFrames['columnNameSpace']
            data = df_.values
            col_index1 = df_.columns.get_loc('Var 1')
            col_index2 = df_.columns.get_loc('Var3')
            col_index3 = df_.columns.get_loc('Var 5 1')
            target_index1 = df_.columns.get_loc('Var 2')
            target_index2 = df_.columns.get_loc('Var 4 A')
            df = EDM.FitCCM(data = data, columns = [col_index1, col_index2, col_index3],
                            target = [target_index1, target_index2],
                            trainSizes = [20, 50, 90], sample = 1,
                            embedDimensions = 5, predictionHorizon = 0, knn = 0, step = -1, seed = 777,
                            embedded = False, validLib = [], includeData = False, noTime = False,
                            ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                            returnObject = False)

        dfv = round(self.ValidationFiles["CCM_Lorenz5D_MV_Space_valid.csv"], 4)

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    # Multiview
    #------------------------------------------------------------
    def test_multiview( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- Multiview ---" )
            df_ = sampleDataFrames['block_3sp']
            data = df_.values
            col_index1 = df_.columns.get_loc('x_t')
            col_index2 = df_.columns.get_loc('y_t')
            col_index3 = df_.columns.get_loc('z_t')
            target_index = df_.columns.get_loc('x_t')
            M = EDM.FitMultiview(data = data,
                                 columns = [col_index1, col_index2, col_index3], target = target_index,
                                 train = [1, 100], test = [101, 198],
                                 D = 0, embedDimensions = 3, predictionHorizon = 1, knn = 0, step = -1,
                                 multiview = 0, exclusionRadius = 0,
                                 trainLib = False, excludeTarget = False,
                                 ignoreNan = True, verbose = False,
                                 numProcess = 4, mpMethod = None, chunksize = 1,
                                 returnObject = False)

        df_pred  = M['Predictions']
        df_combo = M['View'][ ['correlation', 'MAE', 'RMSE'] ]

        # Validate predictions
        dfvp      = self.ValidationFiles["Multiview_pred_valid.csv"]
        predValid = round( dfvp.get('Predictions'), 4 )
        test      = round( df_pred.get('Predictions'), 4 )
        self.assertTrue( predValid.equals( test ) )

        # Validate combinations
        dfvc = round(self.ValidationFiles['Multiview_combos_valid.csv'], 4)
        dfvc = dfvc[ ['correlation', 'MAE', 'RMSE'] ]

        self.assertTrue( dfvc.equals( round( df_combo, 4 ) ) )

    #------------------------------------------------------------
    # EmbedDimension
    #------------------------------------------------------------
    def test_embedDimension( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- EmbedDimension ---" )
            df_ = sampleDataFrames['Lorenz5D']
            data = df_.values
            col_index = df_.columns.get_loc('V1')
            target_index = df_.columns.get_loc('V1')
            df = EDM.FindOptimalEmbeddingDimensionality(data = data, columns = [col_index], target = target_index,
                                                        maxE = 12, train = [1, 500], test=[501, 800],
                                                        predictionHorizon = 15, step = -5, exclusionRadius = 20,
                                                        embedded = False, validLib = [], noTime = False,
                                                        ignoreNan = True, numProcess = 10, mpMethod = None,
                                                        chunksize = 1)

        dfv = self.ValidationFiles["EmbedDim_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-6))

    #------------------------------------------------------------
    # PredictInterval
    #------------------------------------------------------------
    def test_PredictInterval( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- PredictInterval ---" )
            df_ = sampleDataFrames['block_3sp']
            data = df_.values
            col_index = df_.columns.get_loc('x_t')
            target_index = df_.columns.get_loc('x_t')
            df = EDM.FindOptimalPredictionHorizon(data = data, columns = [col_index], target = target_index,
                                                  maxTp = 15, train = [1, 150], test = [151, 198],
                                                  embedDimensions = 3, step = -1,
                                                  embedded = False, validLib = [], noTime = False,
                                                  ignoreNan = True, numProcess = 10, mpMethod = None,
                                                  chunksize = 1)

        dfv = self.ValidationFiles["PredictInterval_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-6))

    #------------------------------------------------------------
    # PredictNonlinear
    #------------------------------------------------------------
    def test_PredictNonlinear( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- Predict ---" )
            df_ = sampleDataFrames['TentMapNoise']
            data = df_.values
            col_index = df_.columns.get_loc('TentMap')
            target_index = df_.columns.get_loc('TentMap')
            df = EDM.FindSMapNeighborhood(data = data, columns = [col_index], target = target_index,
                                          theta = [0.01,0.1,0.3,0.5,0.75,1,1.5,
                                                2,3,4,5,6,7,8,9,10,15,20 ],
                                          train = [1, 500], test = [501,800], embedDimensions = 4,
                                          predictionHorizon = 1, knn = 0, step = -1,
                                          solver = None, embedded = False, validLib = [], noTime = False,
                                          ignoreNan = True, numProcess = 10, mpMethod = None,
                                          chunksize = 1)

        dfv = self.ValidationFiles["PredictNonlinear_valid.csv"].values

        self.assertTrue(numpy.allclose(df, dfv, atol = 1e-6))

    #------------------------------------------------------------
    # Generative mode
    #------------------------------------------------------------
    def test_generate__simplex1( self ):
        """Simplex Generate 1"""
        if self.verbose : print ( "--- Simplex Generate 1 ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('x')
        df = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                            train = [1,200], test = [1,2], embedDimensions = 2, predictionHorizon = 1,
                            knn = 0, step = -1, exclusionRadius = 0,
                            embedded = False, validLib = [], noTime = False, generateSteps = 100,
                            generateConcat = True, verbose = False, ignoreNan = True, returnObject = False)

        self.assertTrue( df.shape == (300,4) )

    #------------------------------------------------------------
    def test_generate_simplex2( self ):
        """Simplex generateSteps 2"""
        if self.verbose : print ( "--- Simplex generateSteps 2 ---" )
        df_ = sampleDataFrames["Lorenz5D"]
        data = df_.values
        col_index = df_.columns.get_loc('V1')
        target_index = df_.columns.get_loc('V1')
        df = EDM.FitSimplex(data = data, columns = [col_index], target = target_index,
                            train = [1, 1000], test = [1, 2], embedDimensions = 5, predictionHorizon = 1,
                            knn = 0, step = -1, exclusionRadius = 0,
                            embedded = False, validLib = [], noTime = False, generateSteps = 100,
                            generateConcat = False, verbose = False, ignoreNan = True, returnObject = False)

        self.assertTrue( df.shape == (100,4) )

    #------------------------------------------------------------
    def test_generate_smap1( self ):
        """DateTime"""
        if self.verbose : print ( "--- SMap Generate ---" )
        df_ = sampleDataFrames["circle"]
        data = df_.values
        col_index = df_.columns.get_loc('x')
        target_index = df_.columns.get_loc('x')
        S = EDM.FitSMap(data = data, columns = [col_index], target = target_index, theta = 3.,
                        train = [1,200], test = [1,2], embedDimensions = 2, predictionHorizon = 1,
                        knn = 0, step = -1, exclusionRadius = 0,
                        solver = None, embedded = False, validLib = [], noTime = False, generateSteps = 100,
                        generateConcat = True, ignoreNan = True, verbose = False, returnObject = False)

        self.assertTrue( S['predictions'].shape == (300,4) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
