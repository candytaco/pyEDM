import unittest
from datetime import datetime
from warnings import filterwarnings, catch_warnings

from numpy import nan, array, array_equal
from pandas import DataFrame, read_csv

import pyEDM as EDM
import pyEDM.Embed


#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_EDM( unittest.TestCase ):
    '''The examples.py and smapSolverTest.py must also run.

    NOTE: Bizarre default of unittest class presumes
          methods names to be run begin with "test_" 
    '''
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
        '''Create dictionary of DataFrame values from file name keys'''
        self.ValidFiles = {}

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
            self.ValidFiles[ file ] = read_csv( filename )

    #------------------------------------------------------------
    # API
    #------------------------------------------------------------
    def test_API_1( self ):
        '''API 1'''
        if self.verbose : print ( " --- API 1 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.FitSimplex(data = df_, columns = 'V1', target = 'V5',
                             train = '1 300', test = '301 310', embedDimensions = 5)

    def test_API_2( self ):
        '''API 2'''
        if self.verbose : print ( "--- API 2 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.FitSimplex(data = df_, columns = ['V1'], target = 'V5',
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_3( self ):
        '''API 3'''
        if self.verbose : print ( "--- API 3 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.FitSimplex(data = df_, columns = ['V1', 'V3'], target = 'V5',
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_4( self ):
        '''API 4'''
        if self.verbose : print ( "--- API 4 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.FitSimplex(data = df_,
                             columns = ['V1','V3'], target = ['V5','V2'],
                             train = [1, 300], test = [301, 310], embedDimensions = 5)

    def test_API_5( self ):
        '''API 5'''
        if self.verbose : print ( "--- API 5 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.FitSimplex(data = df_, columns = 'V1', target = 'V5',
                             train = [1, 300], test = [301, 310], embedDimensions = 5, knn = 0)

    def test_API_6( self ):
        '''API 6'''
        if self.verbose : print ( "--- API 6 ---" )
        df_ = EDM.sampleData['Lorenz5D']
        df  = EDM.FitSimplex(data = df_, columns = 'V1', target = 'V5',
                             train = [1, 300], test = [301, 310], embedDimensions = 5, step = -2)

    def test_API_7( self ):
        '''API 7'''
        if self.verbose : print ( "--- API 7 Column names with space ---" )
        df_ = EDM.sampleData["columnNameSpace"]
        df = EDM.FitSimplex(df_, ['Var 1', 'Var 2'], ["Var 5 1"],
                            [1, 80], [81, 85], 5, 1, 0, -1, 0,
                            False, [], False, 0, False, False, False)

    #------------------------------------------------------------
    # Embed
    #------------------------------------------------------------
    def test_embed( self ):
        '''Embed'''
        if self.verbose : print ( "--- Embed ---" )
        df_ = EDM.sampleData['circle']
        df  = pyEDM.Embed.Embed(df_, 3, -1, "x", False)

    def test_embed2( self ):
        '''Embed multivariate'''
        if self.verbose : print ( "--- Embed multivariate ---" )
        df_ = EDM.sampleData['circle']
        df  = pyEDM.Embed.Embed(df_, 3, -1, ['x', 'y'], False)

    def test_embed2( self ):
        '''Embed multivariate'''
        if self.verbose : print ( "--- Embed includeTime ---" )
        df_ = EDM.sampleData['circle']
        df  = pyEDM.Embed.Embed(df_, 3, -1, ['x', 'y'], True)

    def test_embed3( self ):
        '''Embed from file'''
        if self.verbose : print ( "--- Embed from file ---" )
        df  = pyEDM.Embed.Embed(pathIn = '../data/', dataFile = 'circle.csv',
                                embeddingDimensions = 3, stepSize = -1, columns = ['x', 'y'])

    #------------------------------------------------------------
    # Simplex
    #------------------------------------------------------------
    def test_simplex( self ):
        '''embedded = False'''
        if self.verbose : print ( "--- Simplex embedded = False ---" )
        df_ = EDM.sampleData["block_3sp"]
        df = EDM.FitSimplex(df_, "x_t", "x_t",
                            [1, 100], [101, 195], 3, 1, 0, -1, 0,
                            False, [], False, 0, False, False, False)

        dfv = self.ValidFiles["Smplx_E3_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:95], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:95], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex2( self ):
        '''embedded = True'''
        if self.verbose : print ( "--- Simplex embedded = True ---" )
        df_ = EDM.sampleData["block_3sp"]
        df = EDM.FitSimplex(df_, "x_t y_t z_t", "x_t",
                            [1, 99], [100, 198], 3, 1, 0, -1, 0,
                            True, [], False, 0, False, False, False)

        dfv = self.ValidFiles["Smplx_E3_embd_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:98], 6 )      # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex3( self ):
        '''negative predictionHorizon'''
        if self.verbose : print ( "--- negative predictionHorizon ---" )
        df_ = EDM.sampleData["block_3sp"]
        df = EDM.FitSimplex(df_, "x_t", "y_t",
                            [1, 100], [50, 80], 3, -2, 0, -1, 0,
                            False, [], False, 0, False, False, False)

        dfv = self.ValidFiles["Smplx_negTp_block_3sp_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:98], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex4( self ):
        '''validLib'''
        if self.verbose : print ( "--- validLib ---" )
        df_ = EDM.sampleData["circle"]
        df = EDM.FitSimplex(data = df_, columns = 'x', target = 'x',
                            train = [1,200], test = [1,200], embedDimensions = 2, predictionHorizon = 1,
                            validLib = df_.eval('x > 0.5 | x < -0.5'))

        dfv = self.ValidFiles["Smplx_validLib_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex5( self ):
        '''disjoint train'''
        if self.verbose : print ( "--- disjoint train ---" )
        df_ = EDM.sampleData["circle"]
        df = EDM.FitSimplex(data = df_, columns = 'x', target = 'x',
                            train = [1,40, 50,130], test = [80,170],
                            embedDimensions = 2, predictionHorizon = 1, step = -3)

        dfv = self.ValidFiles["Smplx_disjointLib_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex6( self ):
        '''disjoint test w/ nan'''
        if self.verbose : print ( "--- disjoint test w/ nan ---" )
        df_ = EDM.sampleData["Lorenz5D"]
        df_.iloc[ [8,50,501], [1,2] ] = nan

        df = EDM.FitSimplex(data = df_, columns= 'V1', target = 'V2',
                            embedDimensions = 5, predictionHorizon = 2, train = [1,50,101,200,251,500],
                            test = [1,10,151,155,551,555,881,885,991,1000])

        dfv = self.ValidFiles["Smplx_disjointPred_nan_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:195], 5 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 5 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex7( self ):
        '''exclusion radius'''
        if self.verbose : print ( "--- exclusion radius ---" )
        df_ = EDM.sampleData["circle"]
        df = EDM.FitSimplex(data = df_, columns = 'x', target = 'y',
                            train = [1,100], test = [21,81], embedDimensions = 2, predictionHorizon = 1,
                            exclusionRadius = 5)

        dfv = self.ValidFiles["Smplx_exclRadius_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:60], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:60], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex8( self ):
        '''nan'''
        if self.verbose : print ( "--- nan ---" )
        df_ = EDM.sampleData["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        df = EDM.FitSimplex(dataFrame = dfn, columns = 'x', target = 'y',
                            train = [1,100], test = [1,95], embedDimensions = 2, predictionHorizon = 1)

        dfv = self.ValidFiles["Smplx_nan_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:90], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:90], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex9( self ):
        '''nan'''
        if self.verbose : print ( "--- nan ---" )
        df_ = EDM.sampleData["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        df = EDM.FitSimplex(dataFrame = dfn, columns = 'y', target = 'x',
                            train = [1,200], test = [1,195], embedDimensions = 2, predictionHorizon = 1)

        dfv = self.ValidFiles["Smplx_nan2_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:190], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:190], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex10( self ):
        '''DateTime'''
        if self.verbose : print ( "--- DateTime ---" )
        df_ = EDM.sampleData["SumFlow_1980-2005"]

        df = EDM.FitSimplex(data = df_,
                            columns = 'S12.C.D.S333', target = 'S12.C.D.S333',
                            train = [1,800], test = [801,1001], embedDimensions = 3, predictionHorizon = 1)

        self.assertTrue( isinstance( df['Time'][0],  datetime ) )

        dfv = self.ValidFiles["Smplx_DateTime_valid.csv"]

        S1 = round( dfv.get('Predictions')[1:200], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:200], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_simplex11( self ):
        '''knn = 1'''
        if self.verbose : print ( "--- knn = 1 ---" )
        df_ = EDM.sampleData["Lorenz5D"]

        df = EDM.FitSimplex(data = df_, columns= 'V5', target = 'V5',
                            train = [301,400], test = [350,355],
                            knn = 1, embedded = True, returnObject = True)

        knn = df.knn_neighbors
        knnValid = array( [322,334,362,387,356,355] )[:,None]
        self.assertTrue( array_equal( knn, knnValid ) )

    #------------------------------------------------------------
    def test_simplex12( self ):
        '''exclusion Radius '''
        if self.verbose : print ( "--- exclusion Radius ---" )
        df_ = EDM.sampleData["Lorenz5D"]
        x   = [i+1 for i in range(1000)]
        df_ = DataFrame({'Time':df_['Time'],'X':x,'V1':df_['V1']})

        df = EDM.FitSimplex(data = df_, columns= 'X', target = 'V1',
                            train = [1,100], test = [101,110],
                            embedDimensions = 5, exclusionRadius = 10, returnObject = True)

        knn = df.knn_neighbors[:,0]
        knnValid = array( [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] )
        self.assertTrue( array_equal( knn, knnValid ) )

    #------------------------------------------------------------
    # S-map
    #------------------------------------------------------------
    def test_smap( self ):
        '''SMap'''
        if self.verbose : print ( "--- SMap ---" )
        df_ = EDM.sampleData["circle"]
        S = EDM.FitSMap(data = df_, columns = 'x', target = 'x',
                        train = [1,100], test = [110,160], embedDimensions = 4, predictionHorizon = 1,
                        step = -1, theta = 3.)

        dfv = self.ValidFiles["SMap_circle_E4_valid.csv"]
        df  = S['predictions']

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_smap2( self ):
        '''SMap embedded = True'''
        if self.verbose : print ( "--- SMap embedded = True ---" )
        df_ = EDM.sampleData["circle"]
        S = EDM.FitSMap(data = df_, columns = ['x', 'y'], target = 'x',
                        train = [1,200], test = [1,200], embedDimensions = 2, predictionHorizon = 1,
                        step = -1, embedded = True, theta = 3.)

        dfv  = self.ValidFiles["SMap_circle_E2_embd_valid.csv"]
        
        df  = S['predictions']
        dfc = S['coefficients']

        S1 = round( dfv.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:195], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

        self.assertTrue( dfc['∂x/∂x'].mean().round(5) == 0.99801 )
        self.assertTrue( dfc['∂x/∂y'].mean().round(5) == 0.06311 )

    #------------------------------------------------------------
    def test_smap3( self ):
        '''SMap nan'''
        if self.verbose : print ( "--- SMap nan ---" )
        df_ = EDM.sampleData["circle"]
        dfn = df_.copy()
        dfn.iloc[ [5,6,12], 1 ] = nan
        dfn.iloc[ [10,11,17], 2 ] = nan

        S = EDM.FitSMap(dataFrame = dfn, columns = 'x', target = 'y',
                        train = [1,50], test = [1,50], embedDimensions = 2, predictionHorizon = 1,
                        step = -1, theta = 3.)

        dfv = self.ValidFiles["SMap_nan_valid.csv"]
        df  = S['predictions']

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    def test_smap4( self ):
        '''DateTime'''
        if self.verbose : print ( "--- noTime ---" )
        df_ = EDM.sampleData["circle_noTime"]

        S = EDM.FitSMap(data = df_, columns = 'x', target = 'y',
                        train = [1,100], test = [101,150], embedDimensions = 2,
                        theta = 3, noTime = True)

        dfv = self.ValidFiles["SMap_noTime_valid.csv"]
        df  = S['predictions']

        S1 = round( dfv.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        S2 = round(  df.get('Predictions')[1:50], 6 ) # Skip row 0 Nan
        self.assertTrue( S1.equals( S2 ) )

    #------------------------------------------------------------
    # CCM
    #------------------------------------------------------------
    def test_ccm( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM ---" )
            df_ = EDM.sampleData['sardine_anchovy_sst']
            df = EDM.FitCCM(data = df_, columns = 'anchovy', target = 'np_sst',
                            trainSizes = [10, 20, 30, 40, 50, 60, 70, 75], sample = 100,
                            embedDimensions = 3, predictionHorizon = 0, step = -1, seed = 777)

        dfv = round( self.ValidFiles["CCM_anch_sst_valid.csv"], 2 )

        self.assertTrue( dfv.equals( round( df, 2 ) ) )

    #------------------------------------------------------------
    def test_ccm2( self ):
        '''CCM Multivariate'''
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM multivariate ---" )
            df_ = EDM.sampleData['Lorenz5D']
            df = EDM.FitCCM(data = df_, columns = 'V3 V5', target = 'V1',
                            trainSizes = [20, 200, 500, 950], sample = 30, embedDimensions = 5,
                            predictionHorizon = 10, step = -5, seed = 777)

        dfv = round( self.ValidFiles["CCM_Lorenz5D_MV_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    def test_ccm3( self ):
        '''CCM nan'''
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM nan ---" )
            df_ = EDM.sampleData['circle']
            dfn = df_.copy()
            dfn.iloc[ [5,6,12], 1 ] = nan
            dfn.iloc[ [10,11,17], 2 ] = nan

            df = EDM.FitCCM(dataFrame = dfn, columns = 'x', target = 'y',
                            trainSizes = [10, 190, 10], sample = 20, embedDimensions = 2,
                            predictionHorizon = 5, step = -1, seed = 777)

        dfv = round( self.ValidFiles["CCM_nan_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    def test_ccm4( self ):
        '''CCM Multivariate names with spaces'''
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- CCM multivariate name spaces ---" )
            df_ = EDM.sampleData['columnNameSpace']
            df = EDM.FitCCM(data = df_,
                            columns = ['Var 1','Var3','Var 5 1'],
                            target = ['Var 2','Var 4 A'],
                            trainSizes = [20, 50, 90], sample = 1,
                            embedDimensions = 5, predictionHorizon = 0, step = -1, seed = 777)

        dfv = round( self.ValidFiles["CCM_Lorenz5D_MV_Space_valid.csv"], 4 )

        self.assertTrue( dfv.equals( round( df, 4 ) ) )

    #------------------------------------------------------------
    # Multiview
    #------------------------------------------------------------
    def test_multiview( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- Multiview ---" )
            df_ = EDM.sampleData['block_3sp']
            M = EDM.FitMultiview(data = df_,
                                 columns = "x_t y_t z_t", target = "x_t",
                                 train = [1, 100], test = [101, 198],
                                 D = 0, embedDimensions = 3, predictionHorizon = 1, knn = 0, step = -1,
                                 multiview = 0, exclusionRadius = 0,
                                 trainLib = False, excludeTarget = False,
                                 numProcess = 4, showPlot = False)

        df_pred  = M['Predictions']
        df_combo = M['View'][ ['correlation', 'MAE', 'RMSE'] ]

        # Validate predictions
        dfvp      = self.ValidFiles["Multiview_pred_valid.csv"]
        predValid = round( dfvp.get('Predictions'), 4 )
        test      = round( df_pred.get('Predictions'), 4 )
        self.assertTrue( predValid.equals( test ) )

        # Validate combinations
        dfvc = round( self.ValidFiles['Multiview_combos_valid.csv'], 4 )
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
            df_ = EDM.sampleData['Lorenz5D']
            df = EDM.FindOptimalEmbeddingDimensionality(data = df_, columns= 'V1', target= 'V1',
                                                        maxE = 12, train = [1, 500], test=[501, 800],
                                                        predictionHorizon = 15, step = -5, exclusionRadius = 20,
                                                        numProcess = 10, showPlot = False)

        dfv = round( self.ValidFiles["EmbedDim_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( df, 6 ) ) )

    #------------------------------------------------------------
    # PredictInterval
    #------------------------------------------------------------
    def test_PredictInterval( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- PredictInterval ---" )
            df_ = EDM.sampleData['block_3sp']
            df = EDM.FindOptimalPredictionHorizon(data = df_,
                                                  columns = 'x_t', target='x_t', maxTp = 15,
                                                  train = [1, 150], test = [151, 198], embedDimensions = 3,
                                                  step = -1, numProcess = 10, showPlot=False)
            
        dfv = round( self.ValidFiles["PredictInterval_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( df, 6 ) ) )

    #------------------------------------------------------------
    # PredictNonlinear
    #------------------------------------------------------------
    def test_PredictNonlinear( self ):
        with catch_warnings():
            # Python-3.13 multiprocessing fork DeprecationWarning 
            filterwarnings( "ignore", category = DeprecationWarning )

            if self.verbose : print ( "--- Predict ---" )
            df_ = EDM.sampleData['TentMapNoise']
            df = EDM.FindSMapNeighborhood(data = df_,
                                          columns = 'TentMap', target = 'TentMap',
                                          train = [1, 500], test = [501,800], embedDimensions = 4,
                                          predictionHorizon = 1, step = -1, numProcess = 10,
                                          theta = [0.01,0.1,0.3,0.5,0.75,1,1.5,
                                                2,3,4,5,6,7,8,9,10,15,20 ],
                                          showPlot = False)

        dfv = round( self.ValidFiles["PredictNonlinear_valid.csv"], 6 )

        self.assertTrue( dfv.equals( round( df, 6 ) ) )

    #------------------------------------------------------------
    # Generative mode
    #------------------------------------------------------------
    def test_generate__simplex1( self ):
        '''Simplex Generate 1'''
        if self.verbose : print ( "--- Simplex Generate 1 ---" )
        df_ = EDM.sampleData["circle"]

        df = EDM.FitSimplex(data = df_,
                            columns = 'x', target = 'x',
                            train = [1,200], test = [1,2], embedDimensions = 2,
                            generateSteps = 100, generateConcat = True)

        self.assertTrue( df.shape == (300,4) )

    #------------------------------------------------------------
    def test_generate_simplex2( self ):
        '''Simplex generateSteps 2'''
        if self.verbose : print ( "--- Simplex generateSteps 2 ---" )
        df_ = EDM.sampleData["Lorenz5D"]

        df = EDM.FitSimplex(df_, "V1", "V1",
                            [1, 1000], [1, 2], 5, 1, 0, -1, 0,
                            False, [], False, 100, False, False, False)

        self.assertTrue( df.shape == (100,4) )

    #------------------------------------------------------------
    def test_generate_smap1( self ):
        '''DateTime'''
        if self.verbose : print ( "--- SMap Generate ---" )
        df_ = EDM.sampleData["circle"]

        S = EDM.FitSMap(data = df_,
                        columns = 'x', target = 'x', theta = 3.,
                        train = [1,200], test = [1,2], embedDimensions = 2,
                        generateSteps = 100, generateConcat = True)

        self.assertTrue( S['predictions'].shape == (300,4) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
