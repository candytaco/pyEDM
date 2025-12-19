from .LoadData import sampleData


def Examples():
    '''Canonical EDM API examples'''

    def RunEDM ( cmd ):
        print(cmd)
        print()
        from pyEDM import API as EDM
        df = eval( 'EDM.' + cmd )
        return df

    sampleDataNames = \
        ["TentMap","TentMapNoise","circle","block_3sp","sardine_anchovy_sst"]

    for dataName in sampleDataNames :
        if dataName not in sampleData:
            raise Exception( "Examples(): Failed to find sample data " + \
                             dataName + " in EDM package" )

    #---------------------------------------------------------------
    cmd = str().join(['EmbedDimension( data = sampleData["TentMap"],',
                      ' columns = [1], target = 1,',
                      ' train = [1, 100], test = [201, 500] )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(['PredictInterval( data = sampleData["TentMap"],',
                      ' columns = [1], target = 1,'
                      ' train = [1, 100], test = [201, 500], embedDimensions = 2 )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(
        ['PredictNonlinear( data = sampleData["TentMapNoise"],',
         ' columns = [1], target = 1, '
         ' train = [1, 100], test = [201, 500], embedDimensions = 2 )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    # Tent map simplex : specify multivariable columns embedded = True
    cmd = str().join(['Simplex( data = sampleData["block_3sp"],',
                      ' columns=[1, 4, 7], target=1,'
                      ' train = [1, 99], test = [100, 195],',
                      ' embedDimensions = 3, embedded = True, showPlot = True )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    # Tent map simplex : Embed column x_t to embedDimensions=3, embedded = False
    cmd = str().join(['Simplex( data = sampleData["block_3sp"],',
                      ' columns = [1], target = 1,',
                      ' train = [1, 99], test = [105, 190],',
                      ' embedDimensions = 3, showPlot = True )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(['Multiview( data = sampleData["block_3sp"],',
                      ' columns = [1, 4, 7], target = 1,',
                      ' train = [1, 100], test = [101, 198],',
                      ' D = 0, embedDimensions = 3, predictionHorizon = 1, multiview = 0,',
                      ' trainLib = False, showPlot = True ) '])
    RunEDM( cmd )

    #---------------------------------------------------------------
    # S-map circle : specify multivariable columns embedded = True
    cmd = str().join(['SMap( data = sampleData["circle"],',
                      ' columns = [1, 2], target = 1,'
                      ' train = [1, 100], test = [110, 190], theta = 4, embedDimensions = 2,',
                      ' verbose = False, showPlot = True, embedded = True )'])
    RunEDM( cmd )

    #---------------------------------------------------------------
    cmd = str().join(['CCM( data = sampleData["sardine_anchovy_sst"],',
                      ' columns = [1], target = [4],',
                      ' trainSizes = [10, 70, 10], sample = 50,',
                      ' embedDimensions = 3, predictionHorizon = 0, verbose = False, showPlot = True )'])
    RunEDM( cmd )
