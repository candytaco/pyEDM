#! /usr/bin/env python3

import pyEDM as EDM

#------------------------------------------------------------
#------------------------------------------------------------
def main():
    '''pyEDM examples.'''

    df = EDM.EmbedDimension( dataFrame = EDM.sampleData["TentMap"],
                             columns = "TentMap", target = "TentMap",
                             train = [1, 100], test = [201, 500], maxE = 10,
                             predictionHorizon = 1, step = -1, exclusionRadius = 0,
                             validLib = [], numProcess = 4 )
    
    df = EDM.PredictInterval( dataFrame = EDM.sampleData["TentMap"],
                              columns = "TentMap", target = "TentMap",
                              train = [1, 100], test = [201, 500], maxTp = 10,
                              E = 2, step = -1, exclusionRadius = 0,
                              validLib = [], numProcess = 4 );

    df = EDM.PredictNonlinear( dataFrame = EDM.sampleData[ "TentMapNoise" ],
                               columns = "TentMap", target = "TentMap",
                               train = [1, 100], test = [201, 500], E = 2,
                               predictionHorizon = 1, knn = 0, step = -1,
                               validLib = [], numProcess = 4 )
    
    # Tent map simplex : specify multivariable columns embedded = True
    S = EDM.Simplex( dataFrame = EDM.sampleData[ "block_3sp" ],
                     columns = "x_t y_t z_t", target = "x_t",
                     train = [1, 99], test = [100, 198], E = 3, predictionHorizon = 1,
                     knn = 0, step = -1, exclusionRadius = 0,
                     embedded = True, showPlot = True, validLib = [] )

    # Tent map simplex : Embed column x_t to E=3, embedded = False
    S = EDM.Simplex( dataFrame = EDM.sampleData[ "block_3sp" ],
                     columns = "x_t", target = "x_t",
                     train = [1, 99], test = [100, 198], E = 3, predictionHorizon = 1,
                     knn = 0, step = -1, exclusionRadius = 0,
                     embedded = False, showPlot = True, validLib = [] )

    M = EDM.Multiview( dataFrame = EDM.sampleData[ "block_3sp" ],
                       columns = ["x_t", "y_t", "z_t"], target = "x_t",
                       train = [1, 100], test = [101, 198],
                       D = 0, E = 3, predictionHorizon = 1, knn = 0, step = -1,
                       multiview = 0, exclusionRadius = 0,
                       trainLib = False, excludeTarget = False,
                       numProcess = 4, showPlot = True )

    # S-map circle : specify multivariable columns embedded = True
    S = EDM.SMap( dataFrame = EDM.sampleData[ "circle" ], 
                  columns = ["x", "y"], target = "x",
                  train = [1, 100], test = [101, 198],
                  E = 2, predictionHorizon = 1, knn = 0, step = -1,
                  theta = 4, exclusionRadius = 0,
                  solver = None, embedded = True,
                  validLib = [], showPlot = True )
    
    CM = EDM.CCM( dataFrame = EDM.sampleData[ "sardine_anchovy_sst" ],
                  columns = "anchovy", target = "np_sst",
                  E = 3, predictionHorizon = 0, knn = 0, step = -1, exclusionRadius = 0,
                  libSizes = [10, 70, 10], sample = 100,
                  seed = 0, verbose = False, showPlot = True )

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
