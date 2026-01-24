#! /usr/bin/env python3

from pyEDM import Functions as EDM

#------------------------------------------------------------
#------------------------------------------------------------
def main():
    """pyEDM examples."""

    df_ = EDM.sampleData["TentMap"]
    data = df_.values
    col_index = df_.columns.get_loc('TentMap')
    target_index = df_.columns.get_loc('TentMap')
    df = EDM.FindOptimalEmbeddingDimensionality(data = data,
                                                columns = [col_index], target = target_index,
                                                train = [1, 100], test = [201, 500], maxE = 10,
                                                predictionHorizon = 1, knn = 0, step = -1, exclusionRadius = 0,
                                                embedded = False, validLib = [], noTime = False,
                                                ignoreNan = True, numProcess = 4, mpMethod = None,
                                                chunksize = 1)

    df_ = EDM.sampleData["TentMap"]
    data = df_.values
    col_index = df_.columns.get_loc('TentMap')
    target_index = df_.columns.get_loc('TentMap')
    df = EDM.FindOptimalPredictionHorizon(data = data,
                                          columns = [col_index], target = target_index,
                                          train = [1, 100], test = [201, 500], maxTp = 10,
                                          embedDimensions = 2, knn = 0, step = -1, exclusionRadius = 0,
                                          embedded = False, validLib = [], noTime = False,
                                          ignoreNan = True, numProcess = 4, mpMethod = None,
                                          chunksize = 1);

    df_ = EDM.sampleData["TentMapNoise"]
    data = df_.values
    col_index = df_.columns.get_loc('TentMap')
    target_index = df_.columns.get_loc('TentMap')
    df = EDM.FindSMapNeighborhood(data = data,
                                  columns = [col_index], target = target_index,
                                  train = [1, 100], test = [201, 500], embedDimensions = 2,
                                  predictionHorizon = 1, knn = 0, step = -1,
                                  solver = None, embedded = False, validLib = [], noTime = False,
                                  ignoreNan = True, numProcess = 4, mpMethod = None,
                                  chunksize = 1)

    # Tent map simplex : specify multivariable columns embedded = True
    df_ = EDM.sampleData["block_3sp"]
    data = df_.values
    col_index1 = df_.columns.get_loc('x_t')
    col_index2 = df_.columns.get_loc('y_t')
    col_index3 = df_.columns.get_loc('z_t')
    target_index = df_.columns.get_loc('x_t')
    S = EDM.FitSimplex(data = data,
                       columns = [col_index1, col_index2, col_index3], target = target_index,
                       train = [1, 99], test = [100, 198], embedDimensions = 3, predictionHorizon = 1,
                       knn = 0, step = -1, exclusionRadius = 0,
                       embedded = True, validLib = [], noTime = False, generateSteps = 0,
                       generateConcat = False, verbose = False, ignoreNan = True, returnObject = False)

    # Tent map simplex : Embed column x_t to E=3, embedded = False
    df_ = EDM.sampleData["block_3sp"]
    data = df_.values
    col_index = df_.columns.get_loc('x_t')
    target_index = df_.columns.get_loc('x_t')
    S = EDM.FitSimplex(data = data,
                       columns = [col_index], target = target_index,
                       train = [1, 99], test = [100, 198], embedDimensions = 3, predictionHorizon = 1,
                       knn = 0, step = -1, exclusionRadius = 0,
                       embedded = False, validLib = [], noTime = False, generateSteps = 0,
                       generateConcat = False, verbose = False, ignoreNan = True, returnObject = False)

    df_ = EDM.sampleData["block_3sp"]
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

    # S-map circle : specify multivariable columns embedded = True
    df_ = EDM.sampleData["circle"]
    data = df_.values
    col_index1 = df_.columns.get_loc('x')
    col_index2 = df_.columns.get_loc('y')
    target_index = df_.columns.get_loc('x')
    S = EDM.FitSMap(data = data,
                    columns = [col_index1, col_index2], target = target_index,
                    train = [1, 100], test = [101, 198],
                    embedDimensions = 2, predictionHorizon = 1, knn = 0, step = -1,
                    theta = 4, exclusionRadius = 0,
                    solver = None, embedded = True,
                    validLib = [], noTime = False, generateSteps = 0,
                    generateConcat = False, ignoreNan = True, verbose = False,
                    returnObject = False)

    df_ = EDM.sampleData["sardine_anchovy_sst"]
    data = df_.values
    col_index = df_.columns.get_loc('anchovy')
    target_index = df_.columns.get_loc('np_sst')
    CM = EDM.FitCCM(data = data,
                    columns = [col_index], target = target_index,
                    embedDimensions = 3, predictionHorizon = 0, knn = 0, step = -1, exclusionRadius = 0,
                    trainSizes = [10, 70, 10], sample = 100,
                    seed = 0, embedded = False, validLib = [], includeData = False, noTime = False,
                    ignoreNan = True, mpMethod = None, sequential = False, verbose = False,
                    returnObject = False)

#------------------------------------------------------------
#------------------------------------------------------------
if __name__ == '__main__':
    main()
