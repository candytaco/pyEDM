import functools
import inspect
from .LoadData import sampleData


def print_call(func):
    """Decorator that prints function calls with their arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        arg_parts = []

        if args:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    arg_parts.append(f"{param_names[i]} = {repr(arg)}")
                else:
                    arg_parts.append(repr(arg))

        for key, value in kwargs.items():
            arg_parts.append(f"{key} = {repr(value)}")

        args_str = ", ".join(arg_parts)
        print(f"EDM.{func_name}({args_str})")
        print()

        return func(*args, **kwargs)

    return wrapper


def Examples():
    '''Canonical EDM API examples'''

    from pyEDM import API as EDM

    EmbedDimension = print_call(EDM.EmbedDimension)
    PredictInterval = print_call(EDM.PredictInterval)
    PredictNonlinear = print_call(EDM.PredictNonlinear)
    Simplex = print_call(EDM.Simplex)
    Multiview = print_call(EDM.Multiview)
    SMap = print_call(EDM.SMap)
    CCM = print_call(EDM.CCM)

    sampleDataNames = \
        ["TentMap","TentMapNoise","circle","block_3sp","sardine_anchovy_sst"]

    for dataName in sampleDataNames :
        if dataName not in sampleData:
            raise Exception( "Examples(): Failed to find sample data " + \
                             dataName + " in EDM package" )

    EmbedDimension(data = sampleData["TentMap"],
                   columns = [1], target = 1,
                   train = [1, 100], test = [201, 500])

    PredictInterval(data = sampleData["TentMap"],
                    columns = [1], target = 1,
                    train = [1, 100], test = [201, 500], embedDimensions = 2)

    PredictNonlinear(data = sampleData["TentMapNoise"],
                     columns = [1], target = 1,
                     train = [1, 100], test = [201, 500], embedDimensions = 2)

    # Tent map simplex : specify multivariable columns embedded = True
    Simplex(data = sampleData["block_3sp"],
            columns = [1, 4, 7], target = 1,
            train = [1, 99], test = [100, 195],
            embedDimensions = 3, embedded = True, showPlot = True)

    # Tent map simplex : Embed column x_t to embedDimensions=3, embedded = False
    Simplex(data = sampleData["block_3sp"],
            columns = [1], target = 1,
            train = [1, 99], test = [105, 190],
            embedDimensions = 3, showPlot = True)

    Multiview(data = sampleData["block_3sp"],
              columns = [1, 4, 7], target = 1,
              train = [1, 100], test = [101, 198],
              D = 0, embedDimensions = 3, predictionHorizon = 1, multiview = 0,
              trainLib = False, showPlot = True)

    # S-map circle : specify multivariable columns embedded = True
    SMap(data = sampleData["circle"],
         columns = [1, 2], target = 1,
         train = [1, 100], test = [110, 190], theta = 4, embedDimensions = 2,
         verbose = False, showPlot = True, embedded = True)

    CCM(data = sampleData["sardine_anchovy_sst"],
        columns = [1], target = [4],
        trainSizes = [10, 70, 10], sample = 50,
        embedDimensions = 3, predictionHorizon = 0, verbose = False, showPlot = True)
