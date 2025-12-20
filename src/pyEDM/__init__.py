'''Python tools for EDM'''
from .API import EmbedDimension, PredictInterval, PredictNonlinear
# import EDM functions
from .API import Simplex, SMap, CCM, Multiview
from .Utils import PlotObsPred, PlotCoeff, ComputeError
from .Examples import Examples
from .Utils import SurrogateData
from .Embed import Embed
from .LoadData import sampleData
# Import parameter configuration objects
from .Parameters import (
    EDMParameters,
    DataSplit,
    GenerationParameters,
    SMapParameters,
    CCMParameters,
    MultiviewParameters,
    ExecutionParameters
)
# Import result objects
from .Results import (
    SimplexResult,
    SMapResult,
    CCMResult,
    MultiviewResult
)
# Import visualization functions
from .Visualization import (
    plot_prediction,
    plot_smap_coefficients,
    plot_ccm,
    plot_multiview,
    plot_embed_dimension,
    plot_predict_interval,
    plot_predict_nonlinear
)
# Import execution configuration
from .Execution import ExecutionMode

__version__     = "2.3.2"
__versionDate__ = "2025-11-17"
