"""Python tools for EDM"""
from .Functions import FindOptimalEmbeddingDimensionality, FindOptimalPredictionHorizon, FindSMapNeighborhood
# import EDM functions
from .Functions import FitSimplex, FitSMap, FitCCM, FitMultiview
from .Utils import PlotObsPred, PlotCoeff, ComputeError
from .Examples import Examples
from .Utils import SurrogateData
from .Embed import Embed
from .LoadData import sampleData

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
