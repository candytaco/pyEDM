"""Python tools for EDM"""
from .Functions import FindOptimalEmbeddingDimensionality, FindOptimalPredictionHorizon, FindSMapNeighborhood
# import EDM functions
from .Functions import FitSimplex, FitSMap, FitCCM, FitMultiview
from .Utils import PlotObsPred, PlotCoeff, ComputeError
from .FunctionalExamples import Examples
from .Utils import SurrogateData
from .ExampleData import sampleData

# Import result objects
from pyEDM.EDM.Results import (
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

__version__     = "3"
__versionDate__ = "2026-01-02"
