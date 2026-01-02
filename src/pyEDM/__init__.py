"""Python tools for EDM"""
from .Functions import FindOptimalEmbeddingDimensionality, FindOptimalPredictionHorizon, FindSMapNeighborhood

# provide functional API
from . import Functions
# provide object-based API with train/test splits
from . import Fitters

from .Utils import PlotObsPred, PlotCoeff, ComputeError
from .FunctionalExamples import FunctionalExamples
from .FitterExamples import FitterExamples
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
