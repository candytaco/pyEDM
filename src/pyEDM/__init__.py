'''Python tools for EDM'''
from .API import EmbedDimension, PredictInterval, PredictNonlinear
# import EDM functions
from .API import Simplex, SMap, CCM, Multiview
from .AuxFunc import PlotObsPred, PlotCoeff, ComputeError
from .Examples import Examples
from .AuxFunc import SurrogateData
from .Embed import Embed
from .LoadData import sampleData

__version__     = "2.3.2"
__versionDate__ = "2025-11-17"
