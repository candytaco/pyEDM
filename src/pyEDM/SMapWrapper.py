"""
SMap wrapper for sklearn-like API.
"""
from typing import Optional, List

import numpy

from .EDMWrapper import EDMWrapper
from .SMap import SMap

class SMapWrapper(EDMWrapper):
    """
    Wrapper class for SMap that provides sklearn-like API.
    """

    def __init__(self,
                 XTrain: numpy.ndarray,
                 YTrain: numpy.ndarray,
                 XTest: numpy.ndarray,
                 YTest: numpy.ndarray,
                 Columns: Optional[List[int]] = None,
                 Target: Optional[int] = None,
                 EmbedDimensions: int = 0,
                 PredictionHorizon: int = 1,
                 KNN: int = 0,
                 Step: int = -1,
                 Theta: float = 0.0,
                 ExclusionRadius: int = 0,
                 Embedded: bool = False,
                 TrainTime: Optional[numpy.ndarray] = None,
                 TestTime: Optional[numpy.ndarray] = None,
                 Verbose: bool = False):
        """
        Initialize SMap wrapper with sklearn-style separate arrays.

        Parameters
        ----------
        XTrain : numpy.ndarray
            Training feature data
        YTrain : numpy.ndarray
            Training target data
        XTest : numpy.ndarray
            Test feature data
        YTest : numpy.ndarray
            Test target data
        Columns : list of int, optional
            Column indices to use for embedding
        Target : int, optional
            Target column index
        EmbedDimensions : int, default=0
            Embedding dimension (E)
        PredictionHorizon : int, default=1
            Prediction time horizon (Tp)
        KNN : int, default=0
            Number of nearest neighbors
        Step : int, default=-1
            Time delay step size (tau)
        Theta : float, default=0.0
            S-Map localization parameter
        ExclusionRadius : int, default=0
            Temporal exclusion radius for neighbors
        Embedded : bool, default=False
            Whether data is already embedded
        TrainTime : numpy.ndarray, optional
            Time labels for train data
        TestTime : numpy.ndarray, optional
            Time labels for test data
        Verbose : bool, default=False
            Print diagnostic messages
        """

        super().__init__(XTrain, YTrain, XTest, YTest, TrainTime, TestTime)

        self.Columns = Columns
        self.Target = Target
        self.EmbedDimensions = EmbedDimensions
        self.PredictionHorizon = PredictionHorizon
        self.KNN = KNN
        self.Step = Step
        self.Theta = Theta
        self.ExclusionRadius = ExclusionRadius
        self.Embedded = Embedded
        self.Verbose = Verbose

        self.SMap = None

    def Run(self):
        """
        Run SMap prediction.

        Returns
        -------
        SMapResult
            Prediction results
        """
        Data = self.GetEDMData()
        TrainIndices = self.GetTrainIndices()
        TestIndices = self.GetTestIndices()
        YIndex = self.GetYIndex()
        NoTime = not self.HasTime()

        # Determine columns to use
        XStart, XEnd = self.GetXIndices()
        if self.Columns is not None:
            Columns = [XStart + col for col in self.Columns]
        else:
            Columns = list(range(XStart, XEnd + 1))

        # Determine target
        if self.Target is not None:
            Target = XStart + self.Target
        else:
            Target = YIndex

        self.SMap = SMap(
            data=Data,
            columns=Columns,
            target=Target,
            train=TrainIndices,
            test=TestIndices,
            embedDimensions=self.EmbedDimensions,
            predictionHorizon=self.PredictionHorizon,
            knn=self.KNN,
            step=self.Step,
            theta=self.Theta,
            exclusionRadius=self.ExclusionRadius,
            noTime=NoTime,
            verbose=self.Verbose,
            embedded=self.Embedded
        )

        return self.SMap.Run()
