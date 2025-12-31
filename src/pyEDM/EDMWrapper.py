"""
Base wrapper class for EDM methods that provides sklearn-like API.
"""
from typing import Optional, Tuple

import numpy

from .DataAdapter import DataAdapter

class EDMWrapper:
    """
    Base wrapper class for EDM methods that provides sklearn-like API.

    This class handles the conversion from separate X/Y train/test arrays
    to the EDM single-array format using DataAdapter.
    """

    def __init__(self,
                 XTrain: numpy.ndarray,
                 YTrain: numpy.ndarray,
                 XTest: numpy.ndarray,
                 YTest: numpy.ndarray,
                 TrainTime: Optional[numpy.ndarray] = None,
                 TestTime: Optional[numpy.ndarray] = None):
        """
        Initialize EDM wrapper with sklearn-style separate arrays.

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
        TrainTime : numpy.ndarray, optional
            Time labels for train data
        TestTime : numpy.ndarray, optional
            Time labels for test data
        """

        self.DataAdapter = DataAdapter(XTrain, YTrain, XTest, YTest,
                                      TrainTime, TestTime)

        self.XTrain = XTrain
        self.YTrain = YTrain
        self.XTest = XTest
        self.YTest = YTest
        self.TrainTime = TrainTime
        self.TestTime = TestTime

    def GetEDMData(self) -> numpy.ndarray:
        """
        Get the combined EDM data array.

        Returns
        -------
        numpy.ndarray
            Combined data array in EDM format
        """
        return self.DataAdapter.fullData

    def GetTrainIndices(self) -> Tuple[int, int]:
        """
        Get the train indices for EDM.

        Returns
        -------
        tuple of (int, int)
            Train indices [start, end]
        """
        return self.DataAdapter.TrainIndices

    def GetTestIndices(self) -> Tuple[int, int]:
        """
        Get the test indices for EDM.

        Returns
        -------
        tuple of (int, int)
            Test indices [start, end]
        """
        return self.DataAdapter.TestIndices

    def GetXIndices(self) -> Tuple[int, int]:
        """
        Get the X feature indices.

        Returns
        -------
        tuple of (int, int)
            X indices [start, end]
        """
        return self.DataAdapter.XIndices

    def GetYIndex(self) -> int:
        """
        Get the Y target index.

        Returns
        -------
        int
            Y index
        """
        return self.DataAdapter.YIndex

    def HasTime(self) -> bool:
        """
        Check if data has time column.

        Returns
        -------
        bool
            True if data has time column
        """
        return self.DataAdapter.HasTime
