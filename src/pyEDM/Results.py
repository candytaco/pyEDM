"""Result classes for pyEDM predictions.

This module provides dataclasses for structured prediction results from
different EDM methods. Using result objects instead of conditional return
types provides consistency, self-documentation, and convenient access to
results and metadata.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np


@dataclass(frozen=True)
class SimplexResult:
    """Results from Simplex prediction.

    Attributes
    ----------
    projection : numpy.ndarray
        Array with columns [Time, Observations, Predictions]
    embedDimensions : int
        Embedding dimension used
    predictionHorizon : int
        Prediction horizon used
    """
    projection: np.ndarray
    embedDimensions: int
    predictionHorizon: int

    @property
    def time(self) -> np.ndarray:
        """Time values from projection."""
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """Observed values from projection."""
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """Predicted values from projection."""
        return self.projection[:, 2]

    def compute_error(self) -> Dict[str, float]:
        """Compute prediction error statistics.

        Returns
        -------
        dict
            Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        from .Utils import ComputeError
        return ComputeError(self.observations, self.predictions)


@dataclass(frozen=True)
class SMapResult:
    """Results from S-Map prediction.

    Attributes
    ----------
    projection : numpy.ndarray
        Array with columns [Time, Observations, Predictions]
    coefficients : numpy.ndarray
        S-Map coefficients for each prediction (N_pred, E+1)
    singularValues : numpy.ndarray
        Singular values from SVD for each prediction (N_pred, E+1)
    embedDimensions : int
        Embedding dimension used
    predictionHorizon : int
        Prediction horizon used
    theta : float
        Localization parameter used
    """
    projection: np.ndarray
    coefficients: np.ndarray
    singularValues: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    theta: float

    @property
    def time(self) -> np.ndarray:
        """Time values from projection."""
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """Observed values from projection."""
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """Predicted values from projection."""
        return self.projection[:, 2]

    @property
    def prediction_result(self) -> SimplexResult:
        """Get prediction as SimplexResult for compatibility."""
        return SimplexResult(
            projection=self.projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    def compute_error(self) -> Dict[str, float]:
        """Compute prediction error statistics.

        Returns
        -------
        dict
            Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        from .Utils import ComputeError
        return ComputeError(self.observations, self.predictions)


@dataclass(frozen=True)
class CCMResult:
    """Results from Convergent Cross Mapping.

    Attributes
    ----------
    libMeans : numpy.ndarray
        Mean correlation at each library size.
        Shape (n_lib_sizes, 2 or 3):
        - Column 0: Library size
        - Column 1: Mean correlation for first direction
        - Column 2: Mean correlation for second direction (if applicable)
    embedDimensions : int
        Embedding dimension used
    predictionHorizon : int
        Prediction horizon used
    predictStats1 : numpy.ndarray, optional
        Detailed prediction statistics for first direction
        (only if includeData=True)
    predictStats2 : numpy.ndarray, optional
        Detailed prediction statistics for second direction
        (only if includeData=True)
    """
    libMeans: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    predictStats1: Optional[np.ndarray] = None
    predictStats2: Optional[np.ndarray] = None

    @property
    def library_sizes(self) -> np.ndarray:
        """Library sizes evaluated."""
        return self.libMeans[:, 0]

    @property
    def correlations(self) -> np.ndarray:
        """Correlation values (excludes library size column)."""
        return self.libMeans[:, 1:]


@dataclass(frozen=True)
class MultiviewResult:
    """Results from Multiview prediction.

    Attributes
    ----------
    projection : numpy.ndarray
        Ensemble-averaged prediction array [Time, Observations, Predictions]
    view : list
        Rankings of column combinations. Each element is
        [combo_string, correlation, MAE, CAE, RMSE]
    topRankProjections : dict
        Dictionary mapping column combinations (tuples) to their
        prediction arrays [Time, Observations, Predictions, Variance]
    topRankStats : dict
        Dictionary mapping column combinations (tuples) to their
        error statistics {'correlation', 'MAE', 'CAE', 'RMSE'}
    D : int
        State-space dimension used
    embedDimensions : int
        Embedding dimension for each variable
    predictionHorizon : int
        Prediction horizon used
    """
    projection: np.ndarray
    view: List
    topRankProjections: Dict
    topRankStats: Dict
    D: int
    embedDimensions: int
    predictionHorizon: int

    @property
    def time(self) -> np.ndarray:
        """Time values from projection."""
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """Observed values from projection."""
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """Ensemble-averaged predictions."""
        return self.projection[:, 2]

    @property
    def top_combinations(self) -> List:
        """Get list of top-ranked column combinations."""
        return list(self.topRankProjections.keys())

    def compute_error(self) -> Dict[str, float]:
        """Compute prediction error statistics for ensemble prediction.

        Returns
        -------
        dict
            Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        from .Utils import ComputeError
        return ComputeError(self.observations, self.predictions)

    def get_combination_stats(self, combo: tuple) -> Dict[str, float]:
        """Get error statistics for a specific column combination.

        Parameters
        ----------
        combo : tuple
            Column combination (e.g., (0, 2, 4))

        Returns
        -------
        dict
            Error statistics for this combination
        """
        if combo not in self.topRankStats:
            raise ValueError(f"Combination {combo} not in top-ranked results")
        return self.topRankStats[combo]
