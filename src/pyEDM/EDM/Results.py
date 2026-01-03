"""
Result classes for pyEDM predictions.

This module provides dataclasses for structured prediction results from different EDM methods. Using result objects instead of conditional return types provides consistency, self-documentation, and convenient access to results and metadata.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

from ..Utils import ComputeError


@dataclass(frozen=True)
class SimplexResult:
    """
    Results from Simplex prediction.

    :param projection: Array with columns [Time, Observations, Predictions]
    :param embedDimensions: Embedding dimension used
    :param predictionHorizon: Prediction horizon used
    """
    projection: np.ndarray
    embedDimensions: int
    predictionHorizon: int

    @property
    def time(self) -> np.ndarray:
        """
        Time values from projection.
        """
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from projection.
        """
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Predicted values from projection.
        """
        return self.projection[:, 2]

    def compute_error(self, metric = None) -> float:
        """
        Compute prediction error statistics.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)


@dataclass(frozen=True)
class SMapResult:
    """
    Results from S-Map prediction.

    :param projection: Array with columns [Time, Observations, Predictions]
    :param coefficients: S-Map coefficients for each prediction (N_pred, E+1)
    :param singularValues: Singular values from SVD for each prediction (N_pred, E+1)
    :param embedDimensions: Embedding dimension used
    :param predictionHorizon: Prediction horizon used
    :param theta: Localization parameter used
    """
    projection: np.ndarray
    coefficients: np.ndarray
    singularValues: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    theta: float

    @property
    def time(self) -> np.ndarray:
        """
        Time values from projection.
        """
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from projection.
        """
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Predicted values from projection.
        """
        return self.projection[:, 2]

    @property
    def prediction_result(self) -> SimplexResult:
        """
        Get prediction as SimplexResult for compatibility.
        """
        return SimplexResult(
            projection=self.projection,
            embedDimensions=self.embedDimensions,
            predictionHorizon=self.predictionHorizon
        )

    def compute_error(self, metric = None) -> Dict[str, float]:
        """
        Compute prediction error statistics.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)


@dataclass(frozen=True)
class CCMResult:
    """
    Results from Convergent Cross Mapping.

    :param libMeans: Mean correlation at each library size. Shape (n_lib_sizes, 2 or 3): 
    	Column 0: Library size, 
        Column 1: Mean correlation for first direction, 
        Column 2: Mean correlation for second direction (if applicable)
    :param embedDimensions: Embedding dimension used
    :param predictionHorizon: Prediction horizon used
    :param predictStats1: Detailed prediction statistics for first direction (only if includeData=True)
    :param predictStats2: Detailed prediction statistics for second direction (only if includeData=True)
    """
    libMeans: np.ndarray
    embedDimensions: int
    predictionHorizon: int
    predictStats1: Optional[np.ndarray] = None
    predictStats2: Optional[np.ndarray] = None

    @property
    def library_sizes(self) -> np.ndarray:
        """
        Library sizes evaluated.
        """
        return self.libMeans[:, 0]

    @property
    def correlations(self) -> np.ndarray:
        """
        Correlation values (excludes library size column).
        """
        return self.libMeans[:, 1:]


@dataclass(frozen=True)
class MultiviewResult:
    """
    Results from Multiview prediction.

    :param projection: Ensemble-averaged prediction array [Time, Observations, Predictions]
    :param view: Rankings of column combinations. Each element is [combo_string, correlation, MAE, CAE, RMSE]
    :param topRankProjections: Dictionary mapping column combinations (tuples) to their prediction arrays [Time, Observations, Predictions, Variance]
    :param topRankStats: Dictionary mapping column combinations (tuples) to their error statistics {'correlation', 'MAE', 'CAE', 'RMSE'}
    :param D: State-space dimension used
    :param embedDimensions: Embedding dimension for each variable
    :param predictionHorizon: Prediction horizon used
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
        """
        Time values from projection.
        """
        return self.projection[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from projection.
        """
        return self.projection[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Ensemble-averaged predictions.
        """
        return self.projection[:, 2]

    @property
    def top_combinations(self) -> List:
        """
        Get list of top-ranked column combinations.
        """
        return list(self.topRankProjections.keys())

    def compute_error(self, metric = None) -> Dict[str, float]:
        """
        Compute prediction error statistics for ensemble prediction.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)

    def get_combination_stats(self, combo: tuple) -> Dict[str, float]:
        """
        Get error statistics for a specific column combination.

        :param combo: Column combination (e.g., (0, 2, 4))
        :return: Error statistics for this combination
        :raises ValueError: if combination not in top-ranked results
        """
        if combo not in self.topRankStats:
            raise ValueError(f"Combination {combo} not in top-ranked results")
        return self.topRankStats[combo]


@dataclass(frozen=True)
class MDEResult:
    """
    Results from Multivariate Delay Embedding.

    :param final_forecast: Final forecast array [Time, Observations, Predictions]
    :param selected_features: Column indices of selected features
    :param accuracy: Correlation/MAE at each feature addition step
    :param ccm_values: CCM convergence values for selected features
    """
    final_forecast: np.ndarray
    selected_features: List[int]
    accuracy: List[float]
    ccm_values: List[float]

    @property
    def time(self) -> np.ndarray:
        """
        Time values from forecast.
        """
        return self.final_forecast[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from forecast.
        """
        return self.final_forecast[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Predicted values from forecast.
        """
        return self.final_forecast[:, 2]

    def compute_error(self, metric = None) -> Dict[str, float]:
        """
        Compute prediction error statistics.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)


@dataclass(frozen=True)
class MDECVResult:
    """
    Results from MDE Cross-Validation.

    :param final_forecast: Final forecast array from test set prediction
    :param selected_features: Final selected feature indices
    :param fold_results: Results from each cross-validation fold
    :param accuracy: Test set accuracy for each fold
    :param best_fold: Index of best performing fold
    """
    final_forecast: np.ndarray
    selected_features: List[int]
    fold_results: List[MDEResult]
    accuracy: List[float]
    best_fold: int

    @property
    def time(self) -> np.ndarray:
        """
        Time values from forecast.
        """
        return self.final_forecast[:, 0]

    @property
    def observations(self) -> np.ndarray:
        """
        Observed values from forecast.
        """
        return self.final_forecast[:, 1]

    @property
    def predictions(self) -> np.ndarray:
        """
        Predicted values from forecast.
        """
        return self.final_forecast[:, 2]

    def compute_error(self, metric = None) -> Dict[str, float]:
        """
        Compute prediction error statistics.

        :param metric: Error metric to compute
        :return: Dictionary with keys: 'correlation', 'MAE', 'CAE', 'RMSE'
        """
        return ComputeError(self.observations, self.predictions, metric)
