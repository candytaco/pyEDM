"""Parameter configuration classes for pyEDM.

This module provides dataclasses for organizing and validating parameters
used across different EDM methods. Using parameter objects instead of
individual parameters provides type safety, validation, and a single source
of truth for parameter definitions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy
from .Execution import ExecutionMode


@dataclass
class EDMParameters:
    """Common parameters for all EDM methods.

    Parameters
    ----------
    data : numpy.ndarray
        2D numpy array where column 0 is time (unless noTime=True)
    columns : list of int, optional
        Column indices to use for embedding (defaults to all except time)
    target : int, optional
        Target column index (defaults to column 1)
    embedDimensions : int, default=0
        Embedding dimension (E). If 0, will be set by Validate()
    predictionHorizon : int, default=1
        Prediction time horizon (Tp)
    knn : int, default=0
        Number of nearest neighbors. If 0, will be set to E+1 by Validate()
    step : int, default=-1
        Time delay step size (tau). Negative values indicate lag
    exclusionRadius : int, default=0
        Temporal exclusion radius for neighbors
    embedded : bool, default=False
        Whether data is already embedded
    validLib : list, optional
        Boolean mask for valid library points
    noTime : bool, default=False
        Whether first column is time or data
    ignoreNan : bool, default=True
        Remove NaN values from embedding
    verbose : bool, default=False
        Print diagnostic messages
    """
    data: numpy.ndarray
    columns: Optional[List[int]] = None
    target: Optional[int] = None
    embedDimensions: int = 0
    predictionHorizon: int = 1
    knn: int = 0
    step: int = -1
    exclusionRadius: int = 0
    embedded: bool = False
    validLib: List = field(default_factory=list)
    noTime: bool = False
    ignoreNan: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.data is None:
            raise ValueError("data parameter is required")
        if len(self.data.shape) != 2:
            raise ValueError("data must be a 2D numpy array")


@dataclass
class DataSplit:
    """Train/test split configuration.

    Parameters
    ----------
    train : tuple of (int, int), optional
        Training set indices [start, end]
    test : tuple of (int, int), optional
        Test set indices [start, end]
    """
    train: Optional[Tuple[int, int]] = None
    test: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """Validate split parameters."""
        if self.train is not None:
            if len(self.train) != 2:
                raise ValueError("train must be a tuple of (start, end)")
            if self.train[0] > self.train[1]:
                raise ValueError("train start must be <= train end")

        if self.test is not None:
            if len(self.test) != 2:
                raise ValueError("test must be a tuple of (start, end)")
            if self.test[0] > self.test[1]:
                raise ValueError("test start must be <= test end")


@dataclass
class GenerationParameters:
    """Parameters for iterative generation.

    Parameters
    ----------
    generateSteps : int, default=0
        Number of iterative generation steps. If 0, uses standard prediction.
    generateConcat : bool, default=False
        Whether to concatenate generated predictions
    """
    generateSteps: int = 0
    generateConcat: bool = False

    def __post_init__(self):
        """Validate generation parameters."""
        if self.generateSteps < 0:
            raise ValueError("generateSteps must be non-negative")


@dataclass
class SMapParameters:
    """S-Map specific parameters.

    Parameters
    ----------
    theta : float, default=0.0
        S-Map localization parameter. theta=0 is global linear map,
        larger values increase localization
    solver : object, optional
        Solver to use for S-Map regression. If None, uses numpy.linalg.lstsq.
        Can be any sklearn-compatible regressor.
    """
    theta: float = 0.0
    solver: Optional[object] = None

    def __post_init__(self):
        """Validate S-Map parameters."""
        if self.theta < 0:
            raise ValueError("theta must be non-negative")

        if self.solver is not None:
            # Validate solver is from supported list
            supportedSolvers = [
                'function', 'lstsq',
                'LinearRegression', 'SGDRegressor',
                'Ridge', 'RidgeCV',
                'Lasso', 'LassoCV',
                'Lars', 'LarsCV',
                'LassoLars', 'LassoLarsCV', 'LassoLarsIC',
                'ElasticNet', 'ElasticNetCV',
                'OrthogonalMatchingPursuit',
                'OrthogonalMatchingPursuitCV'
            ]
            if self.solver.__class__.__name__ not in supportedSolvers:
                raise ValueError(
                    f'Invalid solver {self.solver.__class__.__name__}. '
                    f'Supported solvers: {supportedSolvers}'
                )


@dataclass
class CCMParameters:
    """CCM (Convergent Cross Mapping) specific parameters.

    Parameters
    ----------
    trainSizes : list of int, optional
        Library sizes to evaluate [start, stop, increment].
        For example, [10, 100, 10] tests library sizes 10, 20, ..., 100.
    sample : int, default=0
        Number of random samples at each library size. If 0, uses all available.
    seed : int, optional
        Random seed for reproducible sampling
    includeData : bool, default=False
        Whether to include detailed prediction statistics in results
    """
    trainSizes: List[int] = field(default_factory=list)
    sample: int = 0
    seed: Optional[int] = None
    includeData: bool = False

    def __post_init__(self):
        """Validate CCM parameters."""
        if self.sample < 0:
            raise ValueError("sample must be non-negative")

        if len(self.trainSizes) > 0:
            if len(self.trainSizes) != 3:
                raise ValueError(
                    "trainSizes must be [start, stop, increment] "
                    f"but got {len(self.trainSizes)} elements"
                )
            if self.trainSizes[0] <= 0:
                raise ValueError("trainSizes start must be positive")
            if self.trainSizes[1] < self.trainSizes[0]:
                raise ValueError("trainSizes stop must be >= start")
            if self.trainSizes[2] <= 0:
                raise ValueError("trainSizes increment must be positive")


@dataclass
class ExecutionParameters:
    """Execution and multiprocessing parameters.

    Parameters
    ----------
    numProcess : int, default=4
        Number of processes for multiprocessing
    mpMethod : ExecutionMode, optional
        Multiprocessing context method (ExecutionMode.SPAWN, ExecutionMode.FORK, ExecutionMode.FORKSERVER)
        If None, uses platform default
    chunksize : int, default=1
        Chunk size for pool.starmap operations
    sequential : bool, default=False
        Use sequential execution instead of multiprocessing
    """
    numProcess: int = 4
    mpMethod: Optional[ExecutionMode] = None
    chunksize: int = 1
    sequential: bool = False

    def __post_init__(self):
        """Validate execution parameters."""
        if self.numProcess < 1:
            raise ValueError("numProcess must be positive")
        if self.chunksize < 1:
            raise ValueError("chunksize must be positive")
        if self.mpMethod is not None:
            if not isinstance(self.mpMethod, ExecutionMode):
                raise ValueError(f"mpMethod must be an ExecutionMode enum value")


@dataclass
class MultiviewParameters:
    """Multiview specific parameters.

    Parameters
    ----------
    D : int, default=0
        State-space dimension (number of variables to combine for each
        assessment). If 0, defaults to number of columns.
    multiview : int, default=0
        Number of top-ranked D-dimensional predictions for final ensemble
        (parameter k in Ye & Sugihara). If 0, defaults to sqrt(m) where m
        is the number of combinations C(n,D).
    trainLib : bool, default=True
        Evaluation strategy for ranking column combinations:
        - True: Use in-sample evaluation (test=train during Rank phase).
                Faster but may overfit to arbitrary vectors.
        - False: Use proper out-of-sample evaluation with specified train/test.
                More rigorous but computationally expensive.
                Requires explicit train and test parameters.
    excludeTarget : bool, default=False
        Whether to exclude target column from embedding combinations
    """
    D: int = 0
    multiview: int = 0
    trainLib: bool = True
    excludeTarget: bool = False

    def __post_init__(self):
        """Validate Multiview parameters.

        trainLib behavior:
        - True (default): Use in-sample evaluation for ranking top combinations
                         (sets test=train during ranking phase). This is faster
                         but may overfit to arbitrary non-constant vectors.
        - False: Use proper out-of-sample evaluation with specified train/test
                splits for ranking. Requires explicit train and test parameters.
        """
        if self.D < 0:
            raise ValueError("D must be non-negative")
        if self.multiview < 0:
            raise ValueError("multiview must be non-negative")

@dataclass
class MDEParameters:
    """Multivariate Delay Embedding specific parameters.

    Parameters
    ----------
    target : int
        Column index of the target column to forecast
    maxD : int, default=5
        Maximum number of features to select (including target if include_target=True)
    include_target : bool, default=True
        Whether to start with target in feature list
    convergent : bool, default=True
        Whether to use convergence checking for feature selection
    metric : str, default="correlation"
        Metric to use: "correlation" or "MAE"
    batch_size : int, default=1000
        Number of features to process in each parallel batch
    """
    target: int
    maxD: int = 5
    include_target: bool = True
    convergent: bool = True
    metric: str = "correlation"
    batch_size: int = 1000

    def __post_init__(self):
        """Validate MDE parameters."""
        if self.maxD < 1:
            raise ValueError("maxD must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.metric not in ["correlation", "MAE"]:
            raise ValueError("optimize_for must be 'correlation' or 'MAE'")

@dataclass
class MDECVParameters:
    """MDE Cross-Validation specific parameters.

    Parameters
    ----------
    folds : int, default=5
        Number of cross-validation folds
    test_size : float, default=0.2
        Proportion of data to use for test set
    final_feature_mode : str, default="best_fold"
        Method for selecting final features:
        - "best_fold": Use features from best performing fold
        - "frequency": Use most frequent features across folds
        - "best_N": Use top N features based on incremental prediction
    """
    folds: int = 5
    test_size: float = 0.2
    final_feature_mode: str = "best_fold"

    def __post_init__(self):
        """Validate MDECV parameters."""
        if self.folds < 1:
            raise ValueError("folds must be at least 1")
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.final_feature_mode not in ["best_fold", "frequency", "best_N"]:
            raise ValueError("final_feature_mode must be 'best_fold', 'frequency', or 'best_N'")
