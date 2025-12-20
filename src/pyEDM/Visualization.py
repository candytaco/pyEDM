"""Visualization functions for pyEDM results.

This module provides plotting functions for EDM prediction results.
Functions work with both legacy numpy arrays and new Result objects.
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import show, axhline
from typing import Union
import numpy as np


def plot_prediction(result: Union['SimplexResult', 'SMapResult', 'MultiviewResult', np.ndarray],
                   title: str = "",
                   embedDimensions: int = None,
                   predictionHorizon: int = None,
                   block: bool = True):
    """Plot observations vs predictions.

    Parameters
    ----------
    result : SimplexResult, SMapResult, MultiviewResult, or numpy.ndarray
        Result object or legacy numpy array with columns [Time, Observations, Predictions]
    title : str, optional
        Additional title text
    embedDimensions : int, optional
        Embedding dimension (only needed if passing numpy array)
    predictionHorizon : int, optional
        Prediction horizon (only needed if passing numpy array)
    block : bool, default=True
        Whether to block execution when showing plot

    Examples
    --------
    >>> result = Simplex(params, split).run()
    >>> plot_prediction(result)

    >>> # Or with numpy array (legacy)
    >>> plot_prediction(projection_array, embedDimensions=3, predictionHorizon=1)
    """
    from .Utils import ComputeError

    # Handle both Result objects and numpy arrays
    if hasattr(result, 'projection'):
        # It's a Result object
        data = result.projection
        E = result.embedDimensions
        Tp = result.predictionHorizon
    else:
        # It's a numpy array (legacy)
        data = result
        E = embedDimensions or 0
        Tp = predictionHorizon or 0

    # Compute error statistics
    stats = ComputeError(data[:, 1], data[:, 2])

    # Build title
    plot_title = title
    if plot_title:
        plot_title += "\n"
    plot_title += f"Embedding Dims = {E}  predictionHorizon={Tp}  " \
                  f"correlation={round(stats['correlation'], 3)}  " \
                  f"RMSE={round(stats['RMSE'], 3)}"

    # Create plot
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], label='Observations', linewidth=3)
    plt.plot(data[:, 0], data[:, 2], label='Predictions', linewidth=3)
    plt.title(plot_title)
    plt.legend()
    plt.show(block=block)


def plot_smap_coefficients(result: Union['SMapResult', np.ndarray],
                          title: str = "",
                          embedDimensions: int = None,
                          predictionHorizon: int = None,
                          block: bool = True):
    """Plot S-Map coefficients over time.

    Parameters
    ----------
    result : SMapResult or numpy.ndarray
        SMap result object or legacy numpy array with columns [Time, Coeff_0, Coeff_1, ...]
    title : str, optional
        Additional title text
    embedDimensions : int, optional
        Embedding dimension (only needed if passing numpy array)
    predictionHorizon : int, optional
        Prediction horizon (only needed if passing numpy array)
    block : bool, default=True
        Whether to block execution when showing plot

    Examples
    --------
    >>> result = SMap(params, split, smap_params).run()
    >>> plot_smap_coefficients(result)
    """
    # Handle both SMapResult and numpy arrays
    if hasattr(result, 'coefficients'):
        # It's an SMapResult object
        data = result.coefficients
        E = result.embedDimensions
        Tp = result.predictionHorizon
    else:
        # It's a numpy array (legacy)
        data = result
        E = embedDimensions or 0
        Tp = predictionHorizon or 0

    # Build title
    plot_title = title
    if plot_title:
        plot_title += "\n"
    plot_title += f"Embedding Dims = {E}  predictionHorizon={Tp}  S-Map Coefficients"

    # Create subplots for each coefficient
    n_coeff = data.shape[1] - 1 if data.shape[1] > 1 else data.shape[1]

    plt.figure()
    for i in range(1, data.shape[1]):
        plt.subplot(data.shape[1] - 1, 1, i)
        plt.plot(data[:, 0], data[:, i], linewidth=3)
        plt.title(f'Coefficient {i-1}')

    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show(block=block)


def plot_ccm(result: Union['CCMResult', np.ndarray],
            title: str = "",
            embedDimensions: int = None,
            block: bool = True):
    """Plot CCM convergence.

    Parameters
    ----------
    result : CCMResult or numpy.ndarray
        CCM result object or legacy numpy array with columns [LibSize, Correlation_1, Correlation_2]
    title : str, optional
        Additional title text
    embedDimensions : int, optional
        Embedding dimension (only needed if passing numpy array)
    block : bool, default=True
        Whether to block execution when showing plot

    Examples
    --------
    >>> result = CCM(params, ccm_params).run()
    >>> plot_ccm(result)
    """
    # Handle both CCMResult and numpy arrays
    if hasattr(result, 'libMeans'):
        # It's a CCMResult object
        data = result.libMeans
        E = result.embedDimensions
    else:
        # It's a numpy array (legacy)
        data = result
        E = embedDimensions or 0

    # Build title
    plot_title = title or f'E = {E}'

    fig, ax = plt.subplots()

    # Check if we have two directions or one
    if data.shape[1] == 3:
        # CCM of two different variables
        ax.plot(data[:, 0], data[:, 1], linewidth=3, label='Direction 1')
        ax.plot(data[:, 0], data[:, 2], linewidth=3, label='Direction 2')
        ax.legend()
    elif data.shape[1] == 2:
        # CCM of degenerate columns (single direction)
        ax.plot(data[:, 0], data[:, 1], linewidth=3)

    ax.set(xlabel="Library Size",
          ylabel="CCM correlation",
          title=plot_title)
    axhline(y=0, linewidth=1)
    show(block=block)


def plot_multiview(result: Union['MultiviewResult', np.ndarray],
                  title: str = "",
                  block: bool = True):
    """Plot Multiview ensemble prediction.

    Parameters
    ----------
    result : MultiviewResult or numpy.ndarray
        Multiview result object or legacy numpy array
    title : str, optional
        Additional title text
    block : bool, default=True
        Whether to block execution when showing plot

    Examples
    --------
    >>> result = Multiview(params, split, mv_params).run()
    >>> plot_multiview(result)
    """
    # Use plot_prediction for the ensemble result
    if hasattr(result, 'projection'):
        plot_prediction(result, title=title, block=block)
    else:
        plot_prediction(result, title=title, block=block)


# Legacy function names for backward compatibility
def PlotObsPred(data, dataName="", embedDimensions=0, predictionHorizon=0, block=True):
    """Legacy function for plotting observations vs predictions.

    .. deprecated::
        Use plot_prediction() instead.
    """
    plot_prediction(data, title=dataName, embedDimensions=embedDimensions,
                   predictionHorizon=predictionHorizon, block=block)


def PlotCoeff(data, dataName="", embedDimensions=0, predictionHorizon=0, block=True):
    """Legacy function for plotting S-Map coefficients.

    .. deprecated::
        Use plot_smap_coefficients() instead.
    """
    plot_smap_coefficients(data, title=dataName, embedDimensions=embedDimensions,
                          predictionHorizon=predictionHorizon, block=block)
