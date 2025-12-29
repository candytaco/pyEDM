import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pyEDM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def check_convergence(ts_df: pd.DataFrame, target: str, candidate: str, E: int, libSizes: str, plot: bool = False) -> bool:
    """
    Check the convergence of a candidate variable using CCM.
    """
    try:
        logging.info(f"Convergence check for candidate: {candidate}")
        ccm = pyEDM.CCM(
            dataFrame=ts_df,
            E=E,
            columns=candidate,
            target=str(target),
            libSizes=libSizes,
            sample=100,
            showPlot=plot
        )
        ccm = ccm.dropna()
        key = f"{target}:{candidate}"
        y_data = ccm[key].values
        x_data = ccm['LibSize'].values
        x_data = x_data.reshape(len(x_data), 1)
        y_data = y_data.reshape(len(y_data), 1)
        regr = LinearRegression()
        regr.fit(x_data, y_data)
        final_ccm = y_data[-1]
        conv = (regr.coef_ > 0).all()  # True if all coefficients are > 0
    except Exception as e:
        logging.info(f"Error in check_convergence(): {e}")
        conv = False
    return bool(conv), final_ccm

def check_convergence_improved(ts_df: pd.DataFrame, target: str, candidate: str, E: int, libSizes: str, plot: bool = False):

    try:
        ccm = pyEDM.CCM(
            dataFrame=ts_df,
            E=E,
            columns=candidate,
            target=target,
            libSizes=libSizes,
            sample=100,
            showPlot=False
        ).dropna()

        key = f"{target}:{candidate}"
        x = ccm['LibSize'].values
        y = ccm[key].values

        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(x, y, marker='o')
            plt.title(f'CCM convergence: {candidate} → {target}')
            plt.xlabel('Library Size')
            plt.ylabel('Cross Map Skill (ρ)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # New logic
        net_gain = y[-1] - y[0]
        early_gain = y[1] - y[0] if len(y) > 1 else 0
        final_close_to_max = y[-1] >= 0.9 * np.max(y)

        conv = (net_gain > 0.01) and (early_gain > 0) and final_close_to_max
        final_ccm = y[-1]

        logging.info(f"Converged: {conv}, Final CCM: {final_ccm:.4f}")

    except Exception as e:
        logging.info(f"Error in check_convergence(): {e}")
        conv = False
        final_ccm = 0

    return bool(conv), final_ccm


def reset_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset the 'time' column in the DataFrame by dropping any existing 'time'
    column and inserting a new one based on the current row order.
    """
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    df.insert(0, "time", np.arange(len(df)))
    return df

def compute_correlation(observations: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient between observations and predictions.
    """
    if len(observations) == 0 or len(predictions) == 0:
        return np.nan
    corr_matrix = np.corrcoef(observations, predictions)
    return corr_matrix[0, 1]

def compute_cae(observations: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute the cumulative absolute error (CAE) between observations and predictions.
    """
    if len(observations) == 0 or len(predictions) == 0:
        return np.nan
    return np.sum(np.abs(observations - predictions))

def configure_logging(level=logging.INFO):
    """
    Configure logging for the application.
    """
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def plot_training_accuracy(accuracy, target):
    """
    Plot the prediction accuracy during training for a given target.

    Parameters:
    -----------
    accuracy : list or array-like
        List of training accuracy values where each value corresponds to the model's
        prediction accuracy at a given step (e.g., after each feature is added).
    target : str
        The target variable name, which will be included in the plot title.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(accuracy, 'o-')
    plt.title(f"Training Accuracy for {target}")
    plt.xlabel('Number of Features')
    plt.ylabel('Training Accuracy')
    plt.show()


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    # print(f'RMSE: {rmse:.3f}')
    # print(f'MAE: {mae:.3f}')
    # print(f'Correlation: {corr:.3f}')

    return rmse, mae, corr


def estimate_best_params(ts_df: pd.DataFrame, target: str, tau_values=None, max_E: int = 10):
    if tau_values is None:
        tau_values = [-1, -2, -3]

    best_rho = -np.inf
    best_E = None
    best_tau = None

    for tau in tau_values:
        out_E = pyEDM.EmbedDimension(
            dataFrame=ts_df,
            lib=f"1 {len(ts_df)}",
            pred=f"1 {len(ts_df)}",
            tau=tau,
            columns=target,
            target=target,
            maxE=max_E,
            exclusionRadius=0,
            showPlot=False
        )
        E_star = out_E['E'][np.argmax(out_E['rho'])]
        rho = out_E['rho'][np.argmax(out_E['rho'])]
        if rho > best_rho:
            best_rho = rho
            best_E = E_star
            best_tau = tau

    return pd.DataFrame([{
        'target': target,
        'tau': best_tau,
        'E': int(best_E),
        'rho': best_rho
    }])