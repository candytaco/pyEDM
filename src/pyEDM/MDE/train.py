# mde_code/train.py
import pandas as pd
import os
import argparse
import pickle
from mde_code.core import MDE_CV
from mde_code.utils import estimate_best_params

def train_main(input_csv, target: str, output_model: str):
    """
    Train an MDE_CV model by first estimating best E and tau.

    Parameters:
        input_csv (str or pd.DataFrame): Path to CSV file or a loaded DataFrame
        target (str): Target column name
        output_model (str): Path to save the trained model

    Returns:
        Trained MDE_CV model instance
    """
    # Load if input is a path
    if isinstance(input_csv, str):
        df = pd.read_csv(input_csv)
    elif isinstance(input_csv, pd.DataFrame):
        df = input_csv
    else:
        raise TypeError("input_csv must be a file path or a pandas DataFrame.")

    # Estimate parameters
    best = estimate_best_params(df, target)
    E = int(best['E'][0])
    tau = int(best['tau'][0])
    print(f"Training MDE_CV model for target '{target}' with E={E}, tau={tau}")

    mde = MDE_CV(
        Tp=1,
        maxD=50,
        folds=10,
        test_size=0.2,
        plot=False,
        optimize_for="correlation",
        conv=True,
        include_target=False,
        smap=False,
        final_feature_mode="frequency"
    )
    mde.fit(df, target=target, E=E, tau=tau)

    # Save trained model
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    with open(output_model, 'wb') as f:
        pickle.dump(mde, f)
    print(f"Model saved to {output_model}")

    return mde