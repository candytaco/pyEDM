# code/eval.py
import argparse
import pickle
import pandas as pd
from code.utils import regression_metrics

def eval_main(model_pickle: str, ground_truth_csv: str, target: str):
    """
    Load trained MDE_CV model and evaluate predictions on the test set.
    """
    with open(model_pickle, 'rb') as f:
        mde = pickle.load(f)

    preds = mde.predict()
    pred_incremental = mde.predict_incremental(plot=0)
    # Compute and print metrics
    y_true = preds['Observations'].values
    y_pred = preds['Predictions'].values
    rmse, mae, corr = regression_metrics(y_true, y_pred)
    print(f"Evaluation Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Corr: {corr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an MDE_CV model")
    parser.add_argument("--model_pickle", required=True, help="Path to trained model pickle")
    parser.add_argument("--ground_truth_csv", required=True, help="CSV with ground truth (unused if preds contain Observations)")
    parser.add_argument("--target", required=True, help="Target column name (unused)")
    args = parser.parse_args()
    eval_main(args.model_pickle, args.ground_truth_csv, args.target)