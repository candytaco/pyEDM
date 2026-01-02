import numpy as np
import pandas as pd
import os
import pickle
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, KFold
from pyEDM import EmbedDimension
from pyEDM.EDM import Simplex, SMap
import warnings

warnings.filterwarnings(
	"ignore",
	message = "invalid value encountered in divide",
	category = RuntimeWarning
)

from .utils import (
	reset_time_column,
	check_convergence_improved,
	regression_metrics
)


###############################################################################
# 1) parallel_MDE with correlation or MAE, SMap or Simplex, include_target    #
###############################################################################

def parallel_MDE(
		ts_df: pd.DataFrame,
		target: str,
		Tp: int,
		maxD: int,
		tau: int = -1,
		columns: list = None,
		E: int = None,
		lib: str = "1 250",
		pred: str = "251 500",
		conv: bool = True,
		plot: bool = False,
		smap: bool = False,
		include_target: bool = True,
		batch_size: int = 1000,
		optimize_for: str = "correlation"
):
	"""
	Multi-feature selection for EDM using either Simplex or SMap,
	parallelized in batches to evaluate performance by correlation or MAE.

	By default, it starts with either [target] or an empty list
	(depending on include_target) or user-provided columns
	(if columns != None). Then it iteratively adds convergent variables
	up to maxD dimensions.

	Parameters
	----------
	ts_df : pd.DataFrame
		Time-series DataFrame. If it lacks 'time', we insert one for pyEDM.
	target : str
		Name of the target column to forecast.
	Tp : int
		Forecast horizon (time steps ahead).
	maxD : int
		Maximum number of features to select (including the target if include_target=True).
	tau : int, optional
		Time delay for embedding. Default=-1 (rarely used).
	columns : list or None, optional
		Starting columns. If None, the function uses:
		  - [target] if include_target=True
		  - [] otherwise
	E : int or None, optional
		Embedding dimension. If None, automatically determined via EmbedDimension (maxE=20).
	lib : str, optional
		Library indices for training in pyEDM format, e.g. '1 150'.
	pred : str, optional
		Prediction indices in pyEDM format, e.g. '151 428'.
	conv : bool, optional
		If > True, we pick the first candidate that meets check_convergence.
		If == False, pick top scoring candidate without convergence check.
	plot : bool, optional
		Whether to show plots in pyEDM calls. Default=False.
	smap : bool, optional
		If True, use SMap. Otherwise use Simplex. Default=False.
	include_target : bool, optional
		Whether to start with [target] in the feature list. If False,
		the target won't appear among the final features. Default=True.
	batch_size : int, optional
		Number of features to process in each parallel batch. Default=1000.
	optimize_for : str, optional
		Which metric to optimize: 'correlation' (maximize) or 'MAE' (also 'maximize' if using negative MSE).
		Default='correlation'.

	Returns
	-------
	S_final : pd.DataFrame
		The final forecast after selecting up to maxD features (evaluated on the 'pred' portion).
	cols : list
		The final set of features used.
	accuracy : list
		The correlation or MAE at each step of feature addition.
	"""
	if smap:
		print('Using SMap!')
	# Convert columns names to string dtype
	ts_df.columns = ts_df.columns.astype(str)
	target = str(target)
	ccm_values = []
	if 'time' not in ts_df.columns:
		ts_df.insert(0, 'time', np.arange(len(ts_df)))

	# If E is not specified, estimate it via EmbedDimension
	if E is None:
		dim = EmbedDimension(
			dataFrame = ts_df,
			columns = target,
			target = target,
			maxE = 10,
			lib = lib,
			pred = lib,
			Tp = Tp
		)
		E = dim['E'][np.argmax(dim['rho'])]
		print(f'[INFO] Estimated E={E} based on max correlation from EmbedDimension.')

	# Helper: run either Simplex or SMap on training portion
	def run_edm(df, lib_, pred_, E_, Tp_, tau_, col_list, target_, smap_, plot_, embedded):
		"""
		Calls either SMap or Simplex on the library portion (for univariate or partial).
		"""
		if smap_:
			out = SMap(
				dataFrame = df,
				lib = lib_,
				pred = lib_,
				E = E_,
				Tp = Tp_,
				tau = tau_,
				columns = col_list,
				target = target_,
				theta = 1,
				showPlot = plot_,
				embedded = embedded
			)
			return out['predictions']  # extract the 'predictions' DataFrame
		else:
			out = Simplex(
				dataFrame = df,
				lib = lib_,
				pred = lib_,
				E = E_,
				Tp = Tp_,
				tau = tau_,
				columns = col_list,
				target = target_,
				showPlot = plot_,
				embedded = embedded
			)
			return out

	# Decide which columns we start with
	if columns is None:
		cols = [target]
	else:
		cols = [str(c) for c in columns]  # user-provided columns

	# Univariate or initial forecast (training portion only)
	S_univ = run_edm(ts_df, lib, pred, E, Tp, tau, cols, target, smap, plot, embedded = False)
	if not include_target:
		cols = []

	rmse = mae = corr = None

	if not S_univ.empty:
		S_univ.dropna(inplace = True)
		obs_univ, preds_univ = S_univ[['Observations', 'Predictions']].values.T
		rmse, mae, corr = regression_metrics(obs_univ, preds_univ)
		if optimize_for == "correlation":
			rho_univ = corr
		else:
			rho_univ = mae
	else:
		rho_univ = np.nan
	print(f'[INFO] Initial (univariate) {optimize_for}={rho_univ:.4f} with columns={cols}')

	# Build the list of possible columns to add
	def is_excluded(c_):
		if c_ == 'time':
			return True
		if c_ in cols:
			return True
		if (not include_target) and (c_ == target):
			return True
		return False

	remaining_columns = [col for col in ts_df.columns if not is_excluded(col)]
	# Keep track of correlation/MAE as we add features
	accuracy = [rho_univ]
	# For 'check_convergence'
	step = str(round(int(lib.split()[1]) / 5))
	libSizes = f"50 {lib.split()[1]} {step}"

	# print(libSizes)

	# Function to compute correlation or MAE for a batch of columns in parallel
	def compute_metric_batch(batch, selected_cols, optimize_for):
		results = []
		for c_ in batch:
			if smap:
				S_ = SMap(
					dataFrame = ts_df,
					lib = lib,
					pred = lib,
					E = E,
					Tp = Tp,
					tau = tau,
					columns = [c_] + selected_cols,
					target = target,
					theta = 1,
					embedded = True
				)
				S_ = S_['predictions']
			else:
				S_ = Simplex(
					dataFrame = ts_df,
					lib = lib,
					pred = lib,
					E = E,
					Tp = Tp,
					tau = tau,
					columns = [c_] + selected_cols,
					target = target,
					embedded = True
				)
			S_.dropna(inplace = True)
			df_sub = S_[['Observations', 'Predictions']]
			# Compute evaluation metrics on current prediction
			rmse = mae = corr = None
			if df_sub.empty:
				m_val = np.nan
			else:
				o_, p_ = df_sub.values.T
				rmse, mae, corr = regression_metrics(o_, p_)
				if optimize_for == "correlation":
					m_val = corr
				else:
					m_val = mae

			results.append((c_, m_val))
		return results

	# Iteratively add features up to maxD
	while len(cols) < maxD and remaining_columns:
		# Break up 'remaining_columns' into parallel-friendly batches
		batches = [
			remaining_columns[i: i + batch_size]
			for i in range(0, len(remaining_columns), batch_size)
		]
		# Evaluate correlation/MAE for each possible addition in parallel
		batch_results = Parallel(n_jobs = -1)(
			delayed(compute_metric_batch)(batch, cols, optimize_for) for batch in batches
		)
		metric_results = [item for sublist in batch_results for item in sublist]
		# Sort descending by correlation/MAE
		metric_results.sort(key = lambda x: (x[1] if x[1] is not None else -np.inf), reverse = True)

		best_var = None
		best_score = None

		# If conv=True, we use the first 'convergent' variable in sorted order
		if conv:
			for c_, sc_ in metric_results:
				if c_ is None or np.isnan(sc_):
					continue
				# check 'convergence' with CCM
				check = check_convergence_improved(ts_df, target, c_, E, libSizes, plot = plot)
				# check = check_convergence(ts_df, target, c_, E, libSizes, plot=plot)
				if check[0]:
					best_var = c_
					best_score = sc_
					print(
						f'[INFO] #{len(cols)} Adding {c_} with {optimize_for}={best_score:.4f} (convergence {check[1]})')
					ccm_values.append(check[1])
					break
				else:
					remaining_columns.remove(c_)
		else:
			# If conv == False, pick the top metric
			if metric_results and not np.isnan(metric_results[0][1]):
				best_var = metric_results[0][0]
				best_score = metric_results[0][1]
				print(
					f'[INFO] #{len(cols)} Adding {best_var} with {optimize_for}={best_score:.4f} (no convergence check)')

		# If we found something valid, add it
		if best_var:
			cols.append(best_var)
			remaining_columns.remove(best_var)
			accuracy.append(best_score)  # append training accuracy for each added variables
		else:
			# No more convergent variables or all had invalid metric
			print(f'[INFO] Stopping at {len(cols)} features; no next variable chosen.')
			break

	# Final training with the selected columns on the entire library
	print(f'[INFO] Final training on columns={cols} with lib={lib}')
	if smap:
		S_temp = SMap(
			dataFrame = ts_df,
			lib = lib,
			pred = lib,  # training portion
			E = E,
			Tp = Tp,
			tau = tau,
			columns = cols,
			target = target,
			theta = 1,
			embedded = True,
			showPlot = plot
		)
		S_temp = S_temp['predictions']
	else:
		S_temp = Simplex(
			dataFrame = ts_df,
			lib = lib,
			pred = lib,  # training portion
			E = E,
			Tp = Tp,
			tau = tau,
			columns = cols,
			target = target,
			embedded = True,
			showPlot = plot
		)

	# Test on the validation set
	print(f'[INFO] Testing on pred={pred}')
	if smap:
		S_final = SMap(
			dataFrame = ts_df,
			lib = lib,
			pred = pred,
			E = E,
			Tp = Tp,
			tau = tau,
			columns = cols,
			target = target,
			theta = 1,
			embedded = True,
			showPlot = plot
		)
		S_final = S_final['predictions']
	else:
		S_final = Simplex(
			dataFrame = ts_df,
			lib = lib,
			pred = pred,
			E = E,
			Tp = Tp,
			tau = tau,
			columns = cols,
			target = target,
			embedded = True,
			showPlot = plot
		)

	return S_final, cols, accuracy, ccm_values


###############################################################################
# 3) MDE_CV with final_feature_mode                                           #
###############################################################################

class MDE_CV:
	"""
	Cross-validation wrapper that uses parallel_MDE for feature selection in each fold.
	Also allows picking final features by:
	  - best_fold
	  - frequency (most frequent features across folds)
	  - best_N

	"""

	def __init__(
			self,
			Tp: int,
			maxD: int,
			folds: int = 5,
			test_size: float = 0.2,
			plot: bool = False,
			optimize_for: str = "correlation",
			conv: bool = True,
			include_target: bool = True,
			smap: bool = False,
			final_feature_mode: str = "best_fold"  # or "frequency" or "best_N"
	):
		self.Tp = Tp
		self.maxD = maxD
		self.folds = folds
		self.test_size = test_size
		self.plot = plot
		self.optimize_for = optimize_for
		self.conv = conv
		self.include_target = include_target
		self.smap = smap
		self.final_feature_mode = final_feature_mode

		# CV results
		self.selected_features_per_fold = []
		self.rho_test_per_fold = []
		self.fold_details = []
		self.fold_features = []
		self.train_set = None
		self.test_set = None
		self.target = None
		self.E_fold = None

		self.best_fold_idx = None
		self.best_fold_features = None
		self.best_fold_rho = None
		self.frequency_features = None
		self.best_N_features = None
		self.best_N = None
		self.ccm_values = []

		# Classification metrics
		self.corr = None
		self.mae = None
		self.rmse = None

	def fit(self, ts_df: pd.DataFrame, target: str, E: int = None, tau: int = None):
		"""
		Perform cross-validation using parallel_MDE in each fold.
		If folds <= 1, create a single fold with the entire training set.
		"""
		self.target = str(target)
		self.ts_df = ts_df
		ts_df = ts_df.dropna(axis = 1, how = "all")
		ts_df.columns = ts_df.columns.astype(str)

		if tau == None:
			tau = -1
		self.tau = tau
		# Optionally remove existing 'time' column
		if "time" in ts_df.columns:
			ts_df = ts_df.drop(columns = "time")

		# Train/test split
		self.train_set, self.test_set = train_test_split(
			ts_df, test_size = self.test_size, shuffle = False, random_state = 42
		)
		self.train_idx = self.train_set.index[-1]
		self.test_idx = self.test_set.index[0]

		self.train_set = reset_time_column(self.train_set.reset_index(drop = True))
		self.test_set = reset_time_column(self.test_set.reset_index(drop = True))

		# Build fold indices
		if self.folds is None or self.folds <= 1:
			train_idx = np.arange(len(self.train_set))
			fold_indices = [(train_idx, train_idx)]
		else:
			kf = KFold(n_splits = self.folds, shuffle = False)
			fold_indices = [
				(train_idx, val_idx)
				for train_idx, val_idx in kf.split(self.train_set)
			]

		# Clear stored results
		self.fold_details = []
		self.selected_features_per_fold = []
		self.rho_test_per_fold = []

		# Enumerate folds to get a separate fold_idx
		for fold_idx, (train_idx, val_idx) in enumerate(fold_indices, start = 1):
			print(f"\n[INFO] Starting fold {fold_idx}/{len(fold_indices)}")
			result = self._process_fold(fold_idx, train_idx, val_idx, E, tau)
			self.fold_details.append(result)
			self.selected_features_per_fold.append(result["selected_features"])
			self.rho_test_per_fold.append(result["rho_val"])
			self.ccm_values.append(result["ccm_val"])

		# Identify best fold
		self.best_fold_idx = np.nanargmax(self.rho_test_per_fold)
		self.best_fold_rho = self.rho_test_per_fold[self.best_fold_idx]
		self.best_fold_features = self.selected_features_per_fold[self.best_fold_idx]

		print("[INFO] Cross-validation complete.")
		print(f"Best fold index: {self.best_fold_idx} with {self.optimize_for}={self.best_fold_rho:.4f}")
		print(f"Features in best fold: {self.best_fold_features}")

		# If final_feature_mode == "frequency", build top frequency features
		if self.final_feature_mode == "frequency":
			all_feats = [feat for fset in self.selected_features_per_fold for feat in fset]
			feat_counts = pd.Series(all_feats).value_counts()
			top_freq = feat_counts.index[:self.maxD].tolist()
			self.frequency_features = top_freq
			print(f"[INFO] Top {self.maxD} freq features: {self.frequency_features}")

		# If final_feature_mode == "best_N", build top frequency features and selecte best N
		elif self.final_feature_mode == "best_N":
			all_feats = [feat for fset in self.selected_features_per_fold for feat in fset]
			feat_counts = pd.Series(all_feats).value_counts()
			top_freq = feat_counts.index[:self.maxD].tolist()
			self.frequency_features = top_freq
			_, self.best_N = predict_incremental(self,
			                                     plot = 0)  # estimate best N with incremental prediction on the training set
			self.best_N_features = top_freq[:N]
			print(f"[INFO] Top {N} freq features: {self.best_N_features}")

	def _process_fold(self, fold_idx, train_idx, val_idx, E, tau) -> dict:
		train_df = self.train_set.iloc[train_idx].reset_index(drop = True)
		val_df = self.train_set.iloc[val_idx].reset_index(drop = True)
		fold_df = pd.concat([train_df, val_df]).reset_index(drop = True)

		lib = f"1 {len(train_df)}"
		pred = f"{len(train_df) + 1} {len(fold_df)}"

		S_final, chosen_cols, accuracy_list, ccm_values = parallel_MDE(
			ts_df = fold_df,
			target = self.target,
			Tp = self.Tp,
			maxD = self.maxD,
			tau = tau,
			columns = None,
			E = E,
			lib = lib,
			pred = pred,
			conv = self.conv,
			plot = self.plot,
			smap = self.smap,
			include_target = self.include_target,
			optimize_for = self.optimize_for
		)
		rmse = mae = corr = None
		if S_final is not None and not S_final.empty:
			S_final.dropna(inplace = True)
			obs_val = S_final["Observations"]
			preds_val = S_final["Predictions"]
			rmse, mae, corr = regression_metrics(obs_val, preds_val)
			if self.optimize_for == "correlation":
				rho_val = corr
			else:
				print('mae', mae)
				rho_val = mae
		else:
			rho_val = np.nan

		print(f"[INFO] Fold {fold_idx} completed. Final {self.optimize_for}={rho_val:.4f}, features={chosen_cols}")

		return {
			"fold_idx": fold_idx,
			"selected_features": chosen_cols,
			"ccm_val": ccm_values,
			"rho_val": rho_val,
			"corr_val": corr,
			"rmse_val": rmse,
			"mae_val": mae,
		}

	def predict(self) -> pd.DataFrame:
		"""
		Predict using the final chosen feature set on the self.test_set.

		Returns
		-------
		pd.DataFrame
			A DataFrame with columns:
			['Time', 'Observations', 'Predictions']
		"""
		if self.test_set is None:
			print("[WARN] No test_set found. Did you call fit(...) first?")
			return pd.DataFrame()

		# Decide final features
		if self.final_feature_mode == "best_fold":
			features = self.best_fold_features
		elif self.final_feature_mode == "frequency":
			features = self.frequency_features
		else:
			features = self.best_N_features

		self.final_features = features
		# Run prediction on Test set
		test_df = reset_time_column(self.test_set.reset_index(drop = True))
		test_df.columns = test_df.columns.map(str)

		self.train_idx = self.train_set.index[-1]
		self.test_idx = self.test_set.index[0]
		# lib is training set, pred is test set
		if self.smap:
			smap_out = SMap(
				dataFrame = self.ts_df,
				lib = f"1 {self.train_idx}",  # build manifold on training set
				pred = f"{self.train_idx + 1} {len(self.ts_df)}",  # predict test set
				E = len(features),
				tau = self.tau,
				Tp = self.Tp,
				columns = features,
				target = self.target,
				theta = 1,
				showPlot = True,
				embedded = True,
				verbose = False
			)
			df_pred = smap_out["predictions"]
		else:
			simplex_out = Simplex(
				dataFrame = self.ts_df,
				lib = f"1 {self.train_idx}",
				pred = f"{self.train_idx + 1} {len(self.ts_df)}",
				E = len(features),
				tau = self.tau,
				Tp = self.Tp,
				columns = features,
				target = self.target,
				showPlot = True,
				embedded = True,
				verbose = False
			)
			df_pred = simplex_out
		df_pred.dropna(inplace = True)
		df_pred = df_pred[["Time", "Observations", "Predictions"]]
		self.df_pred = df_pred

		if not df_pred.empty:
			rmse, mae, corr = regression_metrics(df_pred["Observations"], df_pred["Predictions"])
			self.mae = mae
			self.rmse = rmse
			self.corr = corr

		return df_pred

	def predict_incremental(self, plot = 0) -> list:
		# on training set, to see how much each feature adds to the prediction
		import matplotlib.pyplot as plt
		if self.final_feature_mode == "best_fold":
			used_features = self.best_fold_features
		else:
			used_features = self.frequency_features or []

		if not used_features:
			print("No features available for incremental prediction.")
			return None

		final_df = pd.concat([self.train_set, self.test_set]).reset_index(drop = True)
		final_df = reset_time_column(final_df)
		lib = f"1 {len(self.train_set)}"

		incremental_scores = []
		current_feature_list = []

		for i, feat in enumerate(used_features, start = 1):
			current_feature_list.append(feat)
			S_test = Simplex(
				dataFrame = final_df,
				lib = lib,
				pred = lib,
				E = self.E_fold,
				Tp = self.Tp,
				columns = current_feature_list,
				target = self.target,
				embedded = True,
				showPlot = False
			)
			S_test.dropna(inplace = True)
			df_test = S_test[["Observations", "Predictions"]]

			if df_test.empty:
				score = np.nan
			else:
				rmse, mae, corr = regression_metrics(df_test["Observations"], df_test["Predictions"])
				incremental_scores.append(corr)

		N = np.argmax(incremental_scores)

		if plot:
			plt.figure()
			plt.plot(range(1, len(used_features) + 1), incremental_scores, marker = 'o')
			plt.xlabel("Number of features")
			plt.ylabel("Score (Test Set)")
			plt.title("Incremental Prediction Score")
			plt.show()

		self.rho_final_test = incremental_scores[-1]
		self.incremental_scores = incremental_scores
		self.best_N = N
		return incremental_scores, N

	def save_results(self, output_dir: str) -> None:
		"""
		Save cross-validation results to a pickle file
		"""
		os.makedirs(output_dir, exist_ok = True)
		results_data = {
			"fold_details": self.fold_details,
			"selected_features_per_fold": self.selected_features_per_fold,
			"rho_test_per_fold": self.rho_test_per_fold,
			"best_fold_features": self.best_fold_features,
			"smap": self.smap,
			"incremental_score": self.incremental_scores,
			"final_features": self.final_features,
			"final_prediction": self.df_pred,
			"mae": self.mae,
			"rmse": self.rmse,
			"corr": self.corr
		}

		if self.frequency_features:
			results_data["frequency_features"] = self.frequency_features

		filename = f"{self.target}_train{len(self.train_set)}_maxD{self.maxD}_Folds{self.folds}_{self.optimize_for}_{self.final_feature_mode}_include_target{self.include_target}_tp{self.Tp}.pkl"

		fname = os.path.join(output_dir, filename)
		with open(fname, "wb") as f:
			pickle.dump(results_data, f)
		print(f"[INFO] Results saved to {fname}")