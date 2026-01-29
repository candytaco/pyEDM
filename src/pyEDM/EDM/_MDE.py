import numba
import numpy
from numba import float64, int32, int64


@numba.jit((float64[:, :], float64[:, :], float64[:, :, :]), nopython = True, parallel = True) # numpy broadcast does not work with numba?
def elementwise_pairwise_distance(a, b, out):
	"""
	Pairwise square euclidean distances between elements of a and b
	along every dimension. Basically an outer subtract.
	:param a:	[n1 x dims] array 1
	:param b:	[n2 x dims] array 2
	:param out:	out array to write to
	"""
	n1 = a.shape[0]
	n2 = b.shape[0]
	dims = a.shape[1]
	for v in numba.prange(dims):
		for j in range(n2):
			for i in range(n1):
				d = a[i, v] - b[j, v]
				out[v, i, j] = d * d


@numba.jit((float64[:, :, :], float64[:, :], float64[:, :, :]), nopython = True, parallel = True)
def increment_pairwise_distance(distances, increments, out):
	"""
	For a set of pairwise distances, increment each slice by the same amount
	i.e. a 2D array broadcast
	:param distances: 	[dims, n1 x n2] set of pairwise distances
	:param increments: 	[n1 x n2] increments
	:param out: 		[dims, n1 x n2] array to write into
	:return:
	"""
	dims = distances.shape[0]
	for v in numba.prange(dims):
		out[v, :, :] = distances[v, :, :] + increments


@numba.jit(nopython = True, parallel = True)
def k_nearest_neighbors(distances, k):
	"""
	K nearnest neighbors along the first dimension of a matrix/tensor
	:param distances:
	:param k:
	:return:
	"""
	return numpy.argsort(distances, axis = 0)[:k, :, :]


@numba.jit(nopython = True, parallel = True)
def fill_sparse_weight_matrix(distances, neighbors, weights, shift):
	"""
	Constructs the sparse weight matrix that maps only the k nearest neighbors for each test sample
	:param distances:
	:param neighbors:
	:param weights:
	:param shift:
	:return:
	"""
	weights *= 0
	n1 = distances.shape[0]
	n2 = distances.shape[1]
	dims = distances.shape[2]
	for v in numba.prange(dims):
		mask = numpy.zeros([n1, n2], dtype = bool)
		mask[neighbors[v, :, :] + shift] = True
		min_dist = numpy.min(distances[mask, v], axis = 0)
		min_dist = numpy.fmax(min_dist, 1e-6)
		weights[mask, v] = numpy.exp(-1 * distances[mask, v] / min_dist[:, None])

@numba.jit((int64[:, :, :], float64[:, :, :], float64[:], float64[:, :]), nopython = True, parallel = True)
def calculate_predictions(neighbors, distances, trainY, predictions):
	V = predictions.shape[1]

	for v in numba.prange(V):
		knn_indices = neighbors[v, :, :]
		knn_dists = distances[v, :, :]

		# Simplex weights: exponential weighting
		minDistances = numpy.zeros(knn_dists.shape[1])
		for i in range(knn_dists.shape[1]):
			lower = numpy.min(knn_dists[:, i])

		# Divide each column of N x k knn_distances by minDistances
		scaledDistances = numpy.divide(knn_dists, minDistances)
		weights = numpy.exp(-scaledDistances)  # Npred x k
		weightRowSum = numpy.sum(weights, axis = 0)  # Npred x 1

		# Projection is average of weighted knn library target values
		# selected = trainY[knn_indices]
		# product = weights * selected
		for i in range(predictions.shape[0]):
			predictions[i, v] = numpy.sum(trainY[knn_indices[i, :]] * weights[i, :]) / weightRowSum[i]


@numba.jit((float64[:], float64[:, :], float64[:]), nopython = True, parallel = True)
def columnwise_correlation(vector, array, out):
	n, m = array.shape

	v_mean = numpy.mean(vector)
	v_centered = vector - v_mean
	v_std = numpy.sqrt(numpy.sum(v_centered ** 2))

	for j in numba.prange(m):
		a_mean = numpy.mean(array[j, :])
		a_centered = array[j, :] - a_mean
		a_std = numpy.sqrt(numpy.sum(a_centered ** 2))

		out[j] = numpy.sum(v_centered * a_centered) / (v_std * a_std)

	return out

@numba.jit(nopython = True)
def floor_array(arr, floor_value):
	result = numpy.empty_like(arr)
	for i in range(arr.size):
		if arr.flat[i] < floor_value:
			arr.flat[i] = floor_value

@numba.jit(nopython = True, parallel=True)
def add_scalar(arr, floor_value):
	result = numpy.empty_like(arr)
	for i in range(arr.size):
		arr.flat[i] += floor_value

@numba.jit(nopython = True, parallel=True)
def min_axis1(arr):
	result = numpy.empty((arr.shape[0], arr.shape[2]), dtype=arr.dtype)
	for i in numba.prange(arr.shape[0]):
		for k in range(arr.shape[2]):
			min_val = arr[i, 0, k]
			for j in range(1, arr.shape[1]):
				if arr[i, j, k] < min_val:
					min_val = arr[i, j, k]
			result[i, k] = min_val
	return result

@numba.jit(nopython = True, parallel=True)
def sum_axis1(arr):
	result = numpy.zeros((arr.shape[0], arr.shape[2]), dtype=arr.dtype)
	for i in numba.prange(arr.shape[0]):
		for j in range(arr.shape[1]):
			for k in range(arr.shape[2]):
				result[i, k] += arr[i, j, k]
	return result

@numba.jit(nopython = True, parallel=True)
def compute_weights(neighborDistances, minDistances):
	result = numpy.empty_like(neighborDistances)
	for i in numba.prange(neighborDistances.shape[0]):
		for j in range(neighborDistances.shape[1]):
			for k in range(neighborDistances.shape[2]):
				result[i, j, k] = numpy.exp(-neighborDistances[i, j, k] / minDistances[i, k])
	return result

@numba.jit(nopython = True, parallel=True)
def compute_predictions(weights, select, weightSum):
	result = numpy.empty((weights.shape[0], weights.shape[2]), dtype=weights.dtype)
	for i in numba.prange(weights.shape[0]):
		for k in range(weights.shape[2]):
			weighted_sum = 0.0
			for j in range(weights.shape[1]):
				weighted_sum += weights[i, j, k] * select[i, j, k]
			result[i, k] = weighted_sum / weightSum[i, k]
	return result

# @numba.jit((float64[:, :, :], float64[:, :], float64[:], int32, int32[:], int32, float64[:, :]),
# 		   nopython=True, parallel=True, nogil=True)
def evaluate_all_candidates_numba(all_distances, current_best,
								  train_y, k, remaining_vars, offset,
								  predictions):
	"""Evaluate all candidate variables using pure numpy operations with GIL released.
	"""
	V = len(remaining_vars)

	for v in numba.prange(V):
		var = remaining_vars[v]

		# Add candidate distance to current best
		distances = current_best + all_distances[var, :, :]

		# Find k nearest neighbors for each test point
		# For each column (test point), find k smallest distances

		# Get all distances for all test points at once
		# Find k nearest neighbors for each test point (column)
		knn_indices = numpy.zeros((k, distances.shape[1]), dtype = numpy.int32)
		for a in range(distances.shape[1]):
			knn_indices[:, a] = numpy.argpartition(distances[:, a], k)[:k, :]
		knn_dists = numpy.take_along_axis(distances, knn_indices, axis=0)
		knn_dists[knn_dists < 1e-6] = 1e-6

		# Simplex weights: exponential weighting
		weights = numpy.exp(-knn_dists / knn_dists)
		weights /= numpy.sum(weights, axis = 1)

		# Weighted predictions for all test points
		predictions[:, v] = numpy.sum(weights * train_y[knn_indices + offset], axis=0)

