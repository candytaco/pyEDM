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
		knn_indices = neighbors[:, v]
		knn_dists = distances[:, v]

		# Simplex weights: exponential weighting
		weights = numpy.exp(-knn_dists / knn_dists)
		weights /= numpy.sum(weights, axis = 0)

		# Weighted predictions for all test points
		predictions[:, v] = numpy.sum(weights * trainY[knn_indices], axis = 0)


@numba.jit((float64[:], float64[:, :], float64[:]), nopython = True, parallel = True)
def columnwise_correlation(vector, array, out):
	n, m = array.shape

	v_mean = numpy.mean(vector)
	v_centered = vector - v_mean
	v_std = numpy.sqrt(numpy.sum(v_centered ** 2))

	for j in numba.prange(m):
		a_mean = numpy.mean(array[:, j])
		a_centered = array[:, j] - a_mean
		a_std = numpy.sqrt(numpy.sum(a_centered ** 2))

		out[j] = numpy.sum(v_centered * a_centered) / (v_std * a_std)

	return out


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
			knn_indices[:, a] = numpy.argsort(distances[:, a])[:k, :]
		knn_dists = numpy.take_along_axis(distances, knn_indices, axis=0)

		# Simplex weights: exponential weighting
		min_dists = numpy.fmax(numpy.min(knn_dists, axis=0), 1e-6)
		weights = numpy.exp(-knn_dists / min_dists)
		weights /= numpy.sum(weights, axis=0)

		# Weighted predictions for all test points
		predictions[:, v] = numpy.sum(weights * train_y[knn_indices + offset], axis=0)

