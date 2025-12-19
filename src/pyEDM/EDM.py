# python modules
from warnings import warn
from datetime import datetime

# package modules
from numpy import any, append, array, concatenate, isnan, zeros

# local modules
import pyEDM.API
import pyEDM.Embed
from .AuxFunc import IsIterable

#--------------------------------------------------------------------
class EDM:
#--------------------------------------------------------------------
    '''EDM class : data container
       Simplex, SMap, CCM inherited from EDM'''

    def __init__( self, dataFrame, name = 'EDM' ):
        self.name = name

        self.Data          = dataFrame # DataFrame
        self.Embedding     = None      # DataFrame, includes nan
        self.Projection    = None      # DataFrame Simplex & SMap output

        self.lib_i         = None  # ndarray library indices
        self.pred_i        = None  # ndarray prediction indices : nan removed
        self.pred_i_all    = None  # ndarray prediction indices : nan included
        self.predList      = []    # list of disjoint pred_i_all
        self.disjointLib   = False # True if disjoint library
        self.libOverlap    = False # True if lib & pred overlap
        self.ignoreNan     = True  # Remove nan from embedding
        self.xRadKnnFactor = 5     # exlcusionRadius knn factor

        self.kdTree        = None  # SciPy KDTree (k-dimensional tree)
        self.knn_neighbors = None  # ndarray (N_pred, knn) sorted
        self.knn_distances = None  # ndarray (N_pred, knn) sorted

        self.projection    = None  # ndarray Simplex & SMap output
        self.variance      = None  # ndarray Simplex & SMap output
        self.targetVec     = None  # ndarray entire record
        self.targetVecNan  = False # True if targetVec has nan : SMap only
        self.time          = None  # ndarray entire record numerically operable

	def FindNeighbors(self):
		# --------------------------------------------------------------------
		'''Use Scipy KDTree to find neighbors

		   Note: If dimensionality is k, the number of points n in
		   the data should be n >> 2^k, otherwise KDTree efficiency is low.
		   k:2^k pairs { 4 : 16, 5 : 32, 7 : 128, 8 : 256, 10 : 1024 }

		   KDTree returns ndarray of knn_neighbors as indices with respect
		   to the data array passed to KDTree, not with respect to the lib_i
		   of embedding[ lib_i ] passed to KDTree. Since lib_i are generally
		   not [0..N] the knn_neighbors need to be adjusted to lib_i reference
		   for use in projections. If the the library is unitary this is
		   a simple shift by lib_i[0]. If the library has disjoint segments
		   or unordered indices, a mapping is needed from KDTree to lib_i.

		   If there are degenerate lib & pred indices the first nn will
		   be the prediction vector itself with distance 0. These are removed
		   to implement "leave-one-out" prediction validation. In this case
		   self.libOverlap is set True and the value of knn is increased
		   by 1 to return an additional nn. The first nn is relplaced by
		   shifting the j = 1:knn+1 knn columns into the j = 0:knn columns.

		   If exlcusionRadius > 0, and, there are degenerate lib & pred
		   indices, or, if there are not degnerate lib & pred but the
		   distance in rows between the lib & pred gap is less than
		   exlcusionRadius, knn_neighbors have to be selected for each
		   pred row to exclude library neighbors within exlcusionRadius.
		   This is done by increasing knn to KDTree.query by a factor of
		   self.xRadKnnFactor, then selecting valid nn.

		   Writes to EDM object:
			 knn_distances : sorted knn distances
			 knn_neighbors : library neighbor rows of knn_distances
		'''
		if self.verbose:
			print(f'{self.name}: FindNeighbors()')

		N_lib_rows = len(self.trainIndices)
		N_pred_rows = len(self.testIndices)

		# Is knn_neighbors exclusionRadius radius adjustment needed?
		exclusionRadius_knn = False

		if self.exclusionRadius > 0:
			if self.libOverlap:
				exclusionRadius_knn = True
			else:
				# If no libOverlap and exclusionRadius is less than the
				# distance in rows between lib : pred, no library neighbor
				# exclusion needed.
				# Find row span between lib & pred
				excludeRow = 0
				if self.testIndices[0] > self.trainIndices[-1]:
					# pred start is beyond lib end
					excludeRow = self.testIndices[0] - self.trainIndices[-1]
				elif self.trainIndices[0] > self.testIndices[-1]:
					# lib start row is beyond pred end
					excludeRow = self.trainIndices[0] - self.testIndices[-1]
				if self.exclusionRadius >= excludeRow:
					exclusionRadius_knn = True

		if len(self.validLib):
			# Convert self.validLib boolean vector to data indices
			data_i = array([i for i in range(self.Data.shape[0])],
			               dtype = int)
			validLib_i = data_i[self.validLib]

			# Filter lib_i to only include valid library points
			lib_i_valid = array([i for i in self.trainIndices if i in validLib_i],
			                    dtype = int)

			if len(lib_i_valid) == 0:
				msg = f'{self.name}: FindNeighbors() : ' + \
				      'No valid library points found. ' + \
				      'All library points excluded by validLib.'
				raise ValueError(msg)

			if len(lib_i_valid) < self.knn:
				msg = f'{self.name}: FindNeighbors() : Only {len(lib_i_valid)} ' + \
				      f'valid library points found, but knn={self.knn}. ' + \
				      'Reduce knn or check validLib.'
				warn(msg)

			# Replace lib_ with lib_i_valid
			self.trainIndices = lib_i_valid

		# Local knn_
		knn_ = self.knn
		if self.libOverlap and not exclusionRadius_knn:
			# Increase knn +1 if libOverlap
			# Returns one more column in knn_distances, knn_neighbors
			# The first nn degenerate with the prediction vector
			# is replaced with the 2nd to knn+1 neighbors
			knn_ = knn_ + 1

		elif exclusionRadius_knn:
			# knn_neighbors exclusionRadius adjustment required
			# Ask for enough knn to discard exclusionRadius neighbors
			# This is controlled by the factor: self.xRadKnnFactor
			# JP : Perhaps easier to just compute all neighbors?
			knn_ = min(knn_ * self.xRadKnnFactor, len(self.trainIndices))

		if len(self.validLib):
			# Have to examine all knn
			knn_ = len(self.trainIndices)

		# -----------------------------------------------
		# Compute KDTree on library of embedding vectors
		# -----------------------------------------------
		self.kdTree = KDTree(self.Embedding[self.trainIndices, :],
		                     leafsize = 20,
		                     compact_nodes = True,
		                     balanced_tree = True)

		# -----------------------------------------------
		# Query prediction set
		# -----------------------------------------------
		numThreads = -1  # Use all CPU threads in kdTree.query
		self.knn_distances, self.knn_neighbors = self.kdTree.query(
			self.Embedding[self.testIndices, :],
			k = knn_, eps = 0, p = 2, workers = numThreads)

		# -----------------------------------------------
		# Shift knn_neighbors to lib_i reference
		# -----------------------------------------------
		# KDTree.query returns knn referenced to embedding[self.lib_i,:]
		# where returned knn_neighbors are indexed from 0 : len( lib_i ).
		# Generally, these are different from the knn that refer to prediction
		# library rows since generally lib != pred. Adjust knn from 0-offset
		# returned by KDTree.query to EDM knn with respect to  embedding rows.
		#
		# If there is only one lib segment with contiguous values, a single
		# adjustment to knn_neighbors based on lib_i[0] suffices
		if not self.disjointLib and \
				self.trainIndices[-1] - self.trainIndices[0] + 1 == len(self.trainIndices):

			self.knn_neighbors = self.knn_neighbors + self.trainIndices[0]

		else:
			# Disjoint library or CCM subset of lib_i.
			# Create mapping from KDTree neighbor indices to knn_neighbors
			knn_lib_map = {}  # keys KDTree index : values lib_i index

			for i in range(len(self.trainIndices)):
				knn_lib_map[i] = self.trainIndices[i]

			# --------------------------------------------------------
			# Function to apply the knn_lib_map in apply_along_axis()
			# --------------------------------------------------------
			def knnMapFunc(knn, knn_lib_map):
				'''Function for apply_along_axis() on knn_neighbors.
				   Maps the KDTree returned knn_neighbor indices to lib_i'''
				out = zeros(len(knn), dtype = int)
				for i in range(len(knn)):
					idx = knn[i]
					out[i] = knn_lib_map[idx]
				return out

			# Apply the knn_lib_map to self.knn_neighbors
			# Use numpy apply_along_axis() to transform knn_neighbors from
			# KDTree indices to lib_i indices using the knn_lib_map
			knn_neighbors_ = zeros(self.knn_neighbors.shape, dtype = int)

			for j in range(self.knn_neighbors.shape[1]):
				knn_neighbors_[:, j] = \
					apply_along_axis(knnMapFunc, 0,
					                 self.knn_neighbors[:, j], knn_lib_map)

			self.knn_neighbors = knn_neighbors_

		if self.knn == 1 and not self.libOverlap:
			# Edge case outside the EDM canon.  KDTree.query() docs:
			# When k == 1, the last dimension of the output is squeezed.
			self.knn_distances = self.knn_distances[:, None]
			self.knn_neighbors = self.knn_neighbors[:, None]

		if self.libOverlap:
			# Remove degenerate knn_distances, knn_neighbors
			# Get first column of knn_neighbors with knn_distance = 0
			knn_neighbors_0 = self.knn_neighbors[:, 0]

			# If self.pred_i == knn_neighbors[:,0], point is degenerate,
			# distance = 0. Create boolean mask array of rows i_overlap
			# True where self.pred_i == knn_neighbors_0
			i_overlap = [i == j for i, j in zip(self.testIndices,
			                                    knn_neighbors_0)]

			# Shift col = 1:knn_ values into col = 0:(J-1)
			# Use 0:(J-1) instead of 0:self.knn since knn_ may be large
			J = self.knn_distances.shape[1]
			self.knn_distances[i_overlap, 0:(J - 1)] = \
				self.knn_distances[i_overlap, 1:knn_]

			self.knn_neighbors[i_overlap, 0:(J - 1)] = \
				self.knn_neighbors[i_overlap, 1:knn_]

			# Delete extra knn_ column
			if not exclusionRadius_knn:
				self.knn_distances = delete(self.knn_distances, self.knn, axis = 1)
				self.knn_neighbors = delete(self.knn_neighbors, self.knn, axis = 1)

		if exclusionRadius_knn:
			# For each pred row find k nn outside exclusionRadius

			# -----------------------------------------------------------
			# Function to select knn from each row of self.knn_neighbors
			# -----------------------------------------------------------
			def ExclusionRad(knnRow, knnDist, excludeRow):
				'''Search excludeRow for each element of knnRow
				   If knnRow is in excludeRow : exclude the neighbor
				   Return knn length arrays of neighbors, distances'''

				knn_neighbors = full(self.knn, -1E6, dtype = int)
				knn_distances = full(self.knn, -1E6, dtype = float)

				k = 0
				for r in range(len(knnRow)):
					if knnRow[r] in excludeRow:
						# this nn is within exlcusionRadius of pred_i
						continue

					knn_neighbors[k] = knnRow[r]
					knn_distances[k] = knnDist[r]
					k = k + 1

					if k == self.knn:
						break

				if -1E6 in knn_neighbors:
					knn_neighbors = knnRow[: self.knn]
					knn_distances = knnDist[: self.knn]
					msg = f'{self.name}: FindNeighbors() : ExclusionRad() ' + \
					      'Failed to find knn outside exclusionRadius ' + \
					      f'{self.exclusionRadius}. Returning orginal knn. ' + \
					      f'Consider to reduce knn {self.knn}.'
					warn(msg)

				return knn_neighbors, knn_distances

			# Call ExclusionRad() on each row
			for i in range(N_pred_rows):
				# Existing knn_neighbors, knn_distances row i with knn_ values
				knn_neighbors_i = self.knn_neighbors[i, :]
				knn_distances_i = self.knn_distances[i, :]

				# Create list excludeRow of lib_i nn to be excluded
				pred_i = self.testIndices[i]
				rowLow = max(self.trainIndices.min(), pred_i - self.exclusionRadius)
				rowHi = min(self.trainIndices.max(), pred_i + self.exclusionRadius)
				excludeRow = [k for k in range(rowLow, rowHi + 1)]

				knn_neighbors, knn_distances = \
					ExclusionRad(knn_neighbors_i, knn_distances_i, excludeRow)

				self.knn_neighbors[i, range(self.knn)] = knn_neighbors
				self.knn_distances[i, range(self.knn)] = knn_distances

			# Delete the extra knn_ columns
			d = [i for i in range(self.knn, self.knn_distances.shape[1])]
			self.knn_distances = delete(self.knn_distances, d, axis = 1)
			self.knn_neighbors = delete(self.knn_neighbors, d, axis = 1)

	# --------------------------------------------------------------------
	# EDM Methods
	# -------------------------------------------------------------------
	def FormatProjection(self):
		# -------------------------------------------------------------------
		'''Create Projection, Coefficients, SingularValues DataFrames
		   AddTime() attempts to extend forecast time if needed

		   NOTE: self.pred_i had all nan removed for KDTree by RemoveNan().
				 self.predList only had leading/trailing embedding nan removed.
				 Here we want to include any nan observation rows so we
				 process predList & pred_i_all, not self.pred_i.
		'''
		if self.verbose:
			print(f'{self.name}: FormatProjection()')

		if len(self.testIndices) == 0:
			msg = f'{self.name}: FormatProjection() No valid prediction indices.'
			warn(msg)

			# Return empty numpy array with shape (0, 4)
			# Columns: Time, Observations, Predictions, Pred_Variance
			self.Projection = empty((0, 4))
			return

		N_dim = self.embedDimensions + 1
		Tp_magnitude = abs(self.predictionHorizon)

		# ----------------------------------------------------
		# Observations: Insert target data in observations
		# ----------------------------------------------------
		outSize = 0

		# Create array of indices into self.targetVec for observations
		obs_i = array([], dtype = int)

		# Process each pred segment in self.predList
		for pred_i in self.predList:
			N_pred = len(pred_i)
			outSize_i = N_pred + Tp_magnitude
			outSize = outSize + outSize_i
			append_i = array([], dtype = int)

			if N_pred == 0:
				# No prediction made for this pred segment
				if self.verbose:
					msg = f'{self.name} FormatProjection(): No prediction made ' + \
					      f'for empty pred in {self.predList}. ' + \
					      'Examine pred, E, tau, Tp parameters and/or nan.'
					print(msg)
				continue

			if self.predictionHorizon == 0:  # Tp = 0
				append_i = pred_i.copy()

			elif self.predictionHorizon > 0:  # Positive Tp
				if pred_i[-1] + self.predictionHorizon < self.targetVec.shape[0]:
					# targetVec data available before end of targetVec
					append_i = append(append_i, pred_i)
					Tp_i = [i for i in range(append_i[-1] + 1,
					                         append_i[-1] + self.predictionHorizon + 1)]
					append_i = append(append_i, array(Tp_i, dtype = int))
				else:
					# targetVec data not available at prediction end
					append_i = append(append_i, pred_i)

			else:  # Negative Tp
				if pred_i[0] + self.predictionHorizon > -1:
					# targetVec data available after begin of pred_i[0]
					append_i = append(append_i, pred_i)
					Tp_i = [i for i in range(pred_i[0] + self.predictionHorizon,
					                         pred_i[0])]
					append_i = append(array(Tp_i, dtype = int), append_i)
				else:
					# targetVec data not available before pred_i[0]
					append_i = append(append_i, pred_i)

			obs_i = append(obs_i, append_i)

		observations = self.targetVec[obs_i, 0]

		# ----------------------------------------------------
		# Projections & variance
		# ----------------------------------------------------
		# Define array's of indices predOut_i, obsOut_i for DataFrame vectors
		predOut_i = array([], dtype = int)
		predOut_0 = 0

		# Process each pred segment in self.predList for predOut_i
		for pred_i in self.predList:
			N_pred = len(pred_i)
			outSize_i = N_pred + Tp_magnitude

			if N_pred == 0:
				# No prediction made for this pred segment
				continue

			if self.predictionHorizon == 0:
				Tp_i = [i for i in range(predOut_0, predOut_0 + N_pred)]
				predOut_i = append(predOut_i, array(Tp_i, dtype = int))
				predOut_0 = predOut_i[-1] + 1

			elif self.predictionHorizon > 0:  # Positive Tp
				Tp_i = [i for i in range(predOut_0 + self.predictionHorizon,
				                         predOut_0 + self.predictionHorizon + N_pred)]
				predOut_i = append(predOut_i, array(Tp_i, dtype = int))
				predOut_0 = predOut_i[-1] + 1

			else:  # Negative Tp
				Tp_i = [i for i in range(predOut_0, predOut_0 + N_pred)]
				predOut_i = append(predOut_i, array(Tp_i, dtype = int))
				predOut_0 = predOut_i[-1] + Tp_magnitude + 1

		# If nan are present, the foregoing can be wrong since it is not
		# known before prediction what lib vectors will produce pred
		# If len( pred_i ) != len( predOut_i ), nan resulted in missing pred
		# Create a map between pred_i_all : predOut_i to create a new/shorter
		# predOut_i mapping pred_i to the output vector predOut_i
		if len(self.testIndices) != len(predOut_i):
			# Map the last predOut_i values since embed shift near data begining
			# can have (E-1)*tau na, but still listed in pred_i_all
			N_ = len(predOut_i)

			if self.embedStep < 0:
				D = dict(zip(self.pred_i_all[-N_:], predOut_i))
			else:
				D = dict(zip(self.pred_i_all[:N_], predOut_i))

			# Reset predOut_i
			predOut_i = [D[i] for i in self.testIndices]

		# Create obsOut_i indices for output vectors in DataFrame
		if self.predictionHorizon > 0:  # Positive Tp
			if obs_i[-1] + self.predictionHorizon > self.Data.shape[0] - 1:
				# Edge case of end of data with positive Tp
				obsOut_i = [i for i in range(len(obs_i))]
			else:
				obsOut_i = [i for i in range(outSize)]

		elif self.predictionHorizon < 1:  # Negative or Zero Tp
			if self.testIndices[0] + self.predictionHorizon < 0:
				# Edge case of start of data with negative Tp
				obsOut_i = [i for i in range(len(obs_i))]

				# Shift obsOut_i values based on leading nan
				shift = Tp_magnitude - self.testIndices[0]
				obsOut_i = obsOut_i + shift
			else:
				obsOut_i = [i for i in range(len(obs_i))]

		obsOut_i = array(obsOut_i, dtype = int)

		# ndarray init to nan
		observationOut = full(outSize, nan)
		projectionOut = full(outSize, nan)
		varianceOut = full(outSize, nan)

		# fill *Out with observed & projected values
		observationOut[obsOut_i] = observations
		projectionOut[predOut_i] = self.projection
		varianceOut[predOut_i] = self.variance

		# ----------------------------------------------------
		# Time
		# ----------------------------------------------------
		self.ConvertTime()

		if self.predictionHorizon == 0 or \
				(self.predictionHorizon > 0 and (self.pred_i_all[-1] + self.predictionHorizon) < len(self.time)) or \
				(self.predictionHorizon < 0 and (self.testIndices[0] + self.predictionHorizon >= 0)):
			# All times present in self.time, copy them
			timeOut = empty(outSize, dtype = self.time.dtype)
			timeOut[obsOut_i] = self.time[obs_i]
		else:
			# Need to pre/append additional times
			timeOut = self.AddTime(Tp_magnitude, outSize, obs_i, obsOut_i)

		# ----------------------------------------------------
		# Output numpy array
		# Shape: (n_samples, 4)
		# Column 0: Time
		# Column 1: Observations
		# Column 2: Predictions
		# Column 3: Pred_Variance
		# ----------------------------------------------------
		self.Projection = column_stack([timeOut, observationOut, projectionOut, varianceOut])

		# ----------------------------------------------------
		# SMap coefficients and singular values
		# ----------------------------------------------------
		if self.name == 'SMap':
			# ndarray init to nan
			coefOut = full((outSize, N_dim), nan)
			SVOut = full((outSize, N_dim), nan)
			# fill coefOut, SVOut with projected values
			coefOut[predOut_i, :] = self.coefficients
			SVOut[predOut_i, :] = self.singularValues

			# Prepend time column to coefficients and singular values
			# Coefficients shape: (n_samples, N_dim + 1)
			# Column 0: Time, Columns 1-N_dim: coefficients
			self.Coefficients = column_stack([timeOut, coefOut])

			# SingularValues shape: (n_samples, N_dim + 1)
			# Column 0: Time, Columns 1-N_dim: singular values
			self.SingularValues = column_stack([timeOut, SVOut])

	# -------------------------------------------------------------------
	def ConvertTime(self):
		# -------------------------------------------------------------------
		'''Replace self.time with ndarray numerically operable values
		   ISO 8601 formats are supported in the time & datetime modules
		'''
		if self.verbose:
			print(f'{self.name}: ConvertTime()')

		time0 = self.time[0]

		# If times are numerically operable, nothing to do.
		if isinstance(time0, int) or isinstance(time0, float) or \
				isinstance(time0, integer) or isinstance(time0, floating) or \
				isinstance(time0, dt.time) or isinstance(time0, dt.datetime):
			return

		# Local copy of time
		time_ = self.time.copy()  # ndarray

		# If times are strings, try to parse into time or datetime
		# If success, replace time with parsed time or datetime array
		if isinstance(time0, str):
			try:
				t0 = dt.time.fromisoformat(time0)
				# Parsed t0 into dt.time OK. Parse the whole vector
				time_ = array([dt.time.fromisoformat(t) for t in time_])
			except ValueError:
				try:
					t0 = dt.datetime.fromisoformat(time0)
					# Parsed t0 into dt.datetime OK. Parse the whole vector
					time_ = array([dt.datetime.fromisoformat(t) for t in time_])
				except ValueError:
					msg = f'{self.name} ConvertTime(): Time values are strings ' + \
					      'but are not ISO 8601 recognized time or datetime.'
					raise RuntimeError(msg)

		# If times were string they have been converted to time or datetime
		# Ensure times can be numerically manipulated, compute deltaT
		try:
			deltaT = time_[1] - time_[0]
		except TypeError:
			msg = f'{self.name} ConvertTime(): Time values not recognized.' + \
			      ' Accepted values are int, float, or string of ISO 8601' + \
			      ' compliant time or datetime.'
			raise RuntimeError(msg)

		# Replace DataFrame derived time with converted time_
		self.time = time_

	# -------------------------------------------------------------------
	def AddTime(self, Tp_magnitude, outSize, obs_i, obsOut_i):
		# -------------------------------------------------------------------
		'''Prepend or append time values to self.time if needed
		   Return timeOut vector with additional Tp points
		'''
		if self.verbose:
			print(f'{self.name}: AddTime()')

		min_pred_i = self.testIndices[0]
		max_pred_i_all = self.pred_i_all[-1]
		deltaT = self.time[1] - self.time[0]

		# First, fill timeOut with times in time
		# timeOut should not be int (np.integer) since they cannot be nan
		time0 = self.time[0]
		if isinstance(time0, int) or isinstance(time0, integer):
			time_dtype = float
		else:
			time_dtype = type(time0)

		timeOut = full(outSize, nan, dtype = time_dtype)

		timeOut[obsOut_i] = self.time[obs_i]

		newTimes = full(Tp_magnitude, nan, dtype = time_dtype)

		if self.predictionHorizon > 0:
			# Tp introduces time values beyond the range of time
			# Generate future times
			lastTime = self.time[max_pred_i_all]
			newTimes[0] = lastTime + deltaT

			for i in range(1, self.predictionHorizon):
				newTimes[i] = newTimes[i - 1] + deltaT

			timeOut[-self.predictionHorizon:] = newTimes

		else:
			# Tp introduces time values before the range of time
			# Generate past times
			newTimes[0] = self.time[min_pred_i] - deltaT
			for i in range(1, Tp_magnitude):
				newTimes[i] = newTimes[i - 1] - deltaT

			newTimes = newTimes[::-1]  # Reverse

			# Shift timeOut values based on leading nan
			shift = Tp_magnitude - self.testIndices[0]
			timeOut[: Tp_magnitude] = newTimes

		return timeOut

	def EmbedData(self):
		'''Embed data : If not embedded call API.Embed()'''

        if not self.embedded :
            self.Embedding = pyEDM.Embed.Embed(dataFrame = self.Data, embeddingDimensions = self.E,
                                               stepSize = self.tau, columns = self.columns)
        else :
            self.Embedding = self.Data[ self.columns ] # Already an embedding 

    #--------------------------------------------------------------------
    def RemoveNan( self ) :
    #--------------------------------------------------------------------
        '''KDTree in Neighbors does not accept nan
           If ignoreNan remove Embedding rows with nan from lib_i, pred_i
        '''
        if self.verbose:
            print( f'{self.name}: RemoveNan()' )

        if self.ignoreNan :
            # type : <class 'pandas.core.series.Series'>
            # Series of bool for all Embedding columns (axis = 1) of lib_i...
            na_lib  = self.Embedding.iloc[self.lib_i,: ].isna().any(axis = 1)
            na_pred = self.Embedding.iloc[self.pred_i,:].isna().any(axis = 1)

            if na_lib.any() :
                if self.name == 'SMap' :
                    original_knn       = self.knn
                    original_lib_i_len = len( self.lib_i )

                # Redefine lib_i excluding nan
                self.lib_i = self.lib_i [ ~na_lib.to_numpy() ]

                # lib_i resized, update SMap self.knn if not user provided
                if self.name == 'SMap' :
                    if original_knn == original_lib_i_len - 1 :
                        self.knn = len( self.lib_i ) - 1

            # Redefine pred_i excluding nan
            if any( na_pred ) :
                self.pred_i = self.pred_i[ ~na_pred ]

            # If targetVec has nan, set flag for SMap internals
            if self.name == 'SMap' :
                if any( isnan( self.targetVec ) ) :
                    self.targetVecNan = True

        self.PredictionValid()

    #--------------------------------------------------------------------
    def CreateIndices( self ):
    #--------------------------------------------------------------------
        '''Populate array index vectors lib_i, pred_i
           Indices specified in list of pairs [ 1,10, 31,40... ]
           where each pair is start:stop span of data rows.
        '''
        if self.verbose:
            print( f'{self.name}: CreateIndices() ' )

        #------------------------------------------------
        # lib_i from lib
        #------------------------------------------------
        # libPairs vector of start, stop index pairs
        if len( self.lib ) % 2 :
            # Odd number of lib
            msg = f'{self.name}: CreateIndices() lib must be an even ' +\
                'number of elements. Lib start : stop pairs'
            raise RuntimeError( msg )

        libPairs = [] # List of 2-tuples of lib indices
        for i in range( 0, len( self.lib ), 2 ) :
            libPairs.append( (self.lib[i], self.lib[i+1]) )

        # Validate end > start
        for libPair in libPairs :
            libStart, libEnd = libPair

            if self.name in [ 'Simplex', 'SMap', 'Multiview' ] :
                # Don't check if CCM since default of "1 1" is used.
                if libStart >= libEnd :
                    msg = f'{self.name}: CreateIndices() lib start ' +\
                        f' {libStart} exceeds lib end {libEnd}.'
                    raise RuntimeError( msg )

            # Disallow indices < 1, the user may have specified 0 start
            if libStart < 1 or libEnd < 1 :
                msg = f'{self.name}: CreateIndices() lib indices ' +\
                    f' less than 1 not allowed.'
                raise RuntimeError( msg )

        # Loop over each lib pair
        # Add rows for library segments, disallowing vectors
        # in disjoint library gap accommodating embedding and Tp
        embedShift = abs( self.tau ) * ( self.E - 1 )
        lib_i_list = list()

        for r in range( len( libPairs ) ) :
            start, stop = libPairs[ r ]

            # Adjust start, stop to enforce disjoint library gaps
            if not self.embedded :
                if self.tau < 0 :
                    start = start + embedShift
                else :
                    stop = stop - embedShift

            if self.Tp < 0 :
                if not self.embedded :
                    start = max( start, start + abs( self.Tp ) - 1 )
            else :
                if ( r == len( libPairs ) - 1 ) :
                    stop = stop - self.Tp

            libPair_i = [ i - 1 for i in range( start, stop + 1 ) ]

            lib_i_list.append( array( libPair_i, dtype = int ) )

        # Concatenate lib_i_list into lib_i
        self.lib_i = concatenate( lib_i_list )

        if len( lib_i_list ) > 1 : self.disjointLib = True

        #------------------------------------------------
        # Validate lib_i: E, tau, Tp combination
        #------------------------------------------------
        if self.name in [ 'Simplex', 'SMap', 'CCM', 'Multiview' ] :
            if self.embedded :
                if len( self.lib_i ) < abs( self.Tp ) :
                    msg = f'{self.name}: CreateIndices(): embbeded True ' +\
                          f'Tp = {self.Tp} is invalid for the library.'
                    raise RuntimeError( msg )
            else :
                vectorStart  = max( [ -embedShift, 0, self.Tp ] )
                vectorEnd    = min( [ -embedShift, 0, self.Tp ] )
                vectorLength = abs( vectorStart - vectorEnd ) + 1

                if vectorLength > len( self.lib_i ) :
                    msg = f'{self.name}: CreateIndices(): Combination of E = '+\
                          f'{self.E}  Tp = {self.Tp}  tau = {self.tau} ' +\
                          'is invalid for the library.'
                    raise RuntimeError( msg )

        #------------------------------------------------
        # pred_i from pred
        #------------------------------------------------
        # predPairs vector of start, stop index pairs
        if len( self.pred ) % 2 :
            # Odd number of pred
            msg = f'{self.name}: CreateIndices() pred must be an even ' +\
                'number of elements. Pred start : stop pairs'
            raise RuntimeError( msg )

        predPairs = [] # List of 2-tuples of pred indices
        for i in range( 0, len( self.pred ), 2 ) :
            predPairs.append( (self.pred[i], self.pred[i+1]) )

        if len( predPairs ) > 1 : self.disjointPred = True

        # Validate end > start
        for predPair in predPairs :
            predStart, predEnd = predPair

            if self.name in [ 'Simplex', 'SMap', 'Multiview' ] :
                # Don't check CCM since default of "1 1" is used.
                if predStart >= predEnd :
                    msg = f'{self.name}: CreateIndices() pred start ' +\
                        f' {predStart} exceeds pred end {predEnd}.'
                    raise RuntimeError( msg )

            # Disallow indices < 1, the user may have specified 0 start
            if predStart < 1 or predEnd < 1 :
                msg = f'{self.name}: CreateIndices() pred indices ' +\
                    ' less than 1 not allowed.'
                raise RuntimeError( msg )

        # Create pred_i indices from predPairs
        for r in range( len( predPairs ) ) :
            start, stop = predPairs[ r ]
            pred_i      = zeros( stop - start + 1, dtype = int )

            i = 0
            for j in range( start, stop + 1 ) :
                pred_i[ i ] = j - 1  # apply zero-offset
                i = i + 1

            self.predList.append( pred_i ) # Append disjoint segment(s)

        # flatten arrays in self.predList for single array self.pred_i
        pred_i_ = []
        for pred_i in self.predList :
            i_      = [i for i in pred_i]
            pred_i_ = pred_i_ + i_

        self.pred_i = array( pred_i_, dtype = int )

        self.PredictionValid()

        self.pred_i_all = self.pred_i.copy() # Before nan are removed

        # Remove embedShift nan from predPairs
        # NOTE : This does NOT redefine self.pred_i, only self.predPairs
        #        self.pred_i is redefined to remove all nan in RemoveNan()
        #        at the API level.
        if not self.embedded :
            # If [0, 1, ... embedShift] nan (negative tau) or
            # [N - embedShift, ... N-1, N]  (positive tau) nan
            # are in pred_i delete elements
            nan_i_start = [ i for i in range( embedShift ) ]
            nan_i_end   = [ self.Data.shape[0]-1-i for i in range(embedShift) ]

            for i in range( len( self.predList ) ) :
                pred_i = self.predList[i]
                nan_i  = None

                if self.tau > 0 :
                    if any( [ i in nan_i_end for i in pred_i ] ) :
                        pred_i_ = [i for i in pred_i if i not in nan_i_end]
                        self.predList[i] = array( pred_i_, dtype = int )
                else :
                    if any( [ i in nan_i_start for i in pred_i ] ) :
                        pred_i_ = [i for i in pred_i if i not in nan_i_start]
                        self.predList[i] = array( pred_i_, dtype = int )

        #------------------------------------------------
        # Validate lib_i pred_i do not exceed data
        #------------------------------------------------
        if self.lib_i[-1] >= self.Data.shape[0] :
            msg = f'{self.name}: CreateIndices() The prediction index ' +\
                f'{self.lib_i[-1]} exceeds the number of data rows ' +\
                f'{self.Data.shape[0]}'
            raise RuntimeError( msg )

        if self.pred_i[-1] >= self.Data.shape[0] :
            msg = f'{self.name}: CreateIndices() The prediction index ' +\
                f'{self.pred_i[-1]} exceeds the number of data rows ' +\
                f'{self.Data.shape[0]}'
            raise RuntimeError( msg )

        #---------------------------------------------------
        # Check for lib : pred overlap for knn leave-one-out
        #---------------------------------------------------
        if len( set( self.lib_i ).intersection( set( self.pred_i ) ) ) :
                self.libOverlap = True

        if self.name == 'SMap' :
            if self.knn < 1 :  # default knn = 0, set knn value to full lib
                self.knn = len( self.lib_i ) - 1

                if self.verbose :
                    msg = f'{self.name} CreateIndices(): ' +\
                        f'Set knn = {self.knn} for SMap.'
                    print( msg, flush = True )

    #--------------------------------------------------------------------
    def PredictionValid( self ) :
    #--------------------------------------------------------------------
        '''Validate there are pred_i to make a prediction
        '''
        if self.verbose:
            print( f'{self.name}: PredictionValid()' )

        if len( self.pred_i ) == 0 :
            msg = f'{self.name}: PredictionValid() No valid prediction ' +\
                'indices. Examine pred, E, tau, Tp parameters and/or nan.'
            warn( msg )

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if self.Data is None :
            raise RuntimeError(f'Validate() {self.name}: dataFrame required.')
        else :
            if not isinstance( self.Data, DataFrame ) :
                raise RuntimeError(f'Validate() {self.name}: dataFrame ' +\
                                   'is not a Pandas DataFrame.')

        if not len( self.columns ) :
            raise RuntimeError( f'Validate() {self.name}: columns required.' )
        if not IsIterable( self.columns ) :
            self.columns = self.columns.split()

        for column in self.columns :
            if not column in self.Data.columns :
                raise RuntimeError( f'Validate() {self.name}: column ' +\
                                    f'{column} not found in dataFrame.' )

        if not len( self.target ) :
            raise RuntimeError( f'Validate() {self.name}: target required.' )
        if not IsIterable( self.target ) :
            self.target = self.target.split()

        for target in self.target :
            if not target in self.Data.columns :
                raise RuntimeError( f'Validate() {self.name}: target ' +\
                                    f'{target} not found in dataFrame.' )

        if not self.embedded :
            if self.tau == 0 :
                raise RuntimeError(f'Validate() {self.name}:' +\
                                   ' tau must be non-zero.')
            if self.E < 1 :
                raise RuntimeError(f'Validate() {self.name}:' +\
                                   f' E = {self.E} is invalid.')

        if self.name != 'CCM' :
            if not len( self.lib ) :
                raise RuntimeError( f'Validate() {self.name}: lib required.' )
            if not IsIterable( self.lib ) :
                self.lib = [ int(i) for i in self.lib.split() ]

            if not len( self.pred ) :
                raise RuntimeError( f'Validate() {self.name}: pred required.' )
            if not IsIterable( self.pred ) :
                self.pred = [ int(i) for i in self.pred.split() ]

        # Set knn default based on E and lib size, E embedded on num columns
        if self.name in [ 'Simplex', 'CCM', 'Multiview' ] :
            # embedded = true: Set E to number of columns
            if self.embedded :
                self.E = len( self.columns )

            # knn not specified : knn set to E+1
            if self.knn < 1 :
                self.knn = self.E + 1

                if self.verbose :
                    msg = f'{self.name} Validate(): Set knn = {self.knn}'
                    print( msg, flush = True )

        if self.name == 'SMap' :
            # embedded = true: Set E to number of columns
            if self.embedded and len( self.columns ) :
                self.E = len( self.columns )

            if not self.embedded and len( self.columns ) > 1 :
                msg = f'{self.name} Validate(): Multivariable S-Map ' +\
                'must use embedded = True to ensure data/dimension '  +\
                'correspondance.'
                raise RuntimeError( msg )

        if self.generateSteps > 0 :
            # univariate only, embedded must be False
            if self.name in [ 'Simplex', 'SMap' ] :

                if self.embedded :
                    msg = f'{self.name} Validate(): generateSteps > 0 ' +\
                        'must use univariate embedded = False.'
                    raise RuntimeError( msg )

                if self.target[0] != self.columns[0] :
                    msg = f'{self.name} Validate(): generateSteps > 0 ' +\
                          f'must use univariate target ({self.target[0]}) ' +\
                          f' == columns ({self.columns[0]}).'
                    raise RuntimeError( msg )

                # If times are datetime, AddTime() fails
                # EDM.time is ndarray storing python datetime
                # In AddTime() datetime, timedelta operations are not compatible
                # with numpy datetime64, timedelta64 : deltaT fails in conversion
                # If times are datetime: raise exception
                if not self.noTime :
                    try:
                        time0 = self.Data.iloc[ 0, 0 ] # self.time not yet
                        dt0   = datetime.fromisoformat( time0 )
                    except :
                        # dt0 is not a datetime assign for finally to pass
                        dt0 = None
                    finally :
                        # if dt0 is datetime, raise exception for noTime = True
                        if isinstance( dt0, datetime ) :
                            msg = f'{self.name} Validate(): generateSteps ' +\
                                'with datetime needs to use noTime = True.'
                            raise RuntimeError( msg )
