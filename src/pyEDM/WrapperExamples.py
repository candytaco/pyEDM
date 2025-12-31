"""
Examples demonstrating the new wrapper classes with sklearn-like API.
"""
from .CCMWrapper import CCMWrapper
from .LoadData import sampleData
from .MultiviewWrapper import MultiviewWrapper
from .SMapWrapper import SMapWrapper
from .SimplexWrapper import SimplexWrapper
from .Visualization import (plot_prediction, plot_smap_coefficients, plot_ccm)


def WrapperExamples():
	"""
	Examples using the new wrapper classes with sklearn-like separate arrays.
	"""

	# Load sample data
	sampleDataNames = ["TentMap", "TentMapNoise", "circle", "block_3sp", "sardine_anchovy_sst"]

	# Example 1: SimplexWrapper with block_3sp data (embedded = True)
	print("Example 1: SimplexWrapper with block_3sp data (embedded = True)")
	print("=" * 60)

	# Split data into separate arrays
	data = sampleData["block_3sp"]
	XTrain = data[1:100, [1, 4, 7]]  # Columns 1, 4, 7 (features), rows 1-99
	YTrain = data[1:100, [1]]  # Target column 1, rows 1-99
	XTest = data[100:196, [1, 4, 7]]  # Columns 1, 4, 7, rows 100-195
	YTest = data[100:196, [1]]  # Target column 1, rows 100-195

	# Create and run SimplexWrapper
	simplexWrapper = SimplexWrapper(
		XTrain = XTrain,
		YTrain = YTrain,
		XTest = XTest,
		YTest = YTest,
		EmbedDimensions = 3,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		Verbose = False,
		Embedded = True
	)

	result = simplexWrapper.Run()
	plot_prediction(result.projection, "SimplexWrapper: block_3sp embedded", embedDimensions = 3)

	# Example 2: SimplexWrapper with block_3sp data (embedded = False)
	print("\nExample 2: SimplexWrapper with block_3sp data (embedded = False)")
	print("=" * 60)

	data = sampleData["block_3sp"]
	XTrain = data[1:100, [1]]  # Column 1 only, rows 1-99
	YTrain = data[1:100, [1]]  # Target column 1, rows 1-99
	XTest = data[105:191, [1]]  # Column 1 only, rows 105-190
	YTest = data[105:191, [1]]  # Target column 1, rows 105-190

	simplexWrapper2 = SimplexWrapper(
		XTrain = XTrain,
		YTrain = YTrain,
		XTest = XTest,
		YTest = YTest,
		EmbedDimensions = 3,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		Verbose = False,
		Embedded = False
	)

	result = simplexWrapper2.Run()
	plot_prediction(result.projection, "SimplexWrapper: block_3sp", embedDimensions = 3)

	# Example 3: MultiviewWrapper with block_3sp data
	print("\nExample 3: MultiviewWrapper with block_3sp data")
	print("=" * 60)

	data = sampleData["block_3sp"]
	XTrain = data[1:101, [1, 4, 7]]  # Columns 1, 4, 7, rows 1-100
	YTrain = data[1:101, [1]]  # Target column 1, rows 1-100
	XTest = data[101:199, [1, 4, 7]]  # Columns 1, 4, 7, rows 101-198
	YTest = data[101:199, [1]]  # Target column 1, rows 101-198

	multiviewWrapper = MultiviewWrapper(
		XTrain = XTrain,
		YTrain = YTrain,
		XTest = XTest,
		YTest = YTest,
		D = 0,
		EmbedDimensions = 3,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		NumMultiview = 0,
		ExclusionRadius = 0,
		TrainLib = False,
		ExcludeTarget = False,
		Verbose = False
	)

	result = multiviewWrapper.Run()
	plot_prediction(result.projection, "MultiviewWrapper: block_3sp", embedDimensions = 3)

	# Example 4: SMapWrapper with circle data
	print("\nExample 4: SMapWrapper with circle data")
	print("=" * 60)

	data = sampleData["circle"]
	XTrain = data[1:101, [1, 2]]  # Columns 1-2 (features), rows 1-100
	YTrain = data[1:101, [1]]  # Target column 1, rows 1-100
	XTest = data[110:191, [1, 2]]  # Columns 1-2, rows 110-190
	YTest = data[110:191, [1]]  # Target column 1, rows 110-190

	smapWrapper = SMapWrapper(
		XTrain = XTrain,
		YTrain = YTrain,
		XTest = XTest,
		YTest = YTest,
		EmbedDimensions = 2,
		PredictionHorizon = 1,
		KNN = 0,
		Step = -1,
		Theta = 4,
		Verbose = False,
		Embedded = True
	)

	result = smapWrapper.Run()
	plot_prediction(result.projection, "SMapWrapper: circle", embedDimensions = 2)
	plot_smap_coefficients(result.coefficients, "SMapWrapper Coefficients", embedDimensions = 2)

	# Example 5: CCMWrapper with sardine_anchovy_sst data
	print("\nExample 5: CCMWrapper with sardine_anchovy_sst data")
	print("=" * 60)

	data = sampleData["sardine_anchovy_sst"]
	XTrain = data[:, [1]]  # Column 1 (sardine), all rows
	YTrain = data[:, [4]]  # Target is same

	ccmWrapper = CCMWrapper(
		XTrain = XTrain,
		YTrain = YTrain,
		TrainSizes = [10, 70, 10],
		numRepeats = 50,
		EmbedDimensions = 3,
		PredictionHorizon = 0,
		Verbose = False
	)

	result = ccmWrapper.Run()
	plot_ccm(result, "CCMWrapper: sardine anchovy sst", embedDimensions = 3)

	print("\nAll wrapper examples completed successfully!")
