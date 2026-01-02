## Empirical Dynamic Modeling (EDM)
---
This package provides a Python toolset for [EDM analysis](http://deepeco.ucsd.edu/nonlinear-dynamics-research/edm/ "EDM @ Sugihara Lab").

Functionality includes:
* Simplex projection ([Sugihara and May 1990](https://www.nature.com/articles/344734a0))
* Sequential Locally Weighted Global Linear Maps (S-Map) ([Sugihara 1994](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1994.0106))
* Multivariate embeddings ([Dixon et. al. 1999](https://science.sciencemag.org/content/283/5407/1528))
* Convergent cross mapping ([Sugihara et. al. 2012](https://science.sciencemag.org/content/338/6106/496))
* Multiview embedding ([Ye and Sugihara 2016](https://science.sciencemag.org/content/353/6302/922))

---
## API Overview

This is forked from the main pyEDM repo and refactored to:
- provide a numpy-native API, removing the pandas dependency
- provide OOP objects that provide a clear train/test and X/Y separation in the arguments
- 
For example usage see:
- `FitterExamples` for OOP objects
- `FunctionalExamples` for the functional API

### 1. Object-Oriented API (New)
The OOP API provides sklearn-like wrappers with explicit train/test separation. These are ideal for machine learning workflows.

```python
from pyEDM import Fitters
import numpy

# Load data
data = numpy.loadtxt('data/TentMap.csv', delimiter = ',')

# Split data
XTrain = data[0:100, 1]
YTrain = data[0:100, 1]
XTest = data[100:200, 1]
YTest = data[100:200, 1]

# Create fitter
fitter = Fitters.SimplexFitter(
    XTrain = XTrain,
    YTrain = YTrain,
    XTest = XTest,
    YTest = YTest,
    EmbedDimensions = 3,
    PredictionHorizon = 1,
    KNN = 4,
    Step = -1
)

# Run prediction
result = fitter.Run()
```

### 2. Functional API (Traditional)
The functional API provides simple functions for EDM analysis. 
These functions accept numpy arrays and return predictions directly.
Other than the numpy-nativeness, these functions largely keep the same
argument syntax as the pyEDM package.

```python
from pyEDM import Functions
import numpy

# Load data
data = numpy.loadtxt('data/TentMap.csv', delimiter = ',')

# Simplex prediction
result = Functions.FitSimplex(
    data = data,
    columns = [1],
    target = 1,
    train = (0, 100),
    test = (100, 200),
    embedDimensions = 3,
    predictionHorizon = 1,
    knn = 4,
    step = -1
)
```


---
## Architecture Overview

### Data Interface
The package now uses a pure NumPy array interface, removing the Pandas dependency. All functions accept and return NumPy arrays, providing better performance and compatibility.

### Codebase Organization

```
src/pyEDM/
├── __init__.py                # Package initialization with API exports
├── Functions.py               # Functional API (FitSimplex, FitSMap, etc.)
├── Fitters/                   # Object-Oriented API wrappers
│   ├── EDMFitter.py           # Base fitter class
│   ├── SimplexFitter.py       # Simplex OOP wrapper
│   ├── SMapFitter.py          # S-Map OOP wrapper
│   ├── CCMFitter.py           # CCM OOP wrapper
│   ├── MDEFitter.py           # MDE OOP wrapper
│   ├── MDEFitterCV.py         # MDE CV OOP wrapper
│   ├── MultiviewFitter.py     # Multiview OOP wrapper
│   └── DataAdapter.py         # Data separation adapter
├── EDM/                       # Core EDM algorithms
│   ├── Simplex.py             # Simplex projection
│   ├── SMap.py                # S-Map implementation
│   ├── CCM.py                 # Convergent Cross Mapping
│   ├── Multiview.py           # Multiview embedding
│   ├── MDE.py                 # Multivariate Delay Embedding
│   ├── MDECV.py               # MDE with cross-validation
│   ├── Embed.py               # Embedding utilities
│   ├── Results.py             # Result objects
│   └── PoolFunc.py            # Parallel processing
├── Utils.py                   # Utility functions and visualization
├── Visualization.py           # Plotting functions
├── ExampleData.py             # Sample datasets
├── FunctionalExamples.py      # Functional API examples
└── FitterExamples.py          # OOP API examples
```

## Available Methods

### Functional API
- `FitSimplex()`: Simplex projection
- `FitSMap()`: S-Map prediction
- `FitCCM()`: Convergent Cross Mapping
- `FitMultiview()`: Multiview embedding
- `FitMDE()`: Multivariate Delay Embedding
- `FindOptimalEmbeddingDimensionality()`: Embedding dimension optimization
- `FindOptimalPredictionHorizon()`: Prediction horizon optimization
- `FindSMapNeighborhood()`: S-Map neighborhood size optimization

### Object-Oriented API
- `SimplexFitter`: Simplex projection wrapper
- `SMapFitter`: S-Map prediction wrapper
- `CCMFitter`: Convergent Cross Mapping wrapper
- `MultiviewFitter`: Multiview embedding wrapper
- `MDEFitter`: MDE wrapper
- `MDEFitterCV`: MDE with cross-validation wrapper


---
### References
Sugihara G. and May R. 1990.  Nonlinear forecasting as a way of distinguishing 
chaos from measurement error in time series. [Nature, 344:734–741](https://www.nature.com/articles/344734a0).

Sugihara G. 1994. Nonlinear forecasting for the classification of natural 
time series. [Philosophical Transactions: Physical Sciences and 
Engineering, 348 (1688) : 477–495](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1994.0106).

Dixon, P. A., M. Milicich, and G. Sugihara, 1999. Episodic fluctuations in larval supply. [Science 283:1528–1530](https://science.sciencemag.org/content/283/5407/1528).

Sugihara G., May R., Ye H., Hsieh C., Deyle E., Fogarty M., Munch S., 2012.
Detecting Causality in Complex Ecosystems. [Science 338:496-500](https://science.sciencemag.org/content/338/6106/496).

Ye H., and G. Sugihara, 2016. Information leverage in interconnected 
ecosystems: Overcoming the curse of dimensionality. [Science 353:922–925](https://science.sciencemag.org/content/353/6302/922).
