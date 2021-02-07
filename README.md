# Simple GMM-HMM models for isolated digit recognition

Python implementation of simple GMM and HMM models for isolated digit recognition.

This implementation contains 3 models:

1. Single Gaussian: Each digit is modeled using a single Gaussian with diagonal 
covariance.
2. Gaussian Mixture Model (GMM): Each digit is modeled using a mixture of Gaussians, 
initialized by perturbing the single Gaussian model.
3. Hidden Markov Model (HMM): Each digit is modeled by an HMM consisting of N states, 
where the emission probability of each state is a single Gaussian with diagonal covariance.

**Disclaimer:** This is an educational implementation and is not expected to be high-performance.

### Installation

To install for usage:

```shell
pip install git+https://github.com/desh2608/gmm-hmm-asr.git
```

To install with tests (for development):

```shell
git clone https://github.com/desh2608/gmm-hmm-asr.git
cd gmm-hmm-asr && pip install -e .
```

### Running the tests

```shell
pytest
```

This will run each of the 3 models end-to-end, and take approximately 2-3 minutes.

### Usage

#### 1. Single Gaussian

To train, first create `train_data` which should be a list of `DataTuple(key,feats,label)` objects. 

```python
from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import SingleGaussTrainer

ndim = 40 # dimensionality of features
DIGITS = ['1','2','3','4','5'] # digits to be recognized

sg_model = SingleGaussTrainer(ndim, DIGITS)
sg_model.train(train_data)
```

For prediction, again create a `test_data` list similar to `train_data`.

```python
preds = sg_model.predict(test_data)
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood
```

#### 2. Gaussian Mixture Model

```python
from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import GMMTrainer

ndim = 40 # dimensionality of features
ncomp = 8 # number of Gaussian components
niter = 10 # number of training iterations
DIGITS = ['1','2','3','4','5'] # digits to be recognized

gmm_model = GMMTrainer(ndim, ncomp, DIGITS)
gmm_model.train(train_data, niter)

preds = gmm_model.predict(test_data)
```

#### 3. Hidden Markov Model

```python
from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import HMMTrainer

ndim = 40 # dimensionality of features
nstate = 5 # number of HMM states
niter = 10 # number of training iterations
DIGITS = ['1','2','3','4','5'] # digits to be recognized

hmm_model = GMMTrainer(ndim, nstate, DIGITS)
hmm_model.train(train_data, niter)

preds = hmm_model.predict(test_data)
```

### Issues

If you find any bugs, please raise an Issue or contact `draj@cs.jhu.edu`.
