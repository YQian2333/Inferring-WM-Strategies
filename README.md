# Pseudo Population and RNN Strategy Analysis

## Tested Environment
- Python 3.10

## Dataset
To create pseudo-populations used for biological data analyses, download dataset files from:
[Inferring Working Memory Strategy](https://figshare.com/projects/Inferring_Working_Memory_Strategy/239771) and refer to:
- [`bioData create pseudoPop.py`](bioData create_pseudoPop.py)

See also the following essential modules used to preprocess our biological data:
- [NeuralProcessingTools](https://github.com/grero/NeuralProcessingTools)
- [DataProcessingTools](https://github.com/grero/DataProcessingTools)

## Training RNNs
To train RNNs with different strategies, refer to:
- [`rnns training.py`](rnns training.py)

## Replicating Results
### Figure 2 and Figure S4
To replicate these results, refer to:
- [`bioData full space decoding.py`](bioData full space decoding.py) (pre-computation)
- [`bioData readout subspace analysis.py`](bioData readout subspace analysis.py) (pre-computation)
- [`rnns evaluation & analysis.py`](rnns evaluation & analysis.py) (pre-computation)
- [`decoding quantification.py`](decoding quantification.py) (plotting and quantification)

### Figure 3 and Figure S3
To replicate these results, refer to:
- [`bioData item subspace analysis.py`](bioData item subspace analysis.py) (pre-computation)
- [`rnns evaluation & analysis.py.py`](rnns_evaluation & analysis.py) (pre-computation)
- [`geometry quantification.py`](geometry quantification.py) (plotting and quantification)

### Figure 4 and Figure S5
To replicate these results, refer to:
- [`bioData readout subspace analysis.py`](bioData readout subspace analysis.py) (pre-computation)
- [`rnns evaluation & analysis.py.py`](rnns evaluation & analysis.py) (pre-computation)
- [`driftDistance quantification.py`](driftDistance quantification.py) (plotting and quantification)

