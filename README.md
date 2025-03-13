# Pseudo Population and RNN Strategy Analysis

## Tested Environment
- Python 3.10

## Dataset
To create pseudo-populations used for biological data analyses, download dataset files from:
[Inferring Working Memory Strategy](https://figshare.com/projects/Inferring_Working_Memory_Strategy/239771) and refer to:
- [`bioData create pseudoPop.py`](/bioData_create_pseudoPop.py)

See also the following essential modules used to preprocess our biological data:
- [NeuralProcessingTools](https://github.com/grero/NeuralProcessingTools)
- [DataProcessingTools](https://github.com/grero/DataProcessingTools)

## Training RNNs
To train RNNs with different strategies, refer to:
- [`rnns training.py`](main/rnns_training.py)

## Replicating Results
### Figure 2 and Figure S4
To replicate these results, refer to:
- [`bioData full space decoding.py`](main/bioData_full_space_decoding.py) (pre-computation)
- [`bioData readout subspace analysis.py`](main/bioData_readout_subspace_analysis.py) (pre-computation)
- [`rnns evaluation & analysis.py`](main/rnns_evaluation_&_analysis.py) (pre-computation)
- [`decoding quantification.py`](main/decoding_quantification.py) (plotting and quantification)

### Figure 3 and Figure S3
To replicate these results, refer to:
- [`bioData item subspace analysis.py`](main/bioData_item_subspace_analysis.py) (pre-computation)
- [`rnns evaluation & analysis.py.py`](main/rnns_evaluation_&_analysis.py) (pre-computation)
- [`geometry quantification.py`](main/geometry_quantification.py) (plotting and quantification)

### Figure 4 and Figure S5
To replicate these results, refer to:
- [`bioData readout subspace analysis.py`](main/bioData_readout_subspace_analysis.py) (pre-computation)
- [`rnns evaluation & analysis.py.py`](main/rnns_evaluation_&_analysis.py) (pre-computation)
- [`driftDistance quantification.py`](main/driftDistance_quantification.py) (plotting and quantification)

