# Pseudo Population and RNN Strategy Analysis

## Tested Environment
- Python: v3.10
- OS: Windows 10/11
- No additional software installation required.

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `scipy`
- `scikit-learn`
- `pytorch`
- `pandas`
- `matplotlib`
- `seaborn`

See also the following essential modules used to preprocess our biological data:
- [NeuralProcessingTools](https://github.com/grero/NeuralProcessingTools)
- [DataProcessingTools](https://github.com/grero/DataProcessingTools)

## Dataset
To create pseudo-populations used for biological data analyses, download dataset files from:
[Inferring Working Memory Strategy](https://figshare.com/projects/Inferring_Working_Memory_Strategy/239771) and refer to:
- [`bioData_create_pseudoPop.py`](bioData_create_pseudoPop.py)
- Approximate time to create 1 pseudoPop (with the default parameters): 3 mins

## Training RNNs
To train RNNs with different strategies, refer to:
- [`rnns_training.py`](rnns_training.py)
Approximate time to train 1 RNN (with the default parameters): 5 mins

## Replicating Results
### Figure 2 and Figure S4
To replicate these results, refer to:
- [`bioData_full_space_decoding.py`](bioData_full_space_decoding.py) (pre-computation)
- [`bioData_readout_subspace_analysis.py`](bioData_readout_subspace_analysis.py) (pre-computation)
- [`rnns_evaluation_&_analysis.py`](rnns_evaluation_&_analysis.py) (pre-computation)
- [`decoding_quantification.py`](decoding_quantification.py) (plotting and quantification)
- Approximate time to compute cross-temporal decodability based on 1 pseudoPop/RNN (with the default parameters): 10 mins

### Figure 3 and Figure S3
To replicate these results, refer to:
- [`bioData_item_subspace_analysis.py`](bioData_item_subspace_analysis.py) (pre-computation)
- [`rnns_evaluation_&_analysis.py.py`](rnns_evaluation_&_analysis.py) (pre-computation)
- [`geometry_quantification.py`](geometry_quantification.py) (plotting and quantification)
- Approximate time to compute geometry analysis based on 1 pseudoPop/RNN (with the default parameters): 3 mins

### Figure 4 and Figure S5
To replicate these results, refer to:
- [`bioData_readout_subspace_analysis.py`](bioData_readout_subspace_analysis.py) (pre-computation)
- [`rnns_evaluation_&_analysis.py.py`](rnns_evaluation_&_analysis.py) (pre-computation)
- [`driftDistance_quantification.py`](driftDistance_quantification.py) (plotting and quantification)
- Approximate time to compute drift distance analysis based on 1 pseudoPop/RNN (with the default parameters): 3 mins
