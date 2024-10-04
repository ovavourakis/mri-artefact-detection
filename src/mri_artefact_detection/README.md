# MRIArtefactDetection

A convolutional neural network to detect the presence of acquisition artefacts in MRI brain image volumes.

## Repository Structure

* `training`: train the model on a dataset and perform inference on the test-set
* `inference`: perform inference on a test dataset using pre-trained weights
* `evaluation`: evaluate model performance on test set
* `deprecated`: older scripts for future reference

## Getting Set Up

1. Install CUDA drivers suitable for you GPU-enabled system; make sure the cuDNN library, as well as a suitable `Python` package manager (we recommend `mamba`) are installed.

2. Create a new `Python 3.10.4` virtual environment. Newer versions of `Python` may work but have not been tested. 

```
mamba create -n mri python=3.10.4 pip
mamba activate mri
```

3. Install GPU-accelerated tensorflow. We recommend doing so via `pip`, as per the [official instructions](https://www.tensorflow.org/install/pip):

```
python3 -m pip install tensorflow[and-cuda]
```

4. Complete the environment with a few more libraries, inlcuding `torchio` for reading and modifying MRI volumes.

```
mamba install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
mamba install torchio
mamba install numpy matplotlib scikit-learn keras pandas seaborn tqdm
```

Specific instructions for training, inference and evaluation can be found in the `READMEs` in the relevant directories.

## Acknowledgements

This project was originally forked from [this repository](https://github.com/AS-Lab/Pizarro-et-al-2023-DL-detects-MRI-artifacts), associated with [Pizarro et al. (2023)](https://doi.org/10.1016/j.media.2023.102942). 
The model was modified slightly from the original specification, then re-implemented and re-trained from scratch.
Training data was taken from:

* Nárai, Á., Hermann, P., Auer, T. et al. (2022) [[paper](https://doi.org/10.1038/s41597-022-01694-8), [dataset](https://doi.org/10.18112/openneuro.ds004173.v1.0.2)]
* Pardoe, H. R., Martin, S. P. (2021) [dataset](https://doi.org/10.18112/openneuro.ds003639.v1.0.0)
* Eichhorn, H. et. al. (2022) [preprint](https://doi.org/10.31234/osf.io/vzh4g), [dataset](https://doi.org/10.18112/openneuro.ds003639.v1.0.0)

This work was very kindly supported by GE Healthcare.
