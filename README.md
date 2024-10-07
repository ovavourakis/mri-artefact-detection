# MRI Artefact Detection

![Tests](
https://github.com/ovavourakis/mri-artefact-detection/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](
https://coveralls.io/repos/github/ovavourakis/mri-artefact-detection/badge.svg?branch=main)](https://coveralls.io/github/ovavourakis/mri-artefact-detection?branch=main)
[![License: MIT](
https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- ![Code style: black](
https://img.shields.io/badge/code%20style-black-000000.svg) -->

A convolutional Bayesian neural network to detect the presence of acquisition artefacts in MRI brain image volumes.

## Key Things To Know

1. Our Bayesian Neural Network model predicts the probability an image contains an artefact, and is trained with BCE loss with predictions stochastically varying across inference runs. 
2. During training, each image is processed once per epoch, while inference involves multiple passes to generate predictions, which can be averaged before or after class thresholding for final results. 
3. Users can choose between binary ('clean' vs. 'artefact') and ternary ('clean', 'artefact', 'manual review needed'; see `eval_theory.pdf`) classification schemes during performance evaluation. The latter will output performance metrics for different threshold choices allowing the user to select an operating point that balances review workload and data accuracy. 
4. Data is split by patient ID for training/validation/testing, and users can define the ratio of clean to artefact images. If the dataset doesn't match this ratio, synthetic images are generated through transformations or artefact simulations. The distribution of synthetic artefacts can be adjusted by the user.

## Installation

```bash
$ pip install git+https://github.com/ovavourakis/mri-artefact-detection.git
```

## Usage

The `mri-artefact-detection` package provides a command-line interface for training the model on your own data (see below for required format), running testset inference, and evaluating model performance.

### Training & Testing

Basic Usage:
```bash
mri-artefact-detection train [ARGUMENTS] [OPTIONS]
```

Required Arguments:
- `--datadir`: directory containing the dataset (structured as described below).
- `--savedir`: directory where training outputs and checkpoints will be saved.

The following options allow you

- to use only specific data subsets during training

    - `--datasets`: comma-separated list of data subset names to be used for training. (default: "artefacts1,artefacts2,artefacts3")
    - `--contrasts`: comma-separated list of MRI contrasts to be considered. (default: "T1wMPR")
    - `--quals`: comma-separated list of quality labels to be considered (default: "clean,exp_artefacts")

- to specify a desired distribution among synthetic artefacts to be introduced (see [`torchio`]() for examples)

    - `--target-clean-ratio`: desired fraction of 'clean' images (default: 0.5)
    - `--random-affine`: probability for random affine transformation (default: 1/12)
    - `--random-elastic-deformation`: probability for random elastic deformation (default: 1/12)
    - `--random-anisotropy`: probability for field anisotropy artefact (default: 1/12)
    - `--rescale-intensity`: probability for uniform intensity artefact (default: 1/12)
    - `--random-motion`: probability for random motion artefact (default: 1/12)
    - `--random-ghosting`: probability for ghosting artefact (default: 1/12)
    - `--random-spike`: probability for spike in K-space artefact (default: 1/12)
    - `--random-bias-field`: probability for bias field artefact (default: 1/12)
    - `--random-blur`: probability for blur artefact (default: 1/12)
    - `--random-noise`: probability for random noise artefact (default: 1/12)
    - `--random-swap`: probability for random swap artefact (default: 1/12)
    - `--random-gamma`: probability for random gamma shift artefact (default: 1/12)

- to alter the number of inference passes per item in the test set 

    - `--mc-runs`: number of Monte Carlo runs on the test set. (default: 20)

Example:
```bash
mri-artefact-detection train --savedir ./output --datadir ./data --datasets artefacts1,artefacts2 --contrasts T1wMPR --quals clean,exp_artefacts --random-ghosting 0.5 --random-anisotropy 0.5 --mc-runs 10
```

### Inference

Standalone inference on a testset, using pre-trained weights is also possible.
This will output to file the model-predicted artefact probability for each image and inference run, and requires as input a tab-separated file containing the filepaths and binary ground-truth labels (clean: 0, artefact: 1) of the image volumes in the testset, formatted as two columns `image`, `bin_gt`:

```
image                    bin_gt
/path/to/image1.nii.gz   0
/path/to/image1.nii.gz   1
```

Usage:
```bash
mri-artefact-detection infer [ARGUMENTS] [OPTIONS]
```

Required Arguments:
- `--savedir`: Directory where inference outputs will be saved.
- `--weights`: Path to the pre-trained model weights.
- `--gt-data`: Path to the ground truth data file.

Options:
- `--mc-runs`: Number of Monte Carlo runs on the test set. (default: 20)

### Evaluation

Performance evaluation requires as input a tab-separated file containing the filepaths, binary ground-truth labels (clean: 0, artefact: 1), and model-predicted artefact probabilities of the image volumes in the testset, formatted as three columns `image`, `bin_gt`, `pred_<run>`:

```
image                    bin_gt  pred_1 pred_2  pred_3
/path/to/image1.nii.gz   0       0.1    0.2     0.1
...
```

Usage:
```bash
mri-artefact-detection eval [ARGUMENTS] [OPTIONS]
``` 

Required Arguments:
- `--model-preds`: Path to the model predictions file.

Options:
- `--ternary`: Flag to indicate ternary classification analysis. If not set, binary analysis is performed by default.

## Dataset Specification

You can train your model on a dataset of a single or multiple MRI contrasts simultaneously.
Currently, the available contrast options are

* `T1wMPR`: T1-weighted MPRAGE
* `T1wTIR`: T1-weighted TIR
* `T2w`: T2-weighted
* `T2starw`: T2*-weighted
* `FLAIR`

Your dataset should contain both clean images and ones with acquisition artefacts (not necessarily in the same proportion), stored in separate folders.
The images should be in `nii` or `nii.gz` format (can be mixed), with filenames beginning with `sub-XYZ_`, where `XYZ` is a unique *subject/patient* ID. 

Images should be structured into sub-datasets (e.g. one for every distinct source). Our data pre-processing pipeline assumes the following overall dataset structure, at the root path `DATADIR`.

```
DATADIR/
    dataset1/
        T1wMPR/
            clean/
                sub-992238_XXXXXXXXXXXX.nii
                sub-992239_YYYYYYYYYYYY.nii.gz
                ...
            exp_artefacts/
                sub-992240_XXXXXXXXXXXX.nii
                ...
        T2w/
            clean/
                ...
            exp_artefacts/
                ...
        ...
    dataset2/
        ...
```

### Image and Batch Sizes

The model assumes image dimensions of $(\textnormal{depth},\textnormal{height},\textnormal{width},\textnormal{channels})=(192,256,256,1)$ and our `DataLoader` simply crops/pads to that size. If your images differ greatly from this, the `DataLoader` and model specification may have to be adjusted.

Batch sizes depend on available GPU memory, and may also be specified in the `DataLoader` instantiation. We found that a batch size of 10 is sufficient for stable training, which–for images of the size above–runs on an A100 with 40GB of GPU memory.

## License

This project is licensed under the terms of the MIT license.

## Credits

`mri_artefact_detection` was created by Odysseas Vavourakis, and Iris Marmouset-De La Taille Tretin with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## Acknowledgements

This project was originally forked from [this repository](https://github.com/AS-Lab/Pizarro-et-al-2023-DL-detects-MRI-artifacts), associated with [Pizarro et al. (2023)](https://doi.org/10.1016/j.media.2023.102942). 
The model was modified slightly from the original specification, then re-implemented from scratch.
Training data was collated from:

* Nárai, Á., Hermann, P., Auer, T. et al. (2022) [[paper](https://doi.org/10.1038/s41597-022-01694-8), [dataset](https://doi.org/10.18112/openneuro.ds004173.v1.0.2)]
* Pardoe, H. R., Martin, S. P. (2021) [dataset](https://doi.org/10.18112/openneuro.ds003639.v1.0.0)
* Eichhorn, H. et. al. (2022) [preprint](https://doi.org/10.31234/osf.io/vzh4g), [dataset](https://doi.org/10.18112/openneuro.ds003639.v1.0.0)

This work was very kindly supported by GE Healthcare.
