# How to Train Your Model

In this directory you fill find everything required to train the model on your own dataset.

* `model.py`: a convolutional Bayesian Neural Net, closely inspired by [Pizarro et al. (2023)](https://doi.org/10.1016/j.media.2023.102942)
* `train_utils.py`: helper classes and functions (notably the `DataLoader` which also performs dataset augmentation with synthetic artefacts or additional clean images, as needed)
* `train.py`: the train-script, with subsequent model evaluation on the test set

## Dataset Specification

You can train your model on a dataset of a single or multiple MRI contrasts simultaneously.
Currently, the available contrast options are

* `T1wMPR`: T1-weighted MPRAGE
* `T1wTIR`: T1-weighted TIR
* `T2w`: T2-weighted
* `T2starw`: T2*-weighted
* `FLAIR`

If more/other contrasts are available, the `DataCrawler` class in `train_utils.py` would have to be modified appropriately.

Your dataset should contain both clean images and ones with acquisition artefacts (not necessarily in the same proportion), stored in separate folders.
The images should be in `nii` or `nii.gz` format (can be mixed), with filenames beginning with `sub-XYZ_`, where `XYZ` is a unique *subject/patient* ID. 

Images should be structured into sub-datasets (e.g. one for each different image source). Our data pre-processing pipeline assumes the following overall dataset structure, at the root path `DATADIR`.

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

## Training

### Constants

Modify the constants at the beginning of `train.py` to define:

* `SAVEDIR`: where to save output files
* `DATADIR`: path to the root of your dataset
* `DATASETS`: which sub-datasets to use (e.g. `['dataset1', 'dataset2', ...]`)
* `CONTRASTS`: which contrasts to use (e.g. `['T1wMPR', 'T2w']`)
* `QUALS`: which image quality categories to use; must currently be set to `['clean', 'exp_artefacts']`

Our `DataLoader` allows the user to specify a desired ratio of clean to artefact-ridden images on which to train. If the desired ratio does not reflect the dataset provided, the `DataLoader` will create synthetic clean images (by applying random flips, rotations and scalings to existing clean images) or synthetic artefacts (by introducing simulated artefacts into existing clean images) until the desired ratio is reached. The distribution from which the synthetic artefacts to be introduced are drawn, can be adjusted in the `ARTEFACT_DISTRO` variable at the top of the `train.py`.

Our model is Bayesian Neural Network; practically speaking we stochastically drop out some of the weights at inference time, which means model predictions on the same image vary between repeated inference runs. Per training epoch, we run each image in the train/validation-set through the network *once* (stochastic, single-shot inference). On the test-set, we run each image through the network `MC_RUNS` times, and output all predictions. These can be collated in different ways to yield a final model prediction (see `evaluation` directory of the repository). You may specify the number of `MC_RUNS` at the top of `train.py`.

### Image and Batch Sizes

Have an idea of the distribution of image sizes in your dataset. The current model assumes image dimensions of $(\textnormal{depth},\textnormal{height},\textnormal{width},\textnormal{channels})=(192,256,256,1)$ and the `DataLoader` simply crops/pads to that size. If your images differ greatly from this, the `DataLoader` and model specification may have to be adjusted. This can be done directly in `train.py`, in the `DataLoader` and `model` instantiations.

Batch sizes depend on available GPU memory, and may also be specified in the `DataLoader` instantiation. We found that a batch size of 10 is sufficient for stable training, which–for images of the size above–runs on an A100 with 40GB of GPU memory.

### Training Procedure

Modify `train.py`, as appropriate for your dataset (see previous sections), then run the script on a GPU-enabled system.

This will 

* train/val/test-split the data *by patient id*, so as to avoid data leakage
* output the test-set ids and ground-truth labels in a `csv` for later reference
* train the model with BCE loss with variable learning rate for 24 epochs, saving checkpoints after every epoch
* stop early, if the validation loss stops improving
* make a plot of the training metrics 
* evaluate the trained model on the test-set, running `MC_RUNS` inferences on each image and outputting each prediction

All outputs are saved to the `SAVEDIR` you specified.

### Further Evaluation

The multiple predictions on the test-set can be further analysed using the scripts in the `evaluation` directory.
