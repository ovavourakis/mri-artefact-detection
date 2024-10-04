# Test-Set Inference

> **Warning:** To predict labels for new images (of unknown artefact status, i.e. not a test set), this script, as well as the DataLoader class, will have to be modified.

This example script 

* reads in the testset (written out during training)
* performs inference using the pre-trained model weights,
* saves the raw inference output to file (required for model evaluation; see `evaluation` directory)
* outputs some simple performance statistics on the test set.

To run test-set inference, modify the `TRAINRUN_DIR` variable at the top of the script to point to your training run's output directory. That directory should contain:

```
TRAINRUN_DIR/
    test_split_gt.tsv               # testset metadata file
    ckpts/
        end_of_epoch_X.keras        # model checkpoint to use for inference
```

`test_split_gt.tsv` is output by the training script during initial data splitting and contains the filepaths and binary ground-truth labels (clean: 0, artefact: 1) of the image volumes in the tests set as two tab-separated columns `image`, `bin_gt`:

```
image   bin_gt
/path/to/image1.nii.gz   0
/path/to/image1.nii.gz   1
```

Run this script from the repository root like:
```
python -m inference.inference
```
