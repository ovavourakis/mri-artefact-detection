"""
Example script to analyse raw model predictions produced by inference/inference.py
on a test set with known labels.

Takes as input a file MODEL_PREDS containing model predictions, as well as the 
ground truth labels (bin_gt) for the test set, as a tab-separated file with the 
following format:

image                           bin_gt      pred_1    ...       pred_20
sub-926536_acq-headmotion1_T1w  1           0.433381  ...  5.049924e-01
sub-926536_acq-standard_T1w     0           0.003448  ...  6.611057e-02

where the pred_X columns are the raw model probability output from different MC samples.
"""

from analysis_utils import *

MODEL_PREDS = '/vols/opig/users/vavourakis/ge_project/trainrun/raw_preds_test.tsv'

ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(MODEL_PREDS)

print('Running analysis for predictive probability := average predicted class')
run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10,  lattice_size=10, 
                    option='mean_class', OUTDIR='analysis_mean_class', init_thresh=0.5, 
                    max_clean_impurity=0.0, min_dirty_impurity=0.95)

print('Running analysis for predictive probability := average probability')
run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10, lattice_size=10, 
                     option='mean_prob', OUTDIR='analysis_mean_prob', 
                     max_clean_impurity=0.0, min_dirty_impurity=0.95)
