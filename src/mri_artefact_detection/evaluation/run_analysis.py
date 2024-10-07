from .analysis_utils import *

def evaluate(MODEL_PREDS: str, TERNARY: bool):
    # load predictions and ground truth
    ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(MODEL_PREDS)

    print('Running analysis for predictive probability := average predicted class')
    if TERNARY:
        run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10,  lattice_size=10, 
                        option='mean_class', OUTDIR='analysis_mean_class', init_thresh=0.5, 
                        max_clean_impurity=0.0, min_dirty_impurity=0.95)
    else:
        run_analysis(raw_model_preds, ground_truth_labels, 
                MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_class', OUTDIR='analysis_mean_class', init_thresh=0.5)

    print('Running analysis for predictive probability := average probability')
    if TERNARY:
        run_ternary_analysis(raw_model_preds, ground_truth_labels, MC=20, nbins=10, lattice_size=10, 
                         option='mean_prob', OUTDIR='analysis_mean_prob', 
                         max_clean_impurity=0.0, min_dirty_impurity=0.95)
    else: 
        run_analysis(raw_model_preds, ground_truth_labels, 
                 MC=20, nbins=10, maxDFFMR=0.3, lattice_size=50, option='mean_prob', OUTDIR='analysis_mean_prob')
