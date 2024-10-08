from .analysis_utils import *


def evaluate(model_preds: str, ternary: bool) -> None:
    """
    Runs analysis on model predictions using either binary or ternary
    classification scheme and both prediction aggregation through
    probability and class averaging.

    :param model_preds: Path to the model predictions file.
    :type model_preds: str
    :param ternary: Flag to determine if ternary analysis should be run.
    :type ternary: bool
    :return: None
    """
    # load predictions and ground truth
    ground_truth_labels, raw_model_preds = load_predictions_and_ground_truth(
        model_preds
    )

    print("Running analysis for predictive probability := average predicted class")
    if ternary:
        run_ternary_analysis(
            raw_model_preds,
            ground_truth_labels,
            MC=20,
            nbins=10,
            lattice_size=10,
            option="mean_class",
            OUTDIR="analysis_mean_class",
            init_thresh=0.5,
            max_clean_impurity=0.0,
            min_dirty_impurity=0.95,
        )
    else:
        run_analysis(
            raw_model_preds,
            ground_truth_labels,
            MC=20,
            nbins=10,
            maxDFFMR=0.3,
            lattice_size=50,
            option="mean_class",
            OUTDIR="analysis_mean_class",
            init_thresh=0.5,
        )

    print("Running analysis for predictive probability := average probability")
    if ternary:
        run_ternary_analysis(
            raw_model_preds,
            ground_truth_labels,
            MC=20,
            nbins=10,
            lattice_size=10,
            option="mean_prob",
            OUTDIR="analysis_mean_prob",
            max_clean_impurity=0.0,
            min_dirty_impurity=0.95,
        )
    else:
        run_analysis(
            raw_model_preds,
            ground_truth_labels,
            MC=20,
            nbins=10,
            maxDFFMR=0.3,
            lattice_size=50,
            option="mean_prob",
            OUTDIR="analysis_mean_prob",
        )
