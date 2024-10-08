import os, math, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm
from typing import Tuple, Union


def load_predictions_and_ground_truth(
    MODEL_PREDS: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read out test set predictions and ground-truth labels from a tab-separated file
    with the following format::

        image                           bin_gt      pred_1    ...       pred_20
        sub-926536_acq-headmotion1_T1w  1           0.433381  ...  5.049924e-01
        sub-926536_acq-standard_T1w     0           0.003448  ...  6.611057e-02

    and split them into separate dataframes.

    :param MODEL_PREDS: Path to the tab-separated file.
    :type MODEL_PREDS: str
    :returns: A tuple containing the ground truth labels and model predictions as dataframes.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    model_preds = pd.read_csv(MODEL_PREDS, sep="\t")
    model_preds.rename(columns={"image": "id"}, inplace=True)
    model_preds.set_index("id", inplace=True)
    ground_truth_labels = model_preds[["bin_gt"]].copy()
    model_preds.drop(columns=["bin_gt"], inplace=True)
    return ground_truth_labels, model_preds


def assign_class(raw_prob: float, thresh: float = 0.5) -> int:
    """
    Binarises a value based on a threshold (0 if below, 1 if above).
    Used to convert raw model probabilities to predicted classes (clean/artefact).

    :param raw_prob: The raw probability value.
    :type raw_prob: float
    :param thresh: The threshold for binarization, defaults to 0.5.
    :type thresh: float, optional
    :returns: The binarized class (0 or 1).
    :rtype: int
    """
    return 1 if raw_prob >= thresh else 0


def compute_mean_class_and_uncertainty(
    raw_model_preds: pd.DataFrame, num_mc_runs: int = 20, thresh: float = 0.5
) -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate model predictions on each sample into a single predictive probability
    via OPTION (A): predictive probability = average predicted class.

    :param raw_model_preds: Dataframe of raw model predictions with format:
                            pred_1    ...       pred_20
                            image
                            sub-926536_acq-headmotion1_T1w  0.433381  ...  5.049924e-01
                            sub-926536_acq-standard_T1w     0.003448  ...  6.611057e-02
    :type raw_model_preds: pd.DataFrame
    :param num_mc_runs: Number of Monte Carlo runs (consider first k predictions per sample).
    :type num_mc_runs: int
    :param thresh: Threshold for class assignment.
    :type thresh: float
    :returns: The mean class for each sample and the standard deviation around that mean.
    :rtype: Tuple[pd.Series, pd.Series]
    """
    model_assigned_classes = raw_model_preds.iloc[:, :num_mc_runs].apply(
        lambda x: x.map(lambda y: assign_class(y, thresh=thresh))
    )
    return model_assigned_classes.mean(axis=1), model_assigned_classes.std(axis=1)


def compute_mean_prob_and_uncertainty(
    raw_model_preds: pd.DataFrame, num_mc_runs: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate model predictions on each sample into a single predictive probability
    via OPTION (B): predictive probability = average predictive probability.

    :param raw_model_preds: Dataframe of raw model predictions with format:
                            pred_1    ...       pred_20
                            image
                            sub-926536_acq-headmotion1_T1w  0.433381  ...  5.049924e-01
                            sub-926536_acq-standard_T1w     0.003448  ...  6.611057e-02
    :type raw_model_preds: pd.DataFrame
    :param num_mc_runs: Number of Monte Carlo runs (consider first k predictions per sample).
    :type num_mc_runs: int
    :returns: The mean probability for each sample and the standard deviation around that mean.
    :rtype: Tuple[pd.Series, pd.Series]
    """
    return raw_model_preds.iloc[:, :num_mc_runs].mean(axis=1), raw_model_preds.iloc[
        :, :num_mc_runs
    ].std(axis=1)


def plot_mean_class_histograms(
    mean: pd.Series, std: pd.Series, num_mc_runs: int, nbins: int, outdir: str
) -> None:
    """
    Plot histograms of the aggregated model predictions and the uncertainty in those aggregated predictions.

    :param mean: The aggregated model predictions for each sample (from compute_mean_{class/prob}_and_uncertainty).
    :type mean: pd.Series
    :param std: The standard deviation around that aggregated prediction (from compute_mean_{class/prob}_and_uncertainty).
    :type std: pd.Series
    :param num_mc_runs: The number of Monte Carlo runs (how many predictions per sample).
    :type num_mc_runs: int
    :param nbins: Number of bins for the histograms.
    :type nbins: int
    :param outdir: Output directory for the plots.
    :type outdir: str
    """
    # plot histogram of mean-class predictions
    plt.hist(mean, bins=nbins)
    plt.xlabel("Mean Class Prediction")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Mean Class Predictions for MC={num_mc_runs}")
    plt.savefig(outdir + f"/hist_mean_class_mc_{num_mc_runs}.png")
    plt.clf()

    # plot histogram of uncertainty in mean_class_predictions
    plt.hist(std, bins=nbins)
    plt.xlabel("Standard Deviation of Mean Class Prediction")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of Mean Class Prediction Uncertainties for MC={num_mc_runs}"
    )
    plt.savefig(outdir + f"/hist_std_mean_class_mc_{num_mc_runs}.png")
    plt.clf()


def plot_predictive_uncertainty_per_bin(
    mean: pd.Series, std: pd.Series, num_mc_runs: int, nbins: int, outdir: str
) -> None:
    """
    Sanity check: Plot the average standard deviation within different bins of the aggregated model predictions
    and the standard deviation around that average.

    We should (by construction) see a concave curve (high standard deviation where the aggregated model
    prediction is close to 0.5, low standard deviation at 0 or 1).

    :param mean: The aggregated model predictions for each sample (from compute_mean_{class/prob}_and_uncertainty).
    :type mean: pd.Series
    :param std: The standard deviation around that aggregated prediction (from compute_mean_{class/prob}_and_uncertainty).
    :type std: pd.Series
    :param num_mc_runs: The number of Monte Carlo runs (how many predictions per sample).
    :type num_mc_runs: int
    :param nbins: Number of bins for the histograms.
    :type nbins: int
    :param outdir: Output directory for the plots.
    :type outdir: str
    """
    # bin the mean-class predictions and calc. average std_dev per bin
    mean_bins = pd.cut(mean, bins=np.linspace(0, 1, nbins + 1), include_lowest=True)
    average_std_per_bin = (
        pd.concat([mean, mean_bins, std], axis=1)
        .groupby(1, observed=False)
        .agg({2: "mean"})
    )
    std_of_mean_std_per_bin = (
        pd.concat([mean, mean_bins, std], axis=1)
        .groupby(1, observed=False)
        .agg({2: "std"})
    )
    # plot the average std_dev against the bin identifiers
    x = np.linspace(0, 10, nbins) * 10
    plt.plot(x, average_std_per_bin)
    plt.fill_between(
        x,
        (average_std_per_bin - std_of_mean_std_per_bin).squeeze(),
        (average_std_per_bin + std_of_mean_std_per_bin).squeeze(),
        alpha=0.3,
    )
    plt.ylim(0, 0.55)
    plt.xlabel("Mean Class Prediction [Percentile]")
    plt.ylabel("Average Prediction StdDev in Percentile")
    plt.title(f"Average StdDev of Mean Class Prediction for MC={num_mc_runs}")
    plt.savefig(outdir + f"/line_std_per_bin_mc_{num_mc_runs}.png")
    plt.clf()


def plot_calibration_plot(
    mean_preds: pd.Series, gt: pd.DataFrame, num_mc_runs: int, nbins: int, outdir: str
) -> None:
    """
    Plot a calibration plot for the model predictions.

    The x-axis is the (aggregated) model prediction for each sample, binned into nbins.
    The y-axis is the average ground-truth positive class frequency in each bin.

    A well-calibrated model should closely follow the identity line (y=x), i.e.
    if the model verdict is that an image contains artefacts with probability p, then
    a fraction p of so-labelled images should indeed contain artefacts (no more, no fewer).

    :param mean_preds: The aggregated model predictions for each sample (from compute_mean_{class/prob}_and_uncertainty).
    :type mean_preds: pd.Series
    :param gt: The ground-truth labels for each sample.
    :type gt: pd.DataFrame
    :param num_mc_runs: The number of Monte Carlo runs (how many predictions per sample).
    :type num_mc_runs: int
    :param nbins: Number of bins for the histograms.
    :type nbins: int
    :param outdir: Output directory for the plots.
    :type outdir: str
    """
    mean_preds_bins = pd.cut(
        mean_preds, bins=np.linspace(0, 1, nbins + 1), include_lowest=True
    )
    preds_and_bins = pd.concat([mean_preds, mean_preds_bins], axis=1)
    merged = pd.merge(preds_and_bins, gt["bin_gt"], left_index=True, right_index=True)
    merged.columns = ["mean_pred", "bin_pred", "gt"]
    average_gt_per_bin = (
        merged.groupby("bin_pred", observed=False).agg({"gt": "mean"}) * 100
    )
    # plot the average ground_truth against the bin identifiers
    x = np.linspace(0, 10, nbins) * 10
    plt.plot(x, average_gt_per_bin)
    # plot the identity line
    plt.plot([0, 100], [0, 100], "k--")
    # add labels
    plt.xlabel("Mean Class Prediction [Percentile]")
    plt.ylabel("Ground-Truth Positive Class Frequency in Percentile")
    plt.title(f"Calibration Plot for MC={num_mc_runs}")
    plt.savefig(outdir + f"/calibration_plot_mc_{num_mc_runs}.png")
    plt.clf()


def merge_predictions_and_gt(
    mean: pd.Series, std: pd.Series, ground_truth_labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge aggregated model predictions and their uncertainties with ground-truth
    labels into a single dataframe. Prediction uncertainties are re-scaled to lie between 0 and 1.

    :param mean: The aggregated model predictions.
    :type mean: pd.Series
    :param std: The standard deviation around that mean.
    :type std: pd.Series
    :param ground_truth_labels: Dataframe of ground-truth labels with format:
                                bin_gt
                                image
                                sub-926536_acq-headmotion1_T1w  1
                                sub-926536_acq-standard_T1w     0
    :type ground_truth_labels: pd.DataFrame
    :returns: A dataframe with merged predictions and ground-truth labels.
    :rtype: pd.DataFrame
    """
    # re-scale std to lie between 0 and 1
    scaled_std = (std - np.min(std)) / (np.max(std) - np.min(std))
    # merge mean-class predictions and uncertainties with ground-truth labels (left join)
    mean_and_std = pd.DataFrame([mean, std, scaled_std]).transpose()
    mean_and_std.columns = ["mean_pred", "std_pred", "scaled_std_pred"]
    merged = pd.merge(
        mean_and_std, ground_truth_labels["bin_gt"], left_index=True, right_index=True
    )
    return merged


def calculate_DFFMR_AP_AUROC(
    merged: pd.DataFrame, eta: float
) -> Tuple[float, Union[float, None], Union[float, None]]:
    """
    Calculate binary classification metrics when classifying images based on
    the aggregated model predictions and their uncertainties.

    Images with high prediction uncertainty sigma > eta are flagged for manual review,
    the rest are classified as 'clean' or 'artefact' based on a single probability threshold.

    Computes:
        - DFFMR: Dataset Fraction Flagged For Manual Review
        - AP: Average Precision (AUPRC)
        - AUROC: Area Under the ROC Curve

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param eta: Uncertainty threshold for flagging images for manual review.
    :type eta: float
    :returns: A tuple containing DFFMR, AP, and AUROC.
    :rtype: Tuple[float, Union[float, None], Union[float, None]]
    """

    # fraction of Dataset Flagged For Manual Review (DFFMR)
    num_images = merged.shape[0]
    num_discarded = sum(merged["scaled_std_pred"] >= eta)
    DFFMR = num_discarded / num_images

    # average precision (AP) on the retained set
    if sum(merged["scaled_std_pred"] < eta) == 0:
        AP, AUC = np.nan, np.nan
    else:
        retained = merged[merged["scaled_std_pred"] < eta]
        AP = average_precision_score(retained["bin_gt"], retained["mean_pred"])
        # area under the roc curve (AUC) on the retained set
        if sum(retained["bin_gt"]) == 0 or sum(retained["bin_gt"]) == len(
            retained["bin_gt"]
        ):
            AUC = np.nan
        else:
            AUC = roc_auc_score(retained["bin_gt"], retained["mean_pred"])

    return DFFMR, AP, AUC


def plot_DFFMR_AP_AUROC(
    merged: pd.DataFrame, num_mc_runs: int, maxDFFMR: float, OUTDIR: str
) -> float:
    """
    Plot binary classification metrics when classifying images based on
    the aggregated model predictions and their uncertainties.

    Images with high prediction uncertainty sigma > eta are flagged for manual
    review. The rest are classified as 'clean' or 'artefact' based on a single
    probability threshold. We compute and plot metrics for a grid of eta values, find
    the eta that achieves a desired maximal DFFMR, and return it.

    Computes and plots (against eta):
        - DFFMR: Dataset fraction flagged for manual review
        - AP: Average precision (AUPRC)
        - AUROC: Area under the ROC curve

    Minimum eta is marked in red on plot.

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param num_mc_runs: Number of Monte Carlo runs (how many predictions per sample).
    :type num_mc_runs: int
    :param maxDFFMR: Maximum acceptable DFFMR.
    :type maxDFFMR: float
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    :returns: The minimum eta that achieves a DFFMR <= maxDFFMR.
    :rtype: float
    """
    etas = np.linspace(0, 1, 100)
    DFFMR_AP = pd.DataFrame([calculate_DFFMR_AP_AUROC(merged, eta) for eta in etas])
    DFFMR_AP.columns = ["DFFMR", "AP", "AUROC"]
    DFFMR_AP.index = etas
    DFFMR_AP.plot()

    plt.axhline(y=maxDFFMR, color="r", linestyle="--")  # horizontal line
    min_eta = DFFMR_AP[DFFMR_AP["DFFMR"] >= maxDFFMR].index[-1]
    plt.axvline(x=min_eta, color="r", linestyle="--")  # vertical line

    # Shade everything to the left of the vertical line in gray
    plt.fill_between(
        DFFMR_AP.index, 0, 1, where=DFFMR_AP.index <= min_eta, color="gray", alpha=0.3
    )

    plt.xlabel("Uncertainty Threshold eta")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1)
    plt.title("Pre-Screening Performance")
    plt.savefig(OUTDIR + f"/eta_preescreening_mc_{num_mc_runs}.png")
    plt.clf()

    return min_eta


def pointwise_metrics(
    retained: pd.DataFrame, theta: float
) -> Tuple[float, float, float, float, float]:
    """
    Calculate pointwise classification metrics for retained images.

    This function computes several classification metrics based on the
    predictions and ground-truth labels of retained images after applying
    an uncertainty threshold. The predictions are binarized using a given
    probability threshold, theta.

    :param retained: DataFrame containing the retained images' predictions
                     and ground-truth labels.
    :type retained: pd.DataFrame
    :param theta: Probability threshold for binarizing predictions.
    :type theta: float
    :returns: A tuple containing F1 score, precision, recall, accuracy, and specificity of the binarized predictions.
    :rtype: Tuple[float, float, float, float, float]
    """
    binarised_preds = retained["mean_pred"] > theta
    tn, fp, fn, tp = confusion_matrix(
        retained["bin_gt"], binarised_preds, labels=[0, 1]
    ).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn > 0) else np.nan
    specificity = tn / (tn + fp) if (tn + fp > 0) else np.nan
    F1 = f1_score(retained["bin_gt"], binarised_preds, zero_division=np.nan)
    precision = precision_score(
        retained["bin_gt"], binarised_preds, zero_division=np.nan
    )
    recall = recall_score(retained["bin_gt"], binarised_preds, zero_division=np.nan)
    return F1, precision, recall, accuracy, specificity


def calculate_gridpoint_metrics(
    merged: pd.DataFrame, eta: float, theta: float
) -> Tuple[
    Union[float, None],
    float,
    Union[float, None],
    float,
    float,
    float,
    float,
    float,
    Union[float, None],
]:
    """
    Calculate metrics for a gridpoint in the eta-theta space.

    Images with high prediction uncertainty sigma > eta are flagged for manual review,
    the rest are classified as 'clean' or 'artefact' based on a single probability
    threshold, theta, on the aggregated model predictions.

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param eta: Uncertainty threshold for flagging images for manual review.
    :type eta: float
    :param theta: Probability threshold for class assignment.
    :type theta: float
    :returns: A tuple containing combined score, DFFMR, UDM, F1, precision, recall, accuracy, specificity, and UDD.
    :rtype: Tuple[Union[float, None], float, Union[float, None], float, float, float, float, float, Union[float, None]]
    """
    # TODO: check if sensible and accurately implemented - amend as appropriate
    # TODO: re-define combined score

    # fraction of Dataset Flagged For Manual Review (DFFMR)
    num_images = merged.shape[0]
    num_discarded = sum(merged["scaled_std_pred"] >= eta)
    DFFMR = num_discarded / num_images

    if DFFMR == 1.0:
        return np.nan, DFFMR, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        # unusable dataset missed (UDM)
        retained = merged[merged["scaled_std_pred"] < eta]
        retained_labeled_clean = retained[retained["mean_pred"] < theta]
        FN = retained_labeled_clean["bin_gt"].sum()
        num_labelled_clean = retained_labeled_clean.shape[0]
        UDM = FN / num_labelled_clean if num_labelled_clean > 0 else np.nan

        # usable dataset discarded (UDD)
        all_discarded = merged[
            (merged["scaled_std_pred"] >= eta) | (merged["mean_pred"] >= theta)
        ]
        UDD = (all_discarded["bin_gt"] == 0).sum() / (merged["bin_gt"] == 0).sum()

        # F1, precision, recall for the retained set
        F1, precision, recall, accuracy, specificity = pointwise_metrics(
            retained, theta
        )

        scores = [DFFMR, UDM, 1 - UDD, 1 - F1, 1 - specificity]
        combined_score = sum(scores)

        return (
            combined_score,
            DFFMR,
            UDM,
            F1,
            precision,
            recall,
            accuracy,
            specificity,
            UDD,
        )


def gridpoint_metrics_tensor(
    merged: pd.DataFrame, min_eta: float, lattice_size: int = 50
) -> pd.DataFrame:
    """
    Calculate metrics for a lattice of gridpoints in the eta-theta space.

    Images with high prediction uncertainty sigma > eta are flagged for manual review,
    the rest are classified as 'clean' or 'artefact' based on a single probability
    threshold, theta, on the aggregated model predictions.

    This function computes the following metrics for a lattice of gridpoints in
    the eta-theta space:
        - DFFMR: Dataset fraction flagged for manual review
        - UDM: Unusable dataset missed (FN)
        - UDD: Usable dataset discarded (FP)
        - F1, precision, recall, accuracy, specificity
        - A combined score that trades off these various metrics
    Prints the optimal gridpoint that minimises the combined score.

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param min_eta: Minimum uncertainty threshold that ensures a desired maximal DFFMR (see plot_DFFMR_AP_AUROC).
    :type min_eta: float
    :param lattice_size: Number of gridpoints for theta (no for eta is proportional).
    :type lattice_size: int
    :returns: A dataframe of metrics for each gridpoint in the eta-theta space.
    :rtype: pd.DataFrame
    """
    etas = np.linspace(min_eta, 1, math.ceil(lattice_size * (1 - min_eta)))
    thetas = np.linspace(0, 1, lattice_size)
    print("Running over eta grid:")
    gridpoint_metrics = pd.DataFrame(
        [
            calculate_gridpoint_metrics(merged, eta, theta)
            for eta in tqdm(etas)
            for theta in thetas
        ]
    )
    gridpoint_metrics.columns = [
        "combined",
        "DFFMR",
        "UDM",
        "F1",
        "precision",
        "recall",
        "accuracy",
        "specificity",
        "UDD",
    ]
    gridpoint_metrics.index = pd.MultiIndex.from_product(
        [etas, thetas], names=["eta", "theta"]
    )

    print("Optimal parameters based on argimin of combined score:")
    print(gridpoint_metrics.iloc[[gridpoint_metrics["combined"].argmin()]])

    return gridpoint_metrics


def plot_gridpoint_metrics(
    gpm_tensor: pd.DataFrame, maxDFFMR: float, num_mc_runs: int, OUTDIR: str
) -> None:
    """
    Plots pre-computed metrics for each gridpoint in the eta-theta space.

    Images with high prediction uncertainty sigma > eta are flagged for manual review,
    the rest are classified as 'clean' or 'artefact' based on a single probability
    threshold, theta, on the aggregated model predictions.

    Plots:
        - A heatmap of each metric in the eta-theta space in areas where DFFMR < maxDFFMR

    :param gpm_tensor: Dataframe of metrics for each gridpoint in the eta-theta space (from gridpoint_metrics_tensor).
    :type gpm_tensor: pd.DataFrame
    :param maxDFFMR: Maximum acceptable DFFMR.
    :type maxDFFMR: float
    :param num_mc_runs: Number of Monte Carlo runs (how many predictions per sample).
    :type num_mc_runs: int
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    """
    # os.makedirs(OUTDIR+'/grids', exist_ok=True)
    gridpoint_metrics = gpm_tensor[gpm_tensor["DFFMR"] < maxDFFMR].reset_index()

    for metric in gridpoint_metrics.columns[2:]:
        gpm = gridpoint_metrics.pivot(index="eta", columns="theta", values=metric)
        gpm = gpm.sort_index(ascending=False, axis=1).sort_index(ascending=True, axis=0)
        plt.figure(figsize=(10, 10))
        # round the axis ticks
        yticks = [f"{y:.2f}" for y in gpm.index]
        xticks = [f"{x:.2f}" for x in gpm.columns]
        # plot a 2D heatmap
        sns.heatmap(
            gpm,
            cmap="coolwarm",
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            xticklabels=xticks,
            yticklabels=yticks,
        )
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        # add labels
        plt.xlabel("Decision Threshold theta")
        plt.ylabel("Uncertainty Threshold eta")
        plt.title(f"{metric} for MC={num_mc_runs}")
        plt.savefig(OUTDIR + f"gridpoint_{metric}_mc_{num_mc_runs}.png")
        plt.close()


def one_stage_screening(merged: pd.DataFrame, OUTDIR: str = "analysis_A_out") -> None:
    """
    Plot binary classification metrics when classifying images based only on
    the aggregated model predictions (ignoring the associated uncertainties; i.e. eta=1).

    Images are classified as either 'clean' or 'artefact' based on a
    single probability threshold.

    Computes and plots:
        - DFFMR: Dataset fraction flagged for manual review (should be 0 always)
        - AP: Average precision (AUPRC)
        - AUROC: Area under the ROC curve
        - UDM: Unusable dataset missed (FN)
        - UDD: Usable dataset discarded (FP)
        - F1, accuracy, specificity
        - A combined score that trades off these various metrics

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    """

    os.makedirs(OUTDIR, exist_ok=True)
    # single-number stats
    DFFMR, AP, AUC = calculate_DFFMR_AP_AUROC(
        merged, eta=1.1
    )  # don't exclude any images
    # threshold-dependent stats
    thetas = np.linspace(0, 1, 100)
    pwm = pd.DataFrame(
        [calculate_gridpoint_metrics(merged, 1.1, theta) for theta in thetas]
    )
    pwm.columns = [
        "combined",
        "DFFMR",
        "UDM (FN)",
        "F1",
        "precision",
        "recall",
        "accuracy",
        "specificity",
        "UDD (FP)",
    ]
    pwm["theta"] = thetas

    # plot the threshold-dependent stats over theta
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # plot 'combined' score in the first subplot
    ax1.plot(pwm["theta"], pwm["combined"])
    ax1.set_ylabel("Combined Score")
    # add single-number stats to the first subplot
    ax1.text(
        0.05,
        0.85,
        f"DFFMR: {DFFMR:.2f}\nAP: {AP:.2f}\nAUROC: {AUC:.2f}",
        transform=ax1.transAxes,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )
    # plot other metrics in the second subplot
    ax2.plot(pwm["theta"], pwm["UDM (FN)"], label="UDM (FN)")
    ax2.plot(pwm["theta"], pwm["UDD (FP)"], label="UDD (FP)")
    ax2.plot(pwm["theta"], pwm["F1"], label="F1")
    ax2.plot(pwm["theta"], pwm["accuracy"], label="Accuracy")
    ax2.plot(pwm["theta"], pwm["specificity"], label="Specificity")
    ax2.set_xlabel("Probability Threshold theta")
    ax2.set_ylabel("Metric Value")
    ax2.legend()

    plt.suptitle("Metrics for One-Stage Screening (All Data Included)")
    plt.tight_layout()
    plt.savefig(OUTDIR + f"/one_stage_screening.png")
    plt.clf()


def two_stage_screening(
    merged: pd.DataFrame,
    MC: int = 20,
    maxDFFMR: float = 0.3,
    lattice_size: int = 50,
    OUTDIR: str = "analysis_A_out",
) -> None:
    """
    Plot binary classification metrics when classifying images based on the aggregated model
    predictions and their uncertainties.

    Images with high prediction uncertainty sigma > eta are flagged for manual review,
    the rest are classified as 'clean' or 'artefact' based on a single probability threshold, theta.

    For a grid of thresholds in the eta-theta space, computes and plots:
        - DFFMR: Dataset fraction flagged for manual review
        - UDM: Unusable dataset missed (FN)
        - UDD: Usable dataset discarded (FP)
        - F1, precision, recall, accuracy, specificity
        - A combined score that trades off these various metrics

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param MC: Number of Monte Carlo runs (how many predictions per sample).
    :type MC: int
    :param maxDFFMR: Maximum acceptable DFFMR.
    :type maxDFFMR: float
    :param lattice_size: Number of gridpoints for theta (no for eta is proportional).
    :type lattice_size: int
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    """

    os.makedirs(OUTDIR, exist_ok=True)
    # eta-dependent stats
    min_eta = plot_DFFMR_AP_AUROC(
        merged, MC, maxDFFMR, OUTDIR
    )  # min_eta to ensure DFFMR <= maxDFFMR
    # eta-and-theta-dependent stats
    gpm_tensor = gridpoint_metrics_tensor(merged, min_eta, lattice_size=lattice_size)
    plot_gridpoint_metrics(gpm_tensor, maxDFFMR, MC, OUTDIR + "/two_stage_screening")


def run_synoptic_analysis(
    raw_model_preds: pd.DataFrame,
    ground_truth_labels: pd.DataFrame,
    MC: int = 20,
    nbins: int = 10,
    option: str = "mean_class",
    OUTDIR: str = "analysis_A_out",
    init_thresh: float = 0.5,
) -> pd.DataFrame:
    """
    Basic analysis pipeline for model predictions on a test set with known labels.
    Aggregates model predictions into a single estimate mu and its uncertainty sigma, per sample.
    Plots histograms of mu and sigma, a sanity check of uncertainty per bin of mu, and a model calibration plot,
    then merges the predictions with ground-truth labels for further analysis.

    :param raw_model_preds: Dataframe of raw model predictions with format:
                            pred_1    ...       pred_20
                            image
                            sub-926536_acq-headmotion1_T1w  0.433381  ...  5.049924e-01
                            sub-926536_acq-standard_T1w     0.003448  ...  6.611057e-02
    :type raw_model_preds: pd.DataFrame
    :param ground_truth_labels: Dataframe of ground-truth labels with format:
                                bin_gt
                                image
                                sub-926536_acq-headmotion1_T1w  1
                                sub-926536_acq-standard_T1w     0
    :type ground_truth_labels: pd.DataFrame
    :param MC: Number of Monte Carlo runs (how many predictions per sample).
    :type MC: int
    :param nbins: Number of bins for the histograms.
    :type nbins: int
    :param option: How to aggregate model predictions (mean_class or mean_prob).
    :type option: str
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    :param init_thresh: Initial threshold for class assignment.
    :type init_thresh: float
    :returns: A dataframe with merged predictions and ground-truth labels.
    :rtype: pd.DataFrame
    """
    # setup
    os.makedirs(OUTDIR, exist_ok=True)
    # analysis
    if option == "mean_class":  # note: the init_thresh kwarg only affects option (A)
        mean, std = compute_mean_class_and_uncertainty(
            raw_model_preds, MC, thresh=init_thresh
        )
    elif option == "mean_prob":
        mean, std = compute_mean_prob_and_uncertainty(raw_model_preds, MC)
    else:
        raise ValueError("analysis option must be either A or B")

    plot_mean_class_histograms(mean, std, MC, nbins, OUTDIR)
    plot_predictive_uncertainty_per_bin(mean, std, MC, nbins, OUTDIR)
    plot_calibration_plot(mean, ground_truth_labels, MC, nbins, OUTDIR)

    merged = merge_predictions_and_gt(mean, std, ground_truth_labels)

    return merged


def run_analysis(
    raw_model_preds: pd.DataFrame,
    ground_truth_labels: pd.DataFrame,
    MC: int = 20,
    nbins: int = 10,
    maxDFFMR: float = 0.3,
    lattice_size: int = 50,
    option: str = "mean_class",
    OUTDIR: str = "analysis_A_out",
    init_thresh: float = 0.5,
) -> None:
    """
    Full binary analysis pipeline for model predictions on a test set with known labels.

    Aggregates model predictions into a single estimate mu and its uncertainty sigma, per sample.
    Then computes metrics and plots results for one-stage screening (probability threshold only)
    and two-stage screening (uncertainty and probability thresholds).

    In both cases the classification of images is binary, into 'clean' or 'artefact', but in the
    second case images with high prediction uncertainty sigma > eta are flagged for manual review
    and removed from the pool before classification.

    :param raw_model_preds: Dataframe of raw model predictions.
    :type raw_model_preds: pd.DataFrame
    :param ground_truth_labels: Dataframe of ground-truth labels.
    :type ground_truth_labels: pd.DataFrame
    :param MC: Number of Monte Carlo runs.
    :type MC: int
    :param nbins: Number of bins for the histograms.
    :type nbins: int
    :param maxDFFMR: Maximum acceptable DFFMR.
    :type maxDFFMR: float
    :param lattice_size: Number of gridpoints for theta.
    :type lattice_size: int
    :param option: How to aggregate model predictions.
    :type option: str
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    :param init_thresh: Initial threshold for class assignment.
    :type init_thresh: float
    """
    merged = run_synoptic_analysis(
        raw_model_preds, ground_truth_labels, MC, nbins, option, OUTDIR, init_thresh
    )
    # one-stage screening (probability threshold only)
    one_stage_screening(merged, OUTDIR=OUTDIR + "/one_stage_screening")
    # two-stage screening (uncertainty and probability thresholds)
    two_stage_screening(
        merged, MC, maxDFFMR, lattice_size, OUTDIR=OUTDIR + "/two_stage_screening"
    )


def split_by_uncertainty(
    merged: pd.DataFrame, eta: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the predictions dataframe into two subsets based on the uncertainty threshold eta.

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param eta: Uncertainty threshold for flagging images for manual review.
    :type eta: float
    :returns: A tuple containing the subset of merged with uncertainty <= eta and the subset with uncertainty > eta.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    unc_pass = merged[merged["scaled_std_pred"] <= eta]
    unc_fail = merged[merged["scaled_std_pred"] > eta]
    return unc_pass, unc_fail


def split_by_prob(
    unc_pass: pd.DataFrame, lower: float, upper: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the predictions dataframe into three subsets based on the probability thresholds lower and upper.

    :param unc_pass: Set of predictions where model is confident in its prediction of probability
                     (from split_by_uncertainty).
    :type unc_pass: pd.DataFrame
    :param lower: Lower probability threshold for class assignment.
    :type lower: float
    :param upper: Upper probability threshold for class assignment.
    :type upper: float
    :returns: A tuple containing the subset of unc_pass with mean_pred < lower, the subset with lower <= mean_pred <= upper, and the subset with mean_pred > upper.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    def_clean = unc_pass[unc_pass["mean_pred"] < lower]
    def_dirty = unc_pass[unc_pass["mean_pred"] > upper]
    prob_fail = unc_pass[
        (unc_pass["mean_pred"] >= lower) & (unc_pass["mean_pred"] <= upper)
    ]
    return def_clean, prob_fail, def_dirty


def ternary_split(
    merged: pd.DataFrame, eta: float, theta: float, tau: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the predictions dataframe into three subsets based on the uncertainty and probability thresholds.

    Images with high prediction uncertainty sigma > eta are flagged for manual review,
    the rest are classified into 'clean', 'artefact', or 'for_review' based on two probability thresholds.

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param eta: Uncertainty threshold for flagging images for manual review.
    :type eta: float
    :param theta: Lower probability threshold for clean class assignment.
    :type theta: float
    :param tau: Upper probability threshold for artefact class assignment.
    :type tau: float
    :returns: A tuple containing the predictions classified as clean, artefacts, flagged for manual review due to model uncertainty, flagged for manual review due to intermediate probability assignment, and the total set flagged for manual review.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    # split into 3 subsets: clean, dirty, for_review
    # also return for_review split by review reason
    unc_pass, unc_fail = split_by_uncertainty(merged, eta)
    def_clean, prob_fail, def_dirty = split_by_prob(unc_pass, theta, tau)
    for_review = pd.concat([unc_fail, prob_fail])

    return def_clean, def_dirty, for_review, unc_fail, prob_fail


def ternary_metrics(
    def_clean: pd.DataFrame,
    def_dirty: pd.DataFrame,
    for_review: pd.DataFrame,
    unc_fail: pd.DataFrame,
) -> Tuple[int, float, int, float, float, int, float]:
    """
    Calculate performance metrics for a pre-computed ternary split of a test set into clean,
    artefact, and for_review subsets.

    Computes:
        - size_clean: Number of labelled-clean images
        - impurity_clean: Fraction of labelled-clean images that are artefacts (FN rate)
        - size_dirty: Number of labelled-artefact images
        - impurity_dirty: Fraction of labelled-artefact images that are clean (FP rate)
        - wrkld_reduction: Fraction of images that are not flagged for manual review
        - size_for_rev: Number of images flagged for manual review
        - frac_dffmred_because_uncertain: Fraction of images flagged for manual review due to model uncertainty

    :param def_clean: Subset of merged with mean_pred < theta (model is confident in its prediction of 'clean').
    :type def_clean: pd.DataFrame
    :param def_dirty: Subset of merged with mean_pred > tau (model is confident in its prediction of 'dirty').
    :type def_dirty: pd.DataFrame
    :param for_review: Subset of merged with theta <= mean_pred <= tau (model-predicted probability is intermediate (but certain)).
    :type for_review: pd.DataFrame
    :param unc_fail: Subset of merged with uncertainty > eta (model is uncertain).
    :type unc_fail: pd.DataFrame
    :returns: A tuple containing size_clean, impurity_clean, size_dirty, impurity_dirty, wrkld_reduction, size_for_rev, and frac_dffmred_because_uncertain.
    :rtype: Tuple[int, float, int, float, float, int, float]
    """
    size_clean, size_dirty, size_for_rev = (
        def_clean.shape[0],
        def_dirty.shape[0],
        for_review.shape[0],
    )
    impurity_clean = def_clean["bin_gt"].mean() if def_clean.shape[0] > 0 else 0
    impurity_dirty = def_dirty["bin_gt"].mean() if def_dirty.shape[0] > 0 else 0

    dffmr = for_review.shape[0] / sum(
        [x.shape[0] for x in [def_clean, def_dirty, for_review]]
    )
    frac_dffmred_because_uncertain = (
        unc_fail.shape[0] / for_review.shape[0] if for_review.shape[0] > 0 else np.nan
    )
    return (
        size_clean,
        impurity_clean,
        size_dirty,
        impurity_dirty,
        1 - dffmr,
        size_for_rev,
        frac_dffmred_because_uncertain,
    )


def ternary_gridpoint_metrics(
    merged: pd.DataFrame, lattice_size: int = 50
) -> pd.DataFrame:
    """
    Calculate performance metrics for a grid of eta, theta, tau values.

    At each grid-point, split testset into clean, artefact, and for_review subsets
    using thresholds eta, theta, tau and the aggregated model prediction.
    Then calculates for that choice of thresholds:
        - size_clean: Number of labelled-clean images
        - impurity_clean: Fraction of labelled-clean images that are artefacts (FN rate)
        - size_dirty: Number of labelled-artefact images
        - impurity_dirty: Fraction of labelled-artefact images that are clean (FP rate)
        - wrkld_reduction: Fraction of images that are not flagged for manual review
        - size_for_rev: Number of images flagged for manual review
        - frac_dffmred_because_uncertain: Fraction of images flagged for manual review due to model uncertainty

    :param merged: Dataframe of merged model predictions and ground-truth labels (from merge_predictions_and_gt).
    :type merged: pd.DataFrame
    :param lattice_size: Number of gridpoints for eta, theta, tau.
    :type lattice_size: int
    :returns: A dataframe of metrics for each gridpoint in the eta-theta-tau space.
    :rtype: pd.DataFrame
    """
    etas = np.linspace(0, 1, lattice_size)
    thetas = np.linspace(0, 1, lattice_size)
    taus_full = np.linspace(0, 1, lattice_size)

    gridpoint_ternary_metrics = pd.DataFrame(
        [
            ternary_metrics(*ternary_split(merged, eta, theta, tau)[:-1])
            for eta in tqdm(etas)
            for theta in thetas
            for tau in taus_full[taus_full >= theta]
        ]
    )
    gridpoint_ternary_metrics.columns = [
        "size_clean",
        "impurity_clean",
        "size_dirty",
        "impurity_dirty",
        "wrkld_reduction",
        "size_for_rev",
        "frac_dffmred_because_uncertain",
    ]
    gridpoint_ternary_metrics.index = pd.MultiIndex.from_tuples(
        [
            (eta, theta, tau)
            for eta in etas
            for theta in thetas
            for tau in taus_full[taus_full >= theta]
        ],
        names=["eta", "theta", "tau"],
    )
    return gridpoint_ternary_metrics


def sort_and_filter_tensor(
    metrics_tensor: pd.DataFrame,
    raw_data: pd.DataFrame,
    max_clean_impurity: float = 0.0,
    min_dirty_impurity: float = 0.95,
    min_wrkld_red: float = 0.5,
    OUTDIR: str = "analysis_A_out",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sanity-checks constraints against input data statistics (output must be better than input).
    max_clean_impurity i.e. max acceptable FN (missed artefacts)
    min_dirty_impurity i.e. max acceptable FP (needlessly rejected scans)

    Sorts the gridpoints by achieved impurity_clean, wrkld_reduction, impurity_dirty.
    Filters out gridpoints that give no workload improvement or do satisfy constraints:

    Writes the full_tensor and filtered_tensor to file.

    :param metrics_tensor: Dataframe of metrics for each gridpoint.
    :type metrics_tensor: pd.DataFrame
    :param raw_data: Dataframe of raw data.
    :type raw_data: pd.DataFrame
    :param max_clean_impurity: Maximum acceptable impurity for clean images.
    :type max_clean_impurity: float
    :param min_dirty_impurity: Minimum acceptable impurity for dirty images.
    :type min_dirty_impurity: float
    :param min_wrkld_red: Minimum acceptable workload reduction.
    :type min_wrkld_red: float
    :param OUTDIR: Output directory for the files.
    :type OUTDIR: str
    :returns: A tuple containing the full tensor and the filtered tensor.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    input_impurity = raw_data["bin_gt"].mean()
    assert (
        max_clean_impurity < input_impurity
    )  # otherwise no point in running the model
    assert (
        min_dirty_impurity > input_impurity
    )  # otherwise no point in running the model

    # sort tensor in order of objective priorities
    metrics_tensor.sort_values(
        ["impurity_clean", "wrkld_reduction", "impurity_dirty"],
        ascending=[True, False, False],
        inplace=True,
    )
    # filter out gridpoints that give no workload improvement
    metrics_tensor = metrics_tensor[metrics_tensor["wrkld_reduction"] > 0]
    # write to file
    with open(OUTDIR + "/full_ternary_grid.txt", "w") as file:
        file.write(metrics_tensor.to_string(index=True, float_format="{:.3f}".format))
    # filter out gridpoints that violate constraints
    within_constraints = metrics_tensor[
        (metrics_tensor["impurity_clean"] <= max_clean_impurity)
        & (metrics_tensor["impurity_dirty"] >= min_dirty_impurity)
        & (metrics_tensor["wrkld_reduction"] >= min_wrkld_red)
    ].copy()
    # sort by highest workload reduction within constraints
    within_constraints.sort_values(["wrkld_reduction"], ascending=[False], inplace=True)
    # write to file
    with open(OUTDIR + "/filtered_ternary_grid.txt", "w") as file:
        file.write(
            within_constraints.to_string(index=True, float_format="{:.3f}".format)
        )

    return metrics_tensor, within_constraints


def run_ternary_analysis(
    raw_model_preds: pd.DataFrame,
    ground_truth_labels: pd.DataFrame,
    MC: int = 20,
    nbins: int = 10,
    lattice_size: int = 50,
    option: str = "mean_class",
    OUTDIR: str = "analysis_A_out",
    init_thresh: float = 0.5,
    max_clean_impurity: float = 0.0,
    min_dirty_impurity: float = 0.95,
    min_wrkld_red: float = 0.5,
) -> None:
    """
    Run a ternary analysis on the model predictions and ground-truth labels.
    Compare `eval_theory.pdf` for more details.

    :param raw_model_preds: Dataframe of raw model predictions.
    :type raw_model_preds: pd.DataFrame
    :param ground_truth_labels: Dataframe of ground-truth labels.
    :type ground_truth_labels: pd.DataFrame
    :param MC: Number of Monte Carlo runs.
    :type MC: int
    :param nbins: Number of bins for the histograms.
    :type nbins: int
    :param lattice_size: Number of gridpoints for eta, theta, tau.
    :type lattice_size: int
    :param option: How to aggregate model predictions.
    :type option: str
    :param OUTDIR: Output directory for the plots.
    :type OUTDIR: str
    :param init_thresh: Initial threshold for class assignment.
    :type init_thresh: float
    :param max_clean_impurity: Maximum acceptable impurity for clean images.
    :type max_clean_impurity: float
    :param min_dirty_impurity: Minimum acceptable impurity for dirty images.
    :type min_dirty_impurity: float
    :param min_wrkld_red: Minimum acceptable workload reduction.
    :type min_wrkld_red: float
    """
    # collate model preds, do basic plots
    merged = run_synoptic_analysis(
        raw_model_preds, ground_truth_labels, MC, nbins, option, OUTDIR, init_thresh
    )
    # get metrics for each possible split of predictions by eta, theta, tau
    metrics_tensor = ternary_gridpoint_metrics(merged, lattice_size)
    # sort and filter tensor
    full_tensor, within_constraints = sort_and_filter_tensor(
        metrics_tensor,
        merged,
        max_clean_impurity,
        min_dirty_impurity,
        min_wrkld_red,
        OUTDIR,
    )
    print(within_constraints.head(1))
