import os, numpy as np, pandas as pd
from .train_utils import *
from .model import *

import tensorflow as tf
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def train_model(
    savedir: str,
    datadir: str,
    datasets: list = ("artefacts" + str(i) for i in [1, 2, 3]),
    contrasts: list = ("T1wMPR"),  # 'T1wTIR', 'T2w', 'T2starw', 'FLAIR'
    quals: list = ("clean", "exp_artefacts"),
    random_affine: float = 1 / 12,
    random_elastic_deformation: float = 1 / 12,
    random_anisotropy: float = 1 / 12,
    rescale_intensity: float = 1 / 12,
    random_motion: float = 1 / 12,
    random_ghosting: float = 1 / 12,
    random_spike: float = 1 / 12,
    random_bias_field: float = 1 / 12,
    random_blur: float = 1 / 12,
    random_noise: float = 1 / 12,
    random_swap: float = 1 / 12,
    random_gamma: float = 1 / 12,
    target_clean_ratio: float = 0.5,  # re-sample training set to this fraction of clean images
    mc_runs: int = 20,  # number of Monte Carlo runs on test set
) -> None:
    """
    Trains a convolutional neural network model for MRI artefact detection.

    :param savedir: Directory where training outputs and checkpoints will be saved.
    :type savedir: str
    :param datadir: Directory containing the dataset.
    :type datadir: str
    :param datasets: List of dataset names to be used for training.
    :type datasets: list
    :param contrasts: List of MRI contrasts to be considered.
    :type contrasts: list
    :param quals: List of quality labels (e.g., 'clean', 'exp_artefacts').
    :type quals: list
    :param random_affine: Distribution weight for RandomAffine artefact.
    :type random_affine: float
    :param random_elastic_deformation: Distribution weight for RandomElasticDeformation artefact.
    :type random_elastic_deformation: float
    :param random_anisotropy: Distribution weight for RandomAnisotropy artefact.
    :type random_anisotropy: float
    :param rescale_intensity: Distribution weight for RescaleIntensity artefact.
    :type rescale_intensity: float
    :param random_motion: Distribution weight for RandomMotion artefact.
    :type random_motion: float
    :param random_ghosting: Distribution weight for RandomGhosting artefact.
    :type random_ghosting: float
    :param random_spike: Distribution weight for RandomSpike artefact.
    :type random_spike: float
    :param random_bias_field: Distribution weight for RandomBiasField artefact.
    :type random_bias_field: float
    :param random_blur: Distribution weight for RandomBlur artefact.
    :type random_blur: float
    :param random_noise: Distribution weight for RandomNoise artefact.
    :type random_noise: float
    :param random_swap: Distribution weight for RandomSwap artefact.
    :type random_swap: float
    :param random_gamma: Distribution weight for RandomGamma artefact.
    :type random_gamma: float
    :param target_clean_ratio: Fraction of clean images to be resampled in the training set.
    :type target_clean_ratio: float
    :param mc_runs: Number of Monte Carlo runs on the test set.
    :type mc_runs: int
    :return: None
    """
    # ENABLE GPU IF PRESENT
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU\n")
    else:
        print("Using CPU\n")

    os.makedirs(savedir, exist_ok=True)

    # get the paths and labels of the real images
    real_image_paths, pids, real_labels = DataCrawler(
        datadir, datasets, contrasts, quals
    ).crawl()

    # split images by patient id
    Xtrain, Xval, Xtest, ytrain, yval, ytest = split_by_patient(
        real_image_paths, pids, real_labels
    )

    for string, y in zip(["train", "val", "test"], [ytrain, yval, ytest]):
        print("number in " + string + " set:", len(y))
        print(string + " class distribution: ", sum(y) / len(y), " percent artefact")

    # instantiate DataLoaders
    artefact_distro = {
        "RandomAffine": random_affine,
        "RandomElasticDeformation": random_elastic_deformation,
        "RandomAnisotropy": random_anisotropy,
        "RescaleIntensity": rescale_intensity,
        "RandomMotion": random_motion,
        "RandomGhosting": random_ghosting,
        "RandomSpike": random_spike,
        "RandomBiasField": random_bias_field,
        "RandomBlur": random_blur,
        "RandomNoise": random_noise,
        "RandomSwap": random_swap,
        "RandomGamma": random_gamma,
    }

    trainloader = DataLoader(
        Xtrain,
        ytrain,
        train_mode=True,
        image_shape=(192, 256, 256),
        batch_size=10,
        target_clean_ratio=target_clean_ratio,
        artef_distro=artefact_distro,
    )
    valloader = DataLoader(
        Xval, yval, train_mode=False, batch_size=15, image_shape=(192, 256, 256)
    )
    testloader = DataLoader(
        Xtest * mc_runs,
        np.array(ytest.tolist() * mc_runs),
        train_mode=False,
        batch_size=15,
        image_shape=(192, 256, 256),
    )

    # write out ground truth for test set
    test_images = [file for sublist in testloader.batches for file in sublist]
    y_true_test = [y for sublist in testloader.labels for y in sublist]
    out = (
        pd.DataFrame({"image": test_images, "bin_gt": y_true_test})
        .groupby("image")
        .agg({"bin_gt": "first"})
    )
    out.to_csv(os.path.join(savedir, "test_split_gt.tsv"), sep="\t")

    # compile model
    model = getConvNet(out_classes=2, input_shape=(192, 256, 256, 1))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="nadam",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc"),
        ],
    )
    print("\n")
    print(model.summary())

    # prepare for training
    checkpoint_dir = os.path.join(savedir, "ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # define callbacks
    callbacks = [
        # keep an eye on model weights norm after every epoch
        PrintModelWeightsNorm(),
        # save model after each epoch
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "end_of_epoch_{epoch}.keras")
        ),
        # reduce learning rate if val_loss doesn't improve for 2 epochs
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            mode="auto",
            min_delta=1e-2,
            cooldown=0,
            min_lr=0.0001,
        ),
        # stop training if val_loss doesn't improve for 5 epochs
        EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=5, verbose=1),
    ]

    # train model
    history = model.fit(
        trainloader,
        validation_data=valloader,
        steps_per_epoch=130,  # batches per epoch
        epochs=24,  # number of epochs
        callbacks=callbacks,
    )

    # save metrics
    pd.DataFrame(
        {
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "train_accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "train_auc": history.history["auroc"],
            "val_auc": history.history["val_auroc"],
            "train_ap": history.history["auprc"],
            "val_ap": history.history["val_auprc"],
        }
    ).to_csv(os.path.join(savedir, "training_metrics.tsv"), sep="\t")
    plot_train_metrics(history, os.path.join(savedir, "train_metrics_plot.png"))

    # evaluate model (multiple MC runs per test image)
    print("#" * 30)
    print("Evaluate on Test Set")
    print("#" * 30)
    # predict on test set, and write out results
    y_pred = model.predict(testloader)  # , use_multiprocessing=True)
    df = pd.DataFrame(
        {"image": test_images, "bin_gt": y_true_test, "y_pred": y_pred[:, 1]}
    )
    df = df.groupby("image").agg({"bin_gt": "first", "y_pred": list})
    df[[f"y_pred{i}" for i in range(mc_runs)]] = pd.DataFrame(
        df["y_pred"].tolist(), index=df.index
    )
    df = df.drop(columns="y_pred")
    df.to_csv(os.path.join(savedir, "raw_preds_test.tsv"), sep="\t")

    # calculate AUROC, AP on each MC run individually
    aurocs, aps = [], []
    for i in range(mc_runs):
        aurocs.append(AUC(curve="ROC")(df["bin_gt"], df[f"y_pred{i}"]))
        aps.append(AUC(curve="PR")(df["bin_gt"], df[f"y_pred{i}"]))
    print("mean AUROC on test:", np.mean(aurocs))
    print("mean AP on test:", np.mean(aps))
