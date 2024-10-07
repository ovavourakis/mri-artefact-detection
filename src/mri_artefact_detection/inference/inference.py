import os, keras, pandas as pd, numpy as np
import tensorflow as tf
from keras.metrics import AUC
from ..training.train_utils import DataLoader

def run_model_inference(savedir, weights, gt_data, mc_runs):
    # ENABLE GPU IF PRESENT
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU\n")
    else:
        print("Using CPU\n")

    os.makedirs(savedir, exist_ok=True)

    # read in test set
    df = pd.read_csv(gt_data, sep='\t')
    df['image'] = df['image'].apply(lambda x: x.replace('/content/drive/MyDrive/', '/Users/odysseasvavourakis/My Drive/'))
    testloader = DataLoader(df['image'].tolist() * mc_runs,
                            np.array(df['bin_gt'].tolist() * mc_runs),
                            train_mode=False,
                            batch_size=15,
                            image_shape=(192, 256, 256))
    test_images = [file for sublist in testloader.batches for file in sublist]
    y_true_test = [y for sublist in testloader.labels for y in sublist]

    # load model
    model = keras.models.load_model(weights)

    # predict on test set, and write out results
    y_pred = model.predict(testloader)
    df = pd.DataFrame({'image': test_images, 
                       'bin_gt': y_true_test, 
                       'y_pred': y_pred[:, 1]})
    df = df.groupby('image').agg({'bin_gt': 'first', 'y_pred': list})
    df[[f'y_pred{i}' for i in range(mc_runs)]] = pd.DataFrame(df['y_pred'].tolist(), index=df.index)
    df = df.drop(columns='y_pred')
    df.to_csv(savedir + 'raw_preds_test.tsv', sep='\t')

    # calculate AUROC, AP on each MC run individually
    aurocs, aps = [], []
    for i in range(mc_runs):
        aurocs.append(AUC(curve='ROC')(df['bin_gt'], df[f'y_pred{i}']))
        aps.append(AUC(curve='PR')(df['bin_gt'], df[f'y_pred{i}']))
    print('mean AUROC on test:', np.mean(aurocs))
    print('mean AP on test:', np.mean(aps))


