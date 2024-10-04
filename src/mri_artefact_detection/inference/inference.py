"""
Load a pre-trained model and perform inference.

run with `python -m inference.inference` from repository root
"""

import torch, os, sys, keras, pandas as pd, numpy as np
import tensorflow as tf
from keras.metrics import AUC

from training.train_utils import DataLoader

TRAINRUN_DIR = '/vols/opig/users/vavourakis/ge_project/trainrun/'
SAVEDIR = TRAINRUN_DIR + 'inference/'
WEIGHTS = TRAINRUN_DIR + 'ckpts/end_of_epoch_3.keras'
GT_DATA = TRAINRUN_DIR + 'test_split_gt.tsv'
MC_RUNS = 20  # number of Monte Carlo runs on test set

# ENABLE GPU IF PRESENT
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU\n")
else:
    print("Using CPU\n")

if __name__ == '__main__':

    os.makedirs(SAVEDIR, exist_ok=True)

    # read in test set
    df = pd.read_csv(GT_DATA, sep='\t')
    df['image']=df['image'].apply(lambda x: x.replace('/content/drive/MyDrive/', '/Users/odysseasvavourakis/My Drive/'))
    testloader = DataLoader(df['image'].tolist()*MC_RUNS,
                            np.array(df['bin_gt'].tolist()*20),
                            train_mode=False,
                            batch_size=15,
                            image_shape = (192,256,256))
    test_images = [file for sublist in testloader.batches for file in sublist]
    y_true_test = [y for sublist in testloader.labels for y in sublist]

    # load model
    model = keras.models.load_model(WEIGHTS)

    # predict on test set, and write out results
    y_pred = model.predict(testloader)
    df = pd.DataFrame({'image': test_images, 
                       'bin_gt': y_true_test, 
                       'y_pred': y_pred[:,1]})
    df = df.groupby('image').agg({'bin_gt': 'first', 'y_pred': list})
    df[[f'y_pred{i}' for i in range(MC_RUNS)]] = pd.DataFrame(df['y_pred'].tolist(), index=df.index)
    df = df.drop(columns='y_pred')
    df.to_csv(SAVEDIR+'raw_preds_test.tsv', sep='\t')

    # calculate AUROC, AP on each MC run individually
    aurocs, aps = [], []
    for i in range(MC_RUNS):
        aurocs.append(AUC(curve='ROC')(df['bin_gt'], df[f'y_pred{i}']))
        aps.append(AUC(curve='PR')(df['bin_gt'], df[f'y_pred{i}']))
    print('mean AUROC on test:', np.mean(aurocs))
    print('mean AP on test:', np.mean(aps))


