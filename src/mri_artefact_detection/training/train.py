'''
    Script to train the model.
'''

# IMPORTS
import os, numpy as np, pandas as pd
from train_utils import *
from model import *

import tensorflow as tf, keras
from keras.metrics import AUC
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# CONSTANTS
SAVEDIR = '/vols/opig/users/vavourakis/ge_project/trainrun'
DATADIR = '/data/nagagpu04/not-backed-up/nvme00/vavourakis/struc_data'
DATASETS = ['artefacts'+str(i) for i in [1,2,3]]
CONTRASTS = ['T1wMPR']#, 'T1wTIR', 'T2w', 'T2starw', 'FLAIR']
QUALS = ['clean', 'exp_artefacts']

ARTEFACT_DISTRO = {
    'RandomAffine' : 1/12,
    'RandomElasticDeformation' : 1/12,
    'RandomAnisotropy' : 1/12,
    'RescaleIntensity' : 1/12,
    'RandomMotion' : 1/12,
    'RandomGhosting' : 1/12,
    'RandomSpike' : 1/12,
    'RandomBiasField' : 1/12,
    'RandomBlur' : 1/12,
    'RandomNoise' : 1/12,
    'RandomSwap' : 1/12,
    'RandomGamma' : 1/12
}

TARGET_CLEAN_RATIO = 0.5 # re-sample training set to this fraction of clean images
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

    # get the paths and labels of the real images
    real_image_paths, pids, real_labels = DataCrawler(DATADIR, DATASETS, CONTRASTS, QUALS).crawl()

    # split images by patient id
    Xtrain, Xval, Xtest, ytrain, yval, ytest = split_by_patient(real_image_paths, pids, real_labels)

    for string, y in zip(['train', 'val', 'test'], [ytrain, yval, ytest]):
        print('number in ' + string + ' set:', len(y))
        print(string + ' class distribution: ', sum(y)/len(y), ' percent artefact')

    # instantiate DataLoaders
    trainloader = DataLoader(Xtrain, ytrain, train_mode=True,
                            image_shape = (192,256,256), batch_size=10,
                            target_clean_ratio=TARGET_CLEAN_RATIO, artef_distro=ARTEFACT_DISTRO)
    valloader = DataLoader(Xval, yval, train_mode=False,
                        batch_size=15, image_shape = (192,256,256))
    testloader = DataLoader(Xtest*MC_RUNS, np.array(ytest.tolist()*MC_RUNS),
                            train_mode=False,
                            batch_size=15, image_shape = (192,256,256))

    # write out ground truth for test set
    test_images = [file for sublist in testloader.batches for file in sublist]
    y_true_test = [y for sublist in testloader.labels for y in sublist]
    out = pd.DataFrame({'image': test_images, 'bin_gt': y_true_test}).groupby('image').agg({'bin_gt': 'first'})
    out.to_csv(os.path.join(SAVEDIR, 'test_split_gt.tsv'), sep='\t')

    # compile model
    model = getConvNet(out_classes=2, input_shape=(192,256,256,1))
    model.compile(loss='categorical_crossentropy',
                optimizer='nadam',
                metrics=['accuracy',
                        AUC(curve='ROC', name='auroc'),
                        AUC(curve='PR', name='auprc')])
    print('\n')
    print(model.summary())

    # prepare for training
    checkpoint_dir = os.path.join(SAVEDIR, "ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # define callbacks
    callbacks = [
        # keep an eye on model weights norm after every epoch
        PrintModelWeightsNorm(),
        # save model after each epoch
        ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "end_of_epoch_{epoch}.keras")), 
        # reduce learning rate if val_loss doesn't improve for 2 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, mode='auto',
                          min_delta=1e-2, cooldown=0, min_lr=0.0001),
        # stop training if val_loss doesn't improve for 5 epochs
        EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=5, verbose=1)
    ]

    # train model
    history = model.fit(trainloader, 
                        validation_data=valloader,
                        steps_per_epoch=130,           # batches per epoch
                        epochs=24,                     # number of epochs
                        callbacks=callbacks)
    
    # save metrics
    pd.DataFrame({
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_auc': history.history['auroc'],
        'val_auc': history.history['val_auroc'],
        'train_ap': history.history['auprc'],
        'val_ap': history.history['val_auprc']
    }).to_csv(os.path.join(SAVEDIR, 'training_metrics.tsv'), sep='\t')
    plot_train_metrics(history, os.path.join(SAVEDIR, 'train_metrics_plot.png'))

    # evaluate model (multiple MC runs per test image)
    print("#"*30)
    print("Evaluate on Test Set")
    print("#"*30)
    # predict on test set, and write out results
    y_pred = model.predict(testloader)#, use_multiprocessing=True)
    df = pd.DataFrame({'image': test_images, 
                       'bin_gt': y_true_test, 
                       'y_pred': y_pred[:,1]})
    df = df.groupby('image').agg({'bin_gt': 'first', 'y_pred': list})
    df[[f'y_pred{i}' for i in range(MC_RUNS)]] = pd.DataFrame(df['y_pred'].tolist(), index=df.index)
    df = df.drop(columns='y_pred')
    df.to_csv(os.path.join(SAVEDIR, 'raw_preds_test.tsv'), sep='\t')

    # calculate AUROC, AP on each MC run individually
    aurocs, aps = [], []
    for i in range(MC_RUNS):
        aurocs.append(AUC(curve='ROC')(df['bin_gt'], df[f'y_pred{i}']))
        aps.append(AUC(curve='PR')(df['bin_gt'], df[f'y_pred{i}']))
    print('mean AUROC on test:', np.mean(aurocs))
    print('mean AP on test:', np.mean(aps))
