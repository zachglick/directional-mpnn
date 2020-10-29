import os, sys, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, Concatenate, Dense, Embedding, Flatten, Input, InputLayer, Lambda, Layer, Reshape, Subtract
tf.keras.backend.set_floatx('float64')

import util

elem_z = {
        'H'  : 1,
        'C'  : 6,
        'N'  : 7,
        'O'  : 8,
        'F'  : 9,
        'P'  : 15,
        'S'  : 16,
        'CL' : 17,
        'BR' : 35,
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an atomic multipole model')
    parser.add_argument('dataset_train',
                        help='Dataset for training (must be in ./data/)')
    parser.add_argument('dataset_val',
                        help='Dataset for validation (must be in ./data/)')
    parser.add_argument('modelname',
                        help='Name for saving model')
    args = parser.parse_args(sys.argv[1:])

    if os.path.isfile(f'models/{args.modelname}.hdf5'):
        print(f'Model models/{args.modelname}.hdf5 already exists!')
        exit()

    # these are defined by the dataset
    pad_dim = 40
    nelem = 36
    
    # this is up to the user
    nembed = 10
    nepochs = 200
    nnodes = [256,128,64]
    nmessage = 3

    # make the model 
    mus = np.linspace(0.8, 5.0, 43)
    etas = np.array([-100.0] * 43)
    model = util.get_model(mus, etas, pad_dim, nelem, nembed, nnodes, nmessage)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=['mse', util.mse_mp, util.mse_mp, util.mse_mp],
                  loss_weights=[1.0, 1.0, 1.0, 1.0],
                  metrics=[util.mae_mp])
    print(model.summary())

    # load data
    RT, ZT, yT = util.get_data(f'/theoryfs2/ds/glick/gits/directional-mpnn/data/{args.dataset_train}.pkl', pad_dim)
    RV, ZV, yV = util.get_data(f'/theoryfs2/ds/glick/gits/directional-mpnn/data/{args.dataset_val}.pkl', pad_dim)

    #RT, ZT, yT = RT[:800], ZT[:800], yT[:800]
    #RV, ZV, yV = RV[:800], ZV[:800], yV[:800]

    # monopole
    yV_ = yV[:,:,0]

    # dipole (mu_x, mu_y, mu_z)
    yV_i_ = yV[:,:,1:4]

    # quadrupole diagonal (Q_xx, Q_yy, Q_zz)
    yV_ii_ = yV[:,:,[4,7,9]]

    # quadrupole off-diagonal (Q_xy, Q_xz, Q_yz)
    yV_ij_ = yV[:,:,[5,6,8]]

    print('Validation Target Magnitudes (MAD):')
    for Z_subset, y_subset in [(ZV, [yV_, yV_i_, yV_ii_, yV_ij_])]:
        y_subset_pos = [ys[Z_subset > 0] for ys in y_subset]
        mad = [np.mean(np.abs(ys - np.mean(ys))) for ys in y_subset_pos]
        print(f'ALL ({y_subset_pos[0].shape[0]:6d}) : q({mad[0]:.4f}) mu({mad[1]:.4f}) Qii({mad[2]:.4f}) Qij({mad[3]:.4f})')
        for name, z in elem_z.items():
            mask = (Z_subset == z)
            y_element = [ys_[mask] for ys_ in y_subset]
            if np.sum(mask) > 0:
                mad = [np.mean(np.abs(ys - np.mean(ys))) for ys in y_element]
            else:
                mad = [np.nan for ys in y_element]
            print(f'{name:3s} ({np.sum(mask):6d}) : q({mad[0]:.4f}) mu({mad[1]:.4f}) Qii({mad[2]:.4f}) Qij({mad[3]:.4f})')
        print()

    print('Fitting Model...')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(f'models/{args.modelname}.hdf5', save_best_only=True, monitor='val_loss', mode='min'),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=(10.0 ** (-1/4)), patience=10, verbose=1, mode='min', min_delta=0, cooldown=0, min_lr=(10.0 ** -5))]
    model.fit(x=util.RotationGenerator(RT, ZT, yT, batch_size=8),
              epochs=nepochs,
              validation_data=([RV, ZV], [yV_, yV_i_, yV_ii_, yV_ij_]),
              callbacks=callbacks,
              verbose=2)

    print('...Done')

    for w in model.get_layer(name='rbf').get_weights():
        print(w)

    yV_pred = model.predict([RV, ZV])

    yV_err = yV_pred[0] - yV_
    yV_i_err = yV_pred[1] - yV_i_
    yV_ii_err = yV_pred[2] - yV_ii_
    yV_ij_err = yV_pred[3] - yV_ij_

    print('Validation Prediction Magnitudes (MAE):')
    for Z_subset, y_subset in [(ZV, [yV_err, yV_i_err, yV_ii_err, yV_ij_err])]:
        y_subset_pos = [ys[Z_subset > 0] for ys in y_subset]
        mae = [np.mean(np.abs(ys)) for ys in y_subset_pos]
        print(f'ALL ({y_subset_pos[0].shape[0]:6d}) : q({mae[0]:.4f}) mu({mae[1]:.4f}) Qii({mae[2]:.4f}) Qij({mae[3]:.4f})')
        for name, z in elem_z.items():
            mask = (Z_subset == z)
            y_element = [ys_[mask] for ys_ in y_subset]
            if np.sum(mask) > 0:
                mae = [np.mean(np.abs(ys)) for ys in y_element]
            else:
                mae = [np.nan for ys in y_element]
            print(f'{name:3s} ({np.sum(mask):6d}) : q({mae[0]:.4f}) mu({mae[1]:.4f}) Qii({mae[2]:.4f}) Qij({mae[3]:.4f})')
        print()
