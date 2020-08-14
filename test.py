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

    parser = argparse.ArgumentParser(description='Test an atomic multipole model')
    parser.add_argument('dataset',
                        help='Dataset for testing (must be in ./data/)')
    parser.add_argument('modelname',
                        help='Name for loading model')
    args = parser.parse_args(sys.argv[1:])

    # these are defined by the dataset
    pad_dim = 40

    # make the model 
    model = tf.keras.models.load_model(f'./models/{args.modelname}.hdf5', custom_objects={'RBFLayer': util.RBFLayer,
                                                                            'mse_mp': util.mse_mp, 
                                                                            'mae_mp' : util.mae_mp})
    print(model.summary())

    R, Z, y = util.get_data(f'/theoryfs2/ds/glick/gits/directional-mpnn/data/{args.dataset}.pkl', pad_dim)

    # monopole
    y_ = y[:,:,0]

    # dipole (mu_x, mu_y, mu_z)
    y_i_ = y[:,:,1:4]

    # quadrupole diagonal (Q_xx, Q_yy, Q_zz)
    y_ii_ = y[:,:,[4,7,9]]

    # quadrupole off-diagonal (Q_xy, Q_xz, Q_yz)
    y_ij_ = y[:,:,[5,6,8]]

    print('Test Data Target Magnitudes (MAD):')
    for Z_subset, y_subset in [(Z, [y_, y_i_, y_ii_, y_ij_])]:
        y_subset_pos = [ys[Z_subset > 0] for ys in y_subset]
        mad = [np.mean(np.abs(ys - np.mean(ys))) for ys in y_subset_pos]
        print(f'ALL ({y_subset_pos[0].shape[0]:5d}) : q({mad[0]:.4f}) mu({mad[1]:.4f}) Qii({mad[2]:.4f}) Qij({mad[3]:.4f})')
        for name, z in elem_z.items():
            mask = (Z_subset == z)
            y_element = [ys_[mask] for ys_ in y_subset]
            mad = [np.mean(np.abs(ys - np.mean(ys))) for ys in y_element]
            print(f'{name:3s} ({np.sum(mask):5d}) : q({mad[0]:.4f}) mu({mad[1]:.4f}) Qii({mad[2]:.4f}) Qij({mad[3]:.4f})')
        print()

    for w in model.get_layer(name='rbf').get_weights():
        print(w)

    y_pred = model.predict([R, Z])
    y_pred2 = np.concatenate([y_pred[0].reshape(-1,pad_dim,1), y_pred[1], y_pred[2], y_pred[3]], axis=2)
    lens = [np.sum(z > 0) for z in Z]
    y_pred2 = [y_pred2[i][:lens[i]] for i in range(len(y_pred2))]
    #df_pred = pd.DataFrame.from_dict(data={'name' : names,
    #                                       'R'    : df['R'].tolist(),
    #                                       'Z'    : df['Z'].tolist(),
    #                                       'cartesian_multipoles' : y_pred2})
    #df_pred.to_pickle(f'S66x8-A_preds.pkl', protocol=4)

    y_err = y_pred[0] - y_
    y_i_err = y_pred[1] - y_i_
    y_ii_err = y_pred[2] - y_ii_
    y_ij_err = y_pred[3] - y_ij_

    #np.save('mpnn_predictions/q_pred.npy', y_pred[0])
    #np.save('mpnn_predictions/mu_pred.npy', y_pred[1])
    #np.save('mpnn_predictions/Qii_pred.npy', y_pred[2])
    #np.save('mpnn_predictions/Qij_pred.npy', y_pred[3])

    #np.save('mpnn_predictions/q_lab.npy', y_)
    #np.save('mpnn_predictions/mu_lab.npy', y_i_)
    #np.save('mpnn_predictions/Qii_lab.npy', y_ii_)
    #np.save('mpnn_predictions/Qij_lab.npy', y_ij_)

    print('Test Data Prediction Magnitudes (MAE):')
    for Z_subset, y_subset in [(Z, [y_err, y_i_err, y_ii_err, y_ij_err])]:
    #for Z_subset, y_subset in [(Z, [y_err, y_i_err, y_ii_err, y_ij_err, y_h_err, y_v_err])]:
        y_subset_pos = [ys[Z_subset > 0] for ys in y_subset]
        mae = [np.mean(np.abs(ys)) for ys in y_subset_pos]
        print(f'ALL ({y_subset_pos[0].shape[0]:5d}) : q({mae[0]:.4f}) mu({mae[1]:.4f}) Qii({mae[2]:.4f}) Qij({mae[3]:.4f})')
        #print(f'ALL ({y_subset_pos[0].shape[0]:5d}) : q({mae[0]:.4f}) mu({mae[1]:.4f}) Qii({mae[2]:.4f}) Qij({mae[3]:.4f}) h({mae[4]:.4f}) v({mae[5]:.4f})')
        for name, z in elem_z.items():
            mask = (Z_subset == z)
            y_element = [ys_[mask] for ys_ in y_subset]
            mae = [np.mean(np.abs(ys)) for ys in y_element]
            print(f'{name:3s} ({np.sum(mask):5d}) : q({mae[0]:.4f}) mu({mae[1]:.4f}) Qii({mae[2]:.4f}) Qij({mae[3]:.4f})')
            #print(f'{name:3s} ({np.sum(mask):5d}) : q({mae[0]:.4f}) mu({mae[1]:.4f}) Qii({mae[2]:.4f}) Qij({mae[3]:.4f}) h({mae[4]:.4f}) v({mae[5]:.4f})')
        print()
