"""
general imports
"""
import argparse
import logging
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from schnet.data import ASEReader, DataProvider
from schnet.forces import predict_energy_forces
from schnet.models import SchNet

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def evaluate(model_path, data_path, indices, energy, forces, name, batch_size=100, atomref=None):
    """main evaluator function"""
    tf.reset_default_graph()
    checkpoint_dir = os.path.join(model_path, 'validation')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    print(ckpt)

    args = np.load(os.path.join(model_path, 'args.npy')).item()

    atomref = None
    try:
        atomref = np.load(atomref)['atom_ref']
        if args.energy == 'energy_U0':
            atomref = atomref[:, 1:2]
        if args.energy == 'energy_U':
            atomref = atomref[:, 2:3]
        if args.energy == 'enthalpy_H':
            atomref = atomref[:, 3:4]
        if args.energy == 'free_G':
            atomref = atomref[:, 4:5]
        if args.energy == 'Cv':
            atomref = atomref[:, 5:6]
    except Exception as err:
        print(err)

    # setup data pipeline
    logging.info('Setup data reader')
    fforces = [forces] if forces != 'none' else []
    data_reader = ASEReader(data_path,
                            [energy],
                            fforces, [(None, 3)])
    data_provider = DataProvider(data_reader, batch_size, indices,
                                 shuffle=False)
    data_batch = data_provider.get_batch()

    logging.info('Setup model')
    schnet = SchNet(args.interactions, args.basis, args.filters, args.cutoff,
                    atomref=atomref,
                    intensive=args.intensive,
                    filter_pool_mode=args.filter_pool_mode)

    # apply model
    e_t = data_batch[energy]
    if forces != 'none':
        f_t = data_batch[forces]
    e_p, f_p = predict_energy_forces(schnet, data_batch)

    aids = []
    e_pred = []
    f_pred = []
    e_list = []
    f_list = []
    count = 0
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        data_provider.create_threads(sess, coord)
        schnet.restore(sess, ckpt)

        for i in range(len(data_provider) // batch_size):
            if forces != 'none':
                e, f, ep, fp, aid = sess.run([e_t, f_t, e_p, f_p, data_batch['aid']])
                f_list.append(f)
                f_pred.append(fp)
            else:
                e, ep, aid = sess.run([e_t, e_p, data_batch['aid']])
            e_list.append(e)
            e_pred.append(ep)
            aids.append(aid)

            count += 1
            if count % 10 == 0:
                print(count)

    e_list = np.hstack(e_list).ravel()
    aids = np.hstack(aids)
    e_pred = np.hstack(e_pred).ravel()
    e_mae = np.mean(np.abs(e_list - e_pred))
    e_rmse = np.sqrt(np.mean(np.square(e_list - e_pred)))

    if forces != 'none':
        f_list = np.vstack(f_list)
        f_pred = np.vstack(f_pred)
        f_mae = np.mean(np.abs(f_list - f_pred[:, 0]))
        f_rmse = np.sqrt(np.mean(np.square(f_list - f_pred[:, 0])))
    else:
        f_list = None
        f_pred = None
        f_mae = 0.
        f_rmse = 0.
    np.savez(os.path.join(model_path, 'results_' + name + '.npz'),
             F=f_list, f_pred=f_pred, E=e_list, e_pred=e_pred, aids=aids)
    return e_mae, e_rmse, f_mae, f_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to directory with models')
    parser.add_argument('data', help='Path to data')
    parser.add_argument('splitdir', help='directory with data splits')
    parser.add_argument('split', help='train / val / test')
    parser.add_argument('--energy', help='Name of run',
                        default='energy_U0')
    parser.add_argument('--forces', help='Name of run',
                        default='none')
    parser.add_argument('--atomref', help='Atom reference file (NPZ)',
                        default=None)
    parser.add_argument('--splitname', help='Name of data split',
                        default=None)
    args = parser.parse_args()

    with open(os.path.join(args.path, 'errors_' + args.split + '.csv'),
              'w',encoding='utf-8') as f:
        f.write('model,energy MAE,energy RMSE,force MAE,force RMSE\n')
        for dir in os.listdir(args.path):
            mdir = os.path.join(args.path, dir)
            if not os.path.isdir(mdir):
                continue
            print(mdir)
            if args.splitname is None:
                SPLIT_NAME = '_'.join(dir.split('_')[9:])
            else:
                SPLIT_NAME = args.splitname
            split_file = os.path.join(args.splitdir, SPLIT_NAME + '.npz')
            indices = np.load(split_file)[args.split]
            print(len(indices)//100)
            try:
                res = evaluate(mdir, args.data, indices,
                           args.energy, args.forces, args.split, atomref=args.atomref)
            except Exception as e:
                print(e)
                continue
            res = [str(np.round(r, 8)) for r in res]
            f.write(dir + ',' + ','.join(res) + '\n')
            print(dir, res)
