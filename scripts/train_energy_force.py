"""
general imports
"""
import argparse
import logging
import os
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from schnet.atoms import stats_per_atom
from schnet.data import ASEReader, DataProvider
from schnet.forces import predict_energy_forces, calculate_errors, \
    collect_summaries
from schnet.models import SchNet
from schnet.nn.train import EarlyStopping, build_train_op

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def train(vari):
    """trainer func"""
    logging.info('Setup directories')
    splitpath = vari.splits
    if not splitpath.endswith('.npz'):
        splitpath += '.npz'
    splitname = splitpath.split('/')[-1].split('.')[0]

    # initialize output files
    name = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}'.format(
        'SchNet', vari.fit_energy, vari.fit_force,
        vari.basis, vari.filters, vari.interactions, vari.cutoff,
        vari.lr, vari.eweight, splitname
    )
    if len(vari.name) > 0:
        name += '_' + vari.name

    outdir = os.path.join(vari.output_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    args_log = os.path.join(outdir, 'args.npy')
    np.save(args_log, vari)

    # setup data pipeline
    logging.info('Setup data reader')
    forces = [vari.forces] if vari.forces != 'none' else []
    data_reader = ASEReader(vari.data,
                            [vari.energy],
                            forces, [(None, 3)])

    if not os.path.exists(splitpath):
        logging.info('Create data splits')
        if vari.ntrain < 0 and vari.nval < 0:
            logging.error('If `splits` does not exist, arguments ' +
                          '`ntrain` and `nval` have to be specified!')
            return

        N = len(data_reader)
        pidx = np.random.permutation(N)
        train_idx = pidx[:vari.ntrain]
        val_idx = pidx[vari.ntrain:vari.ntrain + vari.nval]
        test_idx = pidx[vari.ntrain + vari.nval:]
        np.savez(splitpath, train_idx=train_idx,
                 val_idx=val_idx, test_idx=test_idx)
    else:
        logging.info('Load existing data splits')
        S = np.load(splitpath)
        train_idx = S['train_idx']
        val_idx = S['val_idx']

    atomref = None
    try:
        atomref = np.load(vari.atomref)['atom_ref']
        if vari.energy == 'energy_U0':
            atomref = atomref[:, 1:2]
        if vari.energy == 'energy_U':
            atomref = atomref[:, 2:3]
        if vari.energy == 'enthalpy_H':
            atomref = atomref[:, 3:4]
        if vari.energy == 'free_G':
            atomref = atomref[:, 4:5]
        if vari.energy == 'Cv':
            atomref = atomref[:, 5:6]
    except Exception as err:
        print(err)

    logging.info('Setup train/validation data providers')
    train_provider = DataProvider(data_reader, vari.batch_size,
                                  train_idx)
    train_data = train_provider.get_batch()
    val_provider = DataProvider(data_reader, vari.valbatch, val_idx)
    val_data = val_provider.get_batch()

    logging.info('Collect train data statistics')
    ener = data_reader.get_property(vari.energy, train_idx)
    zvar = data_reader.get_atomic_numbers(train_idx)
    mean_energy_per_atom, stddev_energy_per_atom = \
        stats_per_atom(ener, zvar, vari.intensive, atomref)
    logging.info('Energy statistics: mu/atom=' + str(mean_energy_per_atom) +
                 ', std/atom=' + str(stddev_energy_per_atom))

    logging.info('Setup model')
    schnet = SchNet(vari.interactions, vari.basis, vari.filters, vari.cutoff,
                    mean_energy_per_atom, stddev_energy_per_atom,
                    atomref=atomref, intensive=vari.intensive,
                    filter_pool_mode=vari.filter_pool_mode)

    # apply model
    ep_train, fp_train = predict_energy_forces(schnet, train_data)
    Ep_val, fp_val = predict_energy_forces(schnet, val_data)

    # calculate loss, errors & summaries
    train_loss, train_errors = calculate_errors(ep_train, fp_train, train_data,
                                                vari.energy, vari.forces,
                                                vari.eweight, vari.fit_energy,
                                                vari.fit_force)
    val_loss, val_errors = calculate_errors(Ep_val, fp_val, val_data,
                                            vari.energy, vari.forces,
                                            vari.eweight, vari.fit_energy,
                                            vari.fit_force)

    # setup training
    logging.info('Setup optimizer')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lrate = tf.train.exponential_decay(vari.lr, global_step, 100000, 0.96)

    optimizer = tf.train.AdamOptimizer(lrate)
    train_op = build_train_op(train_loss, optimizer, global_step)

    logging.info('Setup early stopping')
    valbatches = len(val_provider) // vari.valbatch
    trainer = EarlyStopping(outdir, schnet, train_op,
                            train_loss, train_errors,
                            val_loss, val_errors,
                            partial(collect_summaries, vari),
                            save_interval=vari.saveint,
                            validation_interval=vari.valint,
                            validation_batches=valbatches,
                            global_step=global_step)

    with tf.Session() as sess:
        logging.info('Start queue runners')
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        train_provider.create_threads(sess, coord)
        val_provider.create_threads(sess, coord)

        logging.info('Start training')
        trainer.train(sess, coord, vari.max_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        help='Path to training data (ASE DB)')
    parser.add_argument('output_dir',
                        help='Output directory for model and training log.')
    parser.add_argument('splits',
                        help='Path to data splits (is created, if not existing)')
    parser.add_argument('--ntrain', help='Number of training examples',
                        type=int, default=-1)
    parser.add_argument('--nval', help='Number of validation examples',
                        type=int, default=-1)
    parser.add_argument('--energy', help='Name of run',
                        default='energy_U0')
    parser.add_argument('--forces', help='Name of run',
                        default='none')
    parser.add_argument('--filter_pool_mode', help='One out of [sum, mean]',
                        default='sum')
    parser.add_argument('--intensive', action='store_true',
                        help='If intensive, mean pool energy over atoms')
    parser.add_argument('--fit_energy', action='store_true',
                        help='Use additional energy loss')
    parser.add_argument('--fit_force', action='store_true',
                        help='Use additional energy loss')
    parser.add_argument('--atomref', help='Atom reference file (NPZ)',
                        default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size',
                        default=32)
    parser.add_argument('--cutoff', type=float, help='Distance cutoff',
                        default=20.)
    parser.add_argument('--interactions', type=int, help='Distance cutoff',
                        default=6)
    parser.add_argument('--basis', type=int, help='Basis set size',
                        default=64)
    parser.add_argument('--filters', type=int, help='Factor space size',
                        default=64)
    parser.add_argument('--eweight', type=float,
                        help='Number of steps to saturate eloss scale',
                        default=1.)
    parser.add_argument('--max_steps', type=int, help='Number of steps',
                        default=5000000)
    parser.add_argument('--valint', type=int, help='Validation interval',
                        default=5000)
    parser.add_argument('--saveint', type=int, help='Checkpoint interval',
                        default=50000)
    parser.add_argument('--valbatch', type=int,
                        help='Size of validation batches',
                        default=100)
    parser.add_argument('--name', help='Name of run',
                        default='')
    parser.add_argument('--lr', type=float, help='Learning rate',
                        default=1e-3)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)
