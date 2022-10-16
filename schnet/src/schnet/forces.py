import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from schnet.data import get_atoms_input


def predict_energy_forces(model, data):
    """
    Predict energy and forces for a batch of data.
    
    :param model: model to use for prediction
    :param data: batch of data for which we need to predict energy and forces
    :return: predicted energy and forces
    """
    atoms_input = get_atoms_input(data)
    g = tf.get_default_graph()
    with g.gradient_override_map({"Tile": "TileDense"}):
        Ep = model(*atoms_input)
        Fp = -tf.gradients(tf.reduce_sum(Ep), atoms_input[1])[0]
    return Ep, Fp


def calculate_errors(Ep, Fp, data, energy_prop, force_prop,
                     rho, fit_energy=True, fit_forces=True):
    """
    Calculate the errors for the predicted values of energy and forces.

    :param Ep: predicted energy
    :param Fp: predicted forces
    :param data: batch of data for which we need to predict the calculation errors
    :param energy_prop: actual measured chemical energy
    :param force_prop: actual measured forces
    :param rho: weight of energy in the total loss
    :param fit_energy: whether to fit the energy
    :param fit_forces: whether to fit the forces
    :return: loss and errors
    """  
    loss = 0.

    if force_prop != 'none':
        F = data[force_prop]
        fdiff = F - Fp
        fmse = tf.reduce_mean(fdiff ** 2)
        fmae = tf.reduce_mean(tf.abs(fdiff))
        if fit_forces:
            loss += tf.nn.l2_loss(fdiff)
    else:
        fmse = tf.constant(0.)
        fmae = tf.constant(0.)

    E = data[energy_prop]
    ediff = E - Ep
    eloss = tf.nn.l2_loss(ediff)

    emse = tf.reduce_mean(ediff ** 2)
    emae = tf.reduce_mean(tf.abs(ediff))

    if fit_energy:
        loss += rho * eloss

    errors = [emse, emae, fmse, fmae]
    return loss, errors


def collect_summaries(args, loss, errors):
    """
    Collect the summaries for the loss and errors.
    
    :param args: arguments (i.e. the given data)
    :param loss: loss
    :param errors: errors
    :return: total loss and summaries
    """
    emse, emae, fmse, fmae = errors
    vloss = np.sum(loss)

    summary = tf.Summary()
    summary.value.add(tag='loss', simple_value=vloss)
    summary.value.add(tag='total_energy_RMSE',
                      simple_value=np.sqrt(np.mean(emse[0])))
    summary.value.add(tag='total_energy_MAE', simple_value=np.mean(emae))
    if args.forces != 'none':
        summary.value.add(tag='force_RMSE', simple_value=np.sqrt(np.mean(fmse)))
        summary.value.add(tag='force_MAE', simple_value=np.mean(fmae))
    return vloss, summary
