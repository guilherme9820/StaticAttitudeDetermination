import os
import numpy as np
import pandas as pd
import tensorflow as tf
from attitude_determination.test_cases import TestCases
from tenning.generic_utils import build_json_model
from tenning.rotation_utils import average_rotation_svd
from tenning.rotation_utils import axisangle_from_dcm
from src.models import get_attitude_model
from src.utils import read_params
from src.utils import wahba_error


def get_covariance(model, test_case, rounds):

    test_cases = TestCases()

    ref_vectors, body_vectors, stddevs = test_cases(1, test_case)

    body_vectors = tf.cast(body_vectors, "float32")
    ref_vectors = tf.cast(ref_vectors, "float32")

    # Generates equal weights for each sample (star tracking scenario)
    obs = body_vectors.shape[1]
    weights = tf.ones([1, obs, 1]) / obs

    # Builds the profile matrices for each sample
    profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
    profile_matrix = tf.reshape(profile_matrix, [1, -1, 1])

    pred_attitudes = []
    wahba_loss = []
    for _ in range(rounds):
        pred = model(profile_matrix, training=True)

        pred = tf.squeeze(pred)

        # Rotates the reference vector using the predicted attitude matrix to generate the
        # predicted body vectors
        pred_body_vectors = tf.transpose(tf.matmul(pred, ref_vectors[0], transpose_b=True))

        wahba_loss.append(wahba_error(body_vectors[0], pred_body_vectors, stddevs))

        pred_attitudes.append(pred.numpy())

    pred_attitudes = np.asarray(pred_attitudes)
    avg_rotation = average_rotation_svd(pred_attitudes)

    rotation_noise = tf.matmul(avg_rotation, pred_attitudes, transpose_a=True)

    rot_axis, angles = axisangle_from_dcm(rotation_noise)

    euler_vectors = rot_axis * angles[:, tf.newaxis]

    covariance_matrix = tf.matmul(euler_vectors, euler_vectors, transpose_a=True) / rounds

    return covariance_matrix.numpy(), np.mean(wahba_loss)


def set_drop_rate(model, drop_rate):
    for layer in model.layers:
        if "dropout" in layer.name:
            model.get_layer(layer.name).rate = drop_rate


def main(params):

    arch = get_attitude_model()

    model_name = params["model_name"]
    observation = params["observation"]

    for train_dropout in params["dropout"]:

        model_dir = os.path.join(params["model_dir"], f"{model_name}_{observation}_{train_dropout}")
        json_model = os.path.join(model_dir, "architecture.json")
        # Builds the trained model given its weights and architecture
        dropout_model = build_json_model(model_dir, json_model, arch.get_custom_objs())

        results = pd.DataFrame(columns=['train_dropout', 'test_dropout', 'axis', 'var', 'wahba_error'])

        if not os.path.exists(params["csv_file"]):
            results.to_csv(params["csv_file"], index=False)

        for drop_rate in np.arange(0, 21, 2.5):

            set_drop_rate(dropout_model, drop_rate/100)
            covariance_matrix, wahba_error = get_covariance(dropout_model, params["test_case"], params["rounds"])

            var_x = covariance_matrix[0, 0]
            var_y = covariance_matrix[1, 1]
            var_z = covariance_matrix[2, 2]

            results['train_dropout'] = [train_dropout]*3
            results['test_dropout'] = [drop_rate]*3
            results['axis'] = ['x', 'y', 'z']
            results['var'] = [var_x, var_y, var_z]
            results['wahba_error'] = [wahba_error]*3

            results.to_csv(params["csv_file"], mode='a', header=False, index=False)
