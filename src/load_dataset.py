import sys
import os
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
from tenning.data_utils import IteratorBuilder
from tenning.rotation_utils import gen_random_dcm
from tenning.rotation_utils import gen_boresight_vector
from tenning.rotation_utils import gen_rot_quaternion
from tenning.rotation_utils import rotate_vector


class AttitudeDataset(IteratorBuilder):

    def __init__(self,
                 num_samples=2**13,
                 num_observations=4,
                 min_angle=-180,
                 max_angle=180,
                 add_noise=False,
                 std_range=None,
                 parametrization='dcm',
                 **kwargs):
        super().__init__(**kwargs)

        self.obs = num_observations
        self.num_samples = num_samples
        self.add_noise = add_noise
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.std_range = std_range or [1e-6, 0.01]
        self.parametrization = parametrization

        self.gen_data()

    def gen_data(self):

        ref_vectors = gen_boresight_vector(self.num_samples, self.obs, 15)

        if self.parametrization == 'dcm':
            attitudes = gen_random_dcm(self.num_samples,
                                       min_angle=self.min_angle,
                                       max_angle=self.max_angle,
                                       unit='deg').numpy()
        else:
            attitudes = gen_rot_quaternion(self.num_samples,
                                           min_angle=self.min_angle,
                                           max_angle=self.max_angle,
                                           unit='deg').numpy()

        body_vectors = rotate_vector(attitudes, ref_vectors, self.parametrization)

        def gen_noise(stds):

            # Creates an empty array
            noise = np.zeros((0, 3))
            for std in stds:
                # Generates random values from a Gaussian distribution
                temp = np.random.normal(scale=std, size=(1, 3))
                noise = np.vstack([noise, temp])

            return noise

        # Selects standard deviation values from a given range
        # stds = np.random.uniform(-6, -2, size=(self.num_samples, self.obs))
        stds = np.random.uniform(self.std_range[0], self.std_range[1], size=(self.num_samples, self.obs))

        if self.add_noise:

            noise = np.apply_along_axis(gen_noise, axis=1, arr=stds)

            body_vectors += noise

        stds = np.tile(stds[..., np.newaxis], [1, 1, 3])

        if self.parametrization == 'quaternion':
            attitudes = tf.tile(attitudes[..., tf.newaxis], [1, 1, 3])

        dataset = tf.concat([ref_vectors, body_vectors, stds, attitudes], axis=1)  # (samples, 13, 3)

        self.set_dataset(dataset.numpy())

    def yielder(self, data):

        ref_vectors = tf.cast(data[:, :self.obs, :], tf.float32)  # (batch, observations, 3)

        body_vectors = tf.cast(data[:, self.obs:(2*self.obs), :], tf.float32)  # (batch, observations, 3)

        stds = tf.cast(data[:, (2*self.obs):(3*self.obs), 0], tf.float32)  # (batch, observations)

        if self.parametrization == 'dcm':
            true_rotation = tf.cast(data[:, (3*self.obs):, :], tf.float32)  # (batch, 3, 3)
        else:
            true_rotation = tf.cast(data[:, (3*self.obs):, 0], tf.float32)  # (batch, 4)

        return body_vectors, ref_vectors, stds, true_rotation

    def post_process(self, *args):

        return {'body_vectors': args[0],
                'ref_vectors': args[1],
                'stddevs': args[2],
                'true_rotation': args[3]}


def get_handler(params):
    data_handler = AttitudeDataset(num_samples=params["num_samples"],
                                   num_observations=params["num_observations"],
                                   add_noise=params["add_noise"],
                                   std_range=params["std_range"],
                                   val_ratio=params["val_ratio"],
                                   test_ratio=params["test_ratio"],
                                   batch_size=params["batch_size"],
                                   min_angle=params["min_angle"],
                                   max_angle=params["max_angle"],
                                   parametrization=params["parametrization"])

    return data_handler
