from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
import tensorflow as tf
from tenning.base_model import BaseModel
from tenning.rotation_utils import rotation_matrix_from_ortho6d
from tenning.activations import Swish
from tenning.linalg_utils import sym_matrix_from_array
from tenning.rotation_utils import rotate_vector


class MCDAttitudeEstimator(BaseModel):

    def __init__(self,
                 batch_size=None,
                 mcd_rate=0.1,
                 initializer='he_normal',
                 trainable=True,
                 **kwargs):
        super().__init__(trainable=trainable, **kwargs)

        self.batch_size = batch_size
        self.initializer = initializer
        self.mcd_rate = mcd_rate

    def build_model(self):

        input_tensor = Input(shape=[9, 1], batch_size=self.batch_size, name=f"{self.name}/profile_matrix")
        # ref_vectors = Input(shape=[self.observations, 3], batch_size=self.batch_size, name=f"{self.name}/ref_vectors")
        # body_vectors = Input(shape=[self.observations, 3], batch_size=self.batch_size, name=f"{self.name}/body_vectors")

        # input_tensor = tf.keras.layers.Concatenate()([ref_vectors, body_vectors])

        x = Conv1D(64, 9, padding='same', kernel_initializer=self.initializer, name=f"{self.name}/conv1")(input_tensor)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Conv1D(128, 9, padding='same', kernel_initializer=self.initializer, name=f"{self.name}/conv2")(x)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Conv1D(256, 9, kernel_initializer=self.initializer, name=f"{self.name}/conv3")(x)
        x = tf.squeeze(x, axis=1)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Dense(512, kernel_initializer=self.initializer, name=f"{self.name}/dense1")(x)
        x = Swish()(x)

        x = Dropout(self.mcd_rate)(x)
        x = Dense(6, kernel_initializer=self.initializer, name=f"{self.name}/output")(x)
        rotations = rotation_matrix_from_ortho6d(x)  # (batch, 3, 3)

        return {"inputs": input_tensor, "outputs": rotations, "trainable": True}

    @tf.function
    def update_step(self, **data_dict):

        body_vectors = data_dict.get('body_vectors')
        ref_vectors = data_dict.get('ref_vectors')
        true_rotation = data_dict.get('true_rotation')
        # stddevs = data_dict.get('stddevs')

        # # Equation 97 from [Shuster1981]
        # sig_tot = 1. / tf.reduce_sum(1/stddevs**2, axis=1, keepdims=True)
        # # Equation 96 from [Shuster1981]
        # weights = sig_tot / stddevs**2
        # weights = weights[..., tf.newaxis]

        obs = body_vectors.shape[1]
        weights = tf.ones([self.batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [self.batch_size, -1, 1])

        with tf.GradientTape() as tape:

            rotations = self(profile_matrix, training=True)

            predictions = rotate_vector(rotations, ref_vectors, 'dcm')

            total_loss = self.loss(true_rotation, rotations)

        gradients = tape.gradient(total_loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(body_vectors, predictions)
        self.metrics[1].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result(), self.metrics[1].result()]

        return results

    @tf.function
    def predict_step(self, **data_dict):

        body_vectors = data_dict.get('body_vectors')
        ref_vectors = data_dict.get('ref_vectors')
        true_rotation = data_dict.get('true_rotation')
        # stddevs = data_dict.get('stddevs')

        # # Equation 97 from [Shuster1981]
        # sig_tot = 1. / tf.reduce_sum(1/stddevs**2, axis=1, keepdims=True)
        # # Equation 96 from [Shuster1981]
        # weights = sig_tot / stddevs**2
        # weights = weights[..., tf.newaxis]

        obs = body_vectors.shape[1]
        weights = tf.ones([self.batch_size, obs, 1]) / obs

        # Vectorized form of equation 38 from [Shuster1981]
        profile_matrix = tf.matmul(body_vectors, ref_vectors * weights, transpose_a=True)
        profile_matrix = tf.reshape(profile_matrix, [self.batch_size, -1, 1])

        rotations = self(profile_matrix, training=False)

        predictions = rotate_vector(rotations, ref_vectors, 'dcm')

        total_loss = self.loss(true_rotation, rotations)

        self.metrics[0].update_state(body_vectors, predictions)
        self.metrics[1].update_state(true_rotation, rotations)

        results = [total_loss, self.metrics[0].result(), self.metrics[1].result()]

        return results

    def get_config(self):

        config = super().get_config()

        config.update({"batch_size": self.batch_size,
                       "mcd_rate": self.mcd_rate,
                       "initializer": self.initializer,
                       "trainable": self.trainable,
                       "name": self.name})

        return config

    @staticmethod
    def get_custom_objs():
        return {'MCDAttitudeEstimator': MCDAttitudeEstimator,
                'Swish': Swish}


def get_attitude_model():
    return MCDAttitudeEstimator
