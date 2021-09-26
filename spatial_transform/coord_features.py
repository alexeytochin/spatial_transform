import tensorflow as tf


class AddCoordFeatures2D(tf.keras.layers.Layer):
    """
    Implementation of coordinate features form
    https://arxiv.org/pdf/1807.03247.pdf
    """
    def __init__(self, name: str="add_coords_2d"):
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        We add x and y coordinate features to channels.
        The values varies form -1 to 1.

        :param inputs: tf.Tensor, shape = [batch, height, width, channels]
        :return: tf.Tensor, shape = [batch, height, width, channels + 2]
        """
        input_shape = tf.shape(inputs)
        batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]

        x_coords = tf.tile(
            tf.reshape(
                tf.cast(
                    tf.linspace(start=-1, stop=1, num=height),
                    inputs.dtype
                ),
                shape=[1, height, 1, 1]),
            [batch_size, 1, width, 1]
        )
        y_coords = tf.tile(
            tf.reshape(
                tf.cast(
                    tf.linspace(start=-1, stop=1, num=width),
                    inputs.dtype
                ),
                shape=[1, 1, width, 1]
            ),
            [batch_size, height, 1, 1]
        )

        output = tf.concat([inputs, x_coords, y_coords], axis=3)

        return output
