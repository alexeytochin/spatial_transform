from __future__ import annotations
from typing import Tuple

import tensorflow as tf


class SpatialGrid(object):
    """ 2D grid for spatial transform """
    def __init__(self, shape_out: Tuple[int, int], coordinates_tensor: tf.Tensor):
        super().__init__()
        self._shape_out = shape_out
        self._coordinates_tensor = coordinates_tensor

    @property
    def coordinates_tensor(self) -> tf.Tensor:
        """ tf.Tensor, shape = [batch, 2, height * width], dtype = tf.float32 """
        return self._coordinates_tensor

    @property
    def shape_out(self) -> Tuple[int, int]:
        """ Output image shape """
        return self._shape_out

    @property
    def height(self) -> int:
        """ Output image height """
        return self.shape_out[0]

    @property
    def width(self) -> int:
        """ Output image width """
        return self.shape_out[1]


class FlatGrid(SpatialGrid):
    """ Trivial 2D grid for spatial transform. """
    def __init__(self, shape_out: Tuple[int, int], batch_size):
        super().__init__(
            shape_out = shape_out,
            coordinates_tensor = self._get_coordinates_tensor(batch_size=batch_size, shape_out=shape_out)
        )

    @classmethod
    def _get_coordinates_tensor(cls, batch_size: tf.Tensor, shape_out: Tuple[int, int]) -> tf.Tensor:
        """
        :param batch_size: shape = [], dtype = tf.int32
        :return: shape = [batch, 2, height * width], dtype = tf.float32
        """
        height, width = shape_out
        x = tf.linspace(-1, 1, width)
        y = tf.linspace(-1, 1, height)

        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        coordinates = tf.stack([xx, yy])
        coordinates = tf.expand_dims(coordinates, axis=0)
        coordinates = tf.tile(coordinates, [batch_size, 1, 1])
        coordinates = tf.cast(coordinates, dtype=tf.float32)

        return coordinates