from __future__ import annotations
from abc import ABC, abstractmethod
import tensorflow as tf

from spatial_transform.spatail_grid import SpatialGrid


class SpatialInterpolator(ABC):
    """
    Base class for "sampler" object in https://arxiv.org/pdf/1506.02025.pdf terminology
    """
    @abstractmethod
    def interpolate(
            self,
            image: tf.Tensor,
            grid: SpatialGrid,
    ) -> tf.Tensor:
        """
        Interpolate given image along give grid.
        Sized of the output image are specified in :grid parameter.

        :param image: tf.Tensor, shape = [batch, width, heights, channels]
        :param grid: Spatial Grid instance
        :transform_params: SpatialTransformParamsType instance
        :return: tf.Tensor, shape = [batch, width, transformed_grid.heights, transformed_grid.channels]
        """
        raise NotImplementedError()


class BilinearInterpolator(SpatialInterpolator):
    """
    Bilinear transform implementation originally based on
    https://towardsdatascience.com/implementing-spatial-transformer-network-stn-in-tensorflow-bf0dc5055cd5
    """
    def interpolate(
            self,
            image: tf.Tensor,
            grid: SpatialGrid,
    ) -> tf.Tensor:
        image_shape = tf.shape(image)
        height_in, width_in = image_shape[1], image_shape[2]

        coordinates_tensor = tf.transpose(grid.coordinates_tensor, perm=[0, 2, 1])
        coordinates_tensor = tf.reshape(coordinates_tensor, [-1, grid.height, grid.width, 2])

        x_transformed = coordinates_tensor[:, :, :, 0]
        y_transformed = coordinates_tensor[:, :, :, 1]

        x = ((x_transformed + 1.) * tf.cast(width_in, dtype=tf.float32)) * 0.5
        y = ((y_transformed + 1.) * tf.cast(height_in, dtype=tf.float32)) * 0.5

        x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, width_in - 1)
        x1 = tf.clip_by_value(x1, 0, width_in - 1)
        y0 = tf.clip_by_value(y0, 0, height_in - 1)
        y1 = tf.clip_by_value(y1, 0, height_in - 1)
        x = tf.clip_by_value(x, 0, tf.cast(width_in, dtype=tf.float32) - 1.0)
        y = tf.clip_by_value(y, 0, tf.cast(height_in, dtype=tf.float32) - 1)

        Ia = self.advance_indexing(image, x0, y0, grid)
        Ib = self.advance_indexing(image, x0, y1, grid)
        Ic = self.advance_indexing(image, x1, y0, grid)
        Id = self.advance_indexing(image, x1, y1, grid)

        x0 = tf.cast(x0, dtype=tf.float32)
        x1 = tf.cast(x1, dtype=tf.float32)
        y0 = tf.cast(y0, dtype=tf.float32)
        y1 = tf.cast(y1, dtype=tf.float32)

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        return tf.math.add_n([wa * Ia + wb * Ib + wc * Ic + wd * Id])

    def advance_indexing(self, inputs, x, y, transformed_grid: SpatialGrid):
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, transformed_grid.height, transformed_grid.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)
