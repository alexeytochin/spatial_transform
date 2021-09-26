import os
import unittest
import tensorflow as tf

from spatial_transform.interpolation import BilinearInterpolator
from spatial_transform.spatail_grid import FlatGrid
from spatial_transform.spatial_transforms import AffineTransformParams, AffineTransform


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestTransform(unittest.TestCase):
    def test_bilinear_interpolation(self):
        tf.random.set_seed(0)
        interpolator = BilinearInterpolator()
        batch_size = 1
        image_in = tf.reshape(tf.cast(tf.range(25), dtype=tf.float32), shape=[batch_size, 5, 5, 1])
        transformation_params = AffineTransformParams(
            translation = tf.zeros(shape=[batch_size, 2]),
            rotation = tf.eye(2, batch_shape=[batch_size]),
        )
        spatial_transform = AffineTransform()
        grid = FlatGrid(shape_out=(5, 5), batch_size=batch_size)
        transformed_grid = \
            spatial_transform.transform_grid(transformation_params=transformation_params, grid=grid)

        image_out = interpolator.interpolate(image=image_in, grid=transformed_grid)
