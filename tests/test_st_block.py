import os
import unittest
import tensorflow as tf
from tensorflow.python.keras.layers import LayerNormalization

from spatial_transform.coord_features import AddCoordFeatures2D
from spatial_transform.interpolation import BilinearInterpolator
from spatial_transform.localization import StandardConvolutionalLocalizationLayer
from spatial_transform.spatial_transforms import AffineTransform
from spatial_transform.st_blocks import SimpleSpatialTransformBlock


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestSTBlock(unittest.TestCase):
    def test_st_block_forward(self):
        batch_size = 1
        image_in = tf.zeros(shape=[batch_size, 16, 16, 1])
        spatial_transform = AffineTransform()
        st_block = SimpleSpatialTransformBlock(
            localization_layer = StandardConvolutionalLocalizationLayer(
                spatial_transform_params_cls = spatial_transform.param_type,
                init_scale = 2
            ),
            spatial_transform =spatial_transform,
            interpolator= BilinearInterpolator(),
            shape_out = (8, 8),
        )

        image_out = st_block(inputs=image_in)

        self.assertTrue(image_out.shape, (batch_size, 8, 8, 1))