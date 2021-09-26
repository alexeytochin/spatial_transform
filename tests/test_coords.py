import os
import unittest
import tensorflow as tf

from spatial_transform.coord_features import AddCoordFeatures2D


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestCoordFeatures(unittest.TestCase):
    def test_coord_features(self):
        add_coords_layer = AddCoordFeatures2D()
        batch_size = 1
        height = 7
        width = 6
        num_channels = 5
        input = tf.zeros(shape=[batch_size, height, width, num_channels])

        output = add_coords_layer(input)

        self.assertEqual(output.shape, (batch_size, height, width, num_channels + 2))