from abc import ABC, abstractmethod
from typing import Tuple
import tensorflow as tf

from spatial_transform.spatail_grid import FlatGrid
from spatial_transform.interpolation import SpatialInterpolator
from spatial_transform.layers import TensorToTensorLayer, IdentityLayer
from spatial_transform.localization import LocalizationLayer
from spatial_transform.spatial_transforms import SpatialTransformType


class SpatialTransformBlock(TensorToTensorLayer, ABC):
    """
    Interface for Spatial Transform Block
    """
    def __init__(
            self,
            shape_out: Tuple[int, int],
            **kwargs
    ):
        """
        :param shape_out: output image shape
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._shape_out = shape_out

    @abstractmethod
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """
        :param inputs:  tf.Tensor, shape = [batch, height, width, channels], dtype = tf.float32
        :param training: bool
        :return: tf.Tensor, shape = [batch, :param shape_out[0], :param shape_out[1], channels], dtype = tf.float32
        """
        raise NotImplementedError()


class CustomSpatialTransformBlock(SpatialTransformBlock):
    """
    STN-CX block implementation

       +->- [ conv layers ] ->- [ localization_layer ] ->-+
       |                                                  |
    ->-+--------------------->---------------- [ interpolation_layer ] ->-

    """
    def __init__(
            self,
            localization_layer: LocalizationLayer[SpatialTransformType],
            spatial_transform: SpatialTransformType,
            interpolator: SpatialInterpolator,
            conv_layers: TensorToTensorLayer,
            shape_out: Tuple[int, int],
            **kwargs
    ):
        """
        :param localization_layer: Localisation layer parameterized with :param spatial_transform
        :param spatial_transform: Spatial transform type
        :param interpolator: Interpolation type
        :param conv_layers: layer followed by :param localization layer
        :param shape_out: output image shape
        """
        super().__init__(shape_out=shape_out, **kwargs)
        self._localization_layer = localization_layer
        self._spatial_transform = spatial_transform
        self._interpolator = interpolator
        self._conv_layers = conv_layers

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """
        :param inputs: tf.Tensor, shape = [batch, height, width, channels], dtype = tf.float32
        :param training: bool
        :return: tf.Tensor, shape = [batch, height, width, channels], dtype = tf.float32
        """
        batch_size = tf.shape(inputs)[0]
        features = self._conv_layers(inputs=inputs, training=training)
        transformation_params = self.localization_layer(inputs=features, training=training)
        grid = FlatGrid(shape_out=self._shape_out, batch_size=batch_size)
        transformed_grid = \
            self._spatial_transform.transform_grid(transformation_params=transformation_params, grid=grid)
        output = self.interpolator.interpolate(image=inputs, grid=transformed_grid)

        return output

    @property
    def localization_layer(self) -> LocalizationLayer:
        return self._localization_layer

    @property
    def interpolator(self) -> SpatialInterpolator:
        return self._interpolator

    @property
    def conv_layers(self) -> (tf.keras.layers.Layer, TensorToTensorLayer):
        return self._conv_layers


class SimpleSpatialTransformBlock(CustomSpatialTransformBlock):
    """
    STN-C0 block implementation

       +->-[ localization_layer ]->-+
       |                            |
    ->-+------->---------[ interpolation_layer ]->-

    """
    def __init__(
            self,
            localization_layer: LocalizationLayer[SpatialTransformType],
            spatial_transform: SpatialTransformType,
            interpolator: SpatialInterpolator,
            shape_out: Tuple[int, int],
            **kwargs
    ):
        super().__init__(
            localization_layer = localization_layer,
            spatial_transform = spatial_transform,
            interpolator = interpolator,
            conv_layers = IdentityLayer(),
            shape_out = shape_out,
            **kwargs
        )
