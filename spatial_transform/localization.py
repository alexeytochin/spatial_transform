from abc import ABC, abstractmethod
from typing import Generic
import tensorflow as tf

from spatial_transform.coord_features import AddCoordFeatures2D
from spatial_transform.spatial_transforms import SpatialTransformParamsType


class LocalizationLayer(Generic[SpatialTransformParamsType], tf.keras.layers.Layer, ABC):
    """
    Interface for localization layer from
    https://arxiv.org/pdf/1506.02025.pdf .
    The output of self.call method is a SpatialTransformParams object
    containing parameters of specified spatial transform.
    """
    def __init__(self, spatial_transform_params_cls: SpatialTransformParamsType, **kwargs):
        """
        :param spatial_transform_params_cls: type of SpatialTransformParams containing in particular information of
        spatial transform parameters number.
        """
        super().__init__()
        self._spatial_transform_params_cls = spatial_transform_params_cls

    @abstractmethod
    def call(self, inputs: tf.Tensor, training: bool) -> SpatialTransformParamsType:
        """
        :param inputs: tf.Tensor, shape = [batch, height, width, channels], dtype = tf.float32
        :return: SpatialTransformParamsType instance
        """
        raise NotImplementedError()

    @property
    def num_spatial_transform_params(self) -> int:
        return self._spatial_transform_params_cls.get_num_params()


class StandardConvolutionalLocalizationLayer(LocalizationLayer[SpatialTransformParamsType]):
    """
    Implementation of a simple convolutional localization layer.
    """
    def __init__(
            self,
            spatial_transform_params_cls: SpatialTransformParamsType,
            init_scale: float = 1.,
            **kwargs
    ):
        super().__init__(spatial_transform_params_cls=spatial_transform_params_cls, **kwargs)
        init_param_raw_tensor = self._spatial_transform_params_cls.get_init_raw_tensor(scale=init_scale)

        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv1 = tf.keras.layers.Conv2D(16, [5, 5], activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(16, [5, 5], activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(16, activation='relu')
        self.fc2 = tf.keras.layers.Dense(
            self._spatial_transform_params_cls.get_num_params(),
            activation = None,
            bias_initializer = tf.keras.initializers.constant(init_param_raw_tensor),
            kernel_initializer = 'zeros',
        )

    def call(self, inputs: tf.Tensor, training: bool) -> SpatialTransformParamsType:
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta_tensor = self.fc2(x)
        return self._spatial_transform_params_cls.from_raw_tensor(raw_tensor=theta_tensor)


class CoordConvLocalizationLayer(LocalizationLayer[SpatialTransformParamsType]):
    """
    Implementation of a simple convolutional localization layer with coordinates feature,
    see https://arxiv.org/pdf/1807.03247.pdf
    """
    def __init__(self, spatial_transform_params_cls: SpatialTransformParamsType, init_scale: float = 1., **kwargs):
        super().__init__(spatial_transform_params_cls=spatial_transform_params_cls, **kwargs)
        init_param_raw_tensor = self._spatial_transform_params_cls.get_init_raw_tensor(scale=init_scale)

        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv1 = tf.keras.layers.Conv2D(16, [5, 5], activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(16, [5, 5], activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(16, activation='relu')
        self.fc2 = tf.keras.layers.Dense(
            self._spatial_transform_params_cls.get_num_params(),
            activation = None,
            bias_initializer = tf.keras.initializers.constant(init_param_raw_tensor),
            kernel_initializer = 'zeros',
        )

    def call(self, inputs: tf.Tensor, training: bool) -> SpatialTransformParamsType:
        x = inputs
        x = AddCoordFeatures2D()(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = AddCoordFeatures2D()(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = AddCoordFeatures2D()(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta_tensor = self.fc2(x)
        return self._spatial_transform_params_cls.from_raw_tensor(raw_tensor=theta_tensor)


class LargeLocalizationLayer(LocalizationLayer[SpatialTransformParamsType]):
    """
    Implementation of a simple convolutional localization layer with coordinates feature
    that is larger then CoordConvLocalizationLayer
    """
    def __init__(self, spatial_transform_params_cls: SpatialTransformParamsType, init_scale: float = 1., **kwargs):
        super().__init__(spatial_transform_params_cls=spatial_transform_params_cls, **kwargs)
        init_param_raw_tensor = self._spatial_transform_params_cls.get_init_raw_tensor(scale=init_scale)

        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv1 = tf.keras.layers.Conv2D(64, [7, 7], activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(32, [5, 5], activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(
            self._spatial_transform_params_cls.get_num_params(),
            activation = None,
            bias_initializer = tf.keras.initializers.constant(init_param_raw_tensor),
            kernel_initializer = 'zeros',
        )

    def call(self, inputs: tf.Tensor, training: bool) -> SpatialTransformParamsType:
        x = inputs
        x = AddCoordFeatures2D()(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = AddCoordFeatures2D()(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = AddCoordFeatures2D()(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta_tensor = self.fc2(x)
        return self._spatial_transform_params_cls.from_raw_tensor(raw_tensor=theta_tensor)