from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type
import tensorflow as tf
from dataclasses import dataclass, field

from spatial_transform.spatail_grid import SpatialGrid


SpatialTransformParamsType = TypeVar('SpatialTransformParamsType', bound='SpatialTransformParams')


@dataclass
class SpatialTransformParams(ABC):
    num_params: int = field(init=False)

    def __post_init__(self):
        self.num_params = self.get_num_params()

    @classmethod
    @abstractmethod
    def from_raw_tensor(cls: SpatialTransformParamsType, raw_tensor: tf.Tensor) -> SpatialTransformParamsType:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_num_params(self) -> int:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_init_raw_tensor(cls, scale: float) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def batch_size(self) -> tf.Tensor:
        raise NotImplementedError()


SpatialTransformType = TypeVar('SpatialTransformType', bound='SpatialTransform')


class SpatialTransform(Generic[SpatialTransformParamsType], ABC):
    """
    Interface for Spatial Transforms
    """
    param_type: Type[SpatialTransformParamsType]

    @abstractmethod
    def transform_grid(
            self,
            transformation_params: SpatialTransformParamsType,
            grid: SpatialGrid,
    ) -> SpatialGrid:
        """
        Transforms given grid according to this spatial transform.
        :param transformation_params: transform params instance
        :param grid: tf.Tensor, shape = [batch, 2, width * heights], dtype = tf.float32
        :return: tf.Tensor, shape = [batch, 2, width * heights], dtype = tf.float32
        """
        raise NotImplementedError()


@dataclass
class AffineTransformParams(SpatialTransformParams):

    translation: tf.Tensor  # shape = [batch_size, 2]
    rotation: tf.Tensor       # shape = [batch_size, 2, 2]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rotation.shape.assert_is_compatible_with(tf.TensorShape([None, 2, 2]))
        self.translation.shape.assert_is_compatible_with(tf.TensorShape([None, 2]))
        assert self.rotation.shape[0] == self.translation.shape[0]

    @classmethod
    def from_raw_tensor(cls: AffineTransformParams, raw_tensor: tf.Tensor) -> AffineTransformParams:
        raw_tensor.shape.assert_is_compatible_with(tf.TensorShape([None, cls.get_num_params()]))
        return AffineTransformParams(
            translation = tf.reshape(raw_tensor[:,:2], shape=[-1,2]),
            rotation = tf.reshape(raw_tensor[:,2:6], shape=[-1,2,2]),
        )

    @classmethod
    def get_num_params(cls) -> int:
        return 6

    @classmethod
    def get_init_raw_tensor(cls, scale: float) -> int:
        return tf.constant([0.0, 0.0, scale, 0.0, 0.0, scale], dtype=tf.float32)

    @property
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self.translation)[0]


class AffineTransform(SpatialTransform[AffineTransformParams]):
    """
    Affine spatial transform implementation
        x_i -> translation_i + sum_j rotation_ij * x_j,
    """
    param_type = AffineTransformParams

    def transform_grid(
            self,
            transformation_params: AffineTransformParams,
            grid: SpatialGrid,
    ) -> SpatialGrid:
        transformed_coordinates_tensor = \
            tf.matmul(transformation_params.rotation, grid.coordinates_tensor) \
            + tf.expand_dims(transformation_params.translation, axis=2)
        return SpatialGrid(
            shape_out = grid.shape_out,
            coordinates_tensor = transformed_coordinates_tensor
        )


@dataclass
class QuadraticTransformParams(SpatialTransformParams):
    translation: tf.Tensor  # shape = [batch_size, 2]
    rotation: tf.Tensor     # shape = [batch_size, 2, 2]
    quadratic: tf.Tensor    # shape = [batch_size, 2, 2, 2]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.translation.shape.assert_is_compatible_with(tf.TensorShape([None, 2]))
        self.rotation.shape.assert_is_compatible_with(tf.TensorShape([None, 2, 2]))
        self.quadratic.shape.assert_is_compatible_with(tf.TensorShape([None, 2, 2, 2]))

    @classmethod
    def from_raw_tensor(cls: QuadraticTransformParams, raw_tensor: tf.Tensor) -> QuadraticTransformParams:
        raw_tensor.shape.assert_is_compatible_with(tf.TensorShape([None, cls.get_num_params()]))
        return QuadraticTransformParams(
            translation = tf.reshape(raw_tensor[:,:2], shape=[-1,2]),
            rotation = tf.reshape(raw_tensor[:,2:6], shape=[-1,2,2]),
            quadratic = tf.reshape(raw_tensor[:,6:14], shape=[-1,2,2,2]),
        )

    @classmethod
    def get_num_params(cls) -> int:
        return 14

    @classmethod
    def get_init_raw_tensor(cls, scale: float) -> int:
        return tf.constant([0.0, 0.0, scale, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)

    @property
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self.translation)[0]


class QuadraticTransform(SpatialTransform[QuadraticTransformParams]):
    """
    Quadratic spatial transform implementation
        x_i -> translation_i + sum_j rotation_ij * x_j,
        x_i -> translation_i + sum_j rotation_ij * x_j + sum_jk quadratic_ijk * x_j * x_k
    """
    param_type = QuadraticTransformParams

    def transform_grid(
            self,
            transformation_params: QuadraticTransformParams,
            grid: SpatialGrid,
    ) -> SpatialGrid:
        transformed_coordinates_tensor = \
            tf.einsum("bkij,bjn,bin->bkn", transformation_params.quadratic, grid.coordinates_tensor, grid.coordinates_tensor) \
            + tf.matmul(transformation_params.rotation, grid.coordinates_tensor) \
            + tf.expand_dims(transformation_params.translation, axis=2)

        return SpatialGrid(
            shape_out = grid.shape_out,
            coordinates_tensor = transformed_coordinates_tensor
        )