from abc import abstractmethod, ABC
from typing import List
import tensorflow as tf


class TensorToTensorLayer(tf.keras.layers.Layer, ABC):
    """
    Interface for tf.Tensor -> tf.Tensor layer
    """
    @abstractmethod
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        raise NotImplementedError()


class IdentityLayer(TensorToTensorLayer):
    """
    Identity layer on a single tensor
    """
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        return inputs


class LayerChain(tf.keras.layers.Layer):
    """
    Simple chain of given layers
    """
    def __init__(
            self,
            layers: List[TensorToTensorLayer],
            **kwargs
    ):
        super().__init__()
        self._layers = layers

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(inputs=x, training=training)
        return x

    @property
    def layers(self) -> List[TensorToTensorLayer]:
        return self._layers


class RepeatWithSharedWeights(LayerChain):
    """
    Multiplies given layer into a chain of layer with shared weights.
    """
    def __init__(
            self,
            num_repetitions: int,
            layer: TensorToTensorLayer,
            **kwargs
    ):
        super().__init__(layers = [layer] * num_repetitions, **kwargs)