# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CNN model."""

from collections.abc import Callable, Sequence
from typing import Tuple

import einops
from flax import linen as nn
from kauldron.typing import Array, Float, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class CNN(nn.Module):
  """A simple CNN model that returns image features.

  Each layer is conv -> relu -> avg pool.
  Number of layers is specified by conv_channels. Kernel, stride and pooling
  same across all layers.

  Attributes:
    conv_channels: List of integers giving the number of channels in each
      convolutional layer.
    kernel_sizes: List of tuples giving kernel size for convolutions.
    strides: List of tuples giving stride for convolution.
    pool_window_size: Size of pooling along both dims.
    pool_stride_size: Size of stride along both dims.
    downsample: downsampling method.
    kernel_init: initializer for kernel.
    bias_init: initializer for bias.
    activation_fn: activation function after conv layer.
  """

  conv_channels: Sequence[int] = (32, 32, 32, 32)
  kernel_sizes: Sequence[Tuple[int, int]] = ((3, 3), (3, 3), (3, 3), (3, 3))
  strides: Sequence[Tuple[int, int]] = ((1, 1), (1, 1), (1, 1), (1, 1))
  pool_window_size: int = 2
  pool_stride_size: int = 2
  downsample: str | None = 'avg'
  use_deconv_layers: bool = False
  upsample: Sequence[int] | None = None
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  bias_init: nn.initializers.Initializer = nn.initializers.zeros
  activation_fn: Callable[[Array], Array] = nn.relu
  activation_last_layer: bool = True

  @typechecked
  @nn.compact
  def __call__(
      self, inputs: UInt8['*b h w c'] | Float['*b h2 w2 c2']
  ) -> Float['*B h w c']:
    num_layers = len(self.conv_channels)
    if len(self.kernel_sizes) != num_layers:
      raise ValueError(
          f'Number of kernel sizes {len(self.kernel_sizes)} does not match'
          f' number of layers {num_layers}'
      )
    if len(self.strides) != num_layers:
      raise ValueError(
          f'Length of strides {len(self.strides)} does not match number of'
          f' layers {num_layers}'
      )

    window_shape = (self.pool_window_size, self.pool_window_size)
    strides = (self.pool_stride_size, self.pool_stride_size)
    x = inputs
    for i in range(num_layers):
      if not self.use_deconv_layers:
        x = nn.Conv(
            features=self.conv_channels[i],
            kernel_size=self.kernel_sizes[i],
            strides=self.strides[i],
            name=f'conv_{i}',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            padding='SAME',
        )(x)
      else:
        x = nn.ConvTranspose(
            features=self.conv_channels[i],
            kernel_size=self.kernel_sizes[i],
            strides=self.strides[i],
            name=f'conv_{i}',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            padding='SAME',
        )(x)

      # Check whether we apply the activation function on the last layer
      if i != num_layers - 1 or self.activation_last_layer:
        x = self.activation_fn(x)

      if self.downsample == 'avg':
        x = nn.avg_pool(x, window_shape=window_shape, strides=strides)
      if self.downsample == 'max':
        x = nn.max_pool(x, window_shape=window_shape, strides=strides)
      if self.upsample is not None:
        x = einops.repeat(
            x,
            '... h w c -> ... (h rh) (w rw) c',
            rh=self.upsample[i],
            rw=self.upsample[i],
        )

    return x


def get_vd4rl_cnn() -> CNN:
  """Returns CNN from the VD4RL paper.

  Trainer:
  https://github.com/conglu1997/v-d4rl/blob/9dcca0984faaddaaedce6975f4d9e6ef5977ad6f/drqbc/bcagent.py#L14
  Models:
  https://github.com/conglu1997/v-d4rl/blob/9dcca0984faaddaaedce6975f4d9e6ef5977ad6f/drqbc/drqv2.py#L56
  """
  return CNN(
      conv_channels=(32, 32, 32, 32),
      kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3)),
      strides=((2, 2), (1, 1), (1, 1), (1, 1)),
      downsample=None,
      kernel_init=nn.initializers.orthogonal(),
  )


def get_idm_cnn() -> CNN:
  """Returns CNN from the IDM paper.

  See https://arxiv.org/pdf/2305.16985.pdf, Appendix C.2
  """
  return CNN(
      conv_channels=(32, 64, 128, 256),
      kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3)),
      strides=((2, 2), (2, 2), (1, 1), (1, 1)),
      downsample=None,
      kernel_init=nn.initializers.orthogonal(),
      activation_fn=nn.gelu,
  )


def get_decoder_cnn(n_frames: int, n_cameras: int) -> CNN:  # pylint: disable=invalid-name
  """Returns a CNN decoder.

  Inspired from https://arxiv.org/pdf/1910.01741#page=12

  Args:
    n_frames: number of stacked frames
    n_cameras: number of cameras used
  """
  out_channels = 3 * n_cameras * n_frames
  return CNN(
      conv_channels=(128, 64, 32, out_channels),
      kernel_sizes=((3, 3), (3, 3), (3, 3), (3, 3)),
      strides=((1, 1), (1, 1), (2, 2), (2, 2)),
      use_deconv_layers=True,
      downsample=None,
      kernel_init=nn.initializers.orthogonal(),
      activation_fn=nn.gelu,
      activation_last_layer=False,  # no activation for the last layer
  )
