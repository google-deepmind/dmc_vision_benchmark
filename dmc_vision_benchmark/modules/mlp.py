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

"""Simple MLP model."""

from collections.abc import Callable
from typing import Sequence

from flax import linen as nn
from kauldron.typing import Array
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member

ActivationFn = Callable[[Array], Array]


class MLP(nn.Module):
  """MLP.

  Attributes:
    out_dim: output dimnesion.
    features: list of number of features in each layer.
    activation: activation fn to be used in each layer.
    out_activation: activation applied to output before return.
    kernel_init: initializer for kernel.
    bias_init: initializer for bias.
  """

  out_dim: int
  features: Sequence[int] = (256,)
  activation: ActivationFn = nn.relu
  out_activation: ActivationFn | None = None
  kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
  bias_init: nn.initializers.Initializer = nn.initializers.zeros

  @typechecked
  @nn.compact
  def __call__(self, x: Float['*b d']) -> Float['*b {self.out_dim}']:
    for i, feat in enumerate(self.features):
      x = nn.Dense(
          features=feat,
          name=f'dense_{i}',
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
      )(x)
      x = self.activation(x)

    n_layers = len(self.features)
    x = nn.Dense(
        features=self.out_dim,
        name=f'dense_{n_layers}',
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
    )(x)
    if self.out_activation is not None:
      x = self.out_activation(x)
    return x
