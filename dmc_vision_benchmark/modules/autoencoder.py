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

"""Autoencoder model."""

from typing import Union

from flax import linen as nn
import jax
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class AutoEncoder(nn.Module):
  """Implements an autoencoder."""

  obs_encoder: nn.Module
  obs_decoder: nn.Module | None
  state_decoder: nn.Module | None
  stop_gradient_obs_encoder: bool = False

  @typechecked
  @nn.compact
  def __call__(
      self,
      obs: Float["*b j h w d"],
  ) -> dict[str, Union[Float["*b j h w d"], Float["*b n"]] | None]:
    # Encode images
    obs_encoded = self.obs_encoder(obs)
    if self.stop_gradient_obs_encoder:
      obs_encoded = jax.lax.stop_gradient(obs_encoded)

    # Decode states
    states_decoded = None
    if self.state_decoder is not None:
      states_decoded = self.state_decoder(obs_encoded)

    # Decode observations
    obs_decoded = None
    if self.obs_decoder is not None:
      obs_decoded = self.obs_decoder(obs_encoded)
    return {"states_decoded": states_decoded, "obs_decoded": obs_decoded}
