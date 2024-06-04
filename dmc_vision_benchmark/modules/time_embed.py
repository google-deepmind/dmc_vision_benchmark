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

"""Timestep embeddings."""

from __future__ import annotations
from flax import linen as nn
import jax.numpy as jnp
from kauldron.typing import Dtype, Float, Initializer, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class TimestepEmbedding(nn.Module):
  """Learned positional embeddings for timesteps.

  This is not the encoding associated with the context window, but instead
  where the model is in the epsiode, which isn't necessarily the same.

  Attributes:
    max_timestep: Maximum number of timestemps
    emb_dim:  Dimension of the embedding.
    emb_init: Initializer for the position embeddings.
    dtype: Dtype of the position embedding. Default to float32.
  """

  max_timestep: int
  emb_dim: int
  dtype: Dtype = jnp.float32
  emb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @typechecked
  @nn.compact
  def __call__(self, timestep: Int["*b t"]) -> Float["*b t {self.emb_dim}"]:
    """Return learned timestep embeddings.

    Args:
      timestep: Which timesteps to look up.

    Returns:
      Learned position embeddings broadcast to given shape.
    """
    pe = self.param(
        "embeddings",
        self.emb_init,
        (self.max_timestep, self.emb_dim),
        self.dtype,
    )
    return pe[timestep]
