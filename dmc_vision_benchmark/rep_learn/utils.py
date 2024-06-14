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

"""Util functions."""

from typing import cast, Dict

import chex
import jax.numpy as jnp
import orbax.checkpoint


def load_model_params(
    save_to_folder: str,
    pretrained_model_id: int,
) -> Dict[str, jnp.ndarray]:
  """Load the model params from a pretrained model."""
  load_checkpoint_workdir = f"{save_to_folder}/checkpoint/{pretrained_model_id}" # pylint: disable=line-too-long

  # Load the checkpoint_manager
  checkpoint_manager_loaded = orbax.checkpoint.CheckpointManager(
      directory=load_checkpoint_workdir,
      checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
      options=orbax.checkpoint.CheckpointManagerOptions(create=False),
  )
  step = checkpoint_manager_loaded.latest_step()
  restored = checkpoint_manager_loaded.restore(step)
  model_params_loaded = restored["model_params"]
  return model_params_loaded


def square_l2_dist(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
) -> float:
  """Square L2 distance."""
  chex.assert_equal_shape([predictions, targets])

  square_l2_dists = jnp.square(predictions - targets)
  return cast(float, jnp.mean(square_l2_dists))
