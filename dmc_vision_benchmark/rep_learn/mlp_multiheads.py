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

"""Simple inverse model based on image encoder and latent inverse model."""

from collections.abc import Mapping, Sequence
from typing import Union

import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron.typing import Array, Float, Int, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member

from dmc_vision_benchmark.data import dmc_vb_info


class MLPMultiHeads(nn.Module):
  """An MLP with action prediction, inverse dynamics, state prediction and forward model heads.

  Optionally predicts:
    Next action from current observation
    Next action from current and future observation (inverse dynamics model)
    State from current observation.
    Future observation embedding from current observation and action
  """

  obs_encoder: nn.Module
  next_action_predictor: nn.Module | None
  inverse_dynamics_predictor: nn.Module | None
  state_predictor: nn.Module | None
  idm_step_encoder: nn.Module | None
  action_encoder: nn.Module | None
  latent_forward_model: nn.Module | None
  obs_decoder: nn.Module | None
  encode_dim: int

  # For online eval
  domain_name: str
  cameras: Sequence[str]
  obs_vrange: tuple[float, float] = (0, 1)
  pass_state: bool = False

  stop_gradient_obs_encoder: bool = False

  def __hash__(self):
    return id(self)

  @typechecked
  @nn.compact
  def __call__(
      self,
      obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']],
      future_obs: Float['*b n_frames h w c'] | None = None,
      idm_step: Int['*b 1'] | None = None,
      actions: Float['*b 1 n'] | None = None,
  ) -> dict[str, Float['...'] | None]:
    """Runs the MLP."""

    # Step 1. Encode observation
    obs_encoded = self.obs_encoder(obs)
    if self.stop_gradient_obs_encoder:
      obs_encoded = jax.lax.stop_gradient(obs_encoded)
    assert obs_encoded.shape == Shape(f'*b {self.encode_dim}')

    # Step 2. Next action prediction
    pred_actions = None
    if self.next_action_predictor is not None:
      pred_actions = self.next_action_predictor(obs_encoded)

    # Step 3. Inverse dynamics prediction
    inv_model_actions = None
    if future_obs is not None and self.inverse_dynamics_predictor is not None:
      # First encode the future observations
      future_obs_encoded = self.obs_encoder(future_obs)
      if self.stop_gradient_obs_encoder:
        future_obs_encoded = jax.lax.stop_gradient(future_obs_encoded)

      to_concat = [obs_encoded, future_obs_encoded]

      # Second encode the timestep
      if self.idm_step_encoder is not None:
        assert idm_step is not None
        idm_step_encoded = self.idm_step_encoder(idm_step)[:, 0]
        to_concat.append(idm_step_encoded)

      # Concatenate the embeddings
      concat_embeddings = jnp.concatenate(to_concat, axis=-1)
      inv_model_actions = self.inverse_dynamics_predictor(concat_embeddings)

    # Step 4. Predict states from encoded observation
    pred_states = None
    if self.state_predictor is not None:
      pred_states = self.state_predictor(obs_encoded)

    # Step 5. Forward model prediction
    future_obs_encoded = None
    if (
        actions is not None
        and self.latent_forward_model is not None
        and self.action_encoder is not None
    ):
      # Encode the action
      action_encoded = self.action_encoder(actions[:, 0])

      # Apply latent forward model conditioned on action
      concat_embeddings = jnp.concatenate(
          [obs_encoded, action_encoded], axis=-1
      )
      future_obs_encoded = self.latent_forward_model(concat_embeddings)

    # Step 6. Observation reconstruction
    obs_decoded = None
    if self.obs_decoder is not None:
      obs_decoded = self.obs_decoder(obs_encoded)

    return {
        'pred_actions': pred_actions,
        'inv_model_actions': inv_model_actions,
        'pred_states': pred_states,
        'pred_future_obs_encoded': future_obs_encoded,
        'obs_decoded': obs_decoded,
    }

  def get_future_obs_encoded(
      self,
      future_obs: Float['*b n_frames h w c'],
  ) -> dict[str, Float['...'] | None]:
    """Returns the future obs embeddings. Should used the teacher params."""
    return {'future_obs_encoded': self.obs_encoder(future_obs)}

  def plan_actions(
      self,
      step_counter: int,
      observation_history: Array['...'] | Mapping[str, Array['...']],
      action_history: Array['...'] | Mapping[str, Array['...']] | None = None,
      reward_history: Float['t'] | None = None,
  ) -> Mapping[str, Array['...']]:
    """Plan a chunk of actions given history and global context."""

    del step_counter, reward_history, action_history

    if self.pass_state:
      obs = get_state_input(observation_history, self.domain_name)
      obs = normalize_state(obs, self.domain_name)
    else:
      obs = get_rgb_normed(
          observation_history=observation_history,
          vrange=self.obs_vrange,
          cameras=self.cameras,
      )

    model_output = self(obs=obs)

    return {'actions': model_output['pred_actions']}


def get_rgb_normed(
    observation_history: Mapping[str, Array['...']],
    vrange: tuple[float, float],
    cameras: Sequence[str],
) -> Array['...']:
  """Returns the normalized rgb observations."""
  # Process RGB, take last timestep only as framestacking is done by the
  # environment.
  rgbs = [observation_history[camera][-1] for camera in cameras]

  # Add extra batch dimension
  if rgbs[0].ndim == 4:
    rgb = jnp.concatenate(rgbs, axis=-2)  # concatenate along channel dim
    # Stack frames along channel dim
    rgb = einops.rearrange(rgb, 'h w c n -> n h w c')
  else:
    rgb = jnp.stack(rgbs, axis=-1)  # concatenate along channel dim

  rgb = rgb[None]
  rgb_normed = rgb.astype(jnp.float32) / 255.0
  rgb_normed = rgb_normed * (vrange[1] - vrange[0]) + vrange[0]
  return rgb_normed


def normalize_state(state: Array['...'], domain_name: str) -> Array['...']:
  """Normalize state."""
  mean = jnp.array(dmc_vb_info.get_state_mean(domain_name))
  std = jnp.array(dmc_vb_info.get_state_std(domain_name))
  return (state - mean) / std


def get_state_input(
    observation_history: Mapping[str, Array['...']],
    domain_name: str,
) -> Array['...']:
  """Returns the states."""

  # Process states
  state_fields = dmc_vb_info.get_state_fields(domain_name)
  fields = []
  for state_field in state_fields:
    field = observation_history[state_field][-1]
    # Add an extra dimension to scalar fields
    if field.ndim == 1:
      field = [field]
    fields.append(jnp.array(field))
  # Concatenate
  state = jnp.concatenate(fields, axis=0)
  assert state.ndim == 2
  # Concatenate
  state = jnp.concatenate(fields, axis=0)
  assert state.ndim == 2
  # Only keep the last state when we stack frame
  state = state[:, -1]
  # Add extra batch dimension
  state = state[None, None]
  return state
