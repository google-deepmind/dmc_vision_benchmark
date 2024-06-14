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

"""Implements a TD3-BC model from https://arxiv.org/pdf/2106.06860."""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Union

from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron.typing import Array, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member

from dmc_vision_benchmark.rep_learn import mlp_multiheads


class TD3BC(nn.Module):
  """TD3-BC agent.

  See https://arxiv.org/pdf/1802.09477 for TD3
  and https://arxiv.org/pdf/2106.06860 for TD3-BC.
  """

  encode_dim: int
  actions_dim: int
  shared_obs_encoder: nn.Module | Callable[[Any], Any] | None
  actor_encoder_trunk: nn.Module
  critic_encoder_trunk: nn.Module
  actor: nn.Module
  critic1: nn.Module
  critic2: nn.Module

  # Model params
  std_noise_to_actions: float
  noise_clipping: tuple[float, float]
  actions_vrange: tuple[float, float]
  discount: float
  stop_gradient_shared_obs_encoder: bool

  # Online eval params
  domain_name: str
  cameras: Sequence[str]
  obs_vrange: tuple[float, float]
  pass_state: bool

  def __hash__(self):
    return id(self)

  @typechecked
  def __call__(
      self, obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']]
  ) -> dict[str, Float['*b {self.actions_dim}']]:
    """Predicts future actions."""
    obs_encoded = self._run_shared_obs_encoder(obs)
    return self._run_actor(obs_encoded)

  @typechecked
  def initialize(
      self, obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']]
  ) -> dict[
      str,
      Float['*b 1'],
  ]:
    """Initialize all the modules."""
    obs_encoded = self._run_shared_obs_encoder(obs)
    actor_output = self._run_actor(obs_encoded)
    return self._run_critics(
        obs_encoded, actor_output['actions'], only_first_critic=False
    )

  @typechecked
  def _run_shared_obs_encoder(
      self, obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']]
  ) -> Float['...']:
    """Run the shared observations encoder."""
    if self.shared_obs_encoder is None:
      return obs

    obs_encoded = self.shared_obs_encoder(obs)
    if self.stop_gradient_shared_obs_encoder:
      obs_encoded = jax.lax.stop_gradient(obs_encoded)
    return obs_encoded

  @typechecked
  def _run_actor(
      self, obs_encoded: Float['*b n']
  ) -> dict[str, Float['*b {self.actions_dim}']]:
    """Run the actor given shared encoder outputs."""
    actor_obs_encoded = self.actor_encoder_trunk(obs_encoded)
    actions = self.actor(actor_obs_encoded)
    return {'actions': actions}

  @typechecked
  def _run_critics(
      self,
      obs_encoded: Float['*b n'],
      actions: Float['*b {self.actions_dim}'],
      only_first_critic: bool = False,
  ) -> dict[str, Float['*b 1']]:
    """Run the two critics given shared encoder outputs."""
    critic_obs_encoded = self.critic_encoder_trunk(obs_encoded)
    obs_encoded_actions = jnp.concatenate(
        [critic_obs_encoded, actions], axis=-1
    )
    q1 = self.critic1(obs_encoded_actions)
    if only_first_critic:
      return {'q1': q1}

    q2 = self.critic2(obs_encoded_actions)
    return {'q1': q1, 'q2': q2}

  @typechecked
  def compute_critic_outputs(
      self,
      obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']],
      actions: Float['*b 1 n'],
  ) -> dict[str, Float['*b 1']]:
    """Return the two critic outputs."""
    # Encode observations
    obs_encoded = self._run_shared_obs_encoder(obs)
    # Compute both critics outputs
    return self._run_critics(
        obs_encoded, actions[:, 0], only_first_critic=False
    )

  @typechecked
  def compute_critic_target(
      self,
      future_obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']],
      future_rewards: Float['*b 1'],
      is_last: Float['*b 1'],
  ) -> dict[str, Float['*b 1']]:
    """Runs the two critics. To use with teacher params and stop-gradient."""
    # Encode observations
    future_obs_encoded = self._run_shared_obs_encoder(future_obs)

    # Predict actions
    pred_future_actions = self._run_actor(future_obs_encoded)['actions']

    # Add noise to the predictions
    noise_to_actions = (
        jax.random.normal(
            key=self.make_rng('critic_target'), shape=pred_future_actions.shape
        )
        * self.std_noise_to_actions
    )
    clipped_noise_to_actions = jnp.clip(
        noise_to_actions,
        a_min=self.noise_clipping[0],
        a_max=self.noise_clipping[1],
    )
    pred_future_actions = jnp.clip(
        pred_future_actions + clipped_noise_to_actions,
        a_min=self.actions_vrange[0],
        a_max=self.actions_vrange[1],
    )

    # Compute critic target
    future_critic_outputs = self._run_critics(
        future_obs_encoded, pred_future_actions
    )
    target_q = future_rewards + self.discount * (1.0 - is_last) * jnp.minimum(
        future_critic_outputs['q1'], future_critic_outputs['q2']
    )
    return {'target_q': target_q}

  @typechecked
  def compute_actor_outputs(
      self,
      obs: Union[Float['*b n_frames h w c'], Float['*b n_frames m']],
  ) -> dict[str, Union[Float['*b 1'], Float['*b {self.actions_dim}']]]:
    """Returns the actor loss, the actions predicted and the trade-off param."""
    # Encode observations
    obs_encoded = self._run_shared_obs_encoder(obs)

    # Predict actions
    pred_actions = self._run_actor(obs_encoded)['actions']

    # Use the predicted actions with the first critic
    q1 = self._run_critics(obs_encoded, pred_actions, only_first_critic=True)[
        'q1'
    ]
    return {
        'pred_actions': pred_actions,
        'q1': q1,
    }

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
      obs = mlp_multiheads.get_state_input(
          observation_history, self.domain_name
      )
      obs = mlp_multiheads.normalize_state(obs, self.domain_name)
    else:
      obs = mlp_multiheads.get_rgb_normed(
          observation_history=observation_history,
          vrange=self.obs_vrange,
          cameras=self.cameras,
      )

    return self(obs=obs)
