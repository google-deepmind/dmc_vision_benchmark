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

"""Default modules."""

import einops
from flax import linen as nn

from dmc_vision_benchmark.modules import cnn
from dmc_vision_benchmark.modules import mlp


# Part 1. Multihead encoder modules
def get_obs_encoder_default(encode_dim):
  """Default observation encoder."""
  return nn.Sequential([
      # Stack inputs first
      lambda x: einops.rearrange(
          x, "... n_frames h1 w1 c1 -> ... h1 w1 (n_frames c1)"
      ),
      cnn.get_idm_cnn(),
      lambda x: einops.rearrange(x, "... h2 w2 c2 -> ... (h2 w2 c2)"),
      # Next steps as in https://arxiv.org/pdf/2305.16985.pdf
      nn.Dense(
          features=encode_dim,
          name="trunk",
          kernel_init=nn.initializers.orthogonal(),
      ),
      nn.LayerNorm(),
      nn.activation.tanh,
  ])


def get_state_encoder_default(encode_dim):
  """Default state encoder: same without CNN."""
  return nn.Sequential([
      lambda x: einops.rearrange(x, "... n_frames d -> ... (n_frames d)"),
      nn.Dense(
          features=encode_dim,
          name="trunk",
          kernel_init=nn.initializers.orthogonal(),
      ),
      nn.LayerNorm(),
      nn.activation.tanh,
  ])


# Part 2. TD3-BC encoder modules.
# Very similar to part 1, but actor and critic use different trunk networks
def get_shared_obs_encoder_default():
  """Shared part of the observation encoder."""
  return nn.Sequential([
      # Stack inputs first
      lambda x: einops.rearrange(
          x, "... n_frames h1 w1 c1 -> ... h1 w1 (n_frames c1)"
      ),
      cnn.get_idm_cnn(),
      lambda x: einops.rearrange(x, "... h2 w2 c2 -> ... (h2 w2 c2)"),
  ])


def get_shared_state_encoder_default():
  """Shared part of the state encoder."""
  return lambda x: einops.rearrange(x, "... n_frames d -> ... (n_frames d)")


def get_actor_critic_trunk_default(encode_dim):
  return nn.Sequential([  # same as MLPMultiHeads
      nn.Dense(
          features=encode_dim,
          name="trunk",
          kernel_init=nn.initializers.orthogonal(),
      ),
      nn.LayerNorm(),
      nn.activation.tanh,
  ])


# Part 3. MLP modules
def get_mlp_default(out_dim, name):
  return mlp.MLP(
      features=(256, 256),
      out_dim=out_dim,
      activation=nn.relu,
      out_activation=nn.tanh,  # Since actions are between -1 and 1
      name=name,
      kernel_init=nn.initializers.orthogonal(),
  )


def get_state_predictor_default(gt_state_dim):
  return nn.Dense(
      features=gt_state_dim,
      kernel_init=nn.initializers.orthogonal(),
      name="state_predictor",
  )


# Part 4. Decoder modules
def get_obs_decoder_default(frame_stack, n_cameras):
  """Observation decoder inspired from https://arxiv.org/pdf/1910.01741#page=12."""
  return nn.Sequential([
      nn.Dense(
          features=16 * 16 * 256,
          name="trunk",
          kernel_init=nn.initializers.orthogonal(),
      ),
      lambda x: einops.rearrange(
          x, "... (h2 w2 c2) -> ... h2 w2 c2", h2=16, w2=16, c2=256
      ),
      cnn.get_decoder_cnn(n_frames=frame_stack, n_cameras=n_cameras),
      lambda x: einops.rearrange(
          x,
          "... h w (n_frames c) -> ... n_frames h w c",
          n_frames=frame_stack,
      ),
      # Do not use activation for the output
  ])


def get_state_decoder_default(state_dim):
  """Default state decoder."""
  return mlp.MLP(
      features=(256, 256),
      out_dim=state_dim,
      activation=nn.relu,
      out_activation=None,  # Do not use activation for the output
      name="state_decoder",
      kernel_init=nn.initializers.orthogonal(),
  )
