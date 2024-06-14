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

"""Train a decoder given a frozen encoder."""

import functools
from typing import Any, Dict, Iterator

import chex
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import optax
import orbax.checkpoint

from absl import app
from dmc_vision_benchmark.data import load_dmc_vb
from dmc_vision_benchmark.modules import autoencoder
from dmc_vision_benchmark.rep_learn import agent_modules
from dmc_vision_benchmark.rep_learn import utils


_CONFIGS = config_flags.DEFINE_config_file(
    name="config",
    default="config.py",
    help_string="Training configuration",
)


# pylint: disable=invalid-name
# pylint: disable=g-doc-return-or-yield
class DecoderTrainer:
  """Trains the model."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config
    chex.assert_equal(config.learning.batch_size % jax.local_device_count(), 0)

    self.config.obs_preproc.nsteps_idm = 0
    self.config.data.add_rewards = False
    self.config.data.episode_length = self.config.obs_preproc.frame_stack

    # Whether we pass state or observations as inputs
    if self.config.model == "state_decoder":
      self.config.data.add_state = True
    else:
      self.config.data.add_state = False

    # Load the data
    data_loader = load_dmc_vb.load_data(
        **self.config.data, **self.config.obs_preproc
    )
    self.train_dataset = data_loader.train_dataset
    self.eval_dataset = data_loader.eval_dataset  # 5% of the data
    self.gt_state_dim = data_loader.gt_state_dim
    self.cameras = data_loader.cameras
    print("\nDataset loaded")

    # Build the model
    self.model = self._build_decoder()
    print("\nDecoder built")

    self.this_id = id(self)
    print("Model ID: ", self.this_id)
    self.checkpoint_workdir = (
        f"{self.config.save_to_folder}/checkpoint/{self.this_id}"
    )

  def _build_decoder(self) -> Any:
    """Get the model."""
    # Encoder modules
    obs_encoder = agent_modules.get_obs_encoder_default(self.config.encode_dim)
    # Decoder modules
    obs_decoder = agent_modules.get_obs_decoder_default(
        frame_stack=self.config.obs_preproc.frame_stack,
        n_cameras=len(self.cameras),
    )
    state_decoder = agent_modules.get_state_decoder_default(self.gt_state_dim)

    if self.config.model == "obs_decoder":
      return autoencoder.AutoEncoder(
          stop_gradient_obs_encoder=True,
          obs_encoder=obs_encoder,
          obs_decoder=obs_decoder,
          state_decoder=None,
      )
    if self.config.model == "state_decoder":
      return autoencoder.AutoEncoder(
          stop_gradient_obs_encoder=True,
          obs_encoder=obs_encoder,
          obs_decoder=None,
          state_decoder=state_decoder,
      )
    raise ValueError(f"Invalid implementation {self.config.model}")

  @functools.partial(jax.jit, static_argnames="self")
  def _update_mlp(
      self,
      batch_data: Dict[str, jnp.ndarray],
      model_params: Dict[str, jnp.ndarray],
      opt_state: Dict[str, Any],
  ) -> tuple[
      float,
      Dict[str, jnp.ndarray],
      Dict[str, Any],
  ]:
    """Updates the network parameters on a batch."""
    assert hasattr(self, "opt")
    loss, param_grads = jax.value_and_grad(self.loss_on_batch, argnums=1)(
        batch_data, model_params
    )
    # Gradient update
    updates, new_opt_state = self.opt.update(param_grads, opt_state)
    new_model_params = optax.apply_updates(model_params, updates)
    return loss, new_model_params, new_opt_state

  @functools.partial(jax.jit, static_argnames="self")
  def loss_on_batch(
      self,
      batch_data: jnp.ndarray,
      model_params: Dict[str, jnp.ndarray],
  ) -> float:
    """Computes the loss of the network on a given batch."""
    res = self.model.apply(
        variables=model_params,
        obs=batch_data["obs"],
    )
    if self.config.model == "obs_decoder":
      return utils.square_l2_dist(res["obs_decoded"], batch_data["obs"])
    if self.config.model == "state_decoder":
      return utils.square_l2_dist(
          res["states_decoded"], batch_data["states"][:, 0]
      )
    raise ValueError(f"Invalid implementation {self.config.model}")

  def _eval_dataset(
      self,
      dataset: Iterator[Any],
      model_params: Dict[str, jnp.ndarray],
  ) -> float:
    """Computes the loss on a dataset."""
    total_loss = 0.0
    num_batches = 0
    for batch_data in dataset:
      if batch_data["obs"].shape[0] % jax.local_device_count() != 0:
        continue
      loss = self.loss_on_batch(batch_data, model_params)
      total_loss += loss
      num_batches += 1
    return total_loss / num_batches

  def train(self):
    """Trains the network."""
    init_data = next(self.train_dataset)
    num_iters = self.config.learning.num_iters
    model = None

    # Initialize model params, optimizer, and optimizer state
    model_params = self.model.init(
        jax.random.PRNGKey(0),
        obs=init_data["obs"],
    )
    self.opt = optax.adam(learning_rate=self.config.learning.learning_rate)
    opt_state = self.opt.init(model_params)

    # Create checkpoint manager
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        directory=self.checkpoint_workdir,
        checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
        options=orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=2, create=True
        ),
    )

    # Add metrics
    metrics = {"train_loss": {}, "eval_loss": {}}

    # Restore checkpoint
    if checkpoint_manager.latest_step() is not None:
      last_step = checkpoint_manager.latest_step()
      print("\nRestoring checkpoint from step", last_step)
      restored = checkpoint_manager.restore(
          last_step,
          items={
              "it": num_iters,
              "model_params": model_params,
              "metrics": metrics,
          },
      )
      model_params = restored["model_params"]
      num_iters = self.config.learning.num_iters - restored["it"]
      metrics = restored["metrics"]
    else:
      # First iteration
      model_params_loaded = utils.load_model_params(
          save_to_folder=self.config.save_to_folder,
          pretrained_model_id=self.config.pretrained_model_id,
      )
      model_params["params"]["obs_encoder"] = model_params_loaded["params"][
          "obs_encoder"
      ]

    for it in range(num_iters):
      train_batch_data = next(self.train_dataset)
      if train_batch_data["obs"].shape[0] % jax.local_device_count() != 0:
        train_batch_data = next(self.train_dataset)

      loss, model_params, opt_state = self._update_mlp(
          train_batch_data,
          model_params,
          opt_state,
      )
      loss = jax.device_get(loss)
      metrics["train_loss"][it] = loss

      # Evaluate
      if (it + 1) % self.config.learning.eval_every == 0:
        eval_loss = self._eval_dataset(self.eval_dataset, model_params)
        eval_loss = jax.device_get((eval_loss))
        metrics["eval_loss"][it] = eval_loss

      # Save checkpoint
      if it % self.config.learning.checkpoint_every == 0:
        print(f"\nSaving checkpoint at step {it}")
        ckpt = {
            "model_params": model_params,
            "metrics": metrics,
            "opt_state": opt_state,
            "it": it,
        }
        checkpoint_manager.save(it, ckpt)

    return model, metrics


def train(_):
  decoder_trainer = DecoderTrainer(config=_CONFIGS.value)
  decoder_trainer.train()


if __name__ == "__main__":
  app.run(train)
