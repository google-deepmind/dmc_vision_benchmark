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

"""Model trainer."""

import functools
from typing import Any, Dict, Iterator, cast, Union

import chex
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import optax
import orbax.checkpoint

from absl import app
from dmc_vision_benchmark.data import load_dmc_vb
from dmc_vision_benchmark.environments import env_dmc_vb
from dmc_vision_benchmark.environments import env_wrappers
from dmc_vision_benchmark.evals import eval_in_envs
from dmc_vision_benchmark.modules import time_embed
from dmc_vision_benchmark.rep_learn import agent_modules
from dmc_vision_benchmark.rep_learn import mlp_multiheads
from dmc_vision_benchmark.rep_learn import td3_bc
from dmc_vision_benchmark.rep_learn import utils

# Allow rng to not need to communicate for dropout.
# Note that this changes the RNG values produced
jax.config.update("jax_threefry_partitionable", True)


_CONFIGS = config_flags.DEFINE_config_file(
    name="config",
    default="config.py",
    help_string="Training configuration",
)


# pylint: disable=invalid-name
# pylint: disable=g-doc-return-or-yield
class Trainer:
  """Trains the model."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.config = config
    chex.assert_equal(config.learning.batch_size % jax.local_device_count(), 0)

    # Whether we pass state or observations as inputs
    if self.config.model in ["bc_on_state", "td3_bc_on_state"]:
      self.obs_type = "states"
      self.future_obs_type = "future_states"
      self.config.obs_preproc.frame_stack = 1
      self.config.data.add_state = True
      self.config.data.only_state = True
    elif self.config.model == "pretrain_state":
      self.obs_type = "obs"
      self.future_obs_type = "future_obs"
      self.config.data.add_state = True
      self.config.data.only_state = False
    else:
      self.obs_type = "obs"
      self.future_obs_type = "future_obs"
      self.config.data.add_state = False
      self.config.data.only_state = False

    # Update number of IDM steps / whether we use rewards
    if "td3" in self.config.model:
      self.config.obs_preproc.nsteps_idm = 1
      self.config.data.add_rewards = True
    elif self.config.model == "pretrain_lfd":
      self.config.obs_preproc.nsteps_idm = 1
      self.config.data.add_rewards = False
    elif self.config.model == "pretrain_id":
      self.config.data.add_rewards = False
    else:
      self.config.obs_preproc.nsteps_idm = 0
      self.config.data.add_rewards = False

    # Update episode_length
    self.config.data.episode_length = (
        self.config.obs_preproc.frame_stack + self.config.obs_preproc.nsteps_idm
    )

    # Load the data
    data_loader = load_dmc_vb.load_data(
        **self.config.data, **self.config.obs_preproc
    )
    self.train_dataset = data_loader.train_dataset
    self.eval_dataset = data_loader.eval_dataset  # 5% of the data
    self.action_dim = data_loader.action_dim
    self.gt_state_dim = data_loader.gt_state_dim
    self.cameras = data_loader.cameras
    print("\nDataset loaded")

    # Build the model
    self.model = self._build_model()
    print("\nModel built")

    self.this_id = id(self)
    print("Model ID: ", self.this_id)
    self.checkpoint_workdir = (
        f"{self.config.save_to_folder}/checkpoint/{self.this_id}"
    )
    self.online_workdir = (
        f"{self.config.save_to_folder}/online_eval/{self.this_id}"
    )

    # Build the online evaluator
    env = env_wrappers.EvaluationEnvironmentWrapper(
        make_env=env_dmc_vb.make_env,
        make_env_kwargs=dict(
            background_dataset_path=self.config.online_eval.background_dataset_path,
            domain_name=self.config.data.domain_name,
            task_name=self.config.data.task_name,
            action_repeats=self.config.online_eval.action_repeats,
            frame_stack=self.config.obs_preproc.frame_stack,
            height=self.config.data.img_height,
            width=self.config.data.img_width,
            difficulty=self.config.data.difficulty,
            dynamic=self.config.data.dynamic_distractors,
            background_dataset_videos=self.config.online_eval.background_dataset_videos,
            ant_random_start_end=self.config.online_eval.ant_random_start_end,
            propagate_seed_to_env=self.config.online_eval.propagate_seed_to_env,
        ),
        pixels_key=self.config.online_eval.render_camera,
    )

    # Define the online evaluator seeds
    num_online_runs = self.config.online_eval.num_online_runs
    ant_fixed_seed = self.config.data.ant_fixed_seed
    if self.config.data.ant_fixed_seed is not None:
      env_eval_seed_tuples = [(ant_fixed_seed, 0)] * num_online_runs
    else:
      env_eval_seed_tuples = [
          (x, 0) for x in range(int(1e6), int(1e6) + num_online_runs)
      ]

    # Build the online evaluator
    self.online_env = eval_in_envs.EvalInEnv(
        env=env,
        model=self.model,
        workdir=self.online_workdir,
        max_steps=self.config.online_eval.max_episode_length,
        history_length=self.config.online_eval.history_length,
        env_eval_seed_tuples=env_eval_seed_tuples,
        render=True,
        render_resolution=(128, 128),
        render_fps=10,
    )
    print("\nOnline evaluator built")

  def _build_model(self) -> Any:
    """Get the model."""

    # 1. Modules for TD3-BC
    # Encoder modules
    shared_obs_encoder = agent_modules.get_shared_obs_encoder_default()
    shared_state_encoder = agent_modules.get_shared_state_encoder_default()
    actor_encoder_trunk = agent_modules.get_actor_critic_trunk_default(
        self.config.encode_dim
    )
    critic_encoder_trunk = agent_modules.get_actor_critic_trunk_default(
        self.config.encode_dim
    )
    # Actor and critic as in https://arxiv.org/pdf/2106.06860#page=15
    actor_default = agent_modules.get_mlp_default(
        out_dim=self.action_dim, name="actor"
    )
    critic1 = agent_modules.get_mlp_default(out_dim=1, name="critic1")
    critic2 = agent_modules.get_mlp_default(out_dim=1, name="critic2")

    # 2. Modules for MLPMultiHeads
    # Encoder modules
    obs_encoder = agent_modules.get_obs_encoder_default(self.config.encode_dim)
    state_encoder = agent_modules.get_state_encoder_default(
        self.config.encode_dim
    )
    # Decoder module inspired from https://arxiv.org/pdf/1910.01741#page=12
    obs_decoder = agent_modules.get_obs_decoder_default(
        frame_stack=self.config.obs_preproc.frame_stack,
        n_cameras=len(self.cameras),
    )
    # MLP modules as in https://arxiv.org/pdf/2305.16985.pdf
    next_action_predictor = agent_modules.get_mlp_default(
        out_dim=self.action_dim, name="next_action_predictor"
    )
    inverse_dynamics_predictor = agent_modules.get_mlp_default(
        out_dim=self.action_dim, name="inverse_dynamics_predictor"
    )
    action_encoder = agent_modules.get_mlp_default(
        out_dim=self.config.encode_dim, name="action_encoder"
    )
    latent_forward_model = agent_modules.get_mlp_default(
        out_dim=self.config.encode_dim, name="latent_forward_model"
    )
    state_predictor_default = agent_modules.get_state_predictor_default(
        self.gt_state_dim
    )
    # Other
    idm_step_encoder_default = time_embed.TimestepEmbedding(
        max_timestep=self.config.data.episode_length,
        emb_dim=self.config.idm_step_encode_dim,
    )

    def _build_td3_bc(
        shared_obs_encoder,
        pass_state=False,
        stop_gradient_shared_obs_encoder=False,
    ):
      """Builds the TD3-BC agent."""
      return td3_bc.TD3BC(
          encode_dim=self.config.encode_dim,
          actions_dim=self.action_dim,
          shared_obs_encoder=shared_obs_encoder,
          actor_encoder_trunk=actor_encoder_trunk,
          critic_encoder_trunk=critic_encoder_trunk,
          actor=actor_default,
          critic1=critic1,
          critic2=critic2,
          # TD3-BC params
          std_noise_to_actions=self.config.td3_bc.std_noise_to_actions,
          noise_clipping=self.config.td3_bc.noise_clipping,
          actions_vrange=self.config.data.actions_vrange,
          discount=self.config.td3_bc.discount,
          stop_gradient_shared_obs_encoder=stop_gradient_shared_obs_encoder,
          # online eval params
          obs_vrange=self.config.data.obs_vrange,
          domain_name=self.config.data.domain_name,
          cameras=self.cameras,
          pass_state=pass_state,
      )

    def _build_mlp(
        obs_encoder=obs_encoder,
        next_action_predictor=None,
        idm_step_encoder=None,
        inverse_dynamics_predictor=None,
        state_predictor=None,
        action_encoder=None,
        latent_forward_model=None,
        obs_decoder=None,
        stop_gradient_obs_encoder=False,
        pass_state=False,
    ):
      """Builds the MLP agent."""
      return mlp_multiheads.MLPMultiHeads(
          encode_dim=self.config.encode_dim,
          obs_encoder=obs_encoder,
          next_action_predictor=next_action_predictor,
          idm_step_encoder=idm_step_encoder,
          inverse_dynamics_predictor=inverse_dynamics_predictor,
          state_predictor=state_predictor,
          action_encoder=action_encoder,
          latent_forward_model=latent_forward_model,
          obs_decoder=obs_decoder,
          stop_gradient_obs_encoder=stop_gradient_obs_encoder,
          # online eval params
          obs_vrange=self.config.data.obs_vrange,
          domain_name=self.config.data.domain_name,
          cameras=self.cameras,
          pass_state=pass_state,
      )

    # TD3-BC agents
    if self.config.model == "td3_bc":
      return _build_td3_bc(shared_obs_encoder)

    if self.config.model == "td3_bc_on_state":
      return _build_td3_bc(shared_state_encoder, pass_state=True)

    if self.config.model == "td3_bc_w_frozen_encoder":
      return _build_td3_bc(
          shared_obs_encoder, stop_gradient_shared_obs_encoder=True
      )

    # BC agents
    if self.config.model == "bc":
      return _build_mlp(next_action_predictor=next_action_predictor)

    if self.config.model == "bc_on_state":
      return _build_mlp(
          obs_encoder=state_encoder,
          next_action_predictor=next_action_predictor,
          pass_state=True,
      )

    if self.config.model == "bc_w_frozen_encoder":
      return _build_mlp(
          next_action_predictor=next_action_predictor,
          stop_gradient_obs_encoder=True,
      )

    # Pretraining
    if self.config.model == "pretrain_id":
      return _build_mlp(
          inverse_dynamics_predictor=inverse_dynamics_predictor,
          idm_step_encoder=idm_step_encoder_default,
      )

    if self.config.model == "pretrain_lfd":
      return _build_mlp(
          latent_forward_model=latent_forward_model,
          action_encoder=action_encoder,
      )

    if self.config.model == "pretrain_ae":
      return _build_mlp(obs_decoder=obs_decoder)

    if self.config.model == "pretrain_state":
      return _build_mlp(state_predictor=state_predictor_default)

    raise ValueError(f"Invalid implementation {self.config.model}")

  def _update_teacher(
      self,
      teacher_params: Dict[str, jnp.ndarray],
      new_model_params: Dict[str, jnp.ndarray],
  ) -> Dict[str, jnp.ndarray]:
    """Updates the teacher network parameters."""
    return (
        self.config.learning.teacher_damping * teacher_params
        + (1 - self.config.learning.teacher_damping) * new_model_params
    )

  @functools.partial(jax.jit, static_argnames="self")
  def _update_mlp(
      self,
      batch_data: Dict[str, jnp.ndarray],
      model_params: Dict[str, jnp.ndarray],
      teacher_params: Dict[str, jnp.ndarray] | None,
      opt_state: Dict[str, Any],
  ) -> tuple[
      float,
      Dict[str, jnp.ndarray],
      Dict[str, jnp.ndarray] | None,
      Dict[str, Any],
  ]:
    """Updates the network parameters on a batch.

    Args:
      batch_data: Batch data
      model_params: Model parameters
      teacher_params: Optional teacher parameters
      opt_state: Optimizer state

    Returns:
      loss: Loss on the batch
      new_model_params: New model parameters
      new_teacher_params: New teacher parameters
      new_opt_state: New optimizer state
    """
    assert hasattr(self, "opt")
    loss, param_grads = jax.value_and_grad(self.loss_on_batch, argnums=1)(
        batch_data, model_params, teacher_params
    )
    # Gradident update
    updates, new_opt_state = self.opt.update(param_grads, opt_state)
    new_model_params = optax.apply_updates(model_params, updates)

    # Update the teacher network
    if teacher_params is not None:
      new_teacher_params = jax.tree_util.tree_map(
          self._update_teacher,
          teacher_params,
          new_model_params,
      )
    else:
      new_teacher_params = None
    return loss, new_model_params, new_teacher_params, new_opt_state

  @functools.partial(jax.jit, static_argnames=("self", "update_actor"))
  def _update_td3_bc(
      self,
      batch_data: Dict[str, jnp.ndarray],
      model_params: Dict[str, jnp.ndarray],
      teacher_params: Dict[str, jnp.ndarray],
      opt_critic_state: Dict[str, Any],
      opt_actor_state: Dict[str, Any],
      update_actor: bool,
  ) -> tuple[
      Dict[str, float],
      Dict[str, jnp.ndarray],
      Dict[str, jnp.ndarray],
      Dict[str, Any],
      Dict[str, Any],
  ]:
    """Updates the TD3-BC network parameters on a batch.

    Args:
      batch_data: Batch data
      model_params: Model parameters
      teacher_params: Teacher parameters
      opt_critic_state: Critic optimizer state
      opt_actor_state: Actor optimizer state
      update_actor: Whether we update the actor network

    Returns:
      losses: Critic and actor losses on the batch
      new_model_params: Updated model parameters
      new_teacher_params: Updated teacher network parameters
      new_opt_critic_state: Updated critic optimizer state
      new_opt_actor_state: Updated actor optimizer state
    """
    assert hasattr(self, "opt_critic")
    assert hasattr(self, "opt_critic_keys")
    assert hasattr(self, "opt_actor")
    assert hasattr(self, "opt_actor_keys")

    # Step 1. Always update the two critics
    critic_loss, param_grads = jax.value_and_grad(
        self.loss_on_batch_td3_bc, argnums=1
    )(batch_data, model_params, teacher_params, loss_type="critic")

    # Only keep the critic keys
    critic_param_grads = {}
    for key in self.opt_critic_keys:
      critic_param_grads[key] = param_grads["params"][key]
    updates, new_opt_critic_state = self.opt_critic.update(
        critic_param_grads, opt_critic_state
    )

    # Update
    new_model_params = model_params.copy()
    for key in self.opt_critic_keys:
      new_model_params["params"][key] = optax.apply_updates(
          model_params["params"][key], updates[key]
      )

    # Step 2. Periodically update the actor and the teacher network
    if update_actor:
      (
          actor_loss,
          (bc_loss, avg_abs_critic1),
      ), param_grads = jax.value_and_grad(
          self.loss_on_batch_td3_bc, argnums=1, has_aux=True
      )(
          batch_data,
          new_model_params,
          teacher_params=None,
          loss_type="actor",
      )
      # Only keep the actor keys
      actor_param_grads = {}
      for key in self.opt_actor_keys:
        actor_param_grads[key] = param_grads["params"][key]
      updates, new_opt_actor_state = self.opt_actor.update(
          actor_param_grads, opt_actor_state
      )
      # Update the actor
      for key in self.opt_actor_keys:
        new_model_params["params"][key] = optax.apply_updates(
            new_model_params["params"][key], updates[key]
        )

      # Update the teacher network
      new_teacher_params = jax.tree_util.tree_map(
          self._update_teacher,
          teacher_params,
          new_model_params,
      )
      losses = {
          "critic_loss": critic_loss,
          "actor_loss": actor_loss,
          "bc_loss": bc_loss,
          "avg_abs_critic1": avg_abs_critic1,
      }
    else:
      new_teacher_params = teacher_params
      new_opt_actor_state = opt_actor_state
      losses = {"critic_loss": critic_loss}

    return (
        losses,
        new_model_params,
        new_teacher_params,
        new_opt_critic_state,
        new_opt_actor_state,
    )

  @functools.partial(jax.jit, static_argnames="self")
  def loss_on_batch(
      self,
      batch_data: jnp.ndarray,
      model_params: Dict[str, jnp.ndarray],
      teacher_params: Dict[str, jnp.ndarray] | None,
  ) -> float:
    """Computes the loss of the network on a given batch.

    Args:
      batch_data: Batch sentences
      model_params: Network parameters
      teacher_params: Optional teacher parameters

    Returns:
      loss: Loss
    """
    assert "td3" not in self.config.model

    future_obs = (
        batch_data[self.future_obs_type]
        if self.future_obs_type in batch_data
        else None
    )
    idm_step = batch_data["idm_step"] if "idm_step" in batch_data else None
    actions = batch_data["actions"] if "actions" in batch_data else None
    res = self.model.apply(
        variables=model_params,
        obs=batch_data[self.obs_type],
        actions=actions,
        future_obs=future_obs,
        idm_step=idm_step,
    )

    avg_loss = 0.0
    if self.config.model in [
        "bc",
        "bc_on_state",
        "bc_w_frozen_encoder",
    ]:
      bc_loss = utils.square_l2_dist(
          res["pred_actions"], batch_data["actions"][:, 0]
      )
      avg_loss += bc_loss

    if self.config.model == "pretrain_id":
      idm_loss = utils.square_l2_dist(
          res["inv_model_actions"], batch_data["actions"][:, 0]
      )
      avg_loss += idm_loss

    if self.config.model in ["pretrain_lfd"]:
      res_target = jax.lax.stop_gradient(
          self.model.apply(
              variables=teacher_params,
              future_obs=future_obs,
              method="get_future_obs_encoded",
          )
      )
      lfd_loss = utils.square_l2_dist(
          res["pred_future_obs_encoded"], res_target["future_obs_encoded"]
      )
      avg_loss += lfd_loss

    if self.config.model == "pretrain_state":
      state_pred_loss = utils.square_l2_dist(
          res["pred_states"], batch_data["states"][:, 0]
      )
      avg_loss += state_pred_loss

    if self.config.model == "pretrain_ae":
      ae_pred_loss = utils.square_l2_dist(res["obs_decoded"], batch_data["obs"])
      avg_loss += ae_pred_loss

    return avg_loss

  @functools.partial(jax.jit, static_argnames=("self", "loss_type"))
  def loss_on_batch_td3_bc(
      self,
      batch_data: jnp.ndarray,
      model_params: Dict[str, jnp.ndarray],
      teacher_params: Dict[str, jnp.ndarray] | None,
      loss_type: str,
  ) -> Union[float, tuple[float, tuple[float, float]]]:
    """Computes the loss of the TD3-BC agent on a given batch.

    Args:
      batch_data: Batch sentences
      model_params: Network parameters
      teacher_params: Teacher network parameters. Only used for TD3-BC
      loss_type: Actor or critic loss

    Returns:
      loss: Loss
    """
    assert "td3" in self.config.model

    if loss_type == "critic":
      assert teacher_params is not None
      # Critic loss
      res_critics = self.model.apply(
          variables=model_params,
          obs=batch_data[self.obs_type],
          actions=batch_data["actions"],
          method="compute_critic_outputs",
      )
      res_critic_target = jax.lax.stop_gradient(
          self.model.apply(
              variables=teacher_params,
              future_obs=batch_data[self.future_obs_type],
              future_rewards=batch_data["future_rewards"],
              is_last=batch_data["is_last"],
              rngs={"critic_target": jax.random.PRNGKey(1)},
              method="compute_critic_target",
          )
      )
      critic_loss = utils.square_l2_dist(
          res_critics["q1"], res_critic_target["target_q"]
      ) + utils.square_l2_dist(res_critics["q2"], res_critic_target["target_q"])
      return critic_loss

    if loss_type == "actor":
      # Run a new forward pass
      res_actor = self.model.apply(
          variables=model_params,
          obs=batch_data[self.obs_type],
          method="compute_actor_outputs",
      )

      # Trade-off param with stop-gradient
      avg_abs_q1 = jnp.mean(jnp.abs(jax.lax.stop_gradient(res_actor["q1"])))
      avg_abs_critic1 = cast(float, avg_abs_q1) + 1e-9
      beta = self.config.td3_bc.alpha / avg_abs_critic1

      # Actor loss. Careful about the minus sign
      actor_loss = -cast(float, beta * jnp.mean(res_actor["q1"]))
      # Behavioral cloning loss
      bc_loss = utils.square_l2_dist(
          res_actor["pred_actions"], batch_data["actions"][:, 0]
      )
      return actor_loss + bc_loss, (bc_loss, avg_abs_critic1)

    raise ValueError(f"Unknown loss type: {loss_type}")

  def _eval_dataset(
      self,
      dataset: Iterator[Any],
      model_params: Dict[str, jnp.ndarray],
      teacher_params: Dict[str, jnp.ndarray],
  ) -> float:
    """Computes the loss on a dataset.

    Args:
      dataset: Input batch
      model_params: Network parameters
      teacher_params: Teacher parameters

    Returns:
      avg_loss: Averaged loss
    """
    total_loss = 0.0
    num_batches = 0
    for batch_data in dataset:
      if batch_data[self.obs_type].shape[0] % jax.local_device_count() != 0:
        continue
      loss = self.loss_on_batch(batch_data, model_params, teacher_params)
      total_loss += loss
      num_batches += 1
    return total_loss / num_batches

  def train(self):
    """Trains the network."""
    init_data = next(self.train_dataset)
    num_iters = self.config.learning.num_iters

    model = None
    opt_state = None
    opt_critic_state = None
    opt_actor_state = None
    actor_loss = 0
    bc_loss = 0
    avg_abs_critic1 = 0

    # Initialize model params, optimizer, and optimizer state
    if "td3" not in self.config.model:
      future_obs = (
          init_data[self.future_obs_type]
          if self.future_obs_type in init_data
          else None
      )
      idm_step = init_data["idm_step"] if "idm_step" in init_data else None
      actions = init_data["actions"] if "actions" in init_data else None
      model_params = self.model.init(
          jax.random.PRNGKey(0),
          obs=init_data[self.obs_type],
          future_obs=future_obs,
          idm_step=idm_step,
          actions=actions,
      )
      # Initialize the optimizer
      self.opt = optax.adam(learning_rate=self.config.learning.learning_rate)
      opt_state = self.opt.init(model_params)
    else:
      model_params = self.model.init(
          jax.random.PRNGKey(0),
          obs=init_data[self.obs_type],
          method="initialize",
      )
      # Create critic optimizer
      self.opt_critic = optax.adam(
          learning_rate=self.config.learning.learning_rate
      )
      self.opt_critic_keys = ["critic1", "critic2"]
      # Create actor optimizer
      self.opt_actor = optax.adam(
          learning_rate=self.config.learning.learning_rate
      )
      self.opt_actor_keys = ["actor"]

      # Whether we update the encoder with the actor or the critic loss
      encoder_keys = [
          "critic_encoder_trunk",
          "actor_encoder_trunk",
      ]
      if self.config.model == "td3_bc":
        encoder_keys.append("shared_obs_encoder")

      if self.config.td3_bc.loss_for_encoder == "actor":
        self.opt_actor_keys.extend(encoder_keys)
      else:
        self.opt_critic_keys.extend(encoder_keys)

      params_for_opt_critic = {}
      for key in self.opt_critic_keys:
        params_for_opt_critic[key] = model_params["params"][key]
      opt_critic_state = self.opt_critic.init(params_for_opt_critic)

      params_for_opt_actor = {}
      for key in self.opt_actor_keys:
        params_for_opt_actor[key] = model_params["params"][key]
      opt_actor_state = self.opt_actor.init(params_for_opt_actor)

    teacher_params = None
    if self.config.model in ["pretrain_lfd"] or "td3" in self.config.model:
      # Copy the parameters to initialize the teacher network
      teacher_params = jax.tree_util.tree_map(lambda x: x + 0, model_params)

    # Create checkpoint manager
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        directory=self.checkpoint_workdir,
        checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
        options=orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=2, create=True
        ),
    )

    # Add metrics
    metrics = {"train_loss": {}, "eval_loss": {}, "online_eval": {}}

    # Option 1: Initialize the obs encoder from a pretrained model
    if "frozen_encoder" in self.config.model:
      print(
          "\nLoading observation encoder from pretrained model",
          self.config.pretrained_model_id,
      )
      model_params_loaded = utils.load_model_params(
          save_to_folder=self.config.save_to_folder,
          pretrained_model_id=self.config.pretrained_model_id,
      )
      if self.config.model == "bc_w_frozen_encoder":
        model_params["params"]["obs_encoder"] = model_params_loaded["params"][
            "obs_encoder"
        ]
      if self.config.model == "td3_bc_w_frozen_encoder":
        model_params["params"]["shared_obs_encoder"]["layers_1"] = (
            model_params_loaded["params"]["obs_encoder"]["layers_1"]
        )
    # Option 2: Optionally restore checkpoint
    elif not self.config.try_to_restore:
      print("\nNot looking for existing checkpoint")
    elif checkpoint_manager.latest_step() is not None:
      last_step = checkpoint_manager.latest_step()
      print("\nRestoring checkpoint from step", last_step)
      restored = checkpoint_manager.restore(
          last_step,
          items={
              "it": num_iters,
              "model_params": model_params,
              "metrics": metrics,
              "teacher_params": teacher_params,
          },
      )
      model_params = restored["model_params"]
      num_iters = self.config.learning.num_iters - restored["it"]
      metrics = restored["metrics"]
      teacher_params = restored["teacher_params"]

      # Restore both optimizer states for TD3-BC
      if "td3" not in self.config.model:
        restored = checkpoint_manager.restore(
            last_step,
            items={"opt_state": opt_state},
        )
        opt_state = restored["opt_state"]
      else:
        restored = checkpoint_manager.restore(
            last_step,
            items={
                "opt_critic_state": opt_critic_state,
                "opt_actor_state": opt_actor_state,
            },
        )
        opt_critic_state = restored["opt_critic_state"]
        opt_actor_state = restored["opt_actor_state"]
    else:
      print("\nNo checkpoint found at", self.checkpoint_workdir)

    for it in range(num_iters):
      train_batch_data = next(self.train_dataset)
      if (
          train_batch_data[self.obs_type].shape[0] % jax.local_device_count()
          != 0
      ):
        train_batch_data = next(self.train_dataset)

      # This correctly has different samples for different devices.
      if "td3" not in self.config.model:
        assert opt_state is not None
        loss, model_params, teacher_params, opt_state = self._update_mlp(
            train_batch_data,
            model_params,
            teacher_params,
            opt_state,
        )
        loss = jax.device_get(loss)
        metrics["train_loss"][it] = loss
      else:
        assert opt_critic_state is not None
        assert opt_actor_state is not None
        (
            losses,
            model_params,
            teacher_params,
            opt_critic_state,
            opt_actor_state,
        ) = self._update_td3_bc(
            batch_data=train_batch_data,
            model_params=model_params,
            teacher_params=teacher_params,
            opt_critic_state=opt_critic_state,
            opt_actor_state=opt_actor_state,
            # Whether we update the actor network
            update_actor=it % self.config.td3_bc.update_actor_every == 0,
        )
        losses = jax.device_get(losses)
        metrics["train_loss"][it] = losses

      # Do not evaluate for TD3-BC for now
      if (
          (it + 1) % self.config.learning.eval_every == 0
          and "td3" not in self.config.model
      ):
        # Evaluate
        eval_loss = self._eval_dataset(
            self.eval_dataset, model_params, teacher_params
        )
        eval_loss = jax.device_get((eval_loss))
        metrics["eval_loss"][it] = eval_loss

      # Online evaluation
      if it % self.config.learning.online_eval_every == 0:
        if self.config.model not in [
            "pretrain_id",
            "pretrain_lfd",
            "pretrain_ae",
            "pretrain_state",
        ]:
          print(f"\nRunning online evaluation at step {it}")
          online_eval_res, _ = self.online_env.evaluate(
              model_params=model_params,
              step=it,
          )
          reward_mean = jax.device_get(online_eval_res["reward_mean"])
          reward_se = jax.device_get(online_eval_res["reward_se"])
          metrics["online_eval"][it] = {
              "reward_mean": reward_mean,
              "reward_se": reward_se,
          }

      # Save checkpoint
      if it % self.config.learning.checkpoint_every == 0:
        print(f"\nSaving checkpoint at step {it}")
        ckpt = {
            "model_params": model_params,
            "teacher_params": teacher_params,
            "metrics": metrics,
            "it": it,
        }
        if "td3" not in self.config.model:
          ckpt["opt_state"] = opt_state
        else:
          ckpt["opt_critic_state"] = opt_critic_state
          ckpt["opt_actor_state"] = opt_actor_state
        checkpoint_manager.save(it, ckpt)

    return model, metrics


def train(_):
  trainer = Trainer(config=_CONFIGS.value)
  trainer.train()


if __name__ == "__main__":
  app.run(train)
