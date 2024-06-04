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

"""Online evaluation inside simulation environment."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import typing
from typing import Any, Dict, Protocol, TypeVar

import dm_env
from etils import epath
import flax.linen as nn
import jax
import jax.numpy as jnp
from kauldron.typing import Array, Float, ScalarInt, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import mediapy as media
import numpy as np
import tqdm.auto as tqdm


_SelfT = TypeVar('_SelfT')
ElementType = TypeVar('ElementType')


def _deque_append(
    curr_deque: collections.deque[ElementType], element: ElementType
) -> collections.deque[ElementType]:
  curr_deque.append(element)
  return curr_deque


@typing.runtime_checkable
class Environment(Protocol):
  """Protocol for an environment."""

  def seed(self, seed: int | None) -> None:
    """Set environment random seed to be used after the next reset."""

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment."""

  def render(self, mode: str) -> Array['...']:
    """Return the rendered current environment state."""

  def step(
      self, action: Array['...'] | Mapping[str, Array['...']]
  ) -> dm_env.TimeStep:
    """Steps the environment with the given action."""


@typing.runtime_checkable
class Agent(Protocol):
  """Protocol for an agent."""

  def plan_actions(
      self,
      step_counter: ScalarInt,
      observation_history: Array['...'] | Mapping[str, Array['...']],
      action_history: Array['...'] | Mapping[str, Array['...']] | None = None,
      reward_history: Float['history_length'] | None = None,
  ) -> Mapping[str, Array['...']]:
    """Plan a chunk of actions given history and global context."""


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class EnvRunner:
  """Running an agent inside a given environment.

  Attributes:
    env: The environment to run the agent in.
    agent: The agent to run in the environment.
    max_steps: The maximum number of steps to run the agent for.
    history_length: The length of observation and action history passed on to
      the agent for planning.
    render: Whether to render the current observation of the environment.
  """

  env: dm_env.Environment
  agent: nn.Module
  max_steps: int
  history_length: int
  render: bool = False

  def run(
      self,
      model_params: Dict[str, jnp.ndarray],
      env_seed: int,
      eval_seed: int,  # pylint: disable=unused-argument
  ) -> dict[str, Any]:
    """Runs the agent in the environment.

    Args:
      model_params: Model params.
      env_seed: The seed for the environment.
      eval_seed: The seed for the evaluation. Used for random number generation
        in the agent module.

    Returns:
      A dictionary with rewards and videos
    """
    # Check to make sure environment and agent have the expected interface
    assert isinstance(self.env, Environment) and isinstance(
        self.env, dm_env.Environment
    ), (
        'Given environment should be a dm_env.Environment implementing the'
        ' required Environment protocol.'
    )
    assert isinstance(self.agent, Agent) and isinstance(
        self.agent, nn.Module
    ), (
        'Given agent should be an nn.Module implementing the required Agent'
        ' protocol.'
    )
    # Initialize context
    kd_context = collections.defaultdict(collections.deque)

    # Set the seed for the environment
    self.env.seed(env_seed)
    # Reset the environment and get the initial timestep
    timestep = self.env.reset()
    if self.render:
      rgb = self.env.render('rgb_array')
      kd_context['video'].append(rgb)

    # Initialize observation history with the first observation
    observation_history = jax.tree.map(
        lambda x: collections.deque(
            [x for _ in range(self.history_length)], maxlen=self.history_length
        ),
        timestep.observation,
    )
    # Initialize action history with dummy actions
    action_history = jax.tree.map(
        lambda x: collections.deque(
            [x.generate_value() for _ in range(self.history_length)],
            maxlen=self.history_length,
        ),
        self.env.action_spec(),
    )
    reward_history = collections.deque(
        [0.0] * self.history_length, maxlen=self.history_length
    )

    # Run the policy in the environment
    step_counter = 0
    is_done = timestep.last()
    with tqdm.tqdm(total=self.max_steps, desc='EnvRunner') as pbar:
      while not is_done:
        # Plan a chunk of actions given history
        preds = plan_actions(
            model=self.agent,
            params=model_params,
            step_counter=jnp.array(step_counter, jnp.int32),
            observation_history=jax.tree.map(jnp.stack, observation_history),
            action_history=jax.tree.map(jnp.stack, action_history),
            reward_history=jnp.array(reward_history, jnp.float32),
        )
        actions = preds['actions']
        # Execute actions
        for action in actions:
          timestep = self.env.step(action)
          kd_context['rewards'].append(timestep.reward)
          reward_history.append(timestep.reward)

          # Optionally record images
          if self.render:
            rgb = self.env.render('rgb_array')
            kd_context['video'].append(rgb)

          # Update observation and action history
          observation_history = jax.tree.map(
              _deque_append, observation_history, timestep.observation
          )
          action_history = jax.tree.map(_deque_append, action_history, action)
          # Check to see if episode is done
          step_counter += 1
          pbar.update(1)
          pbar.set_postfix(reward=timestep.reward)
          if timestep.last() or (
              self.max_steps is not None and step_counter >= self.max_steps
          ):
            is_done = True
            break

    kd_context = jax.tree.map(
        jnp.stack,
        kd_context,
        is_leaf=lambda x: isinstance(x, collections.deque),
    )
    return kd_context


@functools.partial(jax.jit, static_argnames='model')
@typechecked
def plan_actions(
    model: nn.Module,
    params,  # pylint: disable=missing-arg-type
    step_counter: ScalarInt,
    observation_history: Array['...'] | Mapping[str, Array['...']],
    action_history: Array['...'] | Mapping[str, Array['...']] | None = None,
    reward_history: Array['...'] | None = None,
) -> dict[str, Array['...']]:
  """Functions for planning actions given history."""
  preds = model.apply(
      params,
      step_counter=step_counter,
      observation_history=observation_history,
      action_history=action_history,
      reward_history=reward_history,
      method='plan_actions',
  )
  return preds


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class EvalInEnv:
  """Evaluator for a set of environment and evaluation seed tuples."""

  env: Environment
  model: nn.Module
  workdir: str
  max_steps: int
  history_length: int
  env_eval_seed_tuples: Sequence[tuple[int, int]] = ((42, 42),)
  render: bool = False
  render_resolution: tuple[int, int] | None = None
  render_fps: int = 50
  save_video: bool = True

  @functools.cached_property
  def env_runner(self) -> EnvRunner:
    """Environment runner."""
    env_runner = EnvRunner(
        env=self.env,  # pytype: disable=wrong-arg-types
        agent=self.model,
        history_length=self.history_length,
        max_steps=self.max_steps,
        render=self.render,
    )
    return env_runner

  def evaluate(
      self,
      model_params: Dict[str, jnp.ndarray],
      step: int,
      return_context: bool = False,
      env_eval_seed_tuples: Sequence[tuple[int, int]] | None = None,
      save_video: bool | None = None,
  ) -> tuple[dict[str, float], dict[tuple[int, int], Any]]:
    """Run one full evaluation."""
    all_contexts = {}
    all_reward_sums = []
    base_dir = epath.Path(self.workdir) / f'step{step}'
    if self.render:
      base_dir.mkdir(parents=True, exist_ok=True)

    if env_eval_seed_tuples is None:
      env_eval_seed_tuples = self.env_eval_seed_tuples

    save_video = save_video if save_video else self.save_video
    for _, (env_seed, eval_seed) in enumerate(env_eval_seed_tuples):
      with jax.spmd_mode('allow_all'), jax.transfer_guard('allow'):
        # Run the evaluation for a pair of environment and evaluation seed, and
        # aggregate all relevant info into kd_context.
        kd_context = self.env_runner.run(
            model_params=model_params,
            env_seed=env_seed,
            eval_seed=eval_seed,
        )
        # Save video
        if self.render and save_video:
          if self.render_resolution is not None:
            video = media.resize_video(
                kd_context['video'], self.render_resolution
            )
          else:
            video = kd_context['video']

          video_path = (
              base_dir / f'env_seed_{env_seed}_eval_seed_{eval_seed}.mp4'
          )
          try:
            media.write_video(video_path, video, fps=self.render_fps)
          except:  # pylint: disable=bare-except
            print('Failed to write video to', video_path)

        if return_context:
          all_contexts[(env_seed, eval_seed)] = kd_context

        # Gather metrics and summaries
        reward_sum = sum(kd_context['rewards'])
        all_reward_sums.append(reward_sum)

    metrics = {
        'reward_mean': np.mean(all_reward_sums),
        'reward_se': np.std(all_reward_sums) / np.sqrt(len(all_reward_sums)),
    }
    return metrics, all_contexts
