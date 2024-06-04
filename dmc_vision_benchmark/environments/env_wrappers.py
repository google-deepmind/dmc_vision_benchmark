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

"""DM env wrapper."""

import collections
from collections.abc import Mapping
import dataclasses
import time
from typing import Any, Callable, Iterable, Union

import dm_env
from dm_env import specs as dm_env_specs
from kauldron.typing import Array  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import tree


# Taken from `acme.types`
NestedArray = Any
NestedSpec = Union[
    dm_env_specs.Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],
]


class EnvironmentWrapper(dm_env.Environment):
  """Environment that wraps another environment.

  Taken from `acme.wrappers.base.EnvironmentWrapper`

  This exposes the wrapped environment with the `.environment` property and also
  defines `__getattr__` so that attributes are invisibly forwarded to the
  wrapped environment (and hence enabling duck-typing).
  """

  _environment: dm_env.Environment

  def __init__(self, environment: dm_env.Environment):
    self._environment = environment

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(
          "attempted to get missing private attribute '{}'".format(name)
      )
    return getattr(self._environment, name)

  @property
  def environment(self) -> dm_env.Environment:
    return self._environment

  # The following lines are necessary because methods defined in
  # `dm_env.Environment` are not delegated through `__getattr__`, which would
  # only be used to expose methods or properties that are not defined in the
  # base `dm_env.Environment` class.

  def step(self, action) -> dm_env.TimeStep:
    return self._environment.step(action)

  def reset(self) -> dm_env.TimeStep:
    return self._environment.reset()

  def action_spec(self):
    return self._environment.action_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def observation_spec(self):
    return self._environment.observation_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def close(self):
    return self._environment.close()


class ActionRepeatWrapper(EnvironmentWrapper):
  """Action repeat wrapper.

  Taken from `acme.wrappers.action_repeat.ActionRepeatWrapper`
  """

  def __init__(self, environment: dm_env.Environment, num_repeats: int = 1):
    super().__init__(environment=environment)
    self._num_repeats = num_repeats

  def step(self, action: NestedArray) -> dm_env.TimeStep:
    # Initialize accumulated reward and discount.
    reward = 0.
    discount = 1.

    # Step the environment by repeating action.
    for _ in range(self._num_repeats):
      timestep = self._environment.step(action)

      # Accumulate reward and discount.
      reward += timestep.reward * discount
      discount *= timestep.discount

      # Don't go over episode boundaries.
      if timestep.last():
        break

    # Replace the final timestep's reward and discount.
    return timestep._replace(reward=reward, discount=discount)


@dataclasses.dataclass
class EvaluationEnvironmentWrapper(EnvironmentWrapper):
  """Wrapper for eval with render and seed fns, recreates env on reset."""

  make_env: Callable[..., dm_env.Environment]
  make_env_kwargs: dict[str, Any]
  pixels_key: str = 'pixels'

  _seed: int | None = 0
  _last_obs: Any | None = None

  def __post_init__(self):
    # Make sure we have environment ready if something queries us.
    self.seed(0)
    self.reset()

  def seed(self, seed: int | None) -> None:
    """Set environment random seed to be used after next reset."""
    self._seed = seed

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment."""
    # Some environments depend on the global numpy seed.
    np.random.seed(self._seed)
    kwargs = dict(self.make_env_kwargs)
    kwargs['seed'] = self._seed
    self._environment = self.make_env(**kwargs)

    step = self._environment.reset()
    self._last_obs = step.observation
    return step

  def render(self, mode: str) -> Array['...']:
    """Return the rendered current environment state."""

    del mode  # Unused and unsure what it is for

    assert self._last_obs is not None
    pixels = self._last_obs[self.pixels_key]
    if pixels.ndim == 3:
      return pixels

    if pixels.ndim == 4:  # to extract h w c when using framestacking
      return pixels[..., -1]

    raise ValueError('Unsupported dimension for pixels: %d' % pixels.ndim)

  def step(
      self, action: Array['...'] | Mapping[str, Array['...']]
  ) -> dm_env.TimeStep:
    """Steps the environment with the given action."""

    step = self._environment.step(action)
    self._last_obs = step.observation

    return step


class FrameStackingWrapper(EnvironmentWrapper):
  """Wrapper that stacks observations along a new final axis.

  If no history, copy first observation into history.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      num_frames: int = 4,
      flatten: bool = False,
  ):
    """Initializes a new FrameStackingWrapper.

    Args:
      environment: Environment.
      num_frames: Number frames to stack.
      flatten: Whether to flatten the channel and stack dimensions together.
    """
    super().__init__(environment=environment)
    original_spec = self._environment.observation_spec()
    self._stackers = tree.map_structure(
        lambda _: FrameStacker(num_frames=num_frames, flatten=flatten),
        self._environment.observation_spec(),
    )
    self._observation_spec = tree.map_structure(
        lambda stacker, spec: stacker.update_spec(spec),
        self._stackers,
        original_spec,
    )

  def _process_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = tree.map_structure(
        lambda stacker, x: stacker.step(x), self._stackers, timestep.observation
    )
    return timestep._replace(observation=observation)

  def reset(self) -> dm_env.TimeStep:
    for stacker in tree.flatten(self._stackers):
      stacker.reset()
    return self._process_timestep(self._environment.reset())

  def step(self, action: int) -> dm_env.TimeStep:
    return self._process_timestep(self._environment.step(action))

  def observation_spec(self) -> NestedSpec:
    return self._observation_spec


class FrameStacker:
  """Simple class for frame-stacking observations."""

  def __init__(self, num_frames: int, flatten: bool = False):
    self._num_frames = num_frames
    self._flatten = flatten
    self.reset()

  @property
  def num_frames(self) -> int:
    return self._num_frames

  def reset(self):
    self._stack = collections.deque(maxlen=self._num_frames)

  def step(self, frame: np.ndarray) -> np.ndarray:
    """Append frame to stack and return the stack."""
    if not self._stack:
      # Fill stack with blank frames if empty.
      self._stack.extend([frame] * (self._num_frames - 1))
    self._stack.append(frame)
    stacked_frames = np.stack(self._stack, axis=-1)

    if not self._flatten:
      return stacked_frames
    else:
      new_shape = stacked_frames.shape[:-2] + (-1,)
      return stacked_frames.reshape(*new_shape)

  def update_spec(self, spec: dm_env_specs.Array) -> dm_env_specs.Array:
    if not self._flatten:
      new_shape = spec.shape + (self._num_frames,)
    else:
      new_shape = spec.shape[:-1] + (self._num_frames * spec.shape[-1],)
    return dm_env_specs.Array(shape=new_shape, dtype=spec.dtype, name=spec.name)


class RandomHandInitWrapper(EnvironmentWrapper):
  """Wrapper that randomly initializes the robot hand position.

  If seed is None, use time as seed.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      seed: int | None = None,
      offset: float = 0.05,
  ):
    super().__init__(environment=environment)
    if seed is None:
      seed = int(time.time())
    self._init_rng = np.random.default_rng(seed=seed)
    # hand_init_pos is in metaworld env which is wrapped by metal env
    self._default_hand_pos = self._env.hand_init_pos
    self.offset = np.array([offset, offset, offset])

  def _init_hand(self):
    default = self._default_hand_pos
    init_pos = self._init_rng.uniform(
        low=default - self.offset, high=default + self.offset
    )
    self._env.hand_init_pos = init_pos

  def reset(self) -> dm_env.TimeStep:
    self._init_hand()
    return self._environment.reset()


class FlipImagesWrapper(EnvironmentWrapper):
  """Wrapper that flips images, as metaworld images are upside down."""

  def __init__(self, environment: dm_env.Environment, rgb_key: str):
    super().__init__(environment=environment)
    self.rgb_key = rgb_key

  def _flip_images(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = timestep.observation
    observation[self.rgb_key] = np.flipud(timestep.observation[self.rgb_key])
    return timestep._replace(observation=observation)

  def reset(self) -> dm_env.TimeStep:
    return self._flip_images(self._environment.reset())

  def step(self, action: int) -> dm_env.TimeStep:
    return self._flip_images(self._environment.step(action))


class CropImagesWrapper(EnvironmentWrapper):
  """Wrapper that flips images, as metaworld images are upside down."""

  def __init__(
      self, environment: dm_env.Environment, resolution: int, rgb_key: str
  ):
    super().__init__(environment=environment)
    self._res = int(resolution / 3)  # Used to crop the image
    self.rgb_key = rgb_key
    self._obs_spec = environment.observation_spec()
    self._obs_spec[self.rgb_key] = dm_env.specs.Array(
        shape=(resolution, resolution, 3),
        dtype=self._obs_spec[self.rgb_key].dtype,
        name=self._obs_spec[self.rgb_key].name,
    )

  def _crop_images(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = timestep.observation
    observation[self.rgb_key] = timestep.observation[self.rgb_key][
        2 * self._res : 2 * self._res + 3 * self._res,
        2 * self._res : 2 * self._res + 3 * self._res,
    ]
    return timestep._replace(observation=observation)

  def observation_spec(self):
    return self._obs_spec

  def reset(self) -> dm_env.TimeStep:
    return self._crop_images(self._environment.reset())

  def step(self, action: int) -> dm_env.TimeStep:
    return self._crop_images(self._environment.step(action))
