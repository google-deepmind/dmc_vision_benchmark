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

"""Preprocessing Kauldron independent transforms."""

from typing import Any, Callable

import einops
import jax.tree_util
import numpy as np
import tensorflow as tf

from dmc_vision_benchmark.data import dmc_vb_info


def flatten_with_path(
    pytree,
    *,
    prefix: str = '',
    separator: str | None = '_',
    is_leaf: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
  """Flatten any PyTree / ConfigDict into a dict with 'keys.like[0].this'."""
  flat_tree_items, _ = jax.tree_util.tree_flatten_with_path(
      pytree, is_leaf=is_leaf
  )
  prefix = (jax.tree_util.GetAttrKey(prefix),) if prefix else ()

  def _format_path(jax_path):
    jax_path = tuple([p.key for p in jax_path])
    path = prefix + jax_path
    if separator is None:
      return str(path)
    else:
      return separator.join(str(p) for p in path)

  return {_format_path(jax_path): value for jax_path, value in flat_tree_items}


def create_dm_control_state(
    step,
    domain_name: str,
    target_key: str = 'states',
    normalize: bool = True,
) -> Any:
  """Create state for each VD4RL domain."""
  to_concat = []
  zero_dim_state_fields = dmc_vb_info.get_zero_dim_state_fields(domain_name)
  for field in dmc_vb_info.get_state_fields(domain_name):
    value = step[f'observation_{field}']

    if field in zero_dim_state_fields:
      to_concat.append(value[..., None])
    else:
      to_concat.append(value)

  state = tf.concat(to_concat, axis=-1)
  if normalize:
    mean = tf.convert_to_tensor(
        dmc_vb_info.get_state_mean(domain_name), dtype=tf.float64
    )
    std = tf.convert_to_tensor(
        dmc_vb_info.get_state_std(domain_name), dtype=tf.float64
    )
    state = (state - mean) / std
  step[target_key] = state
  return step


@tf.function
def resize_images(step, keys, img_height, img_width):
  """Rescale images in matching keys.

  Args:
    step: dataset step
    keys: keys containing image values
    img_height: target height in px
    img_width: target width in px

  Returns:
    Updated step with rescaled images.
  """

  def resize(element):
    # Determine resize method based on dtype (e.g. segmentations are int).
    batch_dims = tf.shape(element)[:-3]
    flat_imgs = einops.rearrange(element, '... h w c -> (...) h w c')
    method = 'nearest' if element.dtype.is_integer else 'area'
    resized_imgs = tf.image.resize(
        flat_imgs, (img_height, img_width), method=method
    )
    return tf.reshape(
        resized_imgs,
        tf.concat([batch_dims, tf.shape(resized_imgs)[-3:]], axis=0),
    )

  step_updated = step.copy()
  for k, v in step.items():
    if k in keys:
      step_updated[k] = resize(v)
  return step_updated


@tf.function
def value_range(
    step,
    keys,
    vrange: tuple[float, float],
    in_vrange: tuple[float, float] = (0.0, 255.0),
    clip_values: bool = True,
    dtype: Any = tf.float32,
):
  """Rescale images in matching keys.

  Args:
    step: dataset step
    keys: keys containing image values
    vrange: output range
    in_vrange: input range
    clip_values: whether to clip values
    dtype: dtype

  Returns:
    Updated data with values in range.
  """

  def set_vrange(element):
    element = tf.cast(element, dtype=dtype)

    in_min, in_max = in_vrange
    out_min, out_max = vrange
    element = (element - in_min) / (in_max - in_min)
    element = element * (out_max - out_min) + out_min
    if clip_values:
      element = tf.clip_by_value(element, out_min, out_max)
    return element

  step_updated = step.copy()
  for k, v in step.items():
    if k in keys:
      step_updated[k] = set_vrange(v)
  return step_updated


@tf.function
def rename(step, rename_dict: dict[str, str]):
  """Rename keys in flattened dict.

  Args:
    step: dataset step
    rename_dict: dict of key names to rename

  Returns:
    Updated data with values in range.
  """
  step_updated = step.copy()
  for k, v in step.items():
    if k in rename_dict.keys():
      del step_updated[k]
      step_updated[rename_dict[k]] = v
  return step_updated


@tf.function
def transform_rand_shift(step, keys, img_pad, img_height, img_width):
  """Pad image and take random crop. Equivalent to random shift.

  Inspired by VD4RL BC baseline (DrQv2):
  https://github.com/conglu1997/v-d4rl/blob/9dcca0984faaddaaedce6975f4d9e6ef5977ad6f/drqbc/drqv2.py#L14

  Args:
    step: dataset step
    keys: keys containing image values
    img_pad: padding size.
    img_height: final image height
    img_width: final image width

  Returns:
    Updated epiosde
  """

  def pad(element):
    batch_dims = len(element.shape[:-3])
    padding = ((0, 0),) * batch_dims + (
        (img_pad, img_pad),
        (img_pad, img_pad),
        (0, 0),
    )
    return tf.pad(element, padding, mode='SYMMETRIC')

  def random_crop(element, target_shape):
    """Randomly crop the element to the target shape."""
    shape = tf.shape(element)
    # resolve dynamic portions of self.shape to a static target_shape
    target_shape = _get_target_shape(element, target_shape)
    # compute the range of the offset for the tf.slice
    offset_range = shape - target_shape
    clipped_offset_range = tf.clip_by_value(offset_range, 1, tf.int32.max)
    # randomly sample offsets from the desired range via modulo
    rand_int = tf.random.uniform(
        [shape.shape[0]], maxval=1_000_000, dtype=tf.int32
    )
    offset = tf.where(offset_range > 0, rand_int % clipped_offset_range, 0)
    return tf.slice(element, offset, target_shape)  # crop

  step_updated = step.copy()
  target_shape = (None, img_height, img_width, None)
  for k, v in step.items():
    if k in keys:
      padded_v = pad(v)
      step_updated[k] = random_crop(padded_v, target_shape)
  return step_updated


def _get_target_shape(element, target_shape):
  """Resolve the target_shape."""
  finale_shape = []
  for static_dim, target_dim in zip(element.shape, target_shape):
    if target_dim is not None:
      finale_shape.append(target_dim)
    else:
      finale_shape.append(static_dim)
  return finale_shape


@tf.function
def transform_extract_frame_stack(
    step,
    frame_stack: int,
    add_state: bool = False,
    add_rewards: bool = False,
    add_obs: bool = True,
):
  """Extract frame_stack from action and state.

  Args:
    step: Dataset step
    frame_stack: Number of frames stacked together.
    add_state: Whether we have access to state.
    add_rewards: Whether we keep future rewards.
    add_obs: Whether we keep observations

  Returns:
    The processed episode.
  """
  step_updated = {}

  # Do not process observations
  if add_obs:
    step_updated['obs'] = step['obs']

  # Process actions
  step_updated['actions'] = tf.gather(
      step['actions'], axis=0, indices=(frame_stack - 1,)
  )

  # Optionally process state
  if add_state:
    step_updated['states'] = tf.gather(
        step['states'], axis=0, indices=(frame_stack - 1,)
    )

  # Optionally process future rewards for TD3+BC
  if add_rewards:
    assert step['rewards'].shape[-1] == frame_stack + 1

    step_updated['future_rewards'] = tf.gather(
        step['rewards'], axis=0, indices=(frame_stack,)
    )
    is_last_step = tf.gather(step['is_last'], axis=0, indices=(frame_stack,))
    step_updated['is_last'] = tf.cast(is_last_step, dtype=tf.float32)

  # If we have both states and future rewards, then we use TD3+BC on states
  # and we need to add future states
  if add_state and add_rewards:
    step_updated['future_states'] = tf.gather(
        step['states'], axis=0, indices=(frame_stack,)
    )
  return step_updated


@tf.function
def transform_for_idm(
    step,
    frame_stack: int,
    nsteps_idm: int | None,
    sample_idm_step: int | None,
):
  """Transform observations into inputs for an inverse model.

  Args:
    step: Dataset step
    frame_stack: Number of frames stacked together.
    nsteps_idm: Number of steps we look into the to extract the future
      observations used for inverse dynamics. Note that we also stack future
      frames.
    sample_idm_step: If True, then for each observation, we sample the effective
      nsteps_idm uniformly between 1 and nsteps_idm.

  Returns:
    The processed episode.
  """
  episode_length = frame_stack + nsteps_idm
  assert episode_length > 1

  step_updated = step.copy()

  # Add future observations and idm_step
  if sample_idm_step:
    # Frame index between 1 and num_steps - frame_stack
    start_index = tf.random.uniform(
        shape=(1,),
        minval=1,
        maxval=nsteps_idm + 1,
        dtype=tf.int32,
    )
    step_updated['idm_step'] = start_index

    # Return frame_stack consecutive columns
    indices = tf.range(frame_stack) + start_index

    step_updated['future_obs'] = tf.gather(
        step['obs'],
        axis=0,
        indices=indices,
    )
  else:
    step_updated['future_obs'] = step['obs'][-frame_stack:]
    step_updated['idm_step'] = np.array([nsteps_idm], dtype=np.int32)

  # Add current observations
  step_updated['obs'] = tf.gather(
      step['obs'], axis=0, indices=tf.range(frame_stack)
  )
  return step_updated
