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

"""TFDS data pipeline."""

from typing import Any, List

import rlds
from rlds import rlds_types
import tensorflow as tf
import tensorflow_datasets as tfds


def episode_steps_to_batched_transition(
    episode: dict[str, Any],
    episode_length: int,
    only_first_window: bool,
    shift: int,
    drop_remainder: bool = False,
) -> dict[str, Any]:
  """Returns a new episode as a series of overlapping windows of steps.

  Also adds a timestep.

  Args:
    episode:  Incoming episode
    episode_length:  Length of windows to use
    only_first_window:  Whether to only return first window, as done in Raparthy
      et al.
    shift:  Increment to compute index to select the next element of each batch
    drop_remainder: Whether to drop the last few steps, or keep all of them and
      create batches padded with dummy observations.
  """
  new_episode = dict(episode)

  # Set the batch to episode_length.
  new_episode[rlds.STEPS] = rlds.transformations.batch(
      new_episode[rlds.STEPS],
      shift=shift,
      size=episode_length,
      drop_remainder=drop_remainder,
  )

  # For each episode, only take first episode_length steps.
  if only_first_window:
    new_episode[rlds.STEPS] = new_episode[rlds.STEPS].take(1)

  # Pad and crop each episode
  new_episode[rlds.STEPS] = new_episode[rlds.STEPS].map(
      lambda x: pad_and_crop_episode(x, episode_length)
  )
  return new_episode


def steps_to_transitions(step: rlds_types.Step) -> rlds_types.Step:
  """Convert a batched step (representing a pair of steps) to a transition in SARS format."""
  new_step = {k: step[k][0] for k in step.keys()}
  new_step["next_observation"] = step[rlds.OBSERVATION][1]
  new_step["observations_concat"] = tf.concat(
      [new_step["observation"], new_step["next_observation"]], 0
  )
  new_step["next_is_terminal"] = step[rlds.IS_TERMINAL][1]
  new_step["next_is_last"] = step[rlds.IS_LAST][1]
  return new_step


def episode_steps_to_transitions(
    episode: rlds_types.Episode,
) -> rlds_types.Episode:
  """Convert an episode into a sequence of transitions in SARS format."""
  episode[rlds.STEPS] = rlds.transformations.batch(
      episode[rlds.STEPS], size=2, stride=1, drop_remainder=True
  ).map(steps_to_transitions)
  return episode


@tf.function
def pad_and_crop_episode(
    episode: dict[str, Any], episode_length: int
) -> dict[str, Any]:
  """Returns a new episode where obs, actions, rewards... are of episode_length.

  Pad all the sequences in an episode to be of episode_length.

  Args:
    episode:  Incoming episode
    episode_length:  Length of windows to use
  """
  result = {}

  def pad_and_crop_all_leaves(ep, res):
    """Recursive function to pad and crop the leaves of a tree."""
    for key, value in ep.items():
      res[key] = {}
      if isinstance(value, dict):
        pad_and_crop_all_leaves(value, res[key])
      else:
        if len(value.shape) >= 2:
          # Overwrite the first entry of tf.shape(value)
          padded = tf.zeros(
              tf.tensor_scatter_nd_update(
                  tf.shape(value),
                  [[0]],
                  [episode_length - tf.shape(value)[0]],
              ),
              dtype=value.dtype,
          )
        else:
          padded = tf.zeros(
              episode_length - tf.shape(value)[0], dtype=value.dtype
          )

        # Concatenate the original values with the padded ones
        concat = tf.concat([value, padded], axis=0)

        # Guarantee the shapes returned - required since drop_remainder=False
        res[key] = tf.ensure_shape(concat, (episode_length,) + value.shape[1:])

  pad_and_crop_all_leaves(episode, result)
  return result


def load_from_dir(
    data_dir: str | List[str],
    split: str = "train",
    rng: Any | None = None,
    episode_length: int | None = None,
    only_first_window: bool = False,
    shift: int = 1,
    subsample_proportion: float = 1.0,
    shuffle_buffer_size: (
        int | None
    ) = 10_000,  # ds_shuffle_buffer_size, to shuffle dataset at episode level
    shuffle_files: bool = False,
    trim_batches_to_episode: bool = False,
    final_step_reward_range: tuple[float, float] | None = None,
) -> tf.data.Dataset:
  """Loads dataset from disk, applying sliding window, etc.  See process_ds."""
  if isinstance(data_dir, str):
    dataset_builder = tfds.builder_from_directories([data_dir])
  else:
    dataset_builder = tfds.builder_from_directories(data_dir)

  ds = dataset_builder.as_dataset(
      split=split,
      shuffle_files=shuffle_files,
  )

  return process_ds(
      ds=ds,
      rng=rng,
      episode_length=episode_length,
      only_first_window=only_first_window,
      shift=shift,
      subsample_proportion=subsample_proportion,
      ds_shuffle_buffer_size=shuffle_buffer_size,
      trim_batches_to_episode=trim_batches_to_episode,
      final_step_reward_range=final_step_reward_range,
  )


def process_ds(
    ds: tf.data.Dataset,
    rng: Any | None = None,
    episode_length: int | None = None,
    only_first_window: bool = False,
    shift: int = 1,
    subsample_proportion: float = 1.0,
    ds_shuffle_buffer_size: int | None = 10_000,
    final_step_reward_range: tuple[float, float] | None = None,
    trim_batches_to_episode: bool = False,
) -> tf.data.Dataset:
  """Returns a a single raw dataset processed as needed.

  Args:
    ds:  Dataset to process
    rng:  Random seed for shuffling
    episode_length:  Length of sliding window to use.
    only_first_window:  Whether to only return first sliding window.
    shift:  Increment to compute index to select the next element of each batch
    subsample_proportion:  Proportion of the dataset to use
    ds_shuffle_buffer_size:  Size of the shuffle buffer applied at the dataset
      level.  This is useful when subsampling the data, and when applying the
      sliding window transform, as the sliding window transform makes the data
      stream contain many sequential elements from the same epsiode.   Hence it
      is a lot more memory effective to apply the shuffle before the sliding
      window transform as well as after.  This is in particular a problem in
      datasets like D4RL medium_expert due to the global structure. Redundant if
      neither sliding window transform nor dataset subsampling is used.
    final_step_reward_range:  Range of rewards at the final step of the episode
    trim_batches_to_episode: Avoid creating extra batches with padded features
      beyond episode length.
  """
  # Filter dataset based on the reward at the final step of the episode
  if final_step_reward_range:
    # Defines a condition function.
    def condition(step):
      return step[rlds.REWARD] > 0.0

    # Truncates dataset after the first step with non-zero reward.
    ds = ds.map(
        lambda episode: {
            rlds.STEPS: rlds.transformations.truncate_after_condition(
                episode[rlds.STEPS], condition
            )
        }
    )
    ds = ds.filter(
        lambda x: (
            rlds.transformations.final_step(x[rlds.STEPS])["reward"]
            >= final_step_reward_range[0]
        )
        and (
            rlds.transformations.final_step(x[rlds.STEPS])["reward"]
            <= final_step_reward_range[1]
        )
    ).cache()

  # Shuffle the dataset before sliding window, etc is applied.
  if rng is not None and ds_shuffle_buffer_size is not None:
    rng = rng.fold_in("ds_shuffle_buffer")
    ds = ds.shuffle(buffer_size=ds_shuffle_buffer_size, seed=int(rng.bits()))

  if subsample_proportion < 1.0:
    ds = ds.take(int(len(ds) * subsample_proportion))

  if episode_length:
    # Note!  If do batch inside of map, then flat_map, then metadata for the
    # element spec fails to record the set length of the input.
    # Needs to be combined in a single step.
    ds = ds.flat_map(
        lambda e: episode_steps_to_batched_transition(
            episode=e,
            episode_length=episode_length,
            only_first_window=only_first_window,
            shift=shift,
            drop_remainder=trim_batches_to_episode,
        )[rlds.STEPS]
    )
  else:
    # You can add any other data transformations here.
    # Remove after adding the ops as preprocessing transformations
    ds = ds.flat_map(lambda episode: episode[rlds.STEPS])
  return ds
