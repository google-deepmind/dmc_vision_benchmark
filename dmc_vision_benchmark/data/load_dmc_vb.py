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

"""Load DMC Vision Benchmark dataset."""

import dataclasses
import functools
import os
from typing import Any, Sequence

import rlds
import tensorflow_datasets as tfds

from dmc_vision_benchmark.data import dmc_vb_info
from dmc_vision_benchmark.data import preprocessing
from dmc_vision_benchmark.data import tfds as kd_tfds


@dataclasses.dataclass(frozen=True)
class DataLoader:
  train_dataset: Any
  eval_dataset: Any
  action_dim: int
  gt_state_dim: int
  cameras: Sequence[str]


def load_data(
    domain_name: str,
    target_hidden: bool,
    train_split: str,
    eval_split: str,
    **kwargs,
):
  """Returns a data loader for given domain."""
  ds = make_ds(
      domain_name=domain_name,
      target_hidden=target_hidden,
      split=train_split,
      **kwargs,
  )
  ds = ds.repeat()
  train_dataset = iter(tfds.as_numpy(ds))

  ds = make_ds(
      domain_name=domain_name,
      target_hidden=target_hidden,
      split=eval_split,
      **kwargs,
  )
  eval_dataset = tfds.as_numpy(ds)

  action_dim = dmc_vb_info.get_action_dim(domain_name)
  gt_state_dim = dmc_vb_info.get_state_dim(domain_name)
  cameras = dmc_vb_info.get_camera_fields(domain_name, target_hidden)

  return DataLoader(
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      action_dim=action_dim,
      gt_state_dim=gt_state_dim,
      cameras=cameras,
  )


def path_to_episode_dirs(
    dataset_dir: str,
    domain_name: str,
    task_name: str,
    policy_level: str,
    dynamic_distractors: bool,
    difficulty: str,
    ant_fixed_seed: int | None = None,
) -> str:
  """Returns the directory paths containing the episodes."""

  if domain_name == 'ant':
    if difficulty != 'none':
      raise ValueError('Difficult must be none for ant maze tasks.')
    if task_name not in dmc_vb_info.ANT_MAZE_TASKS:
      raise ValueError(f'task_name {task_name} not in ant maze tasks')

    if ant_fixed_seed is None:
      if policy_level not in ['expert', 'medium']:
        raise ValueError('Only expert/medium data available for antmaze_random')
      return os.path.join(
          dataset_dir,
          'antmaze_random',
          task_name,
          policy_level,
      )
    else:
      if policy_level not in ['expert']:
        raise ValueError('Only expert data available for antmaze_fixed')
      return os.path.join(
          dataset_dir,
          'antmaze_fixed',
          task_name,
          str(ant_fixed_seed),
      )

  else:
    distractor_type = 'dynamic_' if dynamic_distractors else 'static_'
    distractor_type = '' if difficulty == 'none' else distractor_type
    return os.path.join(
        dataset_dir,
        'locomotion',
        f'{domain_name}_{task_name}',
        policy_level,
        f'{distractor_type}{difficulty}',
    )


def make_ds(
    domain_name: str,
    target_hidden: bool,
    dataset_dir: str,
    task_name: str,
    policy_level: str,
    dynamic_distractors: bool,
    difficulty: str,
    add_state: bool,
    only_state: bool,
    add_rewards: bool,
    obs_vrange: tuple[float, float],
    actions_vrange: tuple[float, float],
    img_height: int,
    img_width: int,
    img_pad: int,
    frame_stack: int,
    nsteps_idm: int,
    sample_idm_step: int | None,
    episode_length: int | None,
    batch_size: int,
    shuffle_buffer_size: int | None = 10_000,
    shuffle_files: bool = True,
    split: str = 'train',
    ant_fixed_seed: int | None = None,
):
  """Returns a mixed data pipeline for given DMC-VB dataset."""
  policy_levels = policy_level.split('_')
  splits = split.split('_')
  if len(splits) == 1:
    splits = splits * len(policy_levels)

  data_dirs = []
  for policy_level in policy_levels:
    data_dir = path_to_episode_dirs(
        dataset_dir=dataset_dir,
        domain_name=domain_name,
        task_name=task_name,
        policy_level=policy_level,
        dynamic_distractors=dynamic_distractors,
        difficulty=difficulty,
        ant_fixed_seed=ant_fixed_seed,
    )
    data_dirs.append(data_dir)

  if len(splits) == 1:
    # Load all data together
    dataset = kd_tfds.load_from_dir(
        data_dir=data_dirs,  # Can't be a keyword or it breaks partial.
        split=splits[0],
        episode_length=episode_length,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_files=shuffle_files,
    )
  elif len(splits) == len(policy_levels):
    dataset = None
    for data_dir, split in zip(data_dirs, splits):
      ds = kd_tfds.load_from_dir(
          data_dir=data_dir,  # Can't be a keyword or it breaks partial.
          split=split,
          episode_length=episode_length,
          shuffle_buffer_size=shuffle_buffer_size,
          shuffle_files=shuffle_files,
      )
      if dataset is None:
        dataset = ds
      else:
        dataset = dataset.concatenate(ds)
  else:
    raise ValueError(
        'Split must be either a single split or the same number of splits as'
        ' policy levels.'
    )

  # Flatten by default
  dataset = rlds.transformations.map_steps(
      dataset, transform_step=preprocessing.flatten_with_path
  )

  # Get the list of transformations
  transforms = get_transforms(
      domain_name=domain_name,
      add_state=add_state,
      only_state=only_state,
      add_rewards=add_rewards,
      obs_vrange=obs_vrange,
      actions_vrange=actions_vrange,
      img_height=img_height,
      img_width=img_width,
      img_pad=img_pad,
      frame_stack=frame_stack,
      nsteps_idm=nsteps_idm,
      sample_idm_step=sample_idm_step,
      target_hidden=target_hidden,
  )

  # Apply transormations to each item as needed.
  for transform in transforms:
    dataset = rlds.transformations.map_steps(dataset, transform_step=transform)

  dataset = dataset.batch(batch_size)

  return dataset


def get_transforms(
    domain_name: str,
    add_state: bool,
    only_state: bool,
    add_rewards: bool,
    obs_vrange: tuple[float, float],
    actions_vrange: tuple[float, float],
    img_height: int,
    img_width: int,
    img_pad: int,
    frame_stack: int,
    nsteps_idm: int,
    sample_idm_step: int | None,
    target_hidden: bool,
):
  """Returns a list of transforms to apply to the dataset."""
  # Process actions
  transforms = (
      functools.partial(
          preprocessing.rename,
          rename_dict={
              'action': 'actions',
              'reward': 'rewards',
          },
      ),
      functools.partial(
          preprocessing.value_range,
          keys='actions',
          vrange=actions_vrange,
          in_vrange=(-1.0, 1.0),
      ),
  )

  # Optionally add state
  if add_state:
    transforms += (
        functools.partial(
            preprocessing.create_dm_control_state,
            domain_name=domain_name,
            target_key='states',
            normalize=True,
        ),
    )

  # Optionally do not process observations
  if only_state:
    transforms += (
        functools.partial(
            preprocessing.transform_extract_frame_stack,
            frame_stack=frame_stack,
            add_state=add_state,
            add_rewards=add_rewards,
            add_obs=False,
        ),
    )
    return transforms

  # Process observations
  transforms += (
      functools.partial(
          preprocessing.transform_create_obs,
          domain_name=domain_name,
          target_hidden=target_hidden,
      ),
      functools.partial(
          preprocessing.value_range,
          keys='obs',
          vrange=obs_vrange,
      ),
  )

  # Resize if needed
  if img_height != 64 or img_width != 64:
    transforms += (
        functools.partial(
            preprocessing.resize_images,
            keys='obs',
            img_height=img_height,
            img_width=img_width,
        ),
    )

  # Optionally pad and randomly crop
  if img_pad > 0:
    transforms += (
        functools.partial(
            preprocessing.transform_rand_shift,
            keys='obs',
            img_pad=img_pad,
            img_height=img_height,
            img_width=img_width,
        ),
    )

  # Extract action and state
  transforms += (
      functools.partial(
          preprocessing.transform_extract_frame_stack,
          frame_stack=frame_stack,
          add_state=add_state,
          add_rewards=add_rewards,
          add_obs=True,
      ),
  )

  # Optionally process for IDM
  if nsteps_idm > 0:
    transforms += (
        functools.partial(
            preprocessing.transform_for_idm,
            frame_stack=frame_stack,
            nsteps_idm=nsteps_idm,
            sample_idm_step=sample_idm_step,
        ),
    )
  return transforms
