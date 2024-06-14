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

"""Load env for DMC vision benchmark."""

import dm_env
from dmc_vision_benchmark.environments import antmaze_env
from dmc_vision_benchmark.environments import distracting_suite
from dmc_vision_benchmark.environments import env_wrappers


def make_env(
    background_dataset_path: str,
    domain_name: str,
    task_name: str,
    action_repeats: int = 1,
    frame_stack: int | None = 1,
    height: int = 64,
    width: int = 64,
    difficulty: str | None = None,
    background_dataset_videos: str = 'train',
    dynamic: bool = False,
    seed: int | None = 0,
    ant_random_start_end: bool = True,
    propagate_seed_to_env: bool = True,
) -> dm_env.Environment:
  """Create evaluation environment for DMC_VB, used to generate data.

  Args:
    background_dataset_path: path to dataset used for background distractors.
    domain_name: e.g. cheetah, halfcheetah, hopper, walker2d, ant
    task_name: e.g. walk, run
    action_repeats: number of steps to repeat each action for.
    frame_stack: observation is a stack of n frames.
    height: render height.
    width: render width.
    difficulty: distractor difficulty specified in distracting control suite.
    background_dataset_videos: String ('train'/'val')
    dynamic: Boolean controlling whether distractions are dynamic or static.
    seed: seed to init states.
    ant_random_start_end: Whether to randomize ant mazestart and goal positions.
    propagate_seed_to_env: Whether to propagate seed to ant composer env. If
      False, there is noise in different environment inits with the same seed.

  Returns:
    A dm_env environment.
  """
  if domain_name == 'ant':
    if difficulty != 'none':
      raise ValueError('For ant-maze, difficulty must be none.')

    env = antmaze_env.make_env(
        seed=seed,
        maze_name=task_name,
        train_visual_styles=False,
        random_start_end=ant_random_start_end,
        propagate_seed_to_env=propagate_seed_to_env,
    )
    assert width == 64
    assert height == 64
  else:
    difficulty = None if difficulty == 'none' else difficulty
    render_kwargs = dict(height=height, width=width)
    env = distracting_suite.load(
        domain_name,
        task_name,
        pixels_only=False,
        difficulty=difficulty,
        dynamic=dynamic,
        background_dataset_path=background_dataset_path,
        background_dataset_videos=background_dataset_videos,
        render_kwargs=render_kwargs,
        task_kwargs={'random': seed},
    )
    # Must apply action repeat wrapper first
    if action_repeats > 1:
      env = env_wrappers.ActionRepeatWrapper(
          environment=env, num_repeats=action_repeats
      )
  if frame_stack is not None:
    # Stacks frame. Add an extra dimension when frame_stack=1
    env = env_wrappers.FrameStackingWrapper(
        environment=env, num_frames=frame_stack, flatten=False
    )
  return env
