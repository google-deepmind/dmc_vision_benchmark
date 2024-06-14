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

"""Ant-maze environment construction."""

from typing import Sequence

import dm_env
import numpy as np

from dmc_vision_benchmark.environments import env_wrappers
from dmc_vision_benchmark.environments.ant_maze import level_utils

# Default cameras are all 64x64 image resolution.
# Highres "top_camera" is also available
DEFAULT_CAMERAS = (
    "lowres_top_camera",
    "walker/egocentric_camera",
    "walker/follow_camera",
    "walker/overhead_camera",
)

TRAIN_VISUAL_STYLES = [1, 2, 3, 4]
TEST_VISUAL_STYLES = [0]


def make_env(
    seed: int,
    maze_name: str,
    train_visual_styles: bool,
    random_start_end: bool,
    propagate_seed_to_env: bool,
) -> dm_env.Environment:
  """Simpler make_env that sets default params for data generation and eval."""
  rng = np.random.default_rng(seed)
  visual_style = (
      rng.choice(TRAIN_VISUAL_STYLES)
      if train_visual_styles
      else rng.choice(TEST_VISUAL_STYLES)
  )
  task_kwargs = {"include_distance_observation": True}  # task-specific kwargs
  if random_start_end:
    start_location = "random"
    goal_location = "random"
  else:
    start_location = "fixed"
    goal_location = "fixed"
  return make_env_configurable(
      seed=seed,
      maze_name=maze_name,
      task_name="shortest_path",
      start_location=start_location,
      goal_location=goal_location,
      cameras=DEFAULT_CAMERAS,
      max_episode_duration=60.0,
      use_systematic_seed=False,
      max_target_position_offset=0.5,
      max_walker_position_offset=0.25,
      max_walker_joint_offset=0.0,
      randomize_walker_rotation=True,
      include_absolute_positions=True,
      include_relative_positions=True,
      include_target_activated=True,
      include_hidden_target_observations=True,
      propagate_seed_to_env=propagate_seed_to_env,
      action_repeats=2,
      visual_style=visual_style,
      **task_kwargs,
  )


def make_env_configurable(
    seed: int,
    maze_name: str,
    task_name: str,
    start_location: str,
    goal_location: str,
    cameras: Sequence[str] | None = DEFAULT_CAMERAS,
    max_episode_duration: float = 60.0,
    use_systematic_seed: bool = False,
    max_target_position_offset: float = 0.0,
    max_walker_position_offset: float = 0.0,
    max_walker_joint_offset: float = 0.0,
    randomize_walker_rotation: bool = False,
    include_absolute_positions: bool = False,
    include_relative_positions: bool = False,
    include_distance_observation: bool = False,
    include_target_activated: bool = False,
    include_hidden_target_observations: bool = False,
    propagate_seed_to_env: bool = True,
    action_repeats: int = 1,
    visual_style: int = 0,
) -> dm_env.Environment:
  """Create an environment for a given maze and conditions.

  Args:
    seed: random seed that determines agent start position, target position, and
      environment (mujoco) randomization
    maze_name: name of the maze, either a predefined maze structure in
      maze_arenas.MAZE_SPECS, or "randomNxN_SEED" for a randomly generated maze.
    task_name: name of the task (reward structure), either "distance" or
      "shortest_path"
    start_location: fixed or random
    goal_location: fixed or random, or dir-X where X is a cardinal direction
    cameras: list of camera names to include in the observations. Available
      options are top_camera, walker/egocentric_camera, walker/overhead_camera,
      and walker/follow_camera
    max_episode_duration: maximum episode duration, in seconds. The maximum
      number of steps is maximum_episode_duration / env.control_timestep() (by
      default here, env.control_timestep() == 0.025)
    use_systematic_seed: if true, the seed maps systematically to start and goal
      positions, such that for a set of n contiguous seeds, there will be n
      different start-goal configurations (up to the number of possible
      configurations).
    max_target_position_offset: max offset of the target position
    max_walker_position_offset: max offset of the walker position
    max_walker_joint_offset: max offset of the walker joints
    randomize_walker_rotation: randomize agent spawn rotation
    include_absolute_positions: include ground truth position observations
    include_relative_positions: include ground truth agent-to-target vector obs
    include_distance_observation: include distance_to_target obs
    include_target_activated: include ground truth target_activated obs
    include_hidden_target_observations: include camera observations in which the
      target is always invisible
    propagate_seed_to_env: if true, pass the seed the composer environment. It
      will then be used to initialize the walker joint state. Note this does not
      affect the goal and start location initialization.
    action_repeats: number of times to repeat each action in the environment
    visual_style: number that determines visual style of walls and floor

  Returns:
    env: the environment
  """
  if cameras is None:
    cameras = []

  level_name = level_utils.get_level_name(
      maze_name=maze_name,
      task_name=task_name,
      start=start_location,
      goal=goal_location,
      seed=seed,
  )

  env = level_utils.build_environment_from_level_name(
      level_name=level_name,
      max_episode_duration=max_episode_duration,
      cameras=cameras,
      use_systematic_seed=use_systematic_seed,
      max_target_position_offset=max_target_position_offset,
      max_walker_position_offset=max_walker_position_offset,
      max_walker_joint_offset=max_walker_joint_offset,
      randomize_walker_rotation=randomize_walker_rotation,
      include_absolute_positions=include_absolute_positions,
      include_relative_positions=include_relative_positions,
      include_distance_observation=include_distance_observation,
      include_target_activated=include_target_activated,
      include_hidden_target_observations=include_hidden_target_observations,
      propagate_seed_to_env=propagate_seed_to_env,
      visual_style=visual_style,
  )

  if action_repeats > 1:
    env = env_wrappers.ActionRepeatWrapper(
        environment=env, num_repeats=action_repeats
    )
  return env
