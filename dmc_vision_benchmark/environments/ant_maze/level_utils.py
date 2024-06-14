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

"""Utils for working with maze levels."""

from typing import Any, Sequence

from dm_control import composer

from dmc_vision_benchmark.environments.ant_maze import maze_arenas
from dmc_vision_benchmark.environments.ant_maze import maze_tasks
from dmc_vision_benchmark.environments.ant_maze import walker_utils

_START_LOCATION_SUFFIX = "_start"
_GOAL_LOCATION_SUFFIX = "_goal"

FLOOR_STYLES = ["style_01", "style_02", "style_03", "style_04", "style_05"]
WALL_STYLES = ["style_01", "style_02", "style_03", "style_04", "style_05"]


def get_level_name(
    maze_name: str,
    task_name: str,
    start: str,  # random, fixed, or adjacent
    goal: str,  # random or fixed
    seed: int,
) -> str:
  """Construct the level name from components."""
  start_text = f"{start}{_START_LOCATION_SUFFIX}"
  goal_text = f"{goal}{_GOAL_LOCATION_SUFFIX}"
  return f"{maze_name}/{task_name}/{start_text}/{goal_text}/{seed}"


def extract_conditions_from_level_name(level_name: str) -> dict[str, Any]:
  """Get experimental conditions from the composite level name."""
  conditions = dict(
      goal_location="fixed", start_location="fixed", task_name="shortest_path"
  )
  parts = level_name.split("/")
  if len(parts) > 4:
    conditions["seed"] = int(parts[4])
  if len(parts) > 3:
    conditions["goal_location"] = parts[3].removesuffix(_GOAL_LOCATION_SUFFIX)
  if len(parts) > 2:
    conditions["start_location"] = parts[2].removesuffix(_START_LOCATION_SUFFIX)
  if len(parts) > 1:
    conditions["task_name"] = parts[1]
  conditions["maze_name"] = parts[0]
  return conditions


def build_environment_from_level_name(
    level_name: str,
    max_episode_duration: float = 60.0,
    target_reward: float = 0.0,
    cameras: Sequence[str] | None = None,
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
    visual_style: int = 0,
    task_kwargs: dict[str, Any] | None = None,
) -> composer.Environment:
  """Build the named environment.

  Args:
    level_name: composite name for the maze task, in the form
      MAZE_NAME/TASK_NAME/RANDOM_START/RANDOM_GOAL/SEED
    max_episode_duration: episode time limit, in seconds
    target_reward: reward to provide once the target has been reached; 1.0 =>
      equal to max position reward
    cameras: camera observations to include
    use_systematic_seed: if true, contiguous n seeds produce n different
      combinations of agent and target positions
    max_target_position_offset: max offset of the target position
    max_walker_position_offset: max offset of the walker position
    max_walker_joint_offset: max offset of the walker joints
    randomize_walker_rotation: if true, randomize agent spawn rotation
    include_absolute_positions: if true, include position observations
    include_relative_positions: if true, include agent-to-target vector obs
    include_distance_observation: if true, include distance_to_target
    include_target_activated: include ground truth target_activated obs
    include_hidden_target_observations: if true, include camera observations in
      which the target is always invisible
    propagate_seed_to_env: if true, pass the seed the composer environment. It
      will then be used to initialize the walker joint state. Note this does not
      affect the goal and start location initialization.
    visual_style: number that indicates the combination of visual styles for
      floor and walls
    task_kwargs: additional kwargs for the task constructor

  Returns:
    env: the environment
  """
  conditions = extract_conditions_from_level_name(level_name)
  maze_name = conditions["maze_name"]
  seed = conditions["seed"]
  print(f"Building new environment for level_name={level_name}, seed={seed}")

  task_kwargs = task_kwargs or {}

  cameras = cameras or []

  # Create the walker
  walker, walker_cams = walker_utils.make_ant_walker(
      include_egocentric_camera=("walker/egocentric_camera" in cameras),
      include_overhead_camera=("walker/overhead_camera" in cameras),
      include_follow_camera=("walker/follow_camera" in cameras),
      max_position_offset=max_walker_position_offset,
      max_joint_offset=max_walker_joint_offset,
  )

  # Create the maze arena
  maze = maze_arenas.make_maze(
      name=maze_name,
      seed=seed,
      agent_location=conditions["start_location"],
      target_location=conditions["goal_location"],
      use_systematic_seed=use_systematic_seed,
  )
  arena, arena_cams = maze_arenas.make_maze_arena(
      maze=maze,
      include_highres_top_camera=("top_camera" in cameras),
      include_lowres_top_camera=("lowres_top_camera" in cameras),
      floor_style=FLOOR_STYLES[visual_style % len(FLOOR_STYLES)],
      wall_style=WALL_STYLES[visual_style % len(WALL_STYLES)],
    )

  # Verify cameras
  all_cams = walker_cams + arena_cams
  for camera in cameras:
    assert camera in all_cams, f"Missing requested camera {camera}."

  # Create the task
  task = maze_tasks.make_maze_task(
      task_name=conditions["task_name"],
      walker=walker,
      arena=arena,
      cameras=cameras,
      max_episode_duration=max_episode_duration,
      max_target_offset=max_target_position_offset,
      randomize_start_rotation=randomize_walker_rotation,
      target_reward=target_reward,
      include_absolute_positions=include_absolute_positions,
      include_relative_positions=include_relative_positions,
      include_distance_observation=include_distance_observation,
      include_target_activated=include_target_activated,
      include_hidden_target_observations=include_hidden_target_observations,
      **task_kwargs
  )

  # Create the environment
  random_state = seed if propagate_seed_to_env else None
  env_kwargs = {
      "random_state": random_state,
      "time_limit": max_episode_duration,
  }
  env = composer.Environment(
      task=task, strip_singleton_obs_buffer_dim=True, **env_kwargs
  )

  return env
