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

"""Maze tasks with dense reward, compatible with dm_control/locomotion."""
import functools

from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import ant
from dm_control.mjcf import traversal_utils
import numpy as np

from dmc_vision_benchmark.environments.ant_maze import fast_marching

NullGoalMazeTask = random_goal_maze.NullGoalMaze

Physics = mjcf.physics.Physics


class HideableTargetSphere(target_sphere.TargetSphere):
  """A target sphere prop that can be manually hidden or revealed."""

  _hidden: bool = False

  def set_hidden(self, val: bool):
    self._hidden = val

  def reveal(self, physics):
    self._hidden = False
    physics.bind(self._material).rgba[-1] = 1

  def hide(self, physics):
    self._hidden = True
    physics.bind(self._material).rgba[-1] = 0

  def reset(self, physics):
    self._activated = False
    if not self._hidden:
      self.reveal(physics)


class DenseRewardSingleGoalMazeTask(NullGoalMazeTask):
  """Abstract parent class for all dense-reward single-goal maze tasks."""

  def __init__(
      self,
      walker: ant.Ant,
      maze_arena: mazes.MazeWithTargets,
      cameras: list[str],
      max_episode_duration: float,
      target_reward: float = 1.0,
      aliveness_reward: float = 0.0,
      aliveness_threshold: float = -1.0,  # [-1.0, 0.0]
      enable_global_task_observables: bool = False,
      contact_termination: bool = False,
      target_termination: bool = True,
      max_target_offset: float = 0.0,
      randomize_start_rotation: bool = False,
      include_absolute_positions: bool = False,
      include_relative_positions: bool = False,
      include_distance_observation: bool = False,
      include_target_activated: bool = False,
      include_hidden_target_observations: bool = False,
      verbose: bool = False,
  ):
    """Create a DenseRewardSingleGoalMazeTask.

    Args:
      walker: the ant walker to include in the task
      maze_arena: the maze arena on which to build the task
      cameras: list of camera names
      max_episode_duration: maximum episode duration in seconds
      target_reward: reward to obtain upon having reached the target
      aliveness_reward: reward to receive the walker is 'alive'
      aliveness_threshold: aliveness level above which the walker is considered
        'alive'.  For ant walker, -1 implies always considered alive (Aliveness
        is mainly important for walkers that cannot right themselves)
      enable_global_task_observables: if true, enable additional ground truth
        observations, including the maze grid.
      contact_termination: if true, terminate the episode if non-foot geoms
        contact the ground
      target_termination: if true, terminate episode upon touching the target
      max_target_offset: maximum offset of target position
      randomize_start_rotation: if true, randomize walker start rotation
      include_absolute_positions: if true, include walker_position and
        target_position observations
      include_relative_positions: if true, include target_vector and
        egocentric_target_vector observations
      include_distance_observation: if true, include distance_to_target obs
      include_target_activated: include target_activated boolean, which is true
        when the target is reached
      include_hidden_target_observations: if true, add camera observations in
        which the target is invisible
      verbose: include debugging outputs

    Returns:
      task: the maze task
    """
    is_egocentric_camera_enabled = walker.observables.egocentric_camera.enabled
    super().__init__(
        walker=walker,
        maze_arena=maze_arena,
        aliveness_reward=aliveness_reward,
        aliveness_threshold=aliveness_threshold,
        enable_global_task_observables=enable_global_task_observables,
        contact_termination=contact_termination,
        randomize_spawn_position=True,  # enable randomization across cells
        randomize_spawn_rotation=randomize_start_rotation,
    )
    self._verbose = verbose
    # Note: Whether or not the start position is actually randomized is also
    # determined by the arena ("fixed_agent_location" arg).
    self._max_episode_duration = max_episode_duration
    self._target_reward = target_reward
    self._target_termination = target_termination
    self._max_target_offset = max_target_offset
    walker.observables.egocentric_camera.enabled = is_egocentric_camera_enabled

    self._target = HideableTargetSphere(radius=0.5)
    self._maze_arena.attach(self._target)

    self._add_position_observables(
        include_absolute_positions,
        include_relative_positions,
        include_target_activated,
    )

    if include_hidden_target_observations:
      self._add_hidden_target_observables(cameras)

    if include_distance_observation:
      self._add_distance_observable()

  @property
  def maze_arena(self):
    return self._maze_arena

  @property
  def walker(self):
    return self._walker

  def _get_walker_pos(self, physics: Physics):
    return physics.bind(self._walker.root_body).xpos

  def _get_target_pos(self, physics: Physics):
    return physics.bind(self._target.geom).xpos

  def _get_target_activated(self, _: Physics):
    return self._target.activated

  def _add_position_observables(
      self,
      include_absolute_positions: bool,
      include_relative_positions: bool,
      include_target_activated: bool,
  ):
    """Add absolute and/or relative position observables."""

    def get_vector_to_target(physics: Physics):
      return self._get_target_pos(physics) - self._get_walker_pos(physics)

    if include_absolute_positions:
      # Add walker_position observable
      walker_position_obs = observable.Generic(self._get_walker_pos)
      walker_position_obs.enabled = True
      self._task_observables["walker_position"] = walker_position_obs
      # Add target_position observable
      target_position_obs = observable.Generic(self._get_target_pos)
      target_position_obs.enabled = True
      self._task_observables["target_position"] = target_position_obs

    if include_relative_positions:
      # Add egocentric_target_vector
      self.walker.observables.add_egocentric_vector(
          "egocentric_target_vector",
          observable.MJCFFeature("pos", self._target.geom),
          origin_callable=self._get_walker_pos,
      )
      # Add target_vector
      target_vector_obs = observable.Generic(get_vector_to_target)
      target_vector_obs.enabled = True
      self._task_observables["target_vector"] = target_vector_obs

    if include_target_activated:
      # Add target activated observable
      target_activated_obs = observable.Generic(self._get_target_activated)
      target_activated_obs.enabled = True
      self._task_observables["target_activated"] = target_activated_obs

  def _add_distance_observable(self):

    def _get_distance_to_target(physics: Physics):
      walker_pos = self._get_walker_pos(physics)
      target_pos = self._get_target_pos(physics)
      return np.linalg.norm(target_pos - walker_pos)

    distance_obs = observable.Generic(_get_distance_to_target)
    distance_obs.enabled = True
    self._task_observables["distance_to_target"] = distance_obs

  def _add_hidden_target_observables(self, cameras: list[str]):
    """Add a hidden_target variant for each camera observable."""

    def _get_hidden_target_observation(camera: str, physics: Physics):
      self._target.hide(physics)
      img = self.get_observation(camera, physics)
      self._target.reveal(physics)
      return img

    for cam_name in cameras:
      obs = observable.Generic(
          functools.partial(_get_hidden_target_observation, cam_name)
      )
      obs.enabled = True
      self._task_observables[f"{cam_name}_hidden_target"] = obs

  def _get_max_maze_distance(self):
    h = self.maze_arena.maze.height
    w = self.maze_arena.maze.width
    return np.sqrt(h*h + w*w) * self.maze_arena.xy_scale

  def get_observation(self, name: str, physics: Physics):
    return self.observables[name].observation_callable(physics)()

  def initialize_episode_mjcf(self, random_state):
    super().initialize_episode_mjcf(random_state)
    self._initialize_target_position(random_state)

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    self._target.reset(physics)

  def should_terminate_episode(self, physics: Physics) -> bool:
    if super().should_terminate_episode(physics):
      return True
    if self._target_termination and self._target.activated:
      return True
    if physics.time() >= self._max_episode_duration:
      return True
    return False

  def _get_target_reward(self):
    if self._target.activated:
      return self._target_reward
    else:
      return 0

  def _initialize_target_position(self, random_state: np.random.RandomState):
    pos_index = random_state.randint(0, len(self._maze_arena.target_positions))
    centered_target_position = self._maze_arena.target_positions[pos_index]
    if self._max_target_offset > 0:
      x_offset = random_state.uniform(
          -self._max_target_offset, self._max_target_offset)
      y_offset = random_state.uniform(
          -self._max_target_offset, self._max_target_offset)
      target_offset = np.array([x_offset, y_offset, 0])
      self._target_position = centered_target_position + target_offset
    else:
      self._target_position = centered_target_position
    target_frame = traversal_utils.get_attachment_frame(self._target.mjcf_model)
    target_frame.pos = self._target_position


class DistanceRewardSingleGoalMazeTask(DenseRewardSingleGoalMazeTask):
  """A maze task with reward based on distance from walker to target."""

  def get_reward(self, physics: Physics):
    walker_pos = self._get_walker_pos(physics)
    position_reward = self._get_reward_for_position(walker_pos, physics)
    target_reward = self._get_target_reward()
    reward = position_reward + self._aliveness_reward + target_reward
    if self._verbose or np.isnan(reward):
      print(
          f"reward: pos={position_reward}, alive={self._aliveness_reward},"
          f" target={target_reward}"
      )  # pylint: disable=line-too-long
      assert not np.isnan(reward)
    return reward

  def _get_reward_for_position(self, pos: np.ndarray, physics: Physics):
    distance = self._get_distance_from_position(pos, physics)
    max_distance = self._get_max_maze_distance()
    dist_reward = -distance / max_distance
    return dist_reward

  def _get_distance_from_position(self, pos: np.ndarray, physics: Physics):
    target_pos = self._get_target_pos(physics)
    distance = np.linalg.norm(target_pos - pos)
    return distance


class ShortestPathRewardSingleGoalMazeTask(DenseRewardSingleGoalMazeTask):
  """A maze task with reward based on shortest-path distance to target."""

  def __init__(
      self,
      map_scale: int = 10,
      **kwargs
    ):
    """Initialize a ShortestPathRewardSingleGoalMazeTask.

    Args:
      map_scale: resolution of the arrival map, relative to maze grid
      **kwargs: additional keyword arguments for DenseRewardSingleGoalMazeTask
    """
    super().__init__(**kwargs)
    self._map_scale = map_scale
    self._reset_distance_map()

  def get_reward(self, physics: Physics):
    walker_pos = self._get_walker_pos(physics)
    position_reward = self._get_reward_for_position(walker_pos)
    target_reward = self._get_target_reward()
    if self._verbose:
      print(f"reward: pos={position_reward}, alive={self._aliveness_reward}, target={target_reward}")  # pylint: disable=line-too-long
    return position_reward + self._aliveness_reward + target_reward

  def _get_distance_from_position(self, xy: np.ndarray):
    rc = self._maze_arena.world_to_grid_positions([xy])[0]
    ij = np.round(
        (rc + np.array([0.5, 0.5])) * self._map_scale - np.array((1, 1))
    ).astype(int)
    distance = self._distance_map[ij[0], ij[1]]
    return distance

  def _get_reward_for_position(self, xy: np.ndarray):
    distance = self._get_distance_from_position(xy)
    max_distance = self._get_max_maze_distance()
    dist_reward = -distance / max_distance
    return dist_reward

  def _get_max_maze_distance(self):
    return np.nanmax(self._distance_map)

  def _initialize_target_position(self, random_state: np.random.RandomState):
    super()._initialize_target_position(random_state)
    self._reset_distance_map()

  def _add_distance_observable(self):
    """Add distance_to_target observable based on shortest-path distance."""

    def _get_distance_to_target(physics: Physics):
      walker_position = self._get_walker_pos(physics)
      distance = self._get_distance_from_position(walker_position)
      return distance

    obs = observable.Generic(_get_distance_to_target)
    obs.enabled = True
    self._task_observables["distance_to_target"] = obs

  def _reset_distance_map(self):
    """Reset the distance map according to current maze target location."""
    arrival_map = _get_arrival_map_from_lab_maze(
        self._maze_arena.maze, scale=self._map_scale)
    self._distance_map = (
        arrival_map / self._map_scale * self.maze_arena.xy_scale
    )


def _get_arrival_map_from_lab_maze(
    maze,  # labmaze.fixed_maze.FixedMazeWithRandomGoals
    scale: int = 1,
    return_all_images: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
  """Get the arrival time map from the given labmaze."""
  map_size = np.array(maze.entity_layer.shape) * scale
  floor_mask = np.ones(map_size).astype(bool)
  target_mask = np.zeros(map_size).astype(bool)
  for i in range(floor_mask.shape[0]):
    for j in range(floor_mask.shape[1]):
      if maze.entity_layer[i // scale, j // scale] == "*":
        floor_mask[i, j] = False
      elif maze.entity_layer[i // scale, j // scale] == "G":
        target_mask[i, j] = True
  arrival_map, slowness_map = fast_marching.get_slowness_map_arrival_time(
      floor_mask=floor_mask,
      wave_origin=target_mask,
      safe_distance=scale / 2.0
  )
  if return_all_images:
    all_images = dict(
        floor_mask=floor_mask,
        target_mask=target_mask,
        arrival_map=arrival_map,
        slowness_map=slowness_map,
    )
    return all_images
  else:
    return arrival_map


def make_maze_task(
    task_name: str,
    walker: ant.Ant,
    arena: mazes.MazeWithTargets,
    cameras: list[str],
    **kwargs,
) -> DenseRewardSingleGoalMazeTask:
  """Get the indicated task type with the given walker and maze arena.

  Args:
    task_name: short name for the task, either "distance" or "shortest_path"
    walker: The (ant) walker instance
    arena: The maze arena instance
    cameras: list of camera names
    **kwargs: addition optional keyword args specific to the chosen task

  Returns:
    task: a dense-reward maze task
  """
  task_classes = {
      "distance": DistanceRewardSingleGoalMazeTask,
      "shortest_path": ShortestPathRewardSingleGoalMazeTask,
  }
  task = task_classes[task_name](
      walker=walker,
      maze_arena=arena,
      cameras=cameras,
      **kwargs,
  )
  return task
