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

"""Tools for creating maze arenas."""

import collections

from dm_control import composer
from dm_control.locomotion import arenas
from dm_control.locomotion.arenas import mazes
import labmaze
import numpy as np

# pylint: disable=protected-access


# Labmaze specs, including entity_layer and optional variations_layer
MAZE_SPECS = {
    "empty5x5": dict(
        entity_layer="*****\n*   *\n*   *\n*   *\n*****\n",
    ),
    "empty7x7": dict(
        entity_layer=(
            "*******\n*     *\n*     *\n*     *\n*     *\n*     *\n*******\n"
        ),
    ),
    "simple7x7": dict(
        entity_layer=(
            "*******\n*  *  *\n*  *  *\n*  *  *\n*     *\n*     *\n*******\n"
        ),
    ),
    "easy7x7a": dict(
        entity_layer=(
            "*******\n** *  *\n*  *  *\n*  *  *\n*     *\n*    **\n*******\n"
        )
    ),
    "easy7x7b": dict(
        entity_layer=(
            "*******\n*     *\n*  *  *\n* **  *\n*  ** *\n*     *\n*******\n"
        )
    ),
    "easy7x7c": dict(
        entity_layer=(
            "*******\n*     *\n* *   *\n* *   *\n* *** *\n*     *\n*******\n"
        )
    ),
    "medium7x7a": dict(
        entity_layer=(
            "*******\n*  *  *\n*    **\n*  *  *\n* * * *\n*     *\n*******\n"
        ),
    ),
    "medium7x7b": dict(
        entity_layer=(
            "*******\n*     *\n*** * *\n*     *\n*  ** *\n*     *\n*******\n"
        ),
    ),
    "medium7x7c": dict(
        entity_layer=(
            "*******\n*  *  *\n** *  *\n*     *\n* **  *\n*    **\n*******\n"
        ),
    ),
    "empty9x9": dict(
        entity_layer="\n".join([
            "*********",
            "*       *",
            "*       *",
            "*       *",
            "*       *",
            "*       *",
            "*       *",
            "*       *",
            "*********\n"
        ])
    ),
    "mid9x9a": dict(
        entity_layer="\n".join([
            "*********",
            "*       *",
            "* ***** *",
            "* *     *",
            "* * *** *",
            "* *   * *",
            "* *** * *",
            "*       *",
            "*********\n"
        ])
    ),
    "mid9x9b": dict(
        entity_layer="\n".join([
            "*********",
            "*       *",
            "* ** ** *",
            "*       *",
            "*** *****",
            "*       *",
            "* ****  *",
            "*       *",
            "*********\n"
        ])
    ),
    "mid9x9c": dict(
        entity_layer="\n".join([
            "*********",
            "*       *",
            "* ****  *",
            "* **    *",
            "*    ** *",
            "* ** *  *",
            "*  * ** *",
            "*       *",
            "*********\n"
        ])
    ),
    "empty11x11": dict(
        entity_layer="\n".join([
            "***********",
            "*         *",
            "*         *",
            "*         *",
            "*         *",
            "*         *",
            "*         *",
            "*         *",
            "*         *",
            "*         *",
            "***********\n"
        ])
    ),
    "hard11x11a": dict(
        entity_layer="\n".join([
            "***********",
            "*         *",
            "* ******* *",
            "* *       *",
            "* * *** * *",
            "* * *   * *",
            "* * *** * *",
            "* *     * *",
            "* * ***** *",
            "*         *",
            "***********\n"
        ])
    ),
    "hard11x11b": dict(
        entity_layer="\n".join([
            "***********",
            "*         *",
            "* ******* *",
            "*      *  *",
            "* **** ** *",
            "*   **    *",
            "* *    ** *",
            "* * **  * *",
            "* **** ** *",
            "*         *",
            "***********\n"
        ])
    ),
    "hard11x11c": dict(
        entity_layer="\n".join([
            "***********",
            "*         *",
            "*** ***** *",
            "*     *   *",
            "* *** * ***",
            "*   * *   *",
            "*** * *** *",
            "*   *     *",
            "* ******* *",
            "*         *",
            "***********\n"
        ])
    ),
}


class FixedMazeWithAdjacentGoal(labmaze.base.BaseMaze):
  """A maze in which the goal is adjacent to the agent in a fixed direction."""

  def __init__(
      self,
      direction: str,
      entity_layer: str,
      spawn_token: str = "P",
      object_token: str = "G",
      random_state: np.random.RandomState | None = None,
  ):
    self._direction = direction
    self._entity_layer = labmaze.text_grid.TextGrid(entity_layer)
    self._spawn_token = spawn_token
    self._object_token = object_token
    self._height, self._width = self._entity_layer.shape
    self._random_state = random_state or np.random

    self._variations_layer = self._entity_layer.copy()
    self._variations_layer[:] = "."

    self._floor_cells = []
    for i in range(self._height):
      for j in range(self._width):
        if self._entity_layer[i, j] == " ":
          self._floor_cells.append((i, j))

    self.regenerate()

  def regenerate(self):

    # Clear current entities
    for i, j in self._floor_cells:
      self._entity_layer[i, j] = " "

    # Find a valid pair of entity positions and update the entity layer
    found = False
    while not found:
      # randomly select a cell for the agent
      idx = self._random_state.choice(len(self._floor_cells))
      agent_loc = self._floor_cells[idx]

      # select the appropriate directional cell for the target
      target_loc = self._get_adjacent_cell(agent_loc)

      if target_loc in self._floor_cells:
        found = True
        self._entity_layer[agent_loc[0], agent_loc[1]] = self._spawn_token
        self._entity_layer[target_loc[0], target_loc[1]] = self._object_token

  def _get_adjacent_cell(self, origin):
    offset = (0, 0)
    match self._direction:
      case "N": offset = (-1, 0)
      case "S": offset = (1, 0)
      case "W": offset = (0, -1)
      case "E": offset = (0, 1)
      case "NW": offset = (-1, -1)
      case "NE": offset = (-1, 1)
      case "SW": offset = (1, -1)
      case "SE": offset = (1, 1)
    adjacent_cell = tuple(np.array(origin) + np.array(offset))
    return adjacent_cell

  @property
  def entity_layer(self):
    return self._entity_layer

  @property
  def variations_layer(self):
    return self._variations_layer

  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  @property
  def max_variations(self):
    return 1

  @property
  def max_rooms(self):
    return 1

  @property
  def objects_per_room(self):
    return 1

  @property
  def spawn_token(self):
    return self._spawn_token

  @property
  def object_token(self):
    return self._object_token


def make_maze(
    name: str,
    seed: int | None = None,
    agent_location: str = "fixed",
    target_location: str = "fixed",
    use_systematic_seed: bool = False,
    agent_symbol: str = "P",
    target_symbol: str = "G",
) -> labmaze.FixedMazeWithRandomGoals | FixedMazeWithAdjacentGoal:
  """Construct a labmaze, either from named design or randomNxN.

  Args:
    name: name of a defined maze (see MAZE_SPECS) or 'randomNxN' for an NxN
      random maze.
    seed: random seed for maze generation
    agent_location: agent location condition, either fixed or random
    target_location: target location condition, either fixed, random, or dir-X
      where X is a compass direction (N, S, E, W, NW, NE, SW, SE)
    use_systematic_seed: if true, fixed agent location and target location are
      deterministic functions of the seed value, and n contiguous seeds will
      produce n different combinations.
    agent_symbol: the symbol used in indicate agent position in the maze grid
    target_symbol: the symbol used to indicate target position in the maze grid

  Returns:
    maze: the generated labmaze
  """
  assert agent_location in ["fixed", "random"]
  if "dir-" in target_location:
    maze = make_direction_maze(
        name=name, direction=target_location[len("dir-") :], seed=seed
    )
  elif name.startswith("random_"):
    assert target_location in ["fixed", "random"]
    _, maze_size_str, maze_seed_str = name.split("_")
    height_str, width_str = maze_size_str.split("x")
    maze = make_random_maze(
        size=(int(height_str), int(width_str)),
        maze_seed=int(maze_seed_str),
        entity_seed=seed,
        fixed_agent_location=(agent_location == "fixed"),
        fixed_target_location=(target_location == "fixed"),
        use_systematic_seed=use_systematic_seed,
        agent_symbol=agent_symbol,
        target_symbol=target_symbol,
    )

  else:
    maze = make_named_maze(
        name=name,
        seed=seed,
        agent_location=agent_location,
        target_location=target_location,
        use_systematic_seed=use_systematic_seed,
        agent_symbol=agent_symbol,
        target_symbol=target_symbol,
    )
  return maze


def make_named_maze(
    name: str,
    seed: int | None = None,
    agent_location: str = "fixed",
    target_location: str = "fixed",
    use_systematic_seed: bool = False,
    agent_symbol: str = "P",
    target_symbol: str = "G",
) -> labmaze.FixedMazeWithRandomGoals:
  """Create a maze based on a named template.

  Args:
    name: maze template name
    seed: random seed that affects agent and target locations, -1 or None
      produces random seed
    agent_location: agent start location, either fixed or random
    target_location: target location, either fixed or random
    use_systematic_seed: if true, n contiguous seeds will provide n different
      combinations of fixed agent and target location.
    agent_symbol: symbol for the agent in the entity_layer
    target_symbol: symbol for the target in the entity_layer

  Returns:
    maze: a labmaze
  """
  if seed == -1:
    seed = None
  random_state = np.random.RandomState(seed)

  maze_spec = collections.defaultdict(None, MAZE_SPECS[name])
  entity_layer = maze_spec["entity_layer"]
  variations_layer = maze_spec.get("variations_layer", None)

  target_indices = [
      idx for idx in range(len(entity_layer)) if entity_layer[idx] == " "
  ]
  if target_location == "fixed":
    idx = (
        target_indices[seed % len(target_indices)]
        if use_systematic_seed
        else random_state.choice(target_indices)
    )
    entity_layer = entity_layer[:idx] + target_symbol + entity_layer[idx + 1 :]
    agent_indices = list(set(target_indices) - {idx})
    num_objects = None
  else:
    assert target_location == "random"
    agent_indices = target_indices
    target_indices = [-1]  # len(1) for systematic seed
    num_objects = 1  # 1 randomized target per reset

  if agent_location == "fixed":
    idx = (
        agent_indices[(seed // len(target_indices)) % len(agent_indices)]
        if use_systematic_seed
        else random_state.choice(agent_indices)
    )
    entity_layer = entity_layer[:idx] + agent_symbol + entity_layer[idx + 1 :]
    num_spawns = None
  else:
    assert agent_location == "random"
    num_spawns = len(agent_indices) - 1  # all non-target locations

  maze = labmaze.FixedMazeWithRandomGoals(
      entity_layer=entity_layer,
      variations_layer=variations_layer,
      num_spawns=num_spawns,
      num_objects=num_objects,
      random_state=random_state,
  )
  return maze


def make_direction_maze(
    name: str,
    direction: str,
    seed: int | None = None,
) -> FixedMazeWithAdjacentGoal:
  """Create a maze based on a named template, with adjacent target in a given direction."""
  if seed == -1:
    seed = None
  random_state = np.random.RandomState(seed)
  maze_spec = collections.defaultdict(None, MAZE_SPECS[name])
  entity_layer = maze_spec["entity_layer"]
  maze = FixedMazeWithAdjacentGoal(
      direction=direction, entity_layer=entity_layer, random_state=random_state
  )
  return maze


def make_random_maze(
    size: tuple[int, int],
    maze_seed: int,
    entity_seed: int,
    max_rooms: int = 4,
    fixed_agent_location: bool = False,
    fixed_target_location: bool = True,
    use_systematic_seed: bool = False,
    agent_symbol: str = "P",
    target_symbol: str = "G",
) -> labmaze.FixedMazeWithRandomGoals:
  """Construct a maze via the labmaze generator."""
  random_maze = labmaze.RandomMaze(
      height=size[0],
      width=size[1],
      max_rooms=max_rooms,
      has_doors=False,
      room_min_size=3,
      room_max_size=min(size) // 2,
      spawns_per_room=0,  # will be added below
      objects_per_room=9,  # all tiles in min-size 3x3 room
      simplify=True,
      random_seed=maze_seed,
      spawn_token=agent_symbol,
      object_token=target_symbol,
  )

  entity_layer = str(random_maze.entity_layer)

  random_state = np.random.RandomState(entity_seed)

  if fixed_target_location:
    target_indices = [
        idx
        for idx in range(len(entity_layer))
        if entity_layer[idx] == target_symbol
    ]
    if use_systematic_seed:
      target_index = target_indices[entity_seed % len(target_indices)]
    else:
      target_index = random_state.choice(target_indices)
    for idx in target_indices:
      if idx != target_index:
        entity_layer = (
            entity_layer[:idx] + agent_symbol + entity_layer[idx + 1 :]
        )
    num_objects = None
  else:
    num_objects = 1
    target_indices = [0]  # dummy value, for len calculation below

  agent_indices = [
      idx
      for idx in range(len(entity_layer))
      if entity_layer[idx] == agent_symbol
  ]

  if fixed_agent_location:
    if use_systematic_seed:
      agent_seed = entity_seed // len(target_indices)
      agent_index = agent_indices[agent_seed % len(agent_indices)]
    else:
      agent_index = np.random.choice(agent_indices)
    for idx in agent_indices:
      if idx != agent_index:
        entity_layer = entity_layer[:idx] + " " + entity_layer[idx + 1 :]
    num_spawns = None
  else:
    num_spawns = len(agent_indices)

  maze = labmaze.FixedMazeWithRandomGoals(
      entity_layer=entity_layer,
      variations_layer=str(random_maze.variations_layer),
      num_spawns=num_spawns,
      num_objects=num_objects,
      random_state=random_state,
  )
  return maze


def make_maze_arena(
    maze: labmaze.FixedMazeWithRandomGoals | FixedMazeWithAdjacentGoal,
    include_highres_top_camera: bool = False,
    include_lowres_top_camera: bool = True,
    skybox_style: str = "sky_02",
    floor_style: str = "style_01",
    wall_style: str | dict[str, str] = "style_01",
    xy_scale: float = 2.5,
) -> tuple[mazes.MazeWithTargets, list[str]]:
  """Get an arena for the named maze.

  Args:
    maze: the labmaze from which to create an arena
    include_highres_top_camera: if true, include the top camera
    include_lowres_top_camera: if true, include the lowres top camera
    skybox_style: name of the skybox style (see arenas.labmaze_textures)
    floor_style: name of the floor style (see arenas.labmaze_textures)
    wall_style: name of the wall style (see arenas.labmaze_textures)
    xy_scale: size of each grid cell. Default value ensures that ant walker
      centered in a cell adjacent to the target cannot touch it.

  Returns:
    maze_arena: the maze arena
    camera_names: the camera names (observation keys) defined for this arena
  """
  if isinstance(wall_style, dict):
    wall_textures = {
        key: arenas.labmaze_textures.WallTextures(val)
        for key, val in wall_style.items()
    }
  else:
    wall_textures = arenas.labmaze_textures.WallTextures(wall_style)
  maze_arena = arenas.mazes.MazeWithTargets(
      maze=maze,
      skybox_texture=arenas.labmaze_textures.SkyBox(skybox_style),
      floor_textures=arenas.labmaze_textures.FloorTextures(floor_style),
      wall_textures=wall_textures,
      xy_scale=xy_scale,
  )

  # Add/enable requested cameras
  camera_names = []
  if include_highres_top_camera:
    maze_arena.observables.top_camera.enabled = True
    camera_names.append("top_camera")
  if include_lowres_top_camera:
    cam_name = "lowres_top_camera"
    _create_top_camera(cam_name, maze_arena, image_height=64, image_width=64)
    camera_names.append(cam_name)

  return maze_arena, camera_names


def _create_top_camera(
    cam_name: str,
    maze_arena: arenas.mazes.MazeWithTargets,
    image_height: int,
    image_width: int,
    camera_distance: float = 100,
):
  """Create a centered topdown camera with the given image dimensions."""
  maze_size = max(maze_arena.maze.height, maze_arena.maze.width)
  fovy = (360 / np.pi) * np.arctan2(
      maze_size * maze_arena._xy_scale / 2, camera_distance
  )
  camera = maze_arena.mjcf_model.worldbody.add(
      "camera",
      name=cam_name,
      pos=[0, 0, camera_distance],
      zaxis=[0, 0, 1],
      mode="fixed",
      fovy=fovy,
  )
  observable = composer.observation.observable.MJCFCamera(
      camera, height=image_height, width=image_width
  )
  maze_arena.observables.add_observable(name=cam_name, observable=observable)
