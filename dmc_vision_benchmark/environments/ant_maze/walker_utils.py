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

"""Tools for working with maze walkers."""

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion import walkers
import numpy as np

Physics = mjcf.physics.Physics


# pylint: disable=protected-access


class RandomAntInitializer(walkers.initializers.WalkerInitializer):
  """An initializer that uses the walker-declared upright pose."""

  def __init__(
      self, max_position_offset: float = 0.5, max_joint_offset: float = 0.1
  ):
    """Initializes the random ant initializer.

    Args:
      max_position_offset: The maximum offset to apply to the walker's position.
      max_joint_offset: The maximum offset to apply to the walker's joint
        positions.
    """
    self._position_offset = max_position_offset
    self._joint_offset = max_joint_offset

  def initialize_pose(
      self,
      physics: Physics,
      walker: walkers.ant.Ant,
      random_state: np.random.RandomState,
  ):
    all_joints_binding = physics.bind(walker.mjcf_model.find_all("joint"))
    qpos, xpos, xquat = walker.upright_pose

    # Set joint positions
    if qpos is None:
      qpos = all_joints_binding.qpos0
    qpos_offset = random_state.uniform(
        -self._joint_offset, self._joint_offset, size=(len(qpos),)
    )
    walker.configure_joints(physics, qpos + qpos_offset)

    # Set pose
    xpos_offset = random_state.uniform(
        -self._position_offset,
        self._position_offset,
        size=len(xpos),
    )
    walker.set_pose(physics, position=xpos + xpos_offset, quaternion=xquat)

    # Set zero velocity
    walker.set_velocity(
        physics, velocity=np.zeros(3), angular_velocity=np.zeros(3)
    )


def make_ant_walker(
    include_egocentric_camera=False,
    include_follow_camera=False,
    include_overhead_camera=False,
    exclude_sensor_observations=False,
    exclude_gyro_observations=False,
    max_joint_offset: float = 0.0,
    max_position_offset: float = 0.0,
) -> tuple[walkers.ant.Ant, list[str]]:
  """Ant walker factory function.

  Args:
    include_egocentric_camera: if true, include the egocentric camera
    include_follow_camera: if true, include the follow camera
    include_overhead_camera: if true, include the overhead camera
    exclude_sensor_observations: if true, exclude 'sensor*' observations
    exclude_gyro_observations: if true, exclude 'gyro*' observations
    max_joint_offset: maximum offset to apply to the walker's joint positions.
    max_position_offset: maximum offset to apply to the walker's position.

  Returns:
    walker: An ant walker instance
    camera_names: The camera names (observation keys) defined for this walker
  """
  camera_names = []

  walker = walkers.ant.Ant(
      initializer=RandomAntInitializer(
          max_joint_offset=max_joint_offset,
          max_position_offset=max_position_offset,
      )
  )

  if exclude_sensor_observations:
    for key, observable in walker.observables.as_dict().items():
      if key.startswith("sensor"):
        observable.enabled = False

  if exclude_gyro_observations:
    for key, observable in walker.observables.as_dict().items():
      if key.startswith("gyro"):
        observable.enabled = False

  if include_egocentric_camera:
    camera_names.append("walker/egocentric_camera")
    walker.observables.get_observable("egocentric_camera").enabled = True
  else:
    walker.observables.get_observable("egocentric_camera").enabled = False

  if include_follow_camera:
    cam_name = "follow_camera"
    follow_camera = walker._mjcf_root.worldbody.add(
        "camera", name=cam_name, pos=[-1, 0, 2], zaxis=[-1, 0, 2], mode="fixed"
    )
    walker.observables.add_observable(
        name=cam_name,
        observable=composer.observation.observable.MJCFCamera(
            follow_camera, height=64, width=64
        ),
    )
    camera_names.append(f"walker/{cam_name}")

  if include_overhead_camera:
    cam_name = "overhead_camera"
    overhead_camera = walker._mjcf_root.worldbody.add(
        "camera", name=cam_name, pos=[0, 0, 3], mode="fixed"
    )
    walker.observables.add_observable(
        name=cam_name,
        observable=composer.observation.observable.MJCFCamera(
            overhead_camera, height=64, width=64
        ),
    )
    camera_names.append(f"walker/{cam_name}")

  return walker, camera_names
