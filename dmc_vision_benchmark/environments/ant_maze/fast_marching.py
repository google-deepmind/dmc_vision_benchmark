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

"""Implements Fast Marching techniques for path planning.

This includes a function to visualize the results of fast marching.

Reference:
"Fast Marching Methods in Path Planning"
December 2013, IEEE Robotics & Automation Magazine
https://jvgomez.github.io/files/pubs/fm2star.pdf
"""

from kauldron import typing
import numpy as np
import skfmm


@typing.typechecked
def get_slowness_map(
    floor_mask: typing.Bool["h w"],
    maximum_speed: float = 1.0,
    safe_distance: float = 5.0,
) -> typing.Float["h w"]:
  """Computes the wave propagation speed at all pixel locations.

  This slowness map is used to compute wave arrival time - if the wave has to
  pass through "slow" regions, the arrival time would be later.

  Args:
    floor_mask: A binary map which is True only at locations where the wave can
      have a non-zero speed. In robotics navigational path-planning, this would
      indicate all the floor locations a robot can get to.
    maximum_speed: The maximum speed of wave propagation.
    safe_distance: The pixel distance at which wave propagation takes the
      `maximum_speed`. The speed decays linearly to 0 at distances less than
      this specified value.

  Returns:
    slowness_map: The slowness map giving the wave propagation speed at every
      valid floor mask location.
  """
  if safe_distance > 0:
    # compute the saturated slowness map

    # the speed is constant everywhere
    speed = np.full(floor_mask.shape, fill_value=1.0, dtype=np.float32)

    # run fast marching method to get the slowness map
    slowness_map = skfmm.travel_time(floor_mask, speed, dx=1.0)

    # this slowness map discourages paths being taken close to obstucles
    slowness_map[slowness_map > safe_distance] = safe_distance
    slowness_map = slowness_map / safe_distance * maximum_speed
  else:
    # we have maximum speed at all traversable locations
    slowness_map = floor_mask.astype(np.float32) * maximum_speed

  return slowness_map


@typing.typechecked
def get_slowness_map_arrival_time(
    floor_mask: typing.Bool["h w"],
    wave_origin: typing.Int["*b 2"] | typing.Bool["h w"],
    maximum_speed_or_slowness_map: float | typing.Float["h w"] = 1.0,
    safe_distance: float = 5.0,
) -> tuple[typing.Float["h w"], typing.Float["h w"]]:
  """Computes the Eikonal solution with FM2 saturated variation.

  Given a map of traversable locations (a walkable floor map), this computes the
  time it takes for a wave to propagate from a given location(s) to every given
  location on that map. This is equal to the geodesic distance if the speed of
  propagation is a constant 1 everywhere. We can use the gradient of arrival
  times at every location for a finding a shortest path to the location where
  the wave originated. To make path planning useful for robotics ideally you
  want to stay away from obstacles (where `floor_mask == False`). To help with
  this, the speed of wave propagation is slowed to 0 when approaching obstacles
  (this is the FM2 saturated fast marching method). But when we are at least a
  `safe_distance` away from obstacle, the speed of wave propagation takes a
  constant value. Think of it as increasing material density close to the
  obstacles, making it harder for waves to come close.

  For more details see Sec. IV B. of "Fast Marching Methods in Path Planning"
  referenced above.

  Args:
    floor_mask: A binary map which is True only at locations where the wave can
      have a non-zero speed. In robotics navigational path-planning, this would
      indicate all the floor locations a robot can get to.
    wave_origin: A set of pixel locations on the `floor_mask` or a binary map
      the same size as the `floor_mask`, indicating all the locations where the
      the wave simultaneously originates from. In robotics navigational
      path-planning, this would be the target location that the agent needs to
      get to. Note, if providing multiple locations, the chances of creating a
      problem with multiple local optimas increases. Make sure all the target
      locations are in a compact region in the map. Also make sure all
      `wave_origin` locations are also valid on `floor_mask`.
    maximum_speed_or_slowness_map: The maximum speed of wave propagation. Or if
      a map is given, that is used as the precomputed slowness map itself.
    safe_distance: The pixel distance at which wave propagation takes the
      `maximum_speed`. The speed decays linearly to 0 at distances less than
      this specified value. If `safe_distance = 0` and `maximum_speed = 1`, this
      would give wave arrival times according to the standard fast marching
      method described in Sec. II of  "Fast Marching Methods in Path Planning"
      referenced above.

  Returns:
    wave_arrival_time_map: The map of arrival times. Values are nan wherever the
      wave could not reach (either because `floor_mask == False` or it was
      disconnected from the `wave_origin` location(s)).
    slowness_map: The slowness map giving the wave propagation speed at every
      valid floor mask location.

  Raises:
    ValueError: In case any of the `wave_origin` pixels is in a region where
      `floor_mask == False`.
  """
  if isinstance(maximum_speed_or_slowness_map, (int, float)):
    slowness_map = get_slowness_map(
        floor_mask=floor_mask,
        maximum_speed=maximum_speed_or_slowness_map,
        safe_distance=safe_distance,
    )
  else:
    slowness_map = maximum_speed_or_slowness_map

  # set the phi - which indicates the location(s) where the wave originates from
  # (only zero values matter - they indicate wave source location(s))
  if wave_origin.shape == floor_mask.shape:
    # user has provided a mask for target pixel locations
    assert wave_origin.dtype == bool
    phi = np.array(~wave_origin, dtype=np.float32)
  else:
    # user has given target pixel locations
    assert wave_origin.shape[-1] == 2
    wave_origin = wave_origin.reshape((-1, 2))
    phi = np.ones(floor_mask.shape, dtype=np.float32)
    phi[wave_origin[:, 0], wave_origin[:, 1]] = 0

  # make sure all targets are reachable
  floor_at_origins = floor_mask[phi == 0]
  if not np.all(floor_at_origins):
    raise ValueError(
        "Only some targets are at valid floor map locations: "
        f"{floor_at_origins.sum()} / {floor_at_origins.size}"
    )

  # create a mask where the agent can't go to
  phi = np.ma.MaskedArray(phi, ~floor_mask)

  # run fast marching method with the computed slowness map
  wave_arrival_time_map = skfmm.travel_time(phi, slowness_map, dx=1.0)

  # get a mask telling which locations were reached by the wave
  unreachable_mask = wave_arrival_time_map.mask
  wave_arrival_time_map = wave_arrival_time_map.data
  wave_arrival_time_map[unreachable_mask] = np.nan

  return wave_arrival_time_map, slowness_map
