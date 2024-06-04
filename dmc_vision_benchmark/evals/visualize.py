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

"""Return evaluations of evaluations."""

from collections.abc import Sequence

import numpy as np


def pack_videos(videos: Sequence[np.ndarray]) -> np.ndarray:
  """Returns a single video that packs all smaller videos inside."""

  # Ensure types are the same, and show as a video.
  videos = [
      v if v.dtype == np.uint8 else np.array(v * 255, np.uint8) for v in videos
  ]
  max_time = max(video.shape[0] for video in videos)
  height = videos[0].shape[1]
  width = videos[0].shape[2]
  for video in videos:
    assert video.shape[1:] == (height, width, 3)

  columns = int(np.ceil(len(videos) ** 0.5))
  rows = int(np.ceil(len(videos) / columns))
  joint_frames = np.zeros(
      (
          max_time,
          rows * height + rows - 1,
          columns * width + columns - 1,
          3,
      ),
      np.uint8,
  )
  row_offset = 0
  col_offset = 0
  for video in videos:
    joint_frames[
        : video.shape[0],
        row_offset : row_offset + video.shape[1],
        col_offset : col_offset + video.shape[2],
        :,
    ] = video
    col_offset += video.shape[2] + 1
    if col_offset - 1 == joint_frames.shape[2]:
      col_offset = 0
      row_offset += video.shape[1] + 1
  return joint_frames


def scale_video(video: np.ndarray, scale: int) -> np.ndarray:
  """Returns a video rescaled by the given factor."""
  if scale == 1:
    return video
  return np.kron(video, np.ones((1, scale, scale, 1))).astype(video[0].dtype)
