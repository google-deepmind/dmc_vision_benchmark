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

"""Wrapper for adding distractors to an environment.

Adapted from the distracting control suite:
https://github.com/google-research/google-research/tree/master/distracting_control
"""

import collections
import os
from typing import Any

from dm_control import suite
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite.wrappers import pixels
import dm_env
import mediapy
import numpy as np
import tensorflow as tf

from distracting_control import camera
from distracting_control import color
from distracting_control import suite_utils


SKY_TEXTURE_INDEX = 0
Texture = collections.namedtuple('Texture', ('size', 'address', 'textures'))


def size_and_flatten(
    image: np.ndarray, ref_height: int, ref_width: int
) -> np.ndarray:
  # Resize image if necessary and flatten the result.
  image_height, image_width = image.shape[:2]

  if image_height != ref_height or image_width != ref_width:
    image = tf.cast(tf.image.resize(image, [ref_height, ref_width]), tf.uint8)
  return tf.reshape(image, [-1]).numpy()


def blend_to_background(
    alpha: float, image: np.ndarray, background: np.ndarray
) -> np.ndarray:
  if alpha == 1.0:
    return image
  elif alpha == 0.0:
    return background
  else:
    return (
        alpha * image.astype(np.float32)
        + (1.0 - alpha) * background.astype(np.float32)
    ).astype(np.uint8)


def load(
    domain_name: str,
    task_name: str,
    difficulty: str | None = None,
    dynamic: bool = False,
    background_dataset_path: str | None = None,
    background_dataset_videos: str = 'train',
    camera_kwargs: dict[str, Any] | None = None,
    color_kwargs: dict[str, Any] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    environment_kwargs: dict[str, Any] | None = None,
    visualize_reward: bool = False,
    render_kwargs: dict[str, Any] | None = None,
    pixels_only: bool = True,
    pixels_observation_key: str = 'pixels',
    env_state_wrappers: list[Any] | None = None,
) -> dm_env.Environment:
  """Returns an environment from a domain name, task name and optional settings.

  ```python
  env = suite.load('cartpole', 'balance')
  ```

  Users can also toggle dynamic properties for distractions.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    difficulty: Difficulty for the suite. One of 'easy', 'medium', 'hard'.
    dynamic: Boolean controlling whether distractions are dynamic or static.
    background_dataset_path: Path to the background dataset.
    background_dataset_videos: String ('train'/'val').
    camera_kwargs: Dict, overwrites settings for camera distractions.
    color_kwargs: Dict, overwrites settings for color distractions.
    task_kwargs: Dict, dm control task kwargs.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
    render_kwargs: Dict, render kwargs for pixel wrapper.
    pixels_only: Boolean controlling the exclusion of states in the observation.
    pixels_observation_key: Key in the observation used for the rendered image.
    env_state_wrappers: Env state wrappers to be called before the PixelWrapper.

  Returns:
    The requested environment.
  """

  if difficulty not in [None, 'easy', 'medium', 'hard']:
    raise ValueError("Difficulty should be one of: 'easy', 'medium', 'hard'.")

  render_kwargs = render_kwargs or {}
  if 'camera_id' not in render_kwargs:
    render_kwargs['camera_id'] = 2 if domain_name == 'quadruped' else 0

  assert suite is not None
  env = suite.load(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward,
  )

  # Apply background distractions.
  if difficulty:
    final_background_kwargs = dict()
    if difficulty:
      # Get kwargs for the given difficulty.
      num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
      final_background_kwargs.update(
          suite_utils.get_background_kwargs(
              domain_name,
              num_videos,
              dynamic,
              background_dataset_path,
              background_dataset_videos,
          )
      )
    env = KubricBackgroundEnv(env, **final_background_kwargs)

  # Apply camera distractions.
  if difficulty or camera_kwargs:
    final_camera_kwargs = dict(camera_id=render_kwargs['camera_id'])
    if difficulty:
      # Get kwargs for the given difficulty.
      scale = suite_utils.DIFFICULTY_SCALE[difficulty]
      final_camera_kwargs.update(
          suite_utils.get_camera_kwargs(domain_name, scale, dynamic)
      )
    if camera_kwargs:
      # Overwrite kwargs with those passed here.
      final_camera_kwargs.update(camera_kwargs)
    env = camera.DistractingCameraEnv(env, **final_camera_kwargs)

  # Apply color distractions.
  if difficulty or color_kwargs:
    final_color_kwargs = dict()
    if difficulty:
      # Get kwargs for the given difficulty.
      scale = suite_utils.DIFFICULTY_SCALE[difficulty]
      final_color_kwargs.update(suite_utils.get_color_kwargs(scale, dynamic))
    if color_kwargs:
      # Overwrite kwargs with those passed here.
      final_color_kwargs.update(color_kwargs)
    env = color.DistractingColorEnv(env, **final_color_kwargs)

  if env_state_wrappers is not None:
    for wrapper in env_state_wrappers:
      env = wrapper(env)
  # Apply Pixel wrapper after distractions. This is needed to ensure the
  # changes from the distraction wrapper are applied to the MuJoCo environment
  # before the rendering occurs.
  env = pixels.Wrapper(
      env,
      pixels_only=pixels_only,
      render_kwargs=render_kwargs,
      observation_key=pixels_observation_key,
  )

  return env


class KubricBackgroundEnv(control.Environment):
  """Environment wrapper for Kubric background visual distraction.

  **NOTE**: This wrapper should be applied BEFORE the pixel wrapper to make sure
  the background image changes are applied before rendering occurs.
  """

  def __init__(
      self,
      env: control.Environment,
      dataset_path: str | None = None,
      dataset_videos: str | None = None,
      video_alpha: float = 1.0,
      ground_plane_alpha: float = 1.0,
      num_videos: int | None = None,
      dynamic: bool = False,
      seed: int | None = None,
      shuffle_buffer_size: int | None = None,
  ):
    if not 0 <= video_alpha <= 1:
      raise ValueError('`video_alpha` must be in the range [0, 1]')

    self._env = env
    self._video_alpha = video_alpha
    self._ground_plane_alpha = ground_plane_alpha
    self._random_state = np.random.RandomState(seed=seed)
    self._dynamic = dynamic
    self._shuffle_buffer_size = shuffle_buffer_size
    self._current_img_index = 0
    self._slowdown_video_factor = 10
    self._step_counter = 0

    if not dataset_path or num_videos == 0:
      # Allow running the wrapper without backgrounds to still set the ground
      # plane alpha value.
      self._video_paths = []
    else:
      if dataset_videos in ['train', 'training']:
        dataset_path = dataset_path + '/train'
      elif dataset_videos in ['val', 'validation']:
        dataset_path = dataset_path + '/validation'

      # Get complete paths for all videos.
      dataset_videos = sorted(tf.io.gfile.listdir(dataset_path))
      video_paths = [
          os.path.join(dataset_path, subdir) for subdir in dataset_videos
      ]

      # Optionally use only the first num_paths many paths.
      if num_videos is not None:
        if num_videos > len(video_paths) or num_videos < 0:
          raise ValueError(
              f'`num_bakground_paths` is {num_videos} but '
              'should not be larger than the number of available '
              f'background paths ({len(video_paths)}) and at '
              'least 0.'
          )
        video_paths = video_paths[:num_videos]

      self._video_paths = video_paths

  def reset(self) -> dm_env.TimeStep:
    """Reset the background state."""
    time_step = self._env.reset()
    self._reset_background()
    return time_step

  def _reset_background(self):
    # Make grid semi-transparent.
    if self._ground_plane_alpha is not None:
      self._env.physics.named.model.mat_rgba['grid', 'a'] = (
          self._ground_plane_alpha
      )

    # For some reason the height of the skybox is set to 4800 by default,
    # which does not work with new textures.
    self._env.physics.model.tex_height[SKY_TEXTURE_INDEX] = 800

    # Set the sky texture reference.
    sky_height = self._env.physics.model.tex_height[SKY_TEXTURE_INDEX]
    sky_width = self._env.physics.model.tex_width[SKY_TEXTURE_INDEX]
    sky_size = sky_height * sky_width * 3
    sky_address = self._env.physics.model.tex_adr[SKY_TEXTURE_INDEX]

    sky_texture = self._env.physics.model.tex_rgb[
        sky_address : sky_address + sky_size
    ].astype(np.float32)

    if self._video_paths:

      # Randomly pick a video and load all images.
      video_path = self._random_state.choice(self._video_paths)
      images = mediapy.read_video(video_path)
      if not self._dynamic:
        # Randomly pick a single static frame.
        images = images[self._random_state.randint(len(images))][None]

      # Pick a random starting point and stepping direction.
      self._current_img_index = self._random_state.choice(len(images))
      self._step_direction = self._random_state.choice([-1, 1])

      # Prepare images in the texture format by resizing and flattening.

      # Generate image textures.
      texturized_images = []
      for image in images:
        image_flattened = size_and_flatten(image, sky_height, sky_width)
        new_texture = blend_to_background(
            self._video_alpha, image_flattened, sky_texture
        )
        texturized_images.append(new_texture)

    else:

      self._current_img_index = 0
      texturized_images = [sky_texture]

    self._background = Texture(sky_size, sky_address, texturized_images)
    self._apply()

  def step(self, action: Any) -> dm_env.TimeStep:
    time_step = self._env.step(action)

    if time_step.first():
      self._reset_background()
      return time_step

    if self._dynamic and self._video_paths:
      self._step_counter += 1
      # Move forward / backward in the image sequence by updating the index.
      if self._step_counter % self._slowdown_video_factor == 0:
        self._current_img_index += self._step_direction

        # Start moving forward if we are past the start of the images.
        if self._current_img_index <= 0:
          self._current_img_index = 0
          self._step_direction = abs(self._step_direction)
        # Start moving backwards if we are past the end of the images.
        if self._current_img_index >= len(self._background.textures):
          self._current_img_index = len(self._background.textures) - 1
          self._step_direction = -abs(self._step_direction)

        self._apply()
    return time_step

  def _apply(self):
    """Apply the background texture to the physics."""

    if self._background:
      start = self._background.address
      end = self._background.address + self._background.size
      texture = self._background.textures[self._current_img_index]

      self._env.physics.model.tex_rgb[start:end] = texture
      # Upload the new texture to the GPU. Note: we need to make sure that the
      # OpenGL context belonging to this Physics instance is the current one.
      with self._env.physics.contexts.gl.make_current() as ctx:
        ctx.call(
            mjbindings.mjlib.mjr_uploadTexture,
            self._env.physics.model.ptr,
            self._env.physics.contexts.mujoco.ptr,
            SKY_TEXTURE_INDEX,
        )

  # Forward property and method calls to self._env.
  def __getattr__(self, attr):
    if hasattr(self._env, attr):
      return getattr(self._env, attr)
    raise AttributeError(
        "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
    )
