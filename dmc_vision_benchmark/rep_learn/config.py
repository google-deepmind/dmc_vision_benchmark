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

"""Configuration to train the model."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Training config."""

  config = ml_collections.ConfigDict()
  config.model = "bc"
  config.dataset = "dmc_vb"
  config.encode_dim = 64
  config.idm_step_encode_dim = 64
  config.pretrained_model_id = 1  # optional pretrained model d

  config.save_to_folder = "/tmp/rep_learn"

  config.obs_preproc = ml_collections.ConfigDict()
  config.obs_preproc.frame_stack = 3
  config.obs_preproc.nsteps_idm = 0
  config.obs_preproc.sample_idm_step = False

  config.learning = ml_collections.ConfigDict()
  config.learning.learning_rate = 1e-3
  config.learning.num_iters = 400_001
  config.learning.checkpoint_every = 20_000
  config.learning.eval_every = 20_000
  config.learning.online_eval_every = 20_000
  config.learning.teacher_damping = 0.99

  config.learning.batch_size = 256

  config.data = ml_collections.ConfigDict()
  # Path to the dataset downloaded from the GCP bucket
  config.data.dataset_dir = "/tmp/dmc_vision_bench_data/dmc_vision_benchmark/dmc_vision_benchmark/"  # pylint: disable=line-too-long
  config.data.domain_name = "cheetah"
  config.data.task_name = "run"
  config.data.policy_level = "expert"
  config.data.difficulty = "none"
  config.data.dynamic_distractors = True
  config.data.episode_length = 200
  config.data.batch_size = config.learning.batch_size
  config.data.shuffle_buffer_size = 10_000
  config.data.obs_vrange = (-0.5, 0.5)
  config.data.actions_vrange = (-1.0, 1.0)
  config.data.add_state = False
  config.data.only_state = False
  config.data.add_rewards = False
  config.data.img_height = 64
  config.data.img_width = 64
  config.data.img_pad = 4

  config.online_eval = ml_collections.ConfigDict()
  # Path to the Kubric dataset
  config.online_eval.background_dataset_path = "/tmp/dmc_vision_bench_data/dmc_vision_benchmark/kubric_movi-d/"  # pylint: disable=line-too-long
  config.online_eval.max_episode_length = 500
  config.online_eval.action_repeats = 2
  config.online_eval.history_length = 1
  config.online_eval.num_online_runs = 30
  config.online_eval.background_dataset_videos = "val"

  # As in https://arxiv.org/pdf/2106.06860
  config.td3_bc = ml_collections.ConfigDict()
  config.td3_bc.alpha = 2.5
  config.td3_bc.discount = 0.99
  config.td3_bc.std_noise_to_actions = 0.2
  config.td3_bc.noise_clipping = (-0.5, 0.5)
  config.td3_bc.update_actor_every = 2
  config.td3_bc.loss_for_encoder = "actor"
  return config
