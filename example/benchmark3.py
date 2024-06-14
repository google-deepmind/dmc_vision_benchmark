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

"""Minimal script to kick off Benchmark 3.

For each method, we use one training step and one online evaluation step.

For larger runs, we encourage the user to parallelize the processes on their
hardware.
"""

import copy

from absl import app
from dmc_vision_benchmark.rep_learn import config
from dmc_vision_benchmark.rep_learn import train


def run_benchmark3(_):
  """One step of training."""
  # Load config
  this_config = config.get_config()

  # Shared data parameters
  this_config.data.domain_name = "ant"
  this_config.data.policy_level = "expert"
  this_config.data.difficulty = "none"
  this_config.online_eval.render_camera = "lowres_top_camera"
  this_config.online_eval.max_episode_length = 1_000
  this_config.data.target_hidden = True
  ant_fixed_seed = 1_000_000

  # For simplicity, we only consider one task
  this_config.data.task_name = "empty7x7"

  # # To reproduce Benchmark 3,
  # # we need to loop over antmaze tasks and fixed seeds.
  # from dmc_vision_benchmark.data import dmc_vb_info  # pylint: disable=line-too-long
  # for maze in dmc_vb_info.ANT_MAZE_TASK:
  #   for ant_fixed_seed in [1_000_001, 1_000_002, 1_000_003, 1_000_004, 1_000_005]:
  #     this_config.data.task_name = maze

  # One training step, one eval step
  this_config.online_eval.num_online_runs = 1  # set to 30 in paper
  this_config.learning.num_iters = 1  # set to 400_000 in paper
  this_config.learning.checkpoint_interval = 1  # set to 20_000 in paper
  this_config.learning.online_eval_every = 1  # set to 20_000 in paper
  this_config.learning.eval_every = 10  # skip. set to 20_000 in paper
  this_config.try_to_restore = False  # set to True

  print("\n\n####### NULL + BC (fixed goal) #######\n")
  this_config.model = "bc"
  # Tasks with fixed hidden goals
  this_config.data.train_split = "train[:10]"  # only 10 trajectories
  this_config.online_eval.ant_fixed_seed = ant_fixed_seed
  this_config.online_eval.ant_random_start_end = False
  this_config.online_eval.propagate_seed_to_env = False

  trainer = train.Trainer(copy.deepcopy(this_config))
  _, metrics = trainer.train()
  print("Reward", metrics["online_eval"][0])

  for pretraining in ["pretrain_id", "bc"]:
    name = pretraining.split("_")[-1].upper()
    print(f"\n\n####### {name} (stochastic goals) + BC (fixed goal) #######\n")
    print(
        "\nStep 1. Pretraining a visual encoder on tasks with stochastic"
        " hidden goals."
    )
    this_config.model = pretraining
    # Tasks with stochastic hidden goals
    this_config.data.train_split = "train[:95%]"
    this_config.online_eval.ant_fixed_seed = None
    this_config.online_eval.ant_random_start_end = True
    this_config.online_eval.propagate_seed_to_env = True

    trainer = train.Trainer(copy.deepcopy(this_config))
    _, _ = trainer.train()

    print("\nStep 2. Learning a policy on a task with fixed hidden goal.")
    this_config.model = "bc"
    # Tasks with fixed hidden goals
    this_config.data.train_split = "train[:10]"  # only 10 trajectories
    this_config.online_eval.ant_fixed_seed = ant_fixed_seed
    this_config.online_eval.ant_random_start_end = False
    this_config.online_eval.propagate_seed_to_env = False

    # Load the pretrained model
    this_config.pretrained_model_id = trainer.this_id
    trainer = train.Trainer(copy.deepcopy(this_config))
    _, metrics = trainer.train()
    print("Reward", metrics["online_eval"][0])


if __name__ == "__main__":
  app.run(run_benchmark3)
