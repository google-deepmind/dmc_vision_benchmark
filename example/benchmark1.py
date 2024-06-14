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

"""Minimal script to kick off Benchmark 1.

For each method, we use one training step and one online evaluation step.

For larger runs, we encourage the user to parallelize the processes on their
hardware.
"""

import copy

from absl import app
from dmc_vision_benchmark.rep_learn import config
from dmc_vision_benchmark.rep_learn import train

PAPER_NAMES = {
    "bc": "NULL + BC",
    "bc_on_state": "NULL + BC (state)",
    "td3_bc": "NULL + TD3-BC",
    "pretrain_id": "ID + BC",
    "pretrain_lfd": "LFD + BC",
    "pretrain_ae": "AE + BC",
    "pretrain_state": "State + BC",
}


def run_benchmark1(_):
  """One step of training."""
  # Load config
  this_config = config.get_config()

  # For simplicity, we only consider one task, one distractor, and one policy.
  this_config.data.domain_name = "cheetah"
  this_config.data.task_name = "run"
  this_config.data.policy_level = "expert"
  this_config.data.difficulty = "none"
  this_config.data.dynamic_distractors = True

  # # To reproduce Benchmark 1 for locomotion taks,
  # # we need to loop over tasks, distractors and policies.
  # for domain_name, task_name in [
  #     ("walker", "walk"),
  #     ("cheetah", "run"),
  #     ("humanoid", "walk"),
  # ]:
  #   for difficulty, dynamic_distractors in [
  #       ("none", True),
  #       ("medium", True),
  #       ("medium", False),
  #   ]:
  #     for policy_level in [
  #         "random",
  #         "mixed",
  #         "medium",
  #         "medium_expert",
  #         "expert",
  #     ]:
  #       this_config.data.domain_name = domain_name
  #       this_config.data.task_name = task_name
  #       this_config.data.policy_level = policy_level
  #       this_config.data.difficulty = difficulty
  #       this_config.data.dynamic_distractors = dynamic_distractors

  # # To reproduce Benchmark 1 for antmaze taks,
  # # we need to loop over ant tasks and policies.
  # from dmc_vision_benchmark.data import dmc_vb_info  # pylint: disable=line-too-long
  # for maze in dmc_vb_info.ANT_MAZE_TASK:
  #   for policy_level in ["medium", "expert"]:
  #     this_config.data.domain_name = "ant"
  #     this_config.data.task_name = maze
  #     this_config.data.policy_level = policy_level
  #     this_config.data.difficulty = "none"
  #     # Evaluation parameters
  #     this_config.online_eval.render_camera = "lowres_top_camera"
  #     this_config.online_eval.max_episode_length = 1_000

  # One training step, one eval step
  this_config.online_eval.num_online_runs = 1  # set to 30 in paper
  this_config.learning.num_iters = 1  # set to 400_000 in paper
  this_config.learning.checkpoint_interval = 1  # set to 20_000 in paper
  this_config.learning.online_eval_every = 1  # set to 20_000 in paper
  this_config.learning.eval_every = 10  # skip. set to 20_000 in paper
  this_config.try_to_restore = False  # set to True

  # NULL + {BC, BC (state), TD3-BC}
  for model in ["bc", "bc_on_state", "td3_bc"]:
    print(f"\n\n####### {PAPER_NAMES[model]} #######\n")
    this_config.model = model
    trainer = train.Trainer(copy.deepcopy(this_config))
    _, metrics = trainer.train()
    print("Reward", metrics["online_eval"][0])

  # {ID, LFD, AE} + BC
  for pretraining in [
      "pretrain_id",
      "pretrain_lfd",
      "pretrain_ae",
      "pretrain_state",
  ]:
    print(f"\n\n####### {PAPER_NAMES[pretraining]} #######\n")

    print("\nStep 1. Pretraining a visual encoder.")
    this_config.model = pretraining
    trainer = train.Trainer(copy.deepcopy(this_config))
    _, _ = trainer.train()

    print("\nStep 2. Learning a policy by freezing the visual encoder.")
    this_config.model = "bc_w_frozen_encoder"
    # Load the pretrained model
    this_config.pretrained_model_id = trainer.this_id
    trainer = train.Trainer(copy.deepcopy(this_config))
    _, metrics = trainer.train()
    print("Reward", metrics["online_eval"][0])


if __name__ == "__main__":
  app.run(run_benchmark1)
