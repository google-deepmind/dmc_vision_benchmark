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

"""Minimal script to kick off Benchmark 2.

For each method, we use one training step and one online evaluation step.

For larger runs, we encourage the user to parallelize the processes on their
hardware.
"""

import copy

from absl import app
from dmc_vision_benchmark.rep_learn import config
from dmc_vision_benchmark.rep_learn import train


def run_benchmark2(_):
  """One step of training."""
  # Load config
  this_config = config.get_config()

  # For simplicity, we only consider one task and one distractor
  this_config.data.domain_name = "cheetah"
  this_config.data.task_name = "run"
  this_config.data.difficulty = "none"

  # # To reproduce Benchmark 2,
  # # we need to loop over locomotion tasks and distractors.
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
  #     this_config.data.domain_name = domain_name
  #     this_config.data.task_name = task_name
  #     this_config.data.difficulty = difficulty
  #     this_config.data.dynamic_distractors = dynamic_distractors

  # One training step, one eval step
  this_config.online_eval.num_online_runs = 1  # set to 30 in paper
  this_config.learning.num_iters = 1  # set to 400_000 in paper
  this_config.learning.checkpoint_interval = 1  # set to 20_000 in paper
  this_config.learning.online_eval_every = 1  # set to 20_000 in paper
  this_config.learning.eval_every = 10  # skip. set to 20_000 in paper
  this_config.try_to_restore = False  # set to True

  print("\n\n####### NULL + BC (1% expert) #######\n")
  this_config.model = "bc"
  this_config.data.policy_level = "expert"
  this_config.data.train_split = "train[:1%]"
  trainer = train.Trainer(copy.deepcopy(this_config))
  _, metrics = trainer.train()
  print("Reward", metrics["online_eval"][0])

  print("\n\n####### NULL + BC (mixed + 1% expert) #######\n")
  this_config.data.policy_level = "mixed_expert"
  this_config.data.train_split = "train[:95%]_train[:1%]"
  trainer = train.Trainer(copy.deepcopy(this_config))
  _, metrics = trainer.train()
  print("Reward", metrics["online_eval"][0])

  for pretraining in ["pretrain_id", "bc"]:
    name = pretraining.split("_")[-1].upper()
    print(f"\n\n####### {name} (mixed) + BC (1% expert) #######\n")
    print("\nStep 1. Pretraining a visual encoder on mixed data.")
    this_config.model = pretraining
    this_config.data.policy_level = "mixed"
    this_config.data.train_split = "train[:95%]"
    trainer = train.Trainer(copy.deepcopy(this_config))
    _, _ = trainer.train()

    print("\nStep 2. Learning a policy on limited expert data.")
    this_config.model = "bc"
    this_config.data.policy_level = "expert"
    this_config.data.train_split = "train[:1%]"
    # Load the pretrained model
    this_config.pretrained_model_id = trainer.this_id
    trainer = train.Trainer(copy.deepcopy(this_config))
    _, metrics = trainer.train()
    print("Reward", metrics["online_eval"][0])


if __name__ == "__main__":
  app.run(run_benchmark2)
