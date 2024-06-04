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

"""Example script."""

from absl import app
from dmc_vision_benchmark.rep_learn import config
from dmc_vision_benchmark.rep_learn import train


def train_bc_one_step(_):
  """One step of training."""
  # Load config
  this_config = config.get_config()
  this_config.model = "bc"
  this_config.dataset = "dmc_vb"

  # Modify
  this_config.online_eval.num_online_runs = 2
  this_config.learning.num_iters = 1
  this_config.learning.checkpoint_interval = 1
  this_config.learning.eval_every = 10  # skip
  this_config.learning.online_eval_every = 1

  # Trainer
  trainer = train.Trainer(this_config)
  data_train = next(trainer.train_dataset)
  print("\n#### Batch element and sizes:")
  for k, v in data_train.items():
    print(k, v.shape)

  # One-step training
  _, metrics = trainer.train()

  print("\n#### Metrics after one training step")
  print("Training loss", metrics["train_loss"][0])
  print("Online evaluation metrics", metrics["online_eval"][0])


if __name__ == "__main__":
  app.run(train_bc_one_step)
