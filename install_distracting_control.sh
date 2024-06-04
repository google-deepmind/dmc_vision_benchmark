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

# The distracting_control code release is not available in pip, and specifies
# quite an out-dated version of `dm-control` as a requirement. This script
# downloads the google-research repository, of which distracting_control is
# a part of, creates a simple `setup.py` file for it, and installs it via pip.

#!/bin/bash
set -ex

GR_DOWNLOAD_DIR=$(mktemp -d -u)

git clone https://github.com/google-research/google-research.git $GR_DOWNLOAD_DIR
touch $GR_DOWNLOAD_DIR/distracting_control/__init__.py

# contruct a version from the last commit date/time
GR_DIST_CTRL_VER=$(git -C $GR_DOWNLOAD_DIR log -n 1 --pretty=format:%ci distracting_control | awk '{gsub("-",".",$1); gsub(":",".",$2); print $1 "." $2}')
# create a setup tools file
DC_SETUP_PATH="$GR_DOWNLOAD_DIR/setup.py"
echo "
import setuptools

if __name__ == '__main__':
  setuptools.setup(
      name='distracting_control',
      version='$GR_DIST_CTRL_VER',
      packages=['distracting_control'],
      license='Apache 2.0',
      author='Google Research',
  )
" > $DC_SETUP_PATH
# install distacting_control package
pip install $GR_DOWNLOAD_DIR/
rm -rf $GR_DOWNLOAD_DIR
