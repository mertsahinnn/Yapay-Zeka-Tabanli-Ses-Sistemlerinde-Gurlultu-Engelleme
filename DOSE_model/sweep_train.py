# Copyright 2020 LMNT, Inc. All Rights Reserved.
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
# ==============================================================================

from argparse import ArgumentParser, Namespace
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train
from params import params

import wandb

def main():
  args = Namespace(
    model_dir='/content/weights',
    train_noisy_speech_dir='/content/short-dataset/LFKS/train/noisy',
    train_clean_speech_dir='/content/short-dataset/LFKS/train/clean',
    val_noisy_speech_dir='/content/short-dataset/LFKS/validation/noisy',
    val_clean_speech_dir='/content/short-dataset/LFKS/validation/clean',
    max_steps=1000,
    device_num=0,
    fp16=True,
    restore_model_name=None
)
  train(args, params)


if __name__ == '__main__':
  main()
