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

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train
from params import params

def main(args):
  train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('noisy_speech_dir', # Gürültülü ses dosyalarının bulunduğu dizin
      help='directory containing noisy speech audio files')
  parser.add_argument('clean_speech_dir',   # Temiz ses dosyalarının bulunduğu dizin
      help='directory containing clean speech audio files')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--device_num', default=0, type=int, # Birden fazla gpu olmadi icin device_num 0 olacak
      help='train device number')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--restore_model_name', type = str, default=None, # Yüklemek için kontrol noktası dosya adi
      help='path to a checkpoint file to restore from')

  main(parser.parse_args())
