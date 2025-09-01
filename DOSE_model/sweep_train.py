from argparse import ArgumentParser, Namespace
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train
from params import params

import wandb

def main():
  args = Namespace(
    model_dir='/content/weights',
    train_noisy_speech_dir='/content/datasets/content/LKFS/train/noisy',
    train_clean_speech_dir='/content/datasets/content/LKFS/train/clean',
    val_noisy_speech_dir='/content/datasets/content/LKFS/validation/noisy',
    val_clean_speech_dir='/content/datasets/content/LKFS/validation/clean',
    max_epochs=10,
    device_num=0,
    fp16=True,
    restore_model_name="weights_fixed"
)
  train(args, params)


if __name__ == '__main__':
  main()
