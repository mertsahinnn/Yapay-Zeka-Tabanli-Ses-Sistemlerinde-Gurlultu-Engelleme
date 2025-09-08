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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x



class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=False):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    
    # unconda parametresini sinif ozniteligi olarak sakla
    self.uncond = uncond

    # Conditional mode icin spektrogram projection ekle - residual_channels boyutunda olmalı
    if not uncond:
      self.conditioner_projection = Conv1d(n_mels, residual_channels, 1)  # 2 * residual_channels değil!
    
    
    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner=None):

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    
    # conditional mode: spektrogram  bilgisini ekle
    if not self.uncond and conditioner is not None:
      y = y + self.conditioner_projection(conditioner)

    y = self.dilated_conv(y)
    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    
    
    return (x + residual) / sqrt(2.0), skip

class CompressConcat(nn.Module):
    def __init__(self):
        super(CompressConcat, self).__init__()
        # todo 修改维度,为数据的长度
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x, x_init):
        x_concat = (torch.cat([x, x_init], dim=1))
        return self.conv(x_concat)


class DOSE(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.version46_input_projection = Conv1d(2, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    self.compress_concat = CompressConcat()


    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)
  
  def forward(self, audio, diffusion_step, spectrogram=None):
    audio = audio.unsqueeze(1)  # [batch, 1, time]
    
    # Spektrogram boyutlarını kontrol et ve uygun şekilde yeniden boyutlandır
    processed_spectrogram = None
    if spectrogram is not None:
      if len(spectrogram.shape) == 3:  # [batch, n_mels, time_frames]
        # Mel-spektrogramı ses uzunluğuna göre interpolate et
        target_length = audio.shape[-1]  # ses uzunluğu
        
        # Spektrogramı ses uzunluğuna interpolate et
        spectrogram_interpolated = F.interpolate(
            spectrogram.unsqueeze(1), 
            size=(spectrogram.shape[1], target_length), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)  # [batch, n_mels, target_length]
        
        # ResidualBlock'lar için orijinal mel-spektrogramı kullan
        processed_spectrogram = spectrogram_interpolated
        
        # Audio concatenation için tek kanala indirgenen versiyonu
        spectrogram_for_concat = torch.mean(spectrogram_interpolated, dim=1, keepdim=True)  # [batch, 1, target_length]
        
      elif len(spectrogram.shape) == 2:  # [batch, time_frames]
        spectrogram_for_concat = spectrogram.unsqueeze(1)  # [batch, 1, time_frames]
        
        # Spektrogramı ses uzunluğuna interpolate et
        target_length = audio.shape[-1]
        if spectrogram_for_concat.shape[-1] != target_length:
          spectrogram_for_concat = F.interpolate(
              spectrogram_for_concat.unsqueeze(1), 
              size=target_length, 
              mode='linear', 
              align_corners=False
          ).squeeze(1)  # [batch, 1, target_length]
        
        # ResidualBlock'lar için de aynısını kullan (tek kanal olduğu için)
        processed_spectrogram = spectrogram_for_concat
    
    else:
      # Unconditional mode için noisy audio kullan
      spectrogram_for_concat = audio  # [batch, 1, time]
      processed_spectrogram = None
    
    x = torch.cat([audio, spectrogram_for_concat], dim=1)  # [batch, 2, time]
    x = self.version46_input_projection(x)
    
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    
    skip = None
    for layer in self.residual_layers:
      # ResidualBlock'a doğru boyuttaki spektrogram gönder
      x, skip_connection = layer(x, diffusion_step, processed_spectrogram)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x

