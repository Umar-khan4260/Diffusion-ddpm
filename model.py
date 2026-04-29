# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Paste these classes directly from your notebook cells 4 & 6 ---
# SinusoidalPE, TimeEmb, ResBlock, AttnBlock, Down, Up, UNet, NoiseScheduler

class Config:
    IMAGE_SIZE    = 128
    CHANNELS      = 3
    T             = 300
    BETA_START    = 1e-4
    BETA_END      = 0.02
    NOISE_SCHEDULE= 'cosine'
    BASE_CHANNELS = 128
    TIME_EMB_DIM  = 256

cfg = Config()
