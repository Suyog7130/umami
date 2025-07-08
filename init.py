import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, kl_divergence

import matplotlib.pyplot as plt


from datetime import datetime