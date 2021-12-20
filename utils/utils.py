import numpy as np
import random

import torch

NEPTUNE_API_TOKEN = #
NEPTUNE_PROJECT = #


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
