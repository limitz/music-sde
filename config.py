import os
import glob
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import inline
import scheduler as sched
from collections.abc import Iterable

torch.manual_seed(13)
torch.cuda.manual_seed(13)

class Config(dict):
    def __init__(self):
        super().__init__(
            dict(
                input_path = "/mnt/vdd/music",
                output_path = "/mnt/vdd/working",
                epochs = 1000, 
                cpu_count = mp.cpu_count(),
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0,
                device = "cuda" if torch.cuda.is_available() else "cpu",
                num_workers=4,
                batch_size=4,
                dist_enabled = True,
                dist_worldsize = os.environ["DIST_WORLDSIZE"] if "DIST_WORLDSIZE" in os.environ else (torch.cuda.device_count() if torch.cuda.is_available() else 1),
                dist_port = os.environ["DIST_PORT"] if "DIST_PORT" in os.environ else 23456,
                dist_hostname = os.environ["DIST_HOSTNAME"] if "DIST_HOSTNAME" in os.environ else "localhost",
                rank = dist.get_rank() if dist.is_initialized() else None,
                localrank = dist.get_rank() % torch.cuda.device_count() if (torch.cuda.is_available() and dist.is_initialized()) else None
            ))
    
