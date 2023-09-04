import os
import time
import math
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torchaudio
from torch.utils.data import Dataset, DataLoader

from config import Config
import inline
import scheduler
from model import ScoreMatchingSDE, ReverseSDE
from torch.nn.parallel import DistributedDataParallel

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, repeat=10):
        self.files = glob.glob("/mnt/vdd/music/**/*.wav", recursive=True)
        self.repeat = repeat

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self,idx):
        idx = idx % len(self.files)
        chunk_size = 512 * 512
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        offset = torch.randint(wav.shape[-1]-chunk_size, (1,))
        wav = wav[...,offset:offset+chunk_size]
        wav = (wav - wav.mean()) / wav.std().add(1e-5)
        return wav
        #return torch.where(r > 0, r.sqrt(), -(-r).add(1e-10).sqrt())

def main(config, model):

    dataset_root = "/mnt/vdd/music"
    
    if dist.is_initialized():
        if config["localrank"] == 0:
            dst = MusicDataset()
            dist.barrier()
        else:
            dist.barrier()
            dst = MusicDataset()
    
        st = torch.utils.data.distributed.DistributedSampler(dst, drop_last=True)
        dataloader_train = DataLoader(dst, batch_size=config["batch_size"], num_workers=config["num_workers"], sampler=st, pin_memory=True, persistent_workers=True)

    else:
        dst = MusicDataset()
        dataloader_train = DataLoader(dst, batch_size=config["batch_size"], num_workers=config["num_workers"], pin_memory=True, persistent_workers=True)
    
    nit = len(dataloader_train) * config["epochs"]
    opt = torch.optim.AdamW(model.parameters())
    sched_lr = scheduler.LR(opt, scheduler.Linear, start=1e-7, value=1e-4, final=1e-7, iterations=nit, warmup=100, name="lr")
    
    fwd_model = model.module if dist.is_initialized() else model
    rev_model = ReverseSDE(fwd_model)
    for start_epoch in range(config["epochs"],0,-1):
        path = os.path.join(config["output_path"], config["checkpoint_path"].format(epoch=start_epoch, **config))
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=config["device"])
            sd_model, sd_opt = checkpoint["model"], checkpoint["opt"]
            if sd_model:  
                rev_model.load_state_dict(sd_model)
            if sd_opt: 
                opt.load_state_dict(sd_opt)
            break
    sched_lr.step(start_epoch)

    for e in range(start_epoch, config["epochs"]):
        for training, dl in [(True, dataloader_train)]:
            torch.set_grad_enabled(training)
            model.train(training)
            if dl.sampler: dl.sampler.set_epoch(e)

            window = []
            for i, (batch) in enumerate(dl):
                batch = batch.to(config["device"])
                loss = model(batch)
                loss = loss.mean()

                if training:
                    opt.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                    opt.step()
                    sched_lr.step()

                if dist.is_initialized():
                    loss = loss.detach()
                    dist.all_reduce(loss, dist.ReduceOp.AVG)
                
                window.append(loss.item())
                if len(window) > 1000: window.pop(0)
                
                if config["localrank"] == 0:
                    print(f"E{e+1:03d} {i:06d}/{len(dl)} loss: {sum(window)/len(window):0.06f}", end="        \r")

            if config["localrank"] == 0:
                with torch.no_grad():
                    generated = rev_model.sde_sample_final()
                    torchaudio.save(f"output.{e+1:03d}.wav", generated[0].cpu()/4, 44100, encoding="PCM_F")
                    
                    path = os.path.join(config["output_path"], config["checkpoint_path"].format(epoch=e+1, **config))
                    checkpoint = dict(
                            model=rev_model.state_dict(),
                            opt=opt.state_dict(),
                            loss=sum(window)/len(window),
                            epoch=e+1)
                    torch.save(checkpoint, path)
            
                     

def run_distributed(rank):

    config = Config()
    localrank = rank % config["gpu_count"]
    torch.cuda.set_device("cuda:" + str(localrank))
    config["rank"] = rank
    config["localrank"] = localrank
    print(f"[{rank}] Waiting for peers...")
    dist.init_process_group(
            backend="nccl",
            init_method="tcp://" + config["dist_hostname"] + ":" + str(config["dist_port"]),
            rank=rank,
            world_size= config["dist_worldsize"])
    if localrank == 0:
        print("starting...")
    assert dist.is_initialized()
    model = DistributedDataParallel(ScoreMatchingSDE().cuda(), find_unused_parameters=False)
    main(config, model)

if __name__ == "__main__":
    config = Config()
    if config["device"] == "cpu":
        # CPU path
        model = ScoreMatchingSDE()
        main(config, model)

    elif config["gpu_count"] > 1:
        if config["dist_enabled"]:
            # DistributedDataParallel path
            mp.spawn(run_distributed, nprocs=min(config["dist_worldsize"], config["gpu_count"]))
        else:
            # DataParallel path
            model = DataParallel(ScoreMatchingSDE().cuda())
            main(config, model)
    else:
        # single GPU CUDA path 
        model = ScoreMatchingSDE().cuda()
        main(config, model)

