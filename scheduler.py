import math
import torch
import torch.nn as nn

# Example: scheduler = sched.LRScheduler(optimizer,  sched.Cosine, 0, 0.01, 0.0001, num_batches, warmup = 100)

class SchedulerFunc(object):
    def __init__(self, cb, start, value, final, iterations, warmup = 0, name=None):
        self.start = start
        self.value = value
        self.final = final
        self.iterations = iterations
        self.warmup = warmup
        self.name = name
        self.current = None
        self._step = 0
        self._cb = cb

    def step(self, increment=1, *args, **kwargs):
        self._step += increment
        multiplier = min(1, self._step / (self.warmup+1))
        progress = min(1, max((self._step - self.warmup) / max(self.iterations-self.warmup,1), 0))
        new_value = self._cb(progress, *args, **kwargs) * (self.final - self.value) + self.value
        new_value = multiplier * (new_value - self.start) + self.start
        self.current = new_value
        return self.current

class Linear(SchedulerFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: x, *args, **kwargs)

class Cosine(SchedulerFunc):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda x: (1-math.cos(x * math.pi)) * 0.5, *args, **kwargs)

class Exponential(SchedulerFunc):
    def __init__(self, *args, gamma=2, **kwargs):
        super().__init__(lambda x: (x ** gamma), *args, **kwargs)


class LR(object):
    def __init__(self, optimizer, scheduler, *args, **kwargs):
        self.scheduler = scheduler(*args, **kwargs)
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        v = self.scheduler.step(*args, **kwargs)
        for group in self.optimizer.param_groups:
            factor = group["lr_factor"] if "lr_factor" in group else 1
            group["lr"] = v * factor
        return v
    
    @property
    def name(self): return self.scheduler.name

    @property
    def current(self): return self.scheduler.current

    def get_last_lr(self):
        return [group["lr"] for group in self.optimmizer.param_groups]

class WD(object):
    def __init__(self, optimizer, scheduler, *args, **kwargs):
        self.scheduler = scheduler(*args, **kwargs)
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        v = self.scheduler.step(*args, **kwargs)
        for group in self.optimizer.param_groups:
            factor = group["wd_factor"] if "wd_factor" in group else 1
            group['weight_decay'] = v * factor
        return v

    @property
    def name(self): return self.scheduler.name

    @property
    def current(self): return self.scheduler.current

class Module(object):
    def __init__(self, module, scheduler, *args, callback=None, **kwargs):
        self.scheduler = scheduler(*args, **kwargs)
        self.module = module
        self.callback = callback

    def step(self, *args, **kwargs):
        v = self.scheduler.step(*args, **kwargs)
        for name,module in self.module.named_modules():
            if self.callback is not None:
                self.callback(name, module, v)
        return v
    
    @property
    def name(self): return self.scheduler.name

    @property
    def current(self): return self.scheduler.current

