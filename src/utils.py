"""Small helpers shared by every script: config loading, seeding, logging."""
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

LESIONS = ["EX", "HE", "MA", "SE"]  # channel order of the segmentation model
NUM_GRADES = 5                      # DR grades 0-4 (ungradable is dropped)
MAIN_METRIC = {"grading": "qwk", "segmentation": "mean_aupr"}  # picks the winner


def deep_update(base, new):
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:  # a run file only lists what differs from its base file
        cfg = deep_update(_read_yaml(path.parent / parent), cfg)
    return cfg


def load_config(path, overrides=(), seed=None):
    """Merge a run config over its base.yaml, then apply `a.b=value` overrides."""
    cfg = _read_yaml(path)
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    cfg["base_run"] = cfg["run_name"]
    if seed is not None and seed != cfg["train"]["seed"]:
        cfg["train"]["seed"] = seed
        cfg["run_name"] = f"{cfg['run_name']}_s{seed}"  # extra seeds get their own folder
    return cfg


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    """print() that also appends to <run>/train.log."""

    def __init__(self, path):
        self.file = open(path, "a")

    def __call__(self, msg):
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()
